import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from .common import OUT, SOURCE, ARMS, SEEDS, ROOT, read, write, binding, verify, lock_json, folder

OBS_KEYS = ('intrinsics', 'extrinsics', 'image_size', 'grid_transform',
            'extrinsics_type', 'camera_forward_sign')


def select_camera_order(sample):
    if sample['extrinsics_type'] != 'w2c':
        raise ValueError('This diagnostic expects the frozen HSSD w2c contract')
    c2w = np.linalg.inv(sample['extrinsics'].numpy())
    directions = c2w[:, :3, 3].copy()
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    second = int(np.argmin(directions @ directions[0]))
    if second == 0:
        raise ValueError('No distinct second camera')
    return [0, second] + [i for i in range(8) if i not in (0, second)]


def prepare():
    source = read(SOURCE)
    from pose_aligned_reconstruction.current_30k import SS30K_CHECKPOINT, SLAT30K_CHECKPOINT
    rows = source['samples'][:8]
    sources = [binding(p) for p in sorted(ROOT.glob('*.py'))] + [binding(ROOT / 'run.sh'), binding(ROOT / 'PROTOCOL.md')]
    external = ['pose_aligned_layered_reconstruction/frozen_slat.py',
                'pose_aligned_layered_reconstruction/dino_context.py',
                'pose_aligned_layered_reconstruction/paired_encoder.py',
                'pose_aligned_layered_reconstruction/mesh_metrics.py',
                'pose_aligned_layered_reconstruction/experiments/hssd_operator_audit_20261025/kernel_policy.py',
                'pose_aligned_layered_reconstruction/experiments/round2_v2/memory_efficient_mesh.py',
                'pose_aligned_reconstruction/native_ss_genrecon.py']
    protocol = {'format': 'visual_evidence_probe.v1', 'source': binding(SOURCE), 'rows': rows,
                'indices': list(range(8)), 'arms': list(ARMS), 'seeds': list(SEEDS),
                'ss_checkpoint': binding(SS30K_CHECKPOINT.path), 'slat_checkpoint': binding(SLAT30K_CHECKPOINT.path),
                'sources': sources, 'dependency_sources': [binding(ROOT.parent / p) for p in external],
                'scope': 'eight inspected rendered HSSD objects, exploratory only',
                'gt_frame': 'original native canonical; no alignment', 'confidence': 'not measured'}
    lock_json(OUT / 'protocol.json', protocol)
    for index, row in enumerate(rows):
        sample = torch.load(verify({'path': row['sample'], 'sha256': row['sha256']}), map_location='cpu', weights_only=False)
        order = select_camera_order(sample)
        gt = binding(verify(sample['target_mesh_binding']))
        source_dir = Path(row['sample']).parent
        record = {'index': index, 'uid': row['object_uid'], 'source': row, 'camera_order': order,
                  'gt': gt, 'gt_coords': binding(verify(sample['ss_target_binding'])),
                  'images': [binding(source_dir / f'input_cycles/clean_{i:02d}.png') for i in order],
                  'masks': [binding(source_dir / f'clean_mask_{i:02d}.png') for i in order]}
        lock_json(folder(index) / 'object.json', record)
    write(OUT / 'status.json', {'status': 'prepared', 'objects': 8, 'expected_meshes': 8 * len(ARMS) * len(SEEDS)})
    return protocol


def object_data(index):
    rec = read(folder(index) / 'object.json')
    row = rec['source']
    source = torch.load(verify({'path': row['sample'], 'sha256': row['sha256']}), map_location='cpu', weights_only=False)
    order = rec['camera_order']
    features = source['masked_visual_patch_features'][order].float()
    observations = {key: source[key] for key in OBS_KEYS}
    observations['intrinsics'] = source['intrinsics'][order]
    observations['extrinsics'] = source['extrinsics'][order]
    observations['masked_visual_patch_features'] = features
    return rec, observations


def input_for(index, arm):
    from pose_aligned_layered_reconstruction.dino_context import build_masked_dino_contexts
    rec, obs = object_data(index)
    f = obs['masked_visual_patch_features']
    if arm == 'real2':
        inds = [0, 1]
    elif arm == 'repeat8':
        inds = [0, 1] * 4
    else:
        inds = list(range(8))
    if arm in ('real2', 'repeat8', 'real8'):
        features = f[inds]
    else:
        cache = torch.load(folder(index) / 'render_features.pt', map_location='cpu', weights_only=False)
        if arm == 'gt_render6':
            extra = cache['gt'][2:]
        elif arm in ('donor_a6', 'donor_b6'):
            extra = cache['a' if arm == 'donor_a6' else 'b'][2:]
        else:
            offset = 0 if arm == 'mixed_ab6' else 1
            extra = torch.stack([cache['a' if (i + offset) % 2 == 0 else 'b'][i + 2] for i in range(6)])
        features = torch.cat((f[:2], extra.float()))
    sample = {**obs, 'intrinsics': obs['intrinsics'][inds], 'extrinsics': obs['extrinsics'][inds]}
    sample.update(build_masked_dino_contexts(features))
    sample['visual_patch_features'] = features
    # Explicit inference allowlist: no GT masks, depth, mesh or target latent.
    allowed = {*OBS_KEYS, 'masked_visual_patch_features', 'visual_patch_features', 'stock_condition',
               'slat_condition', 'context_contract'}
    return {key: value for key, value in sample.items() if key in allowed}


def image_for(index, arm, pos):
    rec = read(folder(index) / 'object.json')
    if arm in ('real2', 'repeat8', 'real8') or pos < 2:
        p = pos % 2 if arm == 'repeat8' else pos
        im = Image.open(verify(rec['images'][p])).convert('RGB')
        mask = Image.open(verify(rec['masks'][p])).convert('L')
        return Image.composite(im, Image.new('RGB', im.size), mask)
    key = {'gt_render6': 'gt', 'donor_a6': 'a', 'donor_b6': 'b'}.get(arm)
    if key is None:
        offset = 0 if arm == 'mixed_ab6' else 1
        key = 'a' if (pos - 2 + offset) % 2 == 0 else 'b'
    return Image.open(folder(index) / 'renders' / key / f'{pos:02d}.png').convert('RGB')


def input_gallery(index):
    size, header = 160, 26
    sheet = Image.new('RGB', (8 * size + 150, len(ARMS) * (size + header)), 'white')
    d = ImageDraw.Draw(sheet)
    for row, arm in enumerate(ARMS):
        y = row * (size + header)
        d.text((5, y + 50), arm, fill='black')
        for pos in range(2 if arm == 'real2' else 8):
            im = ImageOps.contain(image_for(index, arm, pos), (size, size))
            x = 150 + pos * size
            sheet.paste(im, (x, y + header))
            d.text((x + 4, y + 4), str(pos), fill='black')
    path = folder(index) / 'inputs.jpg'
    sheet.save(path, quality=90)
    return path
