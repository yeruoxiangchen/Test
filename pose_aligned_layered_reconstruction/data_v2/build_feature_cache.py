"""Materialize paired DINO caches from rendered or composited scene records."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ..paired_encoder import build_frozen_dino_encoder, encode_paired_views
from ..paired_preprocessing import make_masked_rgb
from ..paired_dataset import sha256_file
from ..v2.contracts import SAMPLE_FORMAT, validate_sample
from ..v2.dataset import write_manifest


@torch.no_grad()
def cache_scene(scene_file, ss_target, output, encoder, resume=False):
    scene_file, output = Path(scene_file).resolve(), Path(output)
    receipt_file = Path(str(output) + '.receipt.json')
    if output.exists():
        if not resume or not receipt_file.exists():
            raise FileExistsError(f'cache exists without a reusable receipt: {output}')
        receipt = json.loads(receipt_file.read_text())
        if (receipt['scene_sha256'] != sha256_file(scene_file)
                or receipt['ss_sha256'] != sha256_file(ss_target)
                or receipt['row']['sha256'] != sha256_file(output)):
            raise ValueError('resume cache identity mismatch')
        cached = torch.load(output, map_location='cpu', weights_only=True)
        for binding in cached['source_files']:
            if sha256_file(binding['path']) != binding['sha256']:
                raise ValueError('resume cache source changed')
        return receipt['row']
    meta = json.loads(scene_file.read_text())
    root = scene_file.parent
    raw, masked, visible, amodal = [], [], [], []
    bindings = []
    for view in meta['views']:
        rgb = np.asarray(Image.open(root / view['raw']).convert('RGB'))
        mask = np.asarray(Image.open(root / view['visible']).convert('L')).astype(np.float32) / 255.
        if rgb.shape != (518, 518, 3) or mask.shape != (518, 518):
            raise ValueError('scene sources must be 518x518 with already transformed K')
        raw.append(Image.fromarray(rgb))
        masked.append(Image.fromarray(make_masked_rgb(rgb, mask)))
        visible.append(mask)
        amodal.append(np.asarray(Image.open(root / view['amodal']).convert('L')).astype(np.float32) / 255.)
        bindings.extend({'path': str(root / view[k]), 'sha256': sha256_file(root / view[k])}
                        for k in ('raw', 'visible', 'amodal'))
    features = encode_paired_views(encoder, raw_images=raw, masked_images=masked)
    with np.load(root / meta['camera'], allow_pickle=False) as camera:
        k, extrinsics = torch.from_numpy(camera['intrinsics'].copy()).float(), torch.from_numpy(camera['extrinsics'].copy()).float()
    with np.load(root / meta['layers'], allow_pickle=False) as data:
        labels = {key: torch.from_numpy(np.asarray(data[key]).copy()) for key in data.files}
    with np.load(ss_target, allow_pickle=False) as data:
        # Explicit bridge for official z archives and the V2 target schema.
        key = 'target' if 'target' in data else 'z'
        target = torch.from_numpy(np.asarray(data[key], np.float32).reshape(8, 16, 16, 16).copy())
        coords = torch.from_numpy(np.asarray(data['official_coords'], np.int32).copy())
        roundtrip = float(data['roundtrip_iou'])
    if not 0 < roundtrip <= 1:
        raise ValueError('invalid SS round-trip audit')
    sample = {'format': SAMPLE_FORMAT, **{k: meta[k] for k in ('uid', 'object_uid', 'scene_uid', 'trajectory_uid', 'data_domain')},
        'depth_kind': 'camera_forward_z', 'image_size': (518, 518), 'grid_transform': 'identity',
        'intrinsics': k, 'extrinsics': extrinsics, 'extrinsics_type': meta['extrinsics_type'],
        'camera_forward_sign': float(meta['camera_forward_sign']), 'layer_camera_y_sign': float(meta['layer_camera_y_sign']),
        'raw_visual_patch_features': features['raw_visual_patch_features'].half().cpu(),
        'masked_visual_patch_features': features['masked_visual_patch_features'].half().cpu(),
        'visible_target_mask': torch.from_numpy(np.stack(visible)).half(),
        'amodal_target_mask_gt': torch.from_numpy(np.stack(amodal)).half(),
        'target': target, 'target_coords': coords, 'ss_roundtrip_iou': roundtrip,
        'paired_encoding_contract': features['paired_encoding_contract'],
        'source_to_feature_affines': torch.eye(3).repeat(len(raw), 1, 1),
        'label_geometry_kind': meta['label_geometry_kind'], 'renderer_binding': meta.get('renderer'),
        'source_binding': meta.get('legacy_source_binding', {}),
        'source_files': bindings + [{'path': str(scene_file), 'sha256': sha256_file(scene_file)},
            {'path': str(root / meta['camera']), 'sha256': sha256_file(root / meta['camera'])},
            {'path': str(root / meta['layers']), 'sha256': sha256_file(root / meta['layers'])}],
        'ss_target_binding': {'path': str(Path(ss_target).resolve()), 'sha256': sha256_file(ss_target)},
        'recipe_hash': meta['recipe_hash'], 'crop_kind': meta['crop_kind'], **labels}
    if 'object_from_asset' in meta:
        sample['object_from_asset'] = meta['object_from_asset']
    if 'target_mesh' in meta:
        mesh_path = root / meta['target_mesh']
        sample['target_mesh_binding'] = {'path': str(mesh_path), 'sha256': sha256_file(mesh_path)}
    sample.pop('visible_target_mask_gt', None)
    validate_sample(sample, require_labels=True, require_ss_target=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f'.{os.getpid()}.tmp')
    torch.save(sample, temporary)
    os.replace(temporary, output)
    row = {'uid': sample['uid'], 'object_uid': sample['object_uid'], 'scene_uid': sample['scene_uid'],
           'sample': str(output.resolve()), 'sha256': sha256_file(output)}
    receipt_file.write_text(json.dumps({'row': row, 'scene_sha256': sha256_file(scene_file),
        'ss_sha256': sha256_file(ss_target)}, indent=2))
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes', nargs='+', required=True, help='scene.json paths')
    p.add_argument('--ss_targets', nargs='+', required=True, help='one SS npz per scene, or one shared target')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if len(a.ss_targets) not in (1, len(a.scenes)):
        raise ValueError('SS target count differs')
    root = Path(a.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    encoder = build_frozen_dino_encoder(device=torch.device(a.device))
    rows = [cache_scene(path, a.ss_targets[0 if len(a.ss_targets) == 1 else i], root / f'sample_{i:06d}.pt', encoder)
            for i, path in enumerate(a.scenes)]
    write_manifest(root / 'manifest.json', rows, {'sources': list(map(str, a.scenes)), 'paired_encoder': 'dinov2_vitl14_reg'}, root)


if __name__ == '__main__':
    main()
