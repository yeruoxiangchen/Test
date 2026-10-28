import gc
import time

import numpy as np
import torch

from .common import OUT, ARMS, SEEDS, ROOT, read, write, binding, verify, folder, emit
from .data import object_data, input_for, input_gallery
from .runtime import setup, load_ss, sample_ss, load_slat, mesh_from_support
from .render import render, mask_iou


def result_path(index, arm, seed):
    return folder(index) / 'results' / arm / f'{seed}.json'


def support_path(index, arm, seed):
    return folder(index) / 'results' / arm / f'{seed}_ss.npz'


def verify_result(path):
    record = read(path)
    for key in ('mesh', 'coords', 'gt'):
        verify(record[key])
    return record


@torch.no_grad()
def score(index, arm, seed, runtime, coords, sample, gt, gt_masks, obs, context):
    import trimesh
    from pose_aligned_layered_reconstruction.mesh_metrics import surface_metrics, structure_metrics
    from pose_aligned_layered_reconstruction.evaluate_ss_occupancy import occupancy_metrics
    path = result_path(index, arm, seed)
    if path.exists():
        return verify_result(path)
    start = time.monotonic()
    mesh_path = path.with_suffix('.obj')
    mesh, cfg = mesh_from_support(runtime, sample, coords, index, seed, mesh_path)
    # Score the exported, reviewable OBJ, not an unexported tensor.
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    surface = surface_metrics(mesh, gt, count=50000, seed=20261025)
    _, masks = render(mesh, obs['intrinsics'], obs['extrinsics'], side=128, context=context)
    ious = mask_iou(masks, gt_masks)
    rec = read(folder(index) / 'object.json')
    with np.load(verify(rec['gt_coords'])) as data:
        target_coords = data['official_coords']
    record = {'index': index, 'uid': rec['uid'], 'arm': arm, 'seed': seed, 'success': True,
        'surface': surface, 'structure': structure_metrics(mesh),
        'occupancy': occupancy_metrics(coords, target_coords),
        'silhouette': {'all8': float(ious.mean()), 'source2': float(ious[:2].mean()),
                       'remaining6': float(ious[2:].mean()), 'per_view': ious.tolist()},
        'mesh': binding(mesh_path), 'coords': binding(support_path(index, arm, seed)), 'gt': rec['gt'],
        'ss_context_tokens': sample['stock_condition'].shape[1], 'views': len(sample['intrinsics']),
        'elapsed_slat_decode_score_seconds': time.monotonic() - start, 'cfg': cfg,
        'privileged_gt_inputs': arm == 'gt_render6'}
    write(path, record)
    emit(index, 'mesh_complete', arm=arm, seed=seed, cd=surface['chamfer_l1'], f02=surface['fscore_0p02'])
    return record


@torch.no_grad()
def make_render_features(index, obs, gt, context, device):
    import trimesh
    from pose_aligned_layered_reconstruction.paired_encoder import build_frozen_dino_encoder, _encode_patch_tokens
    path = folder(index) / 'render_features.pt'
    if path.exists():
        verify(read(path.with_suffix('.json'))['features'])
        return
    pipeline = build_frozen_dino_encoder(device=device)
    result, audit = {}, {}
    for key, mesh in [('gt', gt), *[(key, trimesh.load(verify_result(result_path(index, 'real2', seed))['mesh']['path'],
                                    force='mesh', process=False)) for key, seed in [('a', 42), ('b', 43)]]]:
        images, masks = render(mesh, obs['intrinsics'], obs['extrinsics'], side=518, context=context)
        root = folder(index) / 'renders' / key
        root.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            image.save(root / f'{i:02d}.png')
        np.savez_compressed(root / 'masks.npz', masks=masks)
        tokens = []
        for start in range(0, 8, 2):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                tokens.append(_encode_patch_tokens(pipeline, images[start:start + 2]).half().cpu())
        result[key] = torch.cat(tokens)
        audit[key] = {'images': [binding(root / f'{i:02d}.png') for i in range(8)],
                      'masked_pixels': [int(m.sum()) for m in masks]}
    torch.save(result, path)
    write(path.with_suffix('.json'), {'features': binding(path), 'renders': audit,
          'encoding': 'same frozen dinov2_vitl14_reg; encode_image default feature LayerNorm, patches 5:; no per-image crop'})
    del pipeline
    gc.collect(); torch.cuda.empty_cache()


@torch.no_grad()
def run(index):
    import trimesh
    import nvdiffrast.torch as dr
    from pose_aligned_layered_reconstruction.experiments.hssd_operator_audit_20261025.kernel_policy import FixedNative
    rec, obs = object_data(index)
    protocol = read(OUT / 'protocol.json')
    for source in protocol['sources'] + protocol['dependency_sources']:
        verify(source)
    device = setup()
    gt = trimesh.load(verify(rec['gt']), force='mesh', process=False)
    context = dr.RasterizeCudaContext(device=device)
    _, gt_masks = render(gt, obs['intrinsics'], obs['extrinsics'], side=128, context=context)
    from PIL import Image
    _, full_masks = render(gt, obs['intrinsics'], obs['extrinsics'], side=518, context=context)
    source_masks = np.stack([np.asarray(Image.open(verify(m)).convert('L')) > 127 for m in rec['masks']])
    camera_iou = mask_iou(full_masks, source_masks)
    write(folder(index) / 'camera_audit.json', {'per_view_iou': camera_iou.tolist(),
          'minimum': float(camera_iou.min()), 'pass_threshold': .90,
          'passed': bool((camera_iou >= .90).all()), 'pose_fit': False})
    if (camera_iou < .90).any():
        raise ValueError(f'Independent renderer/camera audit failed: {camera_iou.tolist()}')
    try:
        with FixedNative() as policy:
            emit(index, 'donor_ss')
            runtime = load_ss(device)
            write(folder(index) / 'ss_binding.json', runtime['binding'])
            sample = input_for(index, 'real2')
            for seed in SEEDS:
                path = support_path(index, 'real2', seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    sample_ss(runtime, sample, index, seed, path)
            del runtime
            gc.collect(); torch.cuda.empty_cache()
            emit(index, 'donor_meshes')
            slat = load_slat(device)
            write(folder(index) / 'slat_binding.json', slat['binding'])
            for seed in SEEDS:
                with np.load(support_path(index, 'real2', seed)) as data:
                    coords = data['coords']
                score(index, 'real2', seed, slat, coords, sample, gt, gt_masks, obs, context)
            del slat
            gc.collect(); torch.cuda.empty_cache()
            emit(index, 'render_and_encode_donors')
            make_render_features(index, obs, gt, context, device)
            input_gallery(index)
            emit(index, 'all_arms_ss')
            runtime = load_ss(device)
            for arm in ARMS[1:]:
                sample = input_for(index, arm)
                for seed in SEEDS:
                    path = support_path(index, arm, seed)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not path.exists():
                        sample_ss(runtime, sample, index, seed, path)
                        emit(index, 'ss_complete', arm=arm, seed=seed)
            del runtime
            gc.collect(); torch.cuda.empty_cache()
            emit(index, 'all_arms_mesh')
            slat = load_slat(device)
            for arm in ARMS[1:]:
                sample = input_for(index, arm)
                for seed in SEEDS:
                    with np.load(support_path(index, arm, seed)) as data:
                        coords = data['coords']
                    score(index, arm, seed, slat, coords, sample, gt, gt_masks, obs, context)
                    torch.cuda.empty_cache()
            del slat
            gc.collect(); torch.cuda.empty_cache()
            from .report import mesh_gallery
            mesh_gallery(index)
            write(folder(index) / 'kernel_policy.json', policy.manifest())
            emit(index, 'complete', records=len(ARMS) * len(SEEDS))
    except BaseException as e:
        emit(index, 'failed', error=str(e))
        raise
