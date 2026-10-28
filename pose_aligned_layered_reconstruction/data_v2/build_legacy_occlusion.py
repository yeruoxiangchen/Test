"""Same-camera, fixed-3D-occluder augmentation of official paired records."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from ..paired_dataset import canonical_json_sha256, sha256_file
from ..paired_encoder import build_frozen_dino_encoder
from ..prepare_official_experiment import paired_pixels
from ..frozen_slat import load_frozen_slat30k
from ..v2.dataset import SceneDataset
from ..v2.ray_geometry import CameraBundle
from .render_scenes import pack_scene, rasterize, box_occluders
from .build_feature_cache import cache_scene
from ..v2.dataset import write_manifest


def accept_visibility(fractions, policy):
    """Select scene difficulty using visibility only, never reconstruction scores."""
    if policy is None:
        return True
    if (not 1 <= policy['min_strong_views'] <= len(fractions)
            or not 1 <= policy['min_nonempty_views'] <= len(fractions)
            or not 0 < policy['strong_threshold'] < 1
            or not 0 < policy['min_mean_visible'] <= 1 or policy['max_attempts'] < 1):
        raise ValueError('invalid visibility selection policy')
    if not torch.isfinite(fractions).all() or ((fractions < 0) | (fractions > 1)).any():
        raise ValueError('invalid visible fractions')
    return (int((fractions < policy['strong_threshold']).sum()) >= policy['min_strong_views']
            and float(fractions.mean()) >= policy['min_mean_visible']
            and int((fractions > .02).sum()) >= policy['min_nonempty_views'])


def merge_stacks(target, other, layers):
    values = torch.cat((target['layer_depths_gt'], other['layer_depths_gt']), 1)
    owner = torch.cat((torch.ones_like(target['layer_target_gt']), torch.zeros_like(other['layer_target_gt'])), 1)
    values = torch.where(values > 0, values, torch.inf)
    values, order = values.sort(1)
    owner = owner.gather(1, order)
    distinct = torch.isfinite(values)
    distinct[:, 1:] &= (values[:, 1:] - values[:, :-1]) > 1.e-5
    values = torch.where(distinct, values, torch.inf)
    values, order = values.sort(1)
    owner = owner.gather(1, order)
    valid = torch.isfinite(values[:, :layers])
    return {'layer_depths_gt': torch.where(valid, values[:, :layers], 0),
        'layer_target_gt': owner[:, :layers] * valid, 'ray_stop_index_gt': valid.sum(1),
        'ray_overflow_gt': torch.isfinite(values[:, layers])}


@torch.no_grad()
def augment(sample, root, runtime, seed=0, box_count=2, layers=5, visibility_policy=None, cached_target=None):
    import trimesh
    from trellis.modules import sparse as sp
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    device = next(runtime['decoder'].parameters()).device
    source = sample['source_binding']
    for key in ('cache', 'ss', 'slat'):
        if sha256_file(source[key]) != source[key + '_sha256']:
            raise ValueError(f'legacy source changed: {key}')
    cached = torch.load(source['cache'], map_location='cpu', weights_only=False)
    raw, _, original_masks, audit = paired_pixels(source, cached)
    if cached_target is not None:
        if (cached_target['source_slat_sha256'] != source['slat_sha256']
                or cached_target['decoder'] != runtime['binding']['decoder']
                or sha256_file(cached_target['path']) != cached_target['sha256']):
            raise ValueError('cached target source/decoder/content mismatch')
        mesh = trimesh.load(cached_target['path'], process=False, force='mesh')
    else:
        with np.load(source['slat'], allow_pickle=False) as data:
            xyz = torch.from_numpy(np.asarray(data['coords'], np.int32)).to(device)
            feats = torch.from_numpy(np.asarray(data['feats'], np.float32)).to(device)
        sparse = sp.SparseTensor(feats=feats, coords=torch.cat((torch.zeros_like(xyz[:, :1]), xyz), 1))
        decoded = runtime['decoder'](sparse)[0]
        mesh = trimesh.Trimesh(decoded.vertices.cpu().numpy(), decoded.faces.cpu().numpy(), process=False)
    target = pack_scene([mesh], [], device)
    camera = CameraBundle.from_sample(sample, device)
    kwargs = {'image_size': camera.image_size, 'forward_sign': camera.forward_sign}
    target_first = rasterize(target, camera.intrinsics, camera.w2c, layers=1, **kwargs)
    original = original_masks.float() > .5
    td = target_first['layer_depths_gt'][:, 0]
    attempts = []
    max_attempts = visibility_policy['max_attempts'] if visibility_policy else 1
    for attempt in range(max_attempts):
        candidate_seed = seed + attempt * 10000019
        other = pack_scene([], box_occluders(candidate_seed, box_count), device)
        other_first = rasterize(other, camera.intrinsics, camera.w2c, layers=1, rgb=True, **kwargs)
        od = other_first['layer_depths_gt'][:, 0]
        occludes = (od > 0) & ((td == 0) | (od < td))
        visible = original & ~occludes
        fractions = visible.sum((1, 2)) / original.sum((1, 2)).clamp_min(1)
        accepted = accept_visibility(fractions, visibility_policy)
        attempts.append({'seed': candidate_seed, 'visible_fraction': fractions.tolist(), 'accepted': accepted})
        if accepted:
            break
    else:
        (root/'rejected_candidates.json').write_text(json.dumps(attempts, indent=2))
        raise RuntimeError(f'visibility quota not reached for {sample["object_uid"]}; object must not be dropped')
    target_layers = rasterize(target, camera.intrinsics, camera.w2c, raster_side=37, layers=layers + 1, **kwargs)
    other_layers = rasterize(other, camera.intrinsics, camera.w2c, raster_side=37, layers=layers + 1, **kwargs)
    labels = merge_stacks(target_layers, other_layers, layers)
    proxy = target_first['ray_stop_index_gt'] > 0
    td, od = target_first['layer_depths_gt'][:, 0], other_first['layer_depths_gt'][:, 0]
    # Outside original alpha the background is behind the rendered occluder.
    occludes = (od > 0) & ((td == 0) | (od < td))
    visible = original & ~occludes
    rows = []
    for i, image in enumerate(raw):
        rgb = torch.from_numpy(np.asarray(image).copy()).float() / 255.
        rgb = torch.where(occludes[i, ..., None], other_first['rgb'][i], rgb)
        Image.fromarray((rgb.numpy() * 255).round().astype(np.uint8)).save(root / f'raw_{i:02d}.png')
        Image.fromarray((visible[i].numpy() * 255).astype(np.uint8)).save(root / f'visible_{i:02d}.png')
        Image.fromarray((original[i].numpy() * 255).astype(np.uint8)).save(root / f'amodal_{i:02d}.png')
        rows.append({'raw': f'raw_{i:02d}.png', 'visible': f'visible_{i:02d}.png', 'amodal': f'amodal_{i:02d}.png'})
    disagreement = (original != proxy).float()[:, None]
    # Dilate silhouette conflicts and exclude uncertain target/occluder ordering.
    uncertain = F.max_pool2d(disagreement, 9, stride=1, padding=4)[:, 0] > 0
    uncertain |= original & (od > 0) & ((td - od).abs() < .03)
    uncertainty37 = F.adaptive_max_pool2d(uncertain.float()[:, None], (37, 37))[:, 0] > 0
    labels['supervision_valid_mask'] = ~uncertainty37
    np.savez_compressed(root / 'layers.npz', **{k: v.numpy() for k, v in labels.items()})
    # Store original appearance intrinsics; layer_camera_y_sign remains explicit.
    np.savez_compressed(root / 'camera.npz', intrinsics=sample['intrinsics'].numpy(), extrinsics=sample['extrinsics'].numpy())
    recipe = {'seed': seed, 'box_count': box_count, 'layers': layers, 'source_uid': sample['uid']}
    if visibility_policy:
        recipe.update(visibility_policy=visibility_policy, accepted_seed=candidate_seed, selection_attempts=attempts)
    if cached_target is not None:
        recipe['target_mesh_cache'] = cached_target
    meta = {'format': 'posed_scene_source.v2', 'uid': sample['uid'] + f':occlusion:{seed}',
        'object_uid': sample['object_uid'], 'scene_uid': sample['uid'] + f':scene:{seed}',
        'trajectory_uid': sample['uid'] + ':original', 'data_domain': 'legacy_composite',
        'views': rows, 'camera': 'camera.npz', 'layers': 'layers.npz',
        'camera_forward_sign': float(sample['camera_forward_sign']), 'extrinsics_type': sample['extrinsics_type'],
        'layer_camera_y_sign': float(sample.get('layer_camera_y_sign', 1.)), 'depth_kind': 'camera_forward_z',
        'label_geometry_kind': 'target_vae_proxy', 'composition_kind': 'proxy_depth_composite',
        'legacy_source_binding': source, 'recipe': recipe, 'recipe_hash': canonical_json_sha256(recipe),
        'renderer': 'nvdiffrast_proxy_depth_composite', 'crop_kind': 'inherited_original_crop',
        'visible_fraction': (visible.sum((1, 2)) / original.sum((1, 2)).clamp_min(1)).tolist(),
        'supervision_valid_fraction': float((~uncertainty37).float().mean()), 'pixel_audit': audit}
    (root / 'scene.json').write_text(json.dumps(meta, indent=2))
    return root / 'scene.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--max_objects', type=int, default=32)
    p.add_argument('--variants', type=int, default=4)
    p.add_argument('--seed', type=int, default=20261017)
    p.add_argument('--box_count', type=int, default=2)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if min(a.max_objects, a.variants, a.box_count) < 1:
        raise ValueError('counts must be positive')
    device = torch.device('cuda:0' if a.device == 'cuda' else a.device)
    torch.cuda.set_device(device)
    dataset = SceneDataset(a.manifest)
    root = Path(a.output_dir).resolve(); root.mkdir(parents=True, exist_ok=False)
    runtime = load_frozen_slat30k(device)
    encoder = build_frozen_dino_encoder(device=device)
    rows = []
    for i in range(min(a.max_objects, len(dataset))):
        sample = dataset[i]
        for j in range(a.variants):
            folder = root / f'object_{i:05d}_variant_{j:02d}'
            scene_file = augment(sample, folder, runtime, a.seed + i * 1009 + j, a.box_count,
                                 sample['layer_depths_gt'].shape[1])
            rows.append(cache_scene(scene_file, sample['source_binding']['ss'], folder / 'sample.pt', encoder))
    write_manifest(root / 'manifest.json', rows, {'kind': 'legacy_proxy_occlusion', 'seed': a.seed,
        'source_manifest': dataset.identity(), 'variants': a.variants}, root)


if __name__ == '__main__':
    main()
