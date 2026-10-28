"""Frozen SLat feature repair via the decoder's differentiable corner-SDF field.

Only input-supported constraints are used by automatic/random. GT arms remain
privileged diagnostics. No dense FlexiCubes extraction is differentiated: its
corner-SDF is a proxy, and actual decoded meshes are independently evaluated.
"""
import argparse
import gc
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from visual_evidence_probe.experiments.local_repair import config

LATENT_PARAMS = dict(steps=20, lr=.04, normalized_feature_cap=.15, token_radius_voxels=2.,
    checkpoints=[0, 10, 20], field_preservation_weight=.2, feature_prior_weight=.001,
    source='local_geometry_repair_cropped_20261026', field='original decoder averaged corner SDF + original -1/256 bias',
    objective='zero SDF at fixed local target points + field preservation outside selected token support',
    mesh_acceptance='same heldout-from-local-search input NCC and silhouette gate as mesh round')
MESH_SOURCE = config.STORE / 'outputs' / LATENT_PARAMS['source']
OUT = config.STORE / 'outputs/local_slat_repair_20261026'


def field(decoder, z, cache):
    from trellis.models.structured_latent_vae.base import SparseTransformerBase
    from trellis.representations.mesh.utils_cube import construct_voxel_grid
    h = SparseTransformerBase.forward(decoder, z)
    for block in decoder.upsample:
        h = block(h)
    h = decoder.out_layer(h.type(z.dtype))
    if 'coords' not in cache:
        positions, cubes = construct_voxel_grid(h.coords[:, 1:])
        count = torch.bincount(cubes.reshape(-1), minlength=len(positions)).to(h.feats.dtype).clamp_min(1)
        cache.update(coords=h.coords.clone(), positions=positions, cubes=cubes, count=count)
    elif not torch.equal(cache['coords'], h.coords):
        raise ValueError('decoder sparse support changed')
    # Exactly original cube mean of the 8 sdf channels, including bias.
    values = h.feats[:, :8] + decoder.mesh_extractor.sdf_bias
    initial = values.new_zeros(len(cache['positions']))
    reduced = torch.scatter_reduce(initial, 0, cache['cubes'].reshape(-1), values.reshape(-1),
        reduce='mean', include_self=False)
    return reduced


def sdf_at(values, positions, points, res=256):
    # Dense scalar field only (67MB), rather than dense geometry, colors and FlexiCubes autograd.
    side = res + 1
    flat = (positions[:, 0].long() * side + positions[:, 1].long()) * side + positions[:, 2].long()
    dense = values.new_ones(side ** 3).index_copy(0, flat, values).reshape(1, 1, side, side, side)
    grid = (points[:, [2, 1, 0]] * 2).reshape(1, 1, 1, -1, 3)
    return F.grid_sample(dense, grid, align_corners=True, padding_mode='border').reshape(-1)


def input_sample(index):
    from pose_aligned_layered_reconstruction.dino_context import build_masked_dino_contexts
    r = config.read(config.BASE / 'objects' / f'{index:03d}' / 'observations.json')
    cache = torch.load(config.BASE / 'objects' / f'{index:03d}' / 'features.pt', map_location='cpu', weights_only=False)
    features = cache['features'].float()
    return dict(intrinsics=cache['intrinsics'], extrinsics=torch.tensor(r['extrinsics'], dtype=torch.float32),
        image_size=(518, 518), grid_transform='identity', extrinsics_type='w2c', camera_forward_sign=1.,
        **build_masked_dino_contexts(features), visual_patch_features=features)


def register():
    path = OUT / 'protocol.json'
    value = dict(format='local_slat_repair.v1', params=LATENT_PARAMS,
        source=dict(path=str(Path(__file__).resolve()), sha256=config.digest(__file__)),
        mesh_protocol=dict(path=str(MESH_SOURCE / 'protocol.json'), sha256=config.digest(MESH_SOURCE / 'protocol.json')),
        no_training=True, model='frozen paper SLat30K EMA and stock mesh decoder',
        grad_proxy='SDF field ignores small decoder vertex deformation and cube weights; actual mesh not the proxy is the endpoint.',
        original_ss_coordinates_fixed=True, nonselected_feature_rows_frozen=True,
        similarity='reuse same existing input-only global fit; no GT realignment',
        all_runtime_io=str(config.STORE), scope='same inspected exploratory 8 objects x2seeds')
    if path.exists() and config.read(path) != value:
        raise ValueError('latent protocol changed')
    config.write(path, value)


@torch.no_grad()
def recreate(runtime, index, seed):
    import trimesh
    from pose_aligned_layered_reconstruction.frozen_slat import sparse_noise, master_noise, sample_mesh_latent
    path = OUT / 'latents' / f'{index:03d}_{seed}.pt'
    if path.exists():
        from trellis.modules.sparse import SparseTensor
        saved = torch.load(path, map_location='cuda:0', weights_only=False)
        return SparseTensor(feats=saved['features'], coords=saved['coords'])
    with np.load(config.BASE / 'objects' / f'{index:03d}' / f'generated/{seed}_ss.npz') as a:
        coords = a['coords']
    initial = sparse_noise(coords, master_noise(seed, index, runtime['mean'].device))
    start = time.monotonic()
    z, stats = sample_mesh_latent(runtime, input_sample(index), initial)
    mesh = runtime['decoder'](z)[0].to_trimesh(transform_pose=False)
    original = trimesh.load(config.BASE / 'objects' / f'{index:03d}' / f'generated/{seed}.obj',
        process=False, force='mesh', skip_materials=True)
    shape_equal = mesh.vertices.shape == original.vertices.shape and mesh.faces.shape == original.faces.shape
    error = float(np.max(np.abs(mesh.vertices - original.vertices))) if shape_equal else None
    parity = bool(shape_equal and np.array_equal(mesh.faces, original.faces) and error < 2.e-7)
    receipt = dict(index=index, seed=seed, parity=parity, max_vertex_error=error,
        vertex_count=len(mesh.vertices), face_count=len(mesh.faces), sampler=stats,
        elapsed_seconds=time.monotonic() - start)
    config.write(OUT / 'latents' / f'{index:03d}_{seed}.json', receipt)
    if not parity:
        raise ValueError(f'original latent mesh parity failed: {receipt}')
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(features=z.feats.cpu(), coords=z.coords.cpu()), path)
    print(dict(stage='latent_parity', **receipt), flush=True)
    return z


def transform_points(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def optimize(decoder, z, index, seed, method, context):
    import trimesh
    from scipy.spatial import cKDTree
    from visual_evidence_probe.analysis_scripts.local_repair_cropped import observation
    from visual_evidence_probe.experiments.local_repair.runner import raster, silhouette_ious, make_evaluator, load_gt
    from visual_evidence_probe.experiments.local_repair.photometric import prepare_images, validation_cost
    path = OUT / 'objects' / f'{index:03d}' / str(seed) / method
    if (path / 'result.json').exists():
        return
    path.mkdir(parents=True, exist_ok=True)
    source = MESH_SOURCE / 'objects' / f'{index:03d}' / str(seed)
    with np.load(source / f'{method}_constraints.npz') as a:
        c = {key: a[key] for key in a.files}
    if not len(c['face_ids']):
        config.write(path / 'result.json', dict(index=index, seed=seed, method=method,
            status='no_input_constraints', edit=False, selected_step=0))
        return
    start = time.monotonic()
    rec, k, w, images, masks = observation(index)
    mesh = trimesh.load(config.BASE / 'objects' / f'{index:03d}' / f'fitted/{seed}.obj',
        process=False, force='mesh', skip_materials=True)
    fit = config.read(config.BASE / 'objects' / f'{index:03d}' / f'fitted/{seed}.json')
    matrix = np.asarray(fit['chosen']['transform'])
    inverse = np.linalg.inv(matrix)
    targets = transform_points(c['target'], inverse)
    selected_vertex_ids = np.unique(np.asarray(mesh.faces)[c['selected_faces']])
    selected_points = transform_points(np.asarray(mesh.vertices)[selected_vertex_ids], inverse)
    native = (z.coords[:, 1:].float().cpu().numpy() + .5) / 64 - .5
    distance = cKDTree(selected_points).query(native, workers=2)[0]
    active = distance <= LATENT_PARAMS['token_radius_voxels'] / 64
    active_ids = torch.as_tensor(np.flatnonzero(active), device=z.device, dtype=torch.long)
    if not len(active_ids):
        raise ValueError('no latent token within selected region')
    z0 = z.feats.detach().clone()
    std = z0.std(0, keepdim=True).clamp_min(.1)
    delta = torch.nn.Parameter(torch.zeros((len(active_ids), z0.shape[1]), device=z.device))
    optimizer = torch.optim.Adam([delta], lr=LATENT_PARAMS['lr'])
    target = torch.as_tensor(targets, device=z.device, dtype=torch.float32)
    confidence = torch.as_tensor(c['confidence'], device=z.device, dtype=torch.float32).clamp_min(1.e-4)
    confidence /= confidence.mean()
    cache = {}
    with torch.no_grad():
        base_values = field(decoder, z, cache).detach()
        corners = cache['positions'].cpu().numpy() / 256 - .5
        outside = cKDTree(native[active]).query(corners, workers=2)[0] > 3. / 64
        outside = torch.as_tensor(outside, device=z.device)
        initial_target_sdf = sdf_at(base_values, cache['positions'], target)
    rgb, gray, obs, eroded = prepare_images(images, masks, config.PARAMS['side'])
    with np.load(source / 'input_proposals.npz') as a:
        supports = torch.as_tensor(a['validation_support'], device=z.device)
    base_pred = raster(mesh.vertices, mesh.faces, k, w, context, config.PARAMS['side'])
    base_ious = silhouette_ious(base_pred['mask'], masks)
    base_cost = validation_cost(base_pred, gray, eroded, k, w, supports)
    curves, snapshots = [], []
    chosen_step, chosen_cost, chosen_features = 0, base_cost, z0.cpu()

    def latent():
        update = std * LATENT_PARAMS['normalized_feature_cap'] * torch.tanh(delta)
        features = z0.index_add(0, active_ids, update)
        return z.replace(features)

    for step in range(1, LATENT_PARAMS['steps'] + 1):
        optimizer.zero_grad(set_to_none=True)
        current = latent()
        values = field(decoder, current, cache)
        sdf = sdf_at(values, cache['positions'], target)
        data_loss = (sdf.square() * confidence).mean()
        preserve = (values[outside] - base_values[outside]).square().mean() if outside.any() else data_loss * 0
        prior = torch.tanh(delta).square().mean()
        # Scale by initial target SDF to keep the fixed regularizers meaningful across objects.
        scale = initial_target_sdf.square().mean().clamp_min(1.e-5)
        loss = data_loss / scale + LATENT_PARAMS['field_preservation_weight'] * preserve / scale
        loss += LATENT_PARAMS['feature_prior_weight'] * prior
        loss.backward()
        if delta.grad is None or not torch.isfinite(delta.grad).all():
            raise ValueError('decoder latent gradient missing/nonfinite')
        grad = float(delta.grad.norm())
        torch.nn.utils.clip_grad_norm_([delta], 1.)
        optimizer.step()
        row = dict(step=step, loss=float(loss.detach()), target_sdf_mae=float(sdf.detach().abs().mean()),
            outside_field_mse=float(preserve.detach()), gradient_norm=grad)
        curves.append(row)
        del current, values, sdf, loss
        if step in LATENT_PARAMS['checkpoints']:
            with torch.no_grad():
                edited = latent()
                if not torch.equal(edited.feats[~torch.as_tensor(active, device=z.device)], z0[~torch.as_tensor(active, device=z.device)]):
                    raise ValueError('frozen latent rows changed')
                decoded = decoder(edited)[0].to_trimesh(transform_pose=False)
                decoded.apply_transform(matrix)
                pred = raster(decoded.vertices, decoded.faces, k, w, context, config.PARAMS['side'])
                cost = validation_cost(pred, gray, eroded, k, w, supports)
                ious = silhouette_ious(pred['mask'], masks)
                safe = bool((ious >= base_ious - config.PARAMS['max_silhouette_regression']).all())
                accepted = base_cost is not None and cost is not None and safe and cost < base_cost - config.PARAMS['validation_min_gain']
                if accepted and (chosen_cost is None or cost < chosen_cost):
                    chosen_step, chosen_cost, chosen_features = step, cost, edited.feats.cpu()
                decoded.export(path / f'step_{step:03d}.ply', encoding='binary')
                snapshots.append(dict(step=step, validation_cost=cost, silhouette_ious=ious.tolist(), accepted=bool(accepted)))
            print(dict(index=index, seed=seed, method=method, **row, checkpoint=True), flush=True)
    with torch.no_grad():
        final_z = z.replace(chosen_features.to(z.device))
        result_mesh = decoder(final_z)[0].to_trimesh(transform_pose=False)
        result_mesh.apply_transform(matrix)
        result_mesh.export(path / 'accepted.obj', include_texture=False)
        final_raw = decoder(latent())[0].to_trimesh(transform_pose=False)
        final_raw.apply_transform(matrix)
        # GT is loaded after all latent updates and checkpoint selection.
        gt = load_gt(index)
        target_eval = raster(gt.vertices, gt.faces, k, w, context, config.PARAMS['metrics_side'])
        baseline_eval = raster(mesh.vertices, mesh.faces, k, w, context, config.PARAMS['metrics_side'])
        evaluate = make_evaluator(gt, baseline_eval, target_eval, masks)
        accepted_eval = evaluate(result_mesh, raster(result_mesh.vertices, result_mesh.faces, k, w, context, config.PARAMS['metrics_side']))
        proposed_eval = evaluate(final_raw, raster(final_raw.vertices, final_raw.faces, k, w, context, config.PARAMS['metrics_side']))
        baseline_metrics = evaluate(mesh, baseline_eval)
        # Measure actual geometric influence outside the selected spatial region, not merely token equality.
        points = np.asarray(result_mesh.vertices)
        distances, nearest = cKDTree(np.asarray(mesh.vertices)).query(points, workers=2)
        outside_mesh = cKDTree(np.asarray(mesh.vertices)[selected_vertex_ids]).query(points, workers=2)[0] > .05
        leakage = dict(outside_vertices=int(outside_mesh.sum()),
            mean_distance_to_original=float(distances[outside_mesh].mean()) if outside_mesh.any() else None,
            max_distance_to_original=float(distances[outside_mesh].max()) if outside_mesh.any() else None)
        torch.save(dict(features=chosen_features, coords=z.coords.cpu(), active_token_mask=active), path / 'accepted_latent.pt')
    result = dict(index=index, seed=seed, method=method, status='complete', edit=chosen_step > 0,
        selected_step=chosen_step, selected_validation_cost=chosen_cost, baseline_validation_cost=base_cost,
        active_tokens=int(active.sum()), total_tokens=len(active), constraints=len(targets),
        frozen_feature_rows_exact=True, sparse_coordinates_exact=True, model_weights_frozen=True,
        initial_target_sdf_mae=float(initial_target_sdf.abs().mean()), curve=curves, snapshots=snapshots,
        baseline=baseline_metrics, accepted=accepted_eval, proposed=proposed_eval, outside_surface_leakage=leakage,
        elapsed_seconds=time.monotonic() - start)
    config.write(path / 'result.json', result)
    print(dict(stage='latent_complete', index=index, seed=seed, method=method, selected_step=chosen_step,
        seconds=result['elapsed_seconds']), flush=True)


def run(indices, seeds, methods, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    config.setup()
    from visual_evidence_probe.runtime import setup as runtime_setup, load_slat
    from pose_aligned_layered_reconstruction.experiments.hssd_operator_audit_20261025.kernel_policy import FixedNative
    import nvdiffrast.torch as dr
    device = runtime_setup()
    if not config.read(config.STORE / 'io_relocation.json')['cache_ready']:
        raise ValueError('data model caches not ready')
    register()
    context = dr.RasterizeCudaContext(device=device)
    with FixedNative() as policy:
        runtime = load_slat(device)
        config.write(OUT / 'model_binding.json', runtime['binding'])
        latents = {}
        for index in indices:
            for seed in seeds:
                latents[index, seed] = recreate(runtime, index, seed).to('cpu')
        decoder = runtime['decoder']
        del runtime
        gc.collect(); torch.cuda.empty_cache()
        for block in decoder.blocks:
            block.use_checkpoint = True
        for index in indices:
            for seed in seeds:
                z = latents[index, seed].to(device)
                for method in methods:
                    optimize(decoder, z, index, seed, method, context)
                    gc.collect(); torch.cuda.empty_cache()
                del z
        config.write(OUT / 'kernel_policy.json', policy.manifest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indices', default='6')
    parser.add_argument('--seeds', default='42')
    parser.add_argument('--methods', default='automatic,random,oracle_region,oracle_direction')
    parser.add_argument('--gpu', default='2')
    a = parser.parse_args()
    run([int(v) for v in a.indices.split(',')], [int(v) for v in a.seeds.split(',')], a.methods.split(','), a.gpu)


if __name__ == '__main__':
    main()
