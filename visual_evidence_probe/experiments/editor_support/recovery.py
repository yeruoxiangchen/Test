import argparse
import gc

import numpy as np
import torch

from .config import setup, register, PARAMS, OUT, folder, write, read, emit, now
from .data import load_decoder, load_latent, feature_std, constraints, transform, observation, points_transform, active_tokens
from .optimize import decode, optimize, as_mesh, visible_drift, scalar_terms
from .objective import freeze_target


def perturbation(z, active, seed, index):
    generator = torch.Generator(device=z.device).manual_seed(PARAMS['random_seed'] + seed * 1009 + index)
    centers = (z.coords[active, 1:].float() + .5) / 64 - .5
    reference = centers[torch.linspace(0, len(centers) - 1, min(3, len(centers)), device=z.device).long()]
    radial = torch.exp(-torch.cdist(centers, reference).square() / (2 * .06 ** 2))
    coefficients = torch.randn((len(reference), z.feats.shape[1]), device=z.device, generator=generator)
    noise = radial @ coefficients
    noise *= PARAMS['perturb_normalized_rms'] / noise.square().mean().sqrt().clamp_min(1.e-6)
    noise = noise.clamp(-PARAMS['perturb_normalized_max'], PARAMS['perturb_normalized_max'])
    update = torch.zeros_like(z.feats); update[active] = noise * feature_std(z.device)
    return z.replace(z.feats + update), dict(normalized_rms=float(noise.square().mean().sqrt()),
        normalized_max=float(noise.abs().max()), active_tokens=int(active.sum()))


def run_case(decoder, index, seed, context, objectives):
    path = folder(index, seed, 'recovery'); path.mkdir(parents=True, exist_ok=True)
    z = load_latent(index, seed, torch.device('cuda:0'))
    fit = transform(index, seed, z.device)
    rec, k, w, _, _ = observation(index, z.device)
    original = constraints(index, seed)
    native_points = points_transform(torch.tensor(original['target'] - original['displacement'], device=z.device), torch.linalg.inv(fit))
    active = active_tokens(z, native_points.cpu().numpy())
    target_z, perturb = perturbation(z, active, seed, index)
    with torch.no_grad():
        _, _, _, baseline = decode(decoder, z, fit, k, w, context)
        vertices, faces, target_world, target_pred = decode(decoder, target_z, fit, k, w, context)
        target = freeze_target(target_pred)
        base = freeze_target(baseline)
        region = torch.ones_like(target['depth'])
        initial = scalar_terms(base, target, target, region, list(range(4)), controlled=True)
        target_mesh = as_mesh(vertices, faces)
        from pose_aligned_layered_reconstruction.mesh_metrics import sample_surface
        points, _ = sample_surface(target_mesh, 5000, PARAMS['random_seed'] + seed)
        from scipy.spatial import cKDTree
        centers = (z.coords[active, 1:].float().cpu().numpy() + .5) / 64 - .5
        distance = cKDTree(centers).query(points, workers=2)[0]
        points = points[distance <= .05]
        if len(points) < 8: raise ValueError('no local synthetic target surface points')
        points = points[np.linspace(0, len(points) - 1, min(PARAMS['max_constraints'], len(points))).astype(int)]
        target_points = torch.tensor(points, device=z.device, dtype=torch.float32)
        as_mesh(target_world, faces).export(path / 'reachable_target.ply', encoding='binary')
        # Kept only as a diagnostic artifact, not passed to the optimizer.
        torch.save(dict(features=target_z.feats.cpu(), coords=target_z.coords.cpu(), diagnostic_only=True), path / 'hidden_target_latent.pt')
        write(path / 'target.json', dict(index=index, seed=seed, uid=rec['uid'], perturbation=perturb,
            target_representable_by_construction=True, initial_geometry_loss=float(initial['loss']),
            initial_depth_loss=float(initial['local_depth']), target_vs_baseline=visible_drift(target, base),
            supervision='full target mesh renders, four posed views; hidden target feature values not used in optimization'))
    del target_z, vertices, faces, target_pred, target_mesh
    for objective in objectives:
        arm = path / objective
        if (arm / 'result.json').exists():
            previous = read(arm / 'result.json')
            if previous['status'] == 'complete': continue
            write(arm / f'failed_before_retry_{now().replace(":", "-")}.json', previous)
        try:
            accepted, result = optimize(decoder, z, active, fit, k, w, context, target, target, region,
                target_points, list(range(4)), objective, PARAMS['recovery_steps'],
                PARAMS['recovery_checkpoints'], arm, controlled=True)
            with torch.no_grad():
                _, _, _, after = decode(decoder, accepted, fit, k, w, context)
                final = scalar_terms(after, target, target, region, list(range(4)), controlled=True)
                ratio = float(final['local_depth'] / initial['local_depth'].clamp_min(1.e-8))
            result.update(index=index, seed=seed, status='complete', uid=rec['uid'],
                initial_geometry_loss=float(initial['loss']), final_geometry_loss=float(final['loss']),
                initial_depth_loss=float(initial['local_depth']), final_depth_loss=float(final['local_depth']),
                depth_error_reduction=1 - ratio, recovery_passed=bool(ratio <= 1 - PARAMS['recovery_min_depth_reduction']),
                supervised_diagnostic=True, scan_gt_used=False)
        except Exception as error:
            result = dict(index=index, seed=seed, status='failed', objective=objective,
                error=f'{type(error).__name__}: {error}', recovery_passed=False)
            write(arm / 'result.json', result)
            raise
        write(arm / 'result.json', result)
        emit('recovery_arm_complete', index=index, seed=seed, objective=objective,
            passed=result['recovery_passed'], depth_reduction=result['depth_error_reduction'])
        gc.collect(); torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--gpu', default='2')
    p.add_argument('--indices', default='4,6'); p.add_argument('--seeds', default='42,43')
    p.add_argument('--objectives', default='proxy,mesh'); a = p.parse_args()
    setup(a.gpu)
    from visual_evidence_probe.runtime import setup as runtime_setup
    import nvdiffrast.torch as dr
    device = runtime_setup(); register('recovery')
    decoder, binding = load_decoder(device); write(OUT / f'decoder_{a.gpu}.json', binding)
    context = dr.RasterizeCudaContext(device=device)
    from .forward_policy import FixedForwardNative
    with FixedForwardNative():
        for index in map(int, a.indices.split(',')):
            for seed in map(int, a.seeds.split(',')):
                run_case(decoder, index, seed, context, a.objectives.split(','))


if __name__ == '__main__': main()
