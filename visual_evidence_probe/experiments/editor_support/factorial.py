import argparse
import gc
from pathlib import Path
import time

import numpy as np
import torch

from .config import setup, register, PARAMS, OUT, BASE, folder, read, write, emit, now
from .data import load_decoder, load_latent, transform, observation, active_tokens
from .optimize import decode, optimize, as_mesh, visible_drift
from .objective import freeze_target
from .pool import build_pool, sparse_target, save_pool, validation
from .support import expand_support


def mask_ious(pred, baseline):
    return ((pred['mask'] & baseline['mask']).sum((1, 2)) /
        (pred['mask'] | baseline['mask']).sum((1, 2)).clamp_min(1)).cpu().tolist()


def initialization(decoder, z, expanded, added, fit, k, w, context, baseline, path):
    with torch.no_grad():
        _, faces, world, pred = decode(decoder, expanded, fit, k, w, context)
        raw = visible_drift(pred, baseline)
        as_mesh(world, faces).export(path / 'raw_expanded.ply', encoding='binary')
    if not added.any(): return expanded, dict(status='no_support_change', raw_drift=raw, preserved_drift=raw)
    preserved, record = optimize(decoder, expanded, added, fit, k, w, context,
        baseline, baseline, torch.ones_like(baseline['depth']), z.feats.new_empty((0, 3)),
        list(range(4)), 'mesh', PARAMS['initialization_steps'], [0, 10, 20], path / 'preserve',
        controlled=True, initialize=True, cap=PARAMS['initialization_feature_cap'])
    with torch.no_grad():
        _, _, _, pred = decode(decoder, preserved, fit, k, w, context)
        drift = visible_drift(pred, baseline)
    okay = drift['depth_mae'] is not None and drift['depth_mae'] <= PARAMS['initialize_depth_mae']
    okay &= drift['depth_p95'] is not None and drift['depth_p95'] <= PARAMS['initialize_depth_p95']
    okay &= drift['normal_mean_degrees'] is not None and drift['normal_mean_degrees'] <= PARAMS['initialize_normal_degrees']
    okay &= drift['mask_mismatch_max'] <= PARAMS['initialize_mask_regression']
    result = dict(status='valid' if okay else 'init_failure', raw_drift=raw,
        preserved_drift=drift, preservation=record, only_new_feature_rows_updated=True,
        target='original decoded mesh only, no correction target or GT', shared_by_both_objectives=True)
    write(path / 'initialization.json', result)
    torch.save(dict(features=preserved.feats.cpu(), coords=preserved.coords.cpu(), added_mask=added.cpu()), path / 'shared_initialization.pt')
    return preserved, result


def selector(pool, k, w, baseline, initialization_valid):
    def select(pred):
        score = validation(pred, pool, k, w)
        ious = mask_ious(pred, baseline)
        drift = visible_drift(pred, baseline, pool['region'])
        safe = min(ious) >= 1 - PARAMS['max_silhouette_regression']
        safe &= drift['depth_mae'] is not None and drift['depth_mae'] <= PARAMS['initialize_depth_mae']
        safe &= drift['depth_p95'] is not None and drift['depth_p95'] <= PARAMS['initialize_depth_p95']
        improved = score is not None and pool['baseline_validation_cost'] is not None
        improved = improved and score <= pool['baseline_validation_cost'] - PARAMS['validation_min_gain']
        okay = initialization_valid and safe and improved
        return score, okay, dict(input_validation_cost=score, input_validation_improved=bool(improved),
            silhouette_ious_to_original=ious, outside_drift=drift, safety_gate=bool(safe))
    return select


def quality(mesh):
    from pose_aligned_layered_reconstruction.mesh_metrics import structure_metrics
    return structure_metrics(mesh)


def evaluation(gt, original_mesh, original_render, k, w, masks, context, heldout):
    from visual_evidence_probe.experiments.local_repair.runner import raster, make_evaluator
    gt_render = raster(gt.vertices, gt.faces, k, w, context, PARAMS['evaluation_side'])
    evaluate = make_evaluator(gt, original_render, gt_render, masks)
    def score(mesh):
        pred = raster(mesh.vertices, mesh.faces, k, w, context, PARAMS['evaluation_side'])
        metrics = evaluate(mesh, pred)
        metrics['visible']['fixed_view3_depth_mae'] = metrics['visible'].pop('heldout_depth_mae')
        if heldout:
            i = heldout[0]
            common = original_render['mask'][i] & gt_render['mask'][i]
            error = torch.where(pred['mask'][i], (pred['depth'][i] - gt_render['depth'][i]).abs().clamp_max(.1),
                torch.full_like(pred['depth'][i], .1))
            metrics['visible']['edit_holdout_depth_mae'] = float(error[common].mean()) if common.any() else None
        else: metrics['visible']['edit_holdout_depth_mae'] = None
        return metrics
    return score


def run_case(decoder, index, seed, context, gt_cache):
    import trimesh
    path = folder(index, seed); path.mkdir(parents=True, exist_ok=True)
    if (path / 'complete.json').exists(): return
    start = time.monotonic()
    z = load_latent(index, seed, torch.device('cuda:0'))
    fit = transform(index, seed, z.device)
    rec, k, w, images, masks = observation(index, z.device)
    with torch.no_grad():
        _, faces, world, pred = decode(decoder, z, fit, k, w, context)
        baseline = freeze_target(pred)
        original_mesh = as_mesh(world, faces)
        original_mesh.export(path / 'original.obj', include_texture=False)
        original_mesh.export(path / 'original.ply', encoding='binary')
        pool = build_pool(index, seed, baseline, k, w, images, masks, fit)
        save_pool(path, pool)
    outputs = {}
    arms = [(s, o) for s in ('original', 'expanded') for o in ('proxy', 'mesh')]
    original_structure = quality(original_mesh)
    if len(pool['targets']):
        target, _ = sparse_target(baseline, pool, masks, k, w)
        target_points = torch.tensor(pool['native_targets'], device=z.device, dtype=torch.float32)
        expanded, added, expansion = expand_support(z, pool['native_points'], pool['native_targets'],
            pool['confidence'], max_added_tokens=PARAMS['max_added_tokens'], max_fraction=PARAMS['max_added_fraction'])
        ep = path / 'expansion'; ep.mkdir(exist_ok=True)
        write(ep / 'support.json', expansion)
        preserved, init = initialization(decoder, z, expanded, added, fit, k, w, context, baseline, ep)
        for support, objective in arms:
            arm_path = path / f'{support}_{objective}'
            if (arm_path / 'input_selected.json').exists():
                previous = read(arm_path / 'input_selected.json')
                if previous['status'] != 'failed':
                    outputs[f'{support}_{objective}'] = previous; continue
                write(arm_path / f'failed_before_retry_{now().replace(":", "-")}.json', previous)
            current_z = z if support == 'original' else preserved
            init_valid = support == 'original' or init['status'] != 'init_failure'
            editable_points = np.concatenate((pool['native_points'], pool['native_targets']))
            active = active_tokens(current_z, editable_points)
            if support == 'expanded': active |= added
            with torch.no_grad():
                _, f0, v0, step0 = decode(decoder, current_z, fit, k, w, context)
            arm_path.mkdir(parents=True, exist_ok=True)
            as_mesh(v0, f0).export(arm_path / 'step0.ply', encoding='binary')
            try:
                accepted, result = optimize(decoder, current_z, active, fit, k, w, context,
                    target, baseline, pool['region'], target_points, pool['train_views'], objective,
                    PARAMS['formal_steps'], PARAMS['formal_checkpoints'], arm_path,
                    select=selector(pool, k, w, baseline, init_valid))
                accepted_mesh = trimesh.load(arm_path / 'accepted.obj', process=False, force='mesh', skip_materials=True)
                structure = quality(accepted_mesh)
                topology_safe = structure['component_count'] <= original_structure['component_count']
                topology_safe &= structure['nonmanifold_edge_count'] <= original_structure['nonmanifold_edge_count']
                actually_accept = result['best_step'] > 0 and topology_safe and init_valid
                if not actually_accept:
                    old = arm_path / 'accepted.obj'; old.rename(arm_path / 'uncommitted_selection.obj')
                    old.symlink_to(path / 'original.obj')
                    latent_path = arm_path / 'accepted_latent.pt'
                    latent_path.rename(arm_path / 'uncommitted_selection_latent.pt')
                    torch.save(dict(features=z.feats.cpu(), coords=z.coords.cpu()), latent_path)
                result.update(status='init_failure' if not init_valid else ('unable_validation' if pool['status'].startswith('unable') else 'complete'),
                    support=support, objective=objective, initialization_status='original' if support == 'original' else init['status'],
                    accepted_edit=bool(actually_accept), topology_gate=bool(topology_safe),
                    candidate_selected_structure=structure, input_only_selection=True,
                    constraint_count=len(pool['targets']), train_views=pool['train_views'], heldout_views=pool['heldout_views'])
            except Exception as error:
                result = dict(status='failed', error=f'{type(error).__name__}: {error}', support=support,
                    objective=objective, accepted_edit=False, constraint_count=len(pool['targets']))
                write(arm_path / 'input_selected.json', result)
                raise
            write(arm_path / 'input_selected.json', result)
            outputs[f'{support}_{objective}'] = result
            emit('factorial_arm_optimized', index=index, seed=seed, support=support,
                objective=objective, accepted=result['accepted_edit'], init_status=result['initialization_status'])
            gc.collect(); torch.cuda.empty_cache()
    else:
        for support, objective in arms:
            arm_path = path / f'{support}_{objective}'; arm_path.mkdir(parents=True, exist_ok=True)
            for name in ('accepted.obj', 'proposal.ply', 'step0.ply'):
                if not (arm_path / name).exists():
                    (arm_path / name).symlink_to(path / ('original.ply' if name.endswith('.ply') else 'original.obj'))
            result = dict(status='no_input_constraints', support=support, objective=objective,
                accepted_edit=False, constraint_count=0, input_only_selection=True)
            torch.save(dict(features=z.feats.cpu(), coords=z.coords.cpu()), arm_path / 'accepted_latent.pt')
            write(arm_path / 'input_selected.json', result); outputs[f'{support}_{objective}'] = result
    write(path / 'input_sealed.json', dict(time=now(), gt_used=False, arms=list(outputs)))
    # First access to scan GT occurs only after all input-only candidates and choices have been saved.
    if index not in gt_cache:
        gt_cache.clear()
        gt_cache[index] = trimesh.load(BASE / 'objects' / f'{index:03d}' / 'target_canonical.ply',
            process=False, force='mesh', skip_materials=True)
    with torch.no_grad():
        from visual_evidence_probe.experiments.local_repair.runner import raster
        original_render = raster(original_mesh.vertices, original_mesh.faces, k, w, context, PARAMS['evaluation_side'])
        evaluate = evaluation(gt_cache[index], original_mesh, original_render, k, w, masks, context, pool['heldout_views'])
        original_metrics = evaluate(original_mesh)
        for name, result in outputs.items():
            arm_path = path / name
            scored = {}
            for endpoint, filename in (('step0', 'step0.ply'), ('proposal', 'proposal.ply'), ('accepted', 'accepted.obj')):
                if result['status'] == 'no_input_constraints':
                    scored[endpoint] = original_metrics
                    if endpoint == 'proposal': result['proposal_structure'] = original_structure
                else:
                    mesh = trimesh.load(arm_path / filename, process=False, force='mesh', skip_materials=True)
                    scored[endpoint] = evaluate(mesh)
                    if endpoint == 'proposal': result['proposal_structure'] = quality(mesh)
            result.update(index=index, seed=seed, uid=rec['uid'], original_metrics=original_metrics,
                metrics=scored, gt_used_for_evaluation_only=True, original_structure=original_structure)
            write(arm_path / 'result.json', result)
    write(path / 'complete.json', dict(time=now(), index=index, seed=seed, arms=4,
        constraint_count=len(pool['targets']), status=pool['status'], elapsed_seconds=time.monotonic() - start))
    emit('factorial_case_complete', index=index, seed=seed, seconds=time.monotonic() - start)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--gpu', default='3')
    p.add_argument('--indices', default=','.join(map(str, PARAMS['indices'])))
    p.add_argument('--seeds', default='42,43'); a = p.parse_args()
    setup(a.gpu)
    from visual_evidence_probe.runtime import setup as runtime_setup
    from .forward_policy import FixedForwardNative
    import nvdiffrast.torch as dr
    device = runtime_setup(); register('factorial')
    decoder, binding = load_decoder(device); write(OUT / f'decoder_{a.gpu}.json', binding)
    write(OUT / f'hardware_{a.gpu}.json', dict(gpu=torch.cuda.get_device_name(),
        device_index=a.gpu, total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        torch_version=torch.__version__, cuda_version=torch.version.cuda))
    context = dr.RasterizeCudaContext(device=device)
    cache = {}
    with FixedForwardNative():
        for index in map(int, a.indices.split(',')):
            for seed in map(int, a.seeds.split(',')):
                run_case(decoder, index, seed, context, cache)
                gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__': main()
