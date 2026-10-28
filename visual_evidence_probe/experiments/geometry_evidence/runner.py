import argparse
import gc
import time
import traceback

import numpy as np
import torch

from .config import ARMS, OUT, BASE, PARAMS, folder, setup, register, read, write, emit, now
from .targets import seal_input, oracle_targets, novel_cameras
from .optimize import run as optimize
from .metrics import render_mesh, face_region, Evaluator
from visual_evidence_probe.experiments.editor_support.data import load_decoder, load_latent, transform, observation
from visual_evidence_probe.experiments.editor_support.optimize import decode, as_mesh, visible_drift
from visual_evidence_probe.experiments.editor_support.objective import freeze_target
from pose_aligned_layered_reconstruction.mesh_metrics import structure_metrics


def load_obj(path):
    import trimesh
    return trimesh.load(path, process=False, force='mesh', skip_materials=True)


def link(path, source):
    if not path.exists(): path.symlink_to(source)


def noop(path, root, name, status, query_count):
    path.mkdir(parents=True, exist_ok=True)
    for endpoint in ('proposal', 'selected', 'accepted'):
        link(path / f'{endpoint}.obj', root / 'original.obj')
        link(path / f'{endpoint}_latent.pt', root / 'original_latent.pt')
    value = dict(status=status, arm=name, best_step=0, actual_selected_edit=False,
        steps=0, elapsed_seconds=0., peak_allocated_bytes=0,
        query_count=query_count, weights_frozen=True, uses_gt=name != 'input_sparse')
    write(path / 'optimization.json', value)
    return value


@torch.no_grad()
def input_acceptance(name, result, original_structure, baseline, record, images, masks, k, w, context, path):
    receipt_path = path / 'input_acceptance.json'
    if receipt_path.exists(): return read(receipt_path)
    if not result.get('actual_selected_edit', False):
        value = dict(accepted=False, reason='no_selected_edit', uses_gt_for_gate=False)
    else:
        from visual_evidence_probe.experiments.local_repair.photometric import prepare_images
        from visual_evidence_probe.experiments.editor_support.pool import validation
        selected = load_obj(path / 'selected.obj')
        pred = render_mesh(selected, k, w, context)
        structure = structure_metrics(selected)
        drift = visible_drift(pred, baseline, record['region'])
        iou = ((pred['mask'] & baseline['mask']).sum((1, 2)) /
            (pred['mask'] | baseline['mask']).sum((1, 2)).clamp_min(1))
        safe = bool(iou.min() >= 1 - PARAMS['max_silhouette_regression'])
        safe &= drift['depth_mae'] is not None and drift['depth_mae'] <= PARAMS['max_outside_depth_mae']
        safe &= drift['depth_p95'] is not None and drift['depth_p95'] <= PARAMS['max_outside_depth_p95']
        topology = structure['component_count'] <= original_structure['component_count']
        topology &= structure['nonmanifold_edge_count'] <= original_structure['nonmanifold_edge_count']
        score, base_score = None, None
        if record['heldout'] and 'validation_support' in record['saved']:
            _, gray, _, eroded = prepare_images(images, masks, PARAMS['side'])
            support = torch.as_tensor(record['saved']['validation_support'], device=k.device, dtype=torch.bool)
            pool = dict(train_views=record['train'], heldout_views=record['heldout'],
                gray=gray, eroded=eroded, validation_support=support)
            base_score = validation(baseline, pool, k, w)
            score = validation(pred, pool, k, w)
        improved = score is not None and base_score is not None and score <= base_score - PARAMS['validation_min_gain']
        value = dict(accepted=bool(safe and topology and improved), uses_gt_for_gate=False,
            privileged_candidate=name != 'input_sparse', input_validation=score, baseline_validation=base_score,
            validation_available=score is not None, safety_gate=bool(safe), topology_gate=bool(topology),
            selected_structure=structure, outside=drift, silhouette_iou_to_original=iou.cpu().tolist())
    value.update(time=now(), arm=name)
    write(receipt_path, value)
    for extension in ('obj', 'pt'):
        filename = 'accepted.obj' if extension == 'obj' else 'accepted_latent.pt'
        source = (path / ('selected.obj' if extension == 'obj' else 'selected_latent.pt')) if value['accepted'] else (
            path.parent / ('original.obj' if extension == 'obj' else 'original_latent.pt'))
        link(path / filename, source)
    return value


def one_arm(decoder, z, fit, k, w, context, baseline, record, target, name, root, index, seed):
    path = root / name
    saved = path / 'optimization.json'
    if saved.exists() and (path / 'selected.obj').exists(): return read(saved)
    valid = target.get('depth_valid', target['mask']) & target['mask'] & (record['region'] > .5)
    count = int(valid[record['train']].sum())
    if not bool(record['active'].any()) or not bool((record['region'] > .5).any()):
        return noop(path, root, name, 'no_input_region', count)
    if not count:
        return noop(path, root, name, 'no_input_constraints' if name == 'input_sparse' else 'no_oracle_constraints', count)
    result = optimize(decoder, z, fit, k, w, context, baseline, record, target, name, path, index, seed)
    result['query_count'] = count
    write(saved, result)
    return result


def run_case(decoder, index, seed, context, cache):
    root = folder(index, seed); root.mkdir(parents=True, exist_ok=True)
    if (root / 'complete.json').exists():
        emit(index, seed, 'cached_complete'); return
    start = time.monotonic()
    device = torch.device('cuda:0')
    z = load_latent(index, seed, device)
    fit = transform(index, seed, device)
    rec, k, w, images, masks = observation(index, device)
    with torch.no_grad():
        _, faces, world, pred = decode(decoder, z, fit, k, w, context)
        baseline = freeze_target(pred)
        original = as_mesh(world, faces)
        original.export(root / 'original.obj', include_texture=False)
        torch.save(dict(features=z.feats.cpu(), coords=z.coords.cpu()), root / 'original_latent.pt')
        record = seal_input(index, seed, root, baseline, masks, k, w, fit, z, len(faces))
    del pred, faces, world
    original_structure = structure_metrics(original)
    # The complete deployable arm is optimized and its decision sealed before loading GT.
    input_result = one_arm(decoder, z, fit, k, w, context, baseline, record,
        record['target'], 'input_sparse', root, index, seed)
    input_gate = input_acceptance('input_sparse', input_result, original_structure, baseline,
        record, images, masks, k, w, context, root / 'input_sparse')
    write(root / 'input_finished_before_gt.json', dict(time=now(), gt_access_in_input_branch=False,
        gt_resident_from_previous_case=cache.get('index') == index,
        best_step=input_result['best_step'], accepted=input_gate['accepted']))
    emit(index, seed, 'input_arm_sealed_before_gt', constraints=len(record['pixels']))
    nk_array, nw_array, camera_receipt = novel_cameras(k.cpu().numpy(), w.cpu().numpy(), PARAMS['novel_views'])
    nk = torch.as_tensor(nk_array, device=device); nw = torch.as_tensor(nw_array, device=device)
    camera_path = folder(index) / 'novel_cameras.json'
    camera_record = dict(intrinsics=nk_array.tolist(), extrinsics=nw_array.tolist(), **camera_receipt)
    if camera_path.exists() and read(camera_path) != camera_record: raise ValueError('evaluation cameras changed')
    write(camera_path, camera_record)
    if cache.get('index') != index:
        cache.clear(); gc.collect()
        gt_mesh = load_obj(BASE / 'objects' / f'{index:03d}' / 'target_canonical.ply')
        with torch.no_grad():
            gt = freeze_target(render_mesh(gt_mesh, k, w, context))
            ngt = freeze_target(render_mesh(gt_mesh, nk, nw, context))
        cache.update(index=index, mesh=gt_mesh, input=gt, novel=ngt)
        keep = ('depth', 'mask', 'softmask', 'normal', 'normal_valid')
        torch.save(dict(input={key: gt[key].cpu() for key in keep},
            novel={key: ngt[key].cpu() for key in keep}), folder(index) / 'evaluation_targets.pt')
    gt, ngt, gt_mesh = cache['input'], cache['novel'], cache['mesh']
    targets, target_receipt = oracle_targets(baseline, gt, record)
    write(root / 'oracle_targets.json', dict(time=now(), privileged_diagnostic=True, **target_receipt))
    results, gates = dict(input_sparse=input_result), dict(input_sparse=input_gate)
    for name in ARMS:
        if name == 'input_sparse': continue
        results[name] = one_arm(decoder, z, fit, k, w, context, baseline, record,
            targets[name], name, root, index, seed)
        gates[name] = input_acceptance(name, results[name], original_structure, baseline,
            record, images, masks, k, w, context, root / name)
        gc.collect(); torch.cuda.empty_cache()
    # No optimizer or selector can access these evaluation-only renders/metrics.
    emit(index, seed, 'evaluating_sealed_endpoints')
    with torch.no_grad():
        original_novel = render_mesh(original, nk, nw, context)
        novel_region = face_region(original_novel, record['selected_faces'])
        np.savez_compressed(root / 'novel_region.npz', region=novel_region.cpu().numpy())
        evaluate = Evaluator(gt_mesh, original, baseline, gt, record['region'],
            original_novel, ngt, novel_region, k, w, nk, nw, context)
        baseline_metrics = evaluate(original, original=True, structure=original_structure)
        write(root / 'baseline_metrics.json', baseline_metrics)
        for name in ARMS:
            result = results[name]
            path = root / name
            endpoints = {}
            if not result.get('actual_selected_edit', False) and result['steps'] == 0:
                endpoints = {key: baseline_metrics for key in ('proposal', 'selected', 'accepted')}
            else:
                for endpoint in ('selected', 'proposal'):
                    if endpoint == 'selected' and result['best_step'] == 0:
                        endpoints[endpoint] = baseline_metrics
                    elif endpoint == 'proposal' and result['best_step'] == PARAMS['steps']:
                        endpoints[endpoint] = endpoints['selected']
                    else:
                        mesh = load_obj(path / f'{endpoint}.obj')
                        structure = gates[name].get('selected_structure') if endpoint == 'selected' else None
                        endpoints[endpoint] = evaluate(mesh, structure=structure)
                        del mesh
                endpoints['accepted'] = endpoints['selected'] if gates[name]['accepted'] else baseline_metrics
            result.update(index=index, seed=seed, uid=rec['uid'], metrics=endpoints,
                input_acceptance=gates[name], original_metrics=baseline_metrics,
                target_diagnostic=target_receipt, no_new_alignment=True)
            write(path / 'result.json', result)
            emit(index, seed, 'arm_scored', arm=name, best_step=result['best_step'],
                cd_gain=baseline_metrics['surface']['chamfer_l1'] - endpoints['selected']['surface']['chamfer_l1'])
    write(root / 'complete.json', dict(time=now(), index=index, seed=seed, uid=rec['uid'], arms=len(ARMS),
        query_count=len(record['pixels']), elapsed_seconds=time.monotonic() - start,
        gt_privileged_arms=[name for name in ARMS if name != 'input_sparse'], input_arm_sealed_before_gt=True))
    emit(index, seed, 'complete', seconds=time.monotonic() - start)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', default='0'); p.add_argument('--indices', default=','.join(map(str, PARAMS['indices'])))
    p.add_argument('--seeds', default='42,43'); a = p.parse_args()
    setup(a.gpu)
    from visual_evidence_probe.runtime import setup as runtime_setup
    from visual_evidence_probe.experiments.editor_support.forward_policy import FixedForwardNative
    import nvdiffrast.torch as dr
    device = runtime_setup(); register('run')
    decoder, binding = load_decoder(device); write(OUT / f'decoder_{a.gpu}.json', binding)
    write(OUT / f'hardware_{a.gpu}.json', dict(gpu=torch.cuda.get_device_name(), physical_device=a.gpu,
        total_memory=torch.cuda.get_device_properties(0).total_memory, torch=torch.__version__, cuda=torch.version.cuda))
    context = dr.RasterizeCudaContext(device=device)
    cache = {}
    with FixedForwardNative() as policy:
        for index in map(int, a.indices.split(',')):
            for seed in map(int, a.seeds.split(',')):
                try:
                    run_case(decoder, index, seed, context, cache)
                except Exception as error:
                    write(folder(index, seed) / f'failure_{now().replace(":", "-")}.json',
                        dict(time=now(), error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc()))
                    raise
                gc.collect(); torch.cuda.empty_cache()
        write(OUT / f'forward_policy_{a.gpu}.json', policy.manifest())


if __name__ == '__main__': main()
