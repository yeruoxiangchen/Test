import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np
import torch

from .config import BASE, OUT, PARAMS, METHODS, folder, read, write, physical, digest, emit, setup, protocol, now


def observation(index):
    from PIL import Image
    rec = read(BASE / 'objects' / f'{index:03d}' / 'observations.json')
    images = [Image.open(physical(b['path'])).convert('RGB') for b in rec['images']]
    masks = np.stack([np.asarray(Image.open(physical(b['path'])).convert('L')) > 127 for b in rec['masks']])
    return rec, torch.tensor(rec['intrinsics'], device='cuda:0', dtype=torch.float32), \
        torch.tensor(rec['extrinsics'], device='cuda:0', dtype=torch.float32), images, masks


def raster(vertices, faces, k, w, context, side):
    import nvdiffrast.torch as dr
    from visual_evidence_probe.experiments.input_consistency.refine import project_clip
    from .photometric import world_from_depth
    v = torch.as_tensor(vertices, device='cuda:0', dtype=torch.float32)
    f = torch.as_tensor(faces, device='cuda:0', dtype=torch.int32).contiguous()
    rast, _ = dr.rasterize(context, project_clip(v, k, w), f, resolution=(side, side))
    cam_z = v[None] @ w[:, 2, :3, None] + w[:, None, 2, 3, None]
    z, _ = dr.interpolate(cam_z.contiguous(), rast, f)
    hit = rast[..., 3] > 0
    depth = z[..., 0] * hit
    xyz = torch.stack([world_from_depth(depth[i], k[i], w[i])[0] for i in range(len(k))])
    return dict(depth=depth, xyz=xyz * hit[..., None], mask=hit, face_id=rast[..., 3].long() - 1)


def silhouette_ious(pred, masks):
    import torch.nn.functional as F
    target = F.interpolate(torch.as_tensor(masks, device='cuda:0', dtype=torch.float32)[:, None],
        size=pred.shape[-2:], mode='nearest')[:, 0] > .5
    return ((pred & target).sum((1, 2)) / (pred | target).sum((1, 2)).clamp_min(1)).cpu().numpy()


def load_mesh(index, seed):
    import trimesh
    b = read(BASE / 'objects' / f'{index:03d}' / f'fitted/{seed}.json')['mesh']
    path = physical(b['path'])
    if digest(path) != b['sha256']:
        raise ValueError('starting fitted mesh changed')
    mesh = trimesh.load(path, force='mesh', process=False, skip_materials=True)
    return mesh, dict(path=str(path), sha256=b['sha256'])


def load_gt(index):
    import trimesh
    # This canonical target was constructed/verified once by round3; no repeated source scan or hash.
    path = BASE / 'objects' / f'{index:03d}' / 'target_canonical.ply'
    if not path.exists():
        raise FileNotFoundError(f'Waiting for background data migration: {path}')
    return trimesh.load(path, force='mesh', process=False, skip_materials=True)


def arrays(record):
    return {k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in record.items()}


def proposal_diagnostic(proposals, pred, target):
    valid = proposals['valid'] & target['mask'][:3].cpu().numpy()
    before = pred['depth'][:3].cpu().numpy() - target['depth'][:3].cpu().numpy()
    after = before + proposals['delta']
    if not valid.any():
        return dict(proposals=0, before_mae=None, after_mae=None, improved_fraction=None)
    return dict(proposals=int(valid.sum()), before_mae=float(np.abs(before[valid]).mean()),
        after_mae=float(np.abs(after[valid]).mean()), improved_fraction=float((np.abs(after[valid]) < np.abs(before[valid])).mean()),
        correct_direction_fraction=float((before[valid] * proposals['delta'][valid] < 0).mean()))


def make_evaluator(gt, pred_base, target_eval, masks):
    from scipy.spatial import cKDTree
    from pose_aligned_layered_reconstruction.mesh_metrics import sample_surface
    g, gn = sample_surface(gt, PARAMS['surface_samples'], PARAMS['random_seed'])
    tree = cKDTree(g)
    gt_mask = target_eval['mask'].cpu().numpy()
    gt_depth = target_eval['depth'].cpu().numpy()
    original_hit = pred_base['mask'].cpu().numpy()
    original_depth = pred_base['depth'].cpu().numpy()
    common = original_hit & gt_mask
    original_error = np.abs(original_depth - gt_depth)
    bad = common & (original_error > PARAMS['metric_threshold'])
    good = common & ~bad

    def evaluate(mesh, pred):
        p, pn = sample_surface(mesh, PARAMS['surface_samples'], PARAMS['random_seed'])
        pd, pi = tree.query(p, workers=2)
        gd, gi = cKDTree(p).query(g, workers=2)
        surface = dict(chamfer_l1=float(.5 * (pd.mean() + gd.mean())),
            chamfer_l2=float(.5 * (np.square(pd).mean() + np.square(gd).mean())),
            normal_consistency=float(.5 * (np.abs((pn * gn[pi]).sum(1)).mean() + np.abs((gn * pn[gi]).sum(1)).mean())))
        for threshold in (.01, .02, .05):
            precision, recall = float((pd < threshold).mean()), float((gd < threshold).mean())
            surface[f'fscore_{str(threshold).replace(".", "p")}'] = 2 * precision * recall / max(precision + recall, 1.e-12)
        hit, depth = pred['mask'].cpu().numpy(), pred['depth'].cpu().numpy()
        # Fixed initial domains. A disappearing ray incurs 0.1, and cannot vanish from the denominator.
        error = np.where(hit, np.minimum(np.abs(depth - gt_depth), .1), .1)
        visible = dict(common_pixels=int(common.sum()), original_bad_pixels=int(bad.sum()), original_good_pixels=int(good.sum()),
            fixed_common_depth_mae=float(error[common].mean()) if common.any() else None,
            originally_bad_depth_mae=float(error[bad].mean()) if bad.any() else None,
            originally_good_depth_mae=float(error[good].mean()) if good.any() else None,
            lost_original_common_fraction=float((~hit[common]).mean()) if common.any() else None,
            heldout_depth_mae=float(error[3][common[3]].mean()) if common[3].any() else None,
            silhouette_ious=silhouette_ious(pred['mask'], masks).tolist())
        return dict(surface=surface, visible=visible)
    return evaluate


def input_phase(index, seed, context):
    from .photometric import prepare_images, propose, localization_scores, validation_reference, validation_cost
    start = time.monotonic()
    rec, k, w, images, masks = observation(index)
    mesh, source = load_mesh(index, seed)
    pred = raster(mesh.vertices, mesh.faces, k, w, context, PARAMS['side'])
    rgb, gray, obs, eroded = prepare_images(images, masks, PARAMS['side'])
    proposals = propose(pred, gray, eroded, k, w)
    scores = localization_scores(pred, rgb, eroded, k, w)
    supports = validation_reference(pred, gray, eroded, k, w)
    validation = validation_cost(pred, gray, eroded, k, w, supports)
    p = folder(index, seed) / 'input_proposals.npz'; p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **proposals, localization=scores, **arrays(pred), validation_support=supports.cpu().numpy())
    meta = dict(time=now(), source=source, proposals_sha256=digest(p), uses_gt=False,
        train_views=PARAMS['train_views'], validation_view=3, validation_cost=validation,
        validation_support_pixels=int(supports.sum()), valid_proposal_pixels=int(proposals['valid'].sum()),
        elapsed_seconds=time.monotonic() - start)
    write(p.with_suffix('.json'), meta)
    emit(index, seed, 'input_proposals_sealed', **{q: meta[q] for q in ('valid_proposal_pixels', 'validation_support_pixels', 'elapsed_seconds')})
    return rec, k, w, images, masks, mesh, pred, gray, eroded, proposals, scores, supports, validation, meta


def run_case(index, seed, context, gt):
    from .deform import area_region, constraints, solve
    from .photometric import validation_cost
    path = folder(index, seed)
    if (path / 'result.json').exists():
        return read(path / 'result.json')
    total_start = time.monotonic()
    rec, k, w, images, masks, mesh, pred, gray, eroded, proposals, scores, supports, baseline_cost, meta = input_phase(index, seed, context)
    # GT enters only after input proposals and their SHA have been saved.
    target = raster(gt.vertices, gt.faces, k, w, context, PARAMS['side'])
    eval_base = raster(mesh.vertices, mesh.faces, k, w, context, PARAMS['metrics_side'])
    eval_target = raster(gt.vertices, gt.faces, k, w, context, PARAMS['metrics_side'])
    evaluate = make_evaluator(gt, eval_base, eval_target, masks)
    baseline = evaluate(mesh, eval_base)
    baseline_iou = silhouette_ious(pred['mask'], masks)
    face_ids = pred['face_id'][:3].cpu().numpy()
    oracle_score = np.abs(pred['depth'][:3].cpu().numpy() - target['depth'][:3].cpu().numpy())
    oracle_score = np.where(target['mask'][:3].cpu().numpy(), oracle_score, .1)
    oracle_score[~np.isfinite(scores)] = np.nan
    methods = {}
    for method in METHODS:
        begin = time.monotonic()
        random_seed = PARAMS['random_seed'] + index * 1009 + seed
        selected, region = area_region(mesh, face_ids, oracle_score if method.startswith('oracle') else scores, method, random_seed)
        used = proposals
        if method == 'oracle_direction':
            delta = (target['depth'][:3] - pred['depth'][:3]).cpu().numpy()
            valid = target['mask'][:3].cpu().numpy() & np.isfinite(scores) & (np.abs(delta) > .005)
            used = dict(valid=valid, delta=np.clip(delta, -.06, .06), confidence=valid.astype(np.float32))
        c = constraints(mesh, pred, used, selected, k.cpu().numpy(), w.cpu().numpy(), random_seed)
        displacement, solver = solve(mesh, c, selected)
        np.savez_compressed(path / f'{method}_constraints.npz', **c, selected_faces=selected,
            vertex_displacement=displacement)
        candidates = [dict(amplitude=0., validation_cost=baseline_cost, silhouette_ious=baseline_iou.tolist(), accepted=True)]
        chosen_amplitude, chosen_cost = 0., baseline_cost
        for amplitude in PARAMS['amplitudes'][1:]:
            variant = mesh.copy(); variant.vertices = np.asarray(mesh.vertices) + amplitude * displacement
            vp = raster(variant.vertices, variant.faces, k, w, context, PARAMS['side'])
            cost = validation_cost(vp, gray, eroded, k, w, supports)
            iou = silhouette_ious(vp['mask'], masks)
            safe = bool(np.all(iou >= baseline_iou - PARAMS['max_silhouette_regression']))
            accepted = baseline_cost is not None and cost is not None and safe and cost <= baseline_cost - PARAMS['validation_min_gain']
            if accepted and (chosen_cost is None or cost < chosen_cost):
                chosen_cost, chosen_amplitude = cost, amplitude
            candidates.append(dict(amplitude=amplitude, validation_cost=cost,
                silhouette_ious=iou.tolist(), silhouette_gate=safe, accepted=bool(accepted)))
        proposed = mesh.copy(); proposed.vertices = np.asarray(mesh.vertices) + displacement
        accepted_mesh = mesh.copy(); accepted_mesh.vertices = np.asarray(mesh.vertices) + chosen_amplitude * displacement
        proposed.export(path / f'{method}_proposal.ply', encoding='binary')
        accepted_mesh.export(path / f'{method}.obj', include_texture=False)
        pp = raster(proposed.vertices, proposed.faces, k, w, context, PARAMS['metrics_side'])
        ap = raster(accepted_mesh.vertices, accepted_mesh.faces, k, w, context, PARAMS['metrics_side'])
        result = dict(method=method, uses_gt=method.startswith('oracle'), selected_amplitude=chosen_amplitude,
            accepted_edit=chosen_amplitude > 0 and bool(np.any(displacement != 0)), region=region, solver=solver,
            candidates=candidates, proposed=evaluate(proposed, pp), accepted=evaluate(accepted_mesh, ap),
            maximum_displacement=float(np.linalg.norm(chosen_amplitude * displacement, axis=1).max()),
            elapsed_seconds=time.monotonic() - begin)
        write(path / f'{method}.json', result); methods[method] = result
        emit(index, seed, 'mesh_arm_complete', method=method, constraints=solver['constraints'],
            amplitude=chosen_amplitude, cd=result['accepted']['surface']['chamfer_l1'])
    record = dict(time=now(), index=index, uid=rec['uid'], seed=seed, success=True, baseline=baseline,
        input_phase=meta, proposal_diagnostic=proposal_diagnostic(proposals, pred, target), methods=methods,
        elapsed_seconds=time.monotonic() - total_start, no_new_alignment=True, no_training=True)
    write(path / 'result.json', record)
    emit(index, seed, 'complete', seconds=record['elapsed_seconds'])
    return record


def run(indices, seeds, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    setup()
    from visual_evidence_probe.runtime import setup as runtime_setup
    import nvdiffrast.torch as dr
    device = runtime_setup()
    context = dr.RasterizeCudaContext(device=device)
    protocol()
    for index in indices:
        gt = load_gt(index)
        for seed in seeds:
            run_case(index, seed, context, gt)
            gc.collect(); torch.cuda.empty_cache()
        del gt; gc.collect()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['run', 'report'])
    p.add_argument('--indices', default=','.join(map(str, PARAMS['indices'])))
    p.add_argument('--seeds', default='42,43')
    p.add_argument('--gpu', default='3')
    a = p.parse_args()
    if a.action == 'report':
        from .report import main as report
        report()
    else:
        run([int(v) for v in a.indices.split(',')], [int(v) for v in a.seeds.split(',')], a.gpu)


if __name__ == '__main__':
    main()
