import gc
import time

import numpy as np
import torch

from .config import PARAMS, write, emit
from .data import feature_std, points_transform
from .objective import raster_tensor, geometry_loss, freeze_target
from .mesh_backend import decode_tensor_mesh


def decode(decoder, z, fit, k, w, context, side=None):
    from .forward_policy import is_active
    if torch.is_grad_enabled() or is_active():
        vertices, faces = decode_tensor_mesh(decoder, z)
    else:
        from pose_aligned_layered_reconstruction.experiments.hssd_operator_audit_20261025.kernel_policy import FixedNative
        with FixedNative(): vertices, faces = decode_tensor_mesh(decoder, z)
    world = points_transform(vertices, fit)
    pred = raster_tensor(world, faces, k, w, context, side or PARAMS['side'])
    return vertices, faces, world, pred


def as_mesh(vertices, faces):
    import trimesh
    return trimesh.Trimesh(vertices.detach().cpu().numpy(), faces.detach().cpu().numpy(), process=False)


@torch.no_grad()
def visible_drift(pred, base, region=None):
    common = pred['mask'] & base['mask']
    if region is not None: common &= region < .5
    error = (pred['depth'] - base['depth']).abs()[common]
    union = pred['mask'] | base['mask']
    mismatch = (pred['mask'] ^ base['mask']).sum((1, 2)) / union.sum((1, 2)).clamp_min(1)
    cosine = (pred['normal'] * base['normal']).sum(-1).abs().clamp(0, 1)
    angles = torch.acos(cosine[common]) * (180 / np.pi)
    return dict(depth_mae=float(error.mean()) if error.numel() else None,
        depth_p95=float(torch.quantile(error, .95)) if error.numel() else None,
        normal_mean_degrees=float(angles.mean()) if angles.numel() else None,
        mask_mismatch_by_view=mismatch.cpu().tolist(),
        mask_mismatch_max=float(mismatch.max()), fixed_common_pixels=int(common.sum()))


def scalar_terms(pred, target, baseline, region, train, controlled=False):
    weights = dict(protect=PARAMS['protect_weight'], local_normal=.1 if controlled else 0.,
        protect_normal=.1, local_depth=1., local_silhouette=.2, protect_depth=1.,
        protect_silhouette=.2, local_coverage=.2, protect_coverage=.2)
    return geometry_loss(pred, target, baseline, region, train, weights=weights,
        depth_scale=PARAMS['local_depth_scale'])


def optimize(decoder, z, active, fit, k, w, context, target, baseline, region,
             target_points, train, objective, steps, checkpoints, path,
             select=None, controlled=False, cap=None, initialize=False):
    from visual_evidence_probe.analysis_scripts.local_repair_latent import field
    from visual_evidence_probe.analysis_scripts.local_repair_latent_deterministic import sdf_at
    from scipy.spatial import cKDTree
    path.mkdir(parents=True, exist_ok=True)
    features = z.feats.detach().clone()
    ids = torch.nonzero(active, as_tuple=False).flatten()
    if not len(ids): raise ValueError('no editable latent rows')
    std = feature_std(z.device)
    delta = torch.nn.Parameter(torch.zeros((len(ids), features.shape[1]), device=z.device))
    optimizer = torch.optim.Adam([delta], lr=PARAMS['initialization_lr'] if initialize else PARAMS['lr'])
    cap = PARAMS['normalized_feature_cap'] if cap is None else cap
    def current(): return z.replace(features.index_add(0, ids, std * cap * torch.tanh(delta)))
    cache = {}
    base_field, outside, proxy_scale = None, None, None
    with torch.no_grad():
        if objective == 'proxy':
            base_field = field(decoder, z, cache).detach()
            corners = cache['positions'].cpu().numpy() / 256 - .5
            centers = (z.coords[ids, 1:].float().cpu().numpy() + .5) / 64 - .5
            outside = torch.tensor(cKDTree(centers).query(corners, workers=2)[0] > 3 / 64, device=z.device)
            proxy_scale = sdf_at(base_field, cache['positions'], target_points).square().mean().clamp_min(1.e-5)
        _, _, _, initial_pred = decode(decoder, z, fit, k, w, context)
        initial_terms = scalar_terms(initial_pred, target, baseline, region, train, controlled)
        geometry_scale = initial_terms['loss'].clamp_min(1.e-4).detach()
    curve, snapshots = [], []
    best_step, best_score, best_features = 0, None, features.cpu()
    start = time.monotonic(); torch.cuda.reset_peak_memory_stats()

    def snapshot(step):
        nonlocal best_step, best_score, best_features
        with torch.no_grad():
            current_z = current()
            if not torch.equal(current_z.feats[~active], features[~active]): raise ValueError('frozen feature rows changed')
            vertices, faces, world, pred = decode(decoder, current_z, fit, k, w, context)
            terms = scalar_terms(pred, target, baseline, region, train, controlled)
            row = dict(step=step, geometry_loss=float(terms['loss']), local_depth=float(terms['local_depth']),
                outside=visible_drift(pred, baseline, region))
            if select is None:
                score, okay = float(terms['loss']), True
            else:
                score, okay, extra = select(pred)
                row.update(extra)
            row.update(selection_score=score, acceptable=bool(okay))
            if okay and score is not None and (best_score is None or score < best_score):
                best_step, best_score, best_features = step, score, current_z.feats.cpu().clone()
            snapshots.append(row)
            if step == steps:
                as_mesh(world, faces).export(path / 'proposal.ply', encoding='binary')
                torch.save(dict(features=current_z.feats.cpu(), coords=current_z.coords.cpu()), path / 'proposal_latent.pt')
            return row
    snapshot(0)
    for step in range(1, steps + 1):
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        lat = current()
        if objective == 'proxy':
            values = field(decoder, lat, cache)
            residual = sdf_at(values, cache['positions'], target_points)
            data_loss = residual.square().mean() / proxy_scale
            protected = (values[outside] - base_field[outside]).square().mean() / proxy_scale if outside.any() and not controlled else data_loss * 0
            loss = data_loss + PARAMS['proxy_field_weight'] * protected
        else:
            _, _, _, pred = decode(decoder, lat, fit, k, w, context)
            terms = scalar_terms(pred, target, baseline, region, train, controlled)
            loss = terms['loss'] / geometry_scale
            data_loss, protected = terms['local'], terms['protect']
        loss = loss + PARAMS['feature_prior_weight'] * torch.tanh(delta).square().mean()
        # Static scaling transports small full-mesh derivatives through frozen FP16 layers.
        scale = 1024.
        (loss * scale).backward()
        if delta.grad is None: raise ValueError('latent gradient missing')
        delta.grad.div_(scale)
        if not torch.isfinite(delta.grad).all(): raise ValueError('nonfinite full-mesh latent gradient')
        norm = float(delta.grad.norm())
        torch.nn.utils.clip_grad_norm_([delta], 1.)
        optimizer.step()
        row = dict(step=step, loss=float(loss.detach()), data_loss=float(data_loss.detach()),
            protected_loss=float(protected.detach()), gradient_norm=norm,
            step_seconds=time.monotonic() - tick)
        curve.append(row)
        del lat, loss, data_loss, protected
        if objective != 'proxy': del pred, terms
        else: del values, residual
        if step in checkpoints:
            snap = snapshot(step)
            write(path / 'progress.json', dict(curve=curve, snapshots=snapshots))
            emit('editor_checkpoint', arm=str(path.relative_to(path.parents[3])), step=step,
                objective=objective, geometry_loss=snap['geometry_loss'], gradient_norm=norm)
    with torch.no_grad():
        accepted = z.replace(best_features.to(z.device))
        _, faces, world, pred = decode(decoder, accepted, fit, k, w, context)
        as_mesh(world, faces).export(path / 'accepted.obj', include_texture=False)
        torch.save(dict(features=best_features, coords=z.coords.cpu()), path / 'accepted_latent.pt')
    result = dict(objective=objective, steps=steps, best_step=best_step, best_score=best_score,
        curve=curve, snapshots=snapshots, actual_edit=best_step > 0,
        elapsed_seconds=time.monotonic() - start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        active_rows=len(ids), total_rows=len(features), frozen_rows_exact=True,
        static_loss_scale=1024., model_weights_frozen=True)
    write(path / 'optimization.json', result)
    gc.collect(); torch.cuda.empty_cache()
    return accepted, result
