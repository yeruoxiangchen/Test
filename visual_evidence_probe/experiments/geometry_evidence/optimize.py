import gc
import time

import torch

from .config import PARAMS, write, emit
from visual_evidence_probe.experiments.editor_support.data import feature_std
from visual_evidence_probe.experiments.editor_support.optimize import decode, as_mesh, visible_drift
from visual_evidence_probe.experiments.editor_support.objective import geometry_loss


def terms(pred, target, baseline, region, train, normals=False):
    return geometry_loss(pred, target, baseline, region, train,
        weights=dict(local_normal=.1 if normals else 0., protect_normal=.1,
            local_depth=1., local_silhouette=.2, local_coverage=.2,
            protect_depth=1., protect_silhouette=.2, protect_coverage=.2, protect=1.),
        depth_scale=PARAMS['local_depth_scale'], missing_depth=PARAMS['missing_depth'])


def run(decoder, z, fit, k, w, context, baseline, record, target, name, path, index, seed):
    """Optimize against own training evidence, with no evaluation callback."""
    path.mkdir(parents=True, exist_ok=True)
    features = z.feats.detach().clone()
    ids = torch.nonzero(record['active'], as_tuple=False).flatten()
    if not len(ids): raise ValueError('no editable latent rows')
    delta = torch.nn.Parameter(torch.zeros((len(ids), features.shape[1]), device=z.device))
    std = feature_std(z.device)
    optimizer = torch.optim.Adam([delta], lr=PARAMS['lr'])
    def latent():
        return z.replace(features.index_add(0, ids,
            std * PARAMS['normalized_feature_cap'] * torch.tanh(delta)))
    normal = name == 'oracle_dense_normal'
    def loss_terms(pred): return terms(pred, target, baseline, record['region'], record['train'], normal)
    with torch.no_grad():
        first = loss_terms(baseline)
        normalization = first['loss'].clamp_min(1.e-4).detach()
    curve, snapshots = [], []
    best_score, best_step, best_features = None, 0, features.cpu()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()

    def snapshot(step):
        nonlocal best_score, best_step, best_features
        with torch.no_grad():
            current = latent()
            if not torch.equal(current.feats[~record['active']], features[~record['active']]):
                raise ValueError('frozen latent rows changed')
            _, faces, vertices, pred = decode(decoder, current, fit, k, w, context)
            loss = loss_terms(pred)
            score = float(loss['loss'])
            row = dict(step=step, **{key: float(value) for key, value in loss.items()},
                outside=visible_drift(pred, baseline, record['region']))
            if best_score is None or score < best_score:
                best_score, best_step, best_features = score, step, current.feats.cpu().clone()
            snapshots.append(row)
            if step == PARAMS['steps']:
                as_mesh(vertices, faces).export(path / 'proposal.obj', include_texture=False)
                torch.save(dict(features=current.feats.cpu(), coords=current.coords.cpu()), path / 'proposal_latent.pt')
            write(path / 'progress.json', dict(curve=curve, snapshots=snapshots))
            emit(index, seed, 'optimizing', arm=name, step=step, steps=PARAMS['steps'],
                loss=score, best_step=best_step)
    snapshot(0)
    for step in range(1, PARAMS['steps'] + 1):
        optimizer.zero_grad(set_to_none=True)
        current = latent()
        _, _, _, pred = decode(decoder, current, fit, k, w, context)
        values = loss_terms(pred)
        loss = values['loss'] / normalization + PARAMS['feature_prior_weight'] * torch.tanh(delta).square().mean()
        (loss * PARAMS['static_loss_scale']).backward()
        if delta.grad is None: raise ValueError('missing editable-feature gradient')
        delta.grad.div_(PARAMS['static_loss_scale'])
        if not torch.isfinite(delta.grad).all(): raise ValueError('nonfinite editable-feature gradient')
        norm = float(delta.grad.norm())
        if step == 1 and norm == 0 and float(values['loss']) > 1.e-5:
            raise ValueError('nonzero geometry loss has zero editable-feature gradient')
        torch.nn.utils.clip_grad_norm_([delta], 1.)
        optimizer.step()
        curve.append(dict(step=step, loss=float(loss.detach()), gradient_norm=norm,
            local=float(values['local'].detach()), protect=float(values['protect'].detach())))
        del current, pred, values, loss
        if step in PARAMS['checkpoints']: snapshot(step)
    with torch.no_grad():
        selected = z.replace(best_features.to(z.device))
        _, faces, vertices, _ = decode(decoder, selected, fit, k, w, context)
        as_mesh(vertices, faces).export(path / 'selected.obj', include_texture=False)
        torch.save(dict(features=best_features, coords=selected.coords.cpu()), path / 'selected_latent.pt')
    result = dict(status='complete', arm=name, best_step=best_step,
        actual_selected_edit=best_step > 0, best_training_score=best_score,
        initial_training_score=float(first['loss']), steps=PARAMS['steps'],
        snapshots=snapshots, curve=curve, elapsed_seconds=time.monotonic() - start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        active_rows=len(ids), total_rows=len(features), frozen_rows_exact=True,
        normalized_feature_cap=PARAMS['normalized_feature_cap'],
        weights_frozen=True, selection_uses_own_training_target=True,
        novel_views_used_in_selection=False, uses_gt=name != 'input_sparse')
    write(path / 'optimization.json', result)
    del delta, optimizer, selected
    gc.collect(); torch.cuda.empty_cache()
    return result
