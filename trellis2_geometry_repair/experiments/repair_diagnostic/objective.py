"""Uncapped robust depth supervision with a fixed missing-ray penalty."""
import torch

from .config import PARAMS


def charbonnier(error,scale=.03,epsilon=.01):
    return torch.sqrt((error/scale).square()+epsilon**2)-epsilon


def robust_depth(pred,target,baseline,region,views):
    # The query domain, denominator and missing costs are fixed before editing.
    views=torch.as_tensor(views,device=pred['depth'].device,dtype=torch.long)
    ref=target['depth'][views].detach()
    valid=target['depth_valid'][views].detach().bool() & torch.isfinite(ref) & (ref>0)
    weight=region[views].detach()*valid
    error=pred['depth'][views]-torch.nan_to_num(ref)
    old_error=(baseline['depth'][views].detach()-torch.nan_to_num(ref)).abs()
    missing=(old_error+PARAMS['missing_margin']).clamp_min(.1)
    hit=pred['mask'][views] & torch.isfinite(pred['depth'][views]) & (pred['depth'][views]>0)
    cost=charbonnier(error,PARAMS['depth_scale'],PARAMS['charbonnier_epsilon'])
    fixed_missing=charbonnier(missing,PARAMS['depth_scale'],PARAMS['charbonnier_epsilon'])
    cost=torch.where(hit,cost,fixed_missing)
    return (cost*weight).sum()/weight.sum().clamp_min(1)


def terms(pred,target,baseline,region,views):
    from visual_evidence_probe.experiments.editor_support.objective import geometry_loss
    # Keep the existing rendering, normal, silhouette and coverage definitions.
    # Only the depth objective changes; old experiments keep their own function.
    result=geometry_loss(pred,target,baseline,region,views,depth_scale=PARAMS['depth_scale'],
          weights=dict(local_depth=0.,local_silhouette=.2,local_coverage=.2,local_normal=0.,
                       protect_depth=0.,protect_silhouette=.2,protect_normal=.1,protect_coverage=.2,protect=1.))
    local=robust_depth(pred,target,baseline,region,views)
    protected=dict(baseline,depth_valid=baseline['mask'])
    outside=robust_depth(pred,protected,baseline,1-region,views)
    result.update(local_depth=local,protect_depth=outside,
                  local=result['local']+local,protect=result['protect']+outside,
                  loss=result['loss']+local+outside)
    return result


def summarize_errors(pred,target,region,views):
    view=torch.as_tensor(views,device=pred['depth'].device,dtype=torch.long)
    valid=target['depth_valid'][view] & (region[view]>.5)
    hit=pred['mask'][view]&valid
    delta=(pred['depth'][view]-target['depth'][view])[hit]
    if not len(delta):return dict(queries=int(valid.sum()),hits=0)
    absolute=delta.abs()
    return dict(queries=int(valid.sum()),hits=len(delta),raw_mae=float(absolute.mean()),
                signed_mean=float(delta.mean()),median=float(absolute.median()),
                p95=float(torch.quantile(absolute,.95)),clipped_fraction=float((absolute>=.1).float().mean()),
                near_cap_fraction=float((absolute>=.09).float().mean()),
                lost_fraction=float((valid&~pred['mask'][view]).sum()/valid.sum().clamp_min(1)))
