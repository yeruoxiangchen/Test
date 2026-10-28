import torch
from visual_evidence_probe.experiments.dense_evidence.optimize import loss_terms as original_loss


def terms(pred,target,baseline,record):
    result=original_loss(pred,target,baseline,record)
    if 'depth_weight' not in target:return result
    train=record['train'];region=record['region'][train]
    domain=target['depth_valid'][train]&target['mask'][train]
    fixed=region*domain*target['depth_weight'][train]
    depth=(pred['depth'][train]-target['depth'][train]).abs().clamp_max(.1)
    depth=torch.where(pred['mask'][train],depth,torch.full_like(depth,.1))
    weighted=(depth/.03*fixed).sum()/fixed.sum().clamp_min(1.)
    difference=weighted-result['local_depth']
    result.update(loss=result['loss']+difference,local=result['local']+difference,local_depth=weighted)
    return result
