"""Proper input-only similarity candidates, with a retained original fit."""
import numpy as np
import torch

from .config import PARAMS
from .evidence import mask_cost
from ...geometry import transform,render,cube_rotations,similarity,mask_iou
from visual_evidence_probe.experiments.local_repair import photometric as photo
from visual_evidence_probe.experiments.dense_evidence.matching import configured


def photo_score(pred,data):
    if int(data['supports'].sum())<8:return None
    with configured(data['train'],data['heldout'][0]):
        return photo.validation_cost(pred,data['gray'],data['eroded'],data['k'],data['w'],data['supports'])


def depth_cost(pred,trusted,baseline,train):
    valid=trusted['valid'][train];count=int(valid.sum())
    if count<PARAMS['min_trusted_pixels']:return pred['depth'].sum()*0
    target=trusted['depth'][train]
    residual=pred['depth'][train]-target
    missing=((baseline['depth'][train]-target).abs()+.05).clamp_min(.1).detach()
    error=torch.where(pred['mask'][train],residual.abs(),missing)
    return ((torch.sqrt((error/.03).square()+.01**2)-.01)*valid).sum()/valid.sum()


def refine(data,trusted,context):
    base=data['old_baseline'];old=data['old_matrix'];device=old.device
    views=list(range(len(data['k'])));coarse=[]
    def loss(pred):return mask_cost(pred,data['observed'],views)+.05*depth_cost(pred,trusted,base,data['train'])
    with torch.no_grad():
        old_ious=np.asarray(mask_iou(base['mask'],data['observed']));old_photo=photo_score(base,data)
        initial=float(loss(base));coarse.append((initial,old.clone(),'original'))
        for index,r in enumerate(cube_rotations()):
            m=old.clone();m[:3,:3]=old[:3,:3]@torch.tensor(r,device=device)
            pred=render(transform(data['native'],m),data['faces'],data['k'],data['w'],context,96)
            observed=torch.nn.functional.interpolate(data['observed'][:,None].float(),size=(96,96),mode='area')[:,0]>.5
            score=float(mask_cost(pred,observed,views));coarse.append((score,m,index))
    # Coarse scores are only orientation proposals, never compared to the final
    # full-resolution training objective. Always retain the exact original.
    finalists=sorted(coarse[1:],key=lambda x:x[0])[:PARAMS['registration_finalists']]
    candidates=[dict(matrix=old.clone(),cost=initial,origin='original')]
    for _,start,rid in finalists:
        raw=torch.nn.Parameter(torch.zeros(7,device=device));optimizer=torch.optim.Adam([raw],lr=PARAMS['registration_lr'])
        for step in range(PARAMS['registration_steps']+1):
            rotation,scale,translation=similarity(raw,torch.eye(3,device=device))
            m=torch.eye(4,device=device);m[:3,:3]=scale*rotation@start[:3,:3]
            m[:3,3]=scale*(rotation@start[:3,3])+translation
            pred=render(transform(data['native'],m),data['faces'],data['k'],data['w'],context)
            value=loss(pred)
            if step%10==0:candidates.append(dict(matrix=m.detach().clone(),cost=float(value.detach()),origin=f'rotation_{rid}_step_{step}'))
            if step<PARAMS['registration_steps']:
                optimizer.zero_grad(set_to_none=True);(value+.001*raw.square().mean()).backward();optimizer.step()
        del optimizer,raw
    checked=[];eligible=[]
    with torch.no_grad():
        for candidate in candidates:
            m=candidate['matrix'];pred=render(transform(data['native'],m),data['faces'],data['k'],data['w'],context)
            ious=np.asarray(mask_iou(pred['mask'],data['observed']));p=photo_score(pred,data)
            safe=bool((ious-old_ious).min()>=-.005)
            if p is not None and old_photo is not None:safe &= p<=old_photo+.002
            improved=candidate['cost']<initial-1.e-4
            row=dict(origin=candidate['origin'],cost=candidate['cost'],mean_iou=float(ious.mean()),
                     photo=p,safe=safe,improved=improved,matrix=m.cpu().tolist())
            checked.append(row)
            if safe and improved:eligible.append(candidate)
        chosen=min(eligible,key=lambda c:c['cost']) if eligible else candidates[0]
    matrix=chosen['matrix']
    return matrix,dict(gt_used=False,old_matrix=old.cpu().tolist(),matrix=matrix.cpu().tolist(),
           changed=not torch.equal(matrix,old),candidates=checked,initial_cost=initial,selected_origin=chosen['origin'],
           proper_rotation=True,all_input_masks_used=True,heldout_rgb_for_selection_only=True)
