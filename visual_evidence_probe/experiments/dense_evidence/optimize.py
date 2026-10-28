import gc
import os
import time

import numpy as np
import torch

from .config import PARAMS,folder,write,emit
from .gates import component_summary,select
from .matching import configured
from visual_evidence_probe.experiments.editor_support.data import feature_std
from visual_evidence_probe.experiments.editor_support.optimize import decode,as_mesh,visible_drift
from visual_evidence_probe.experiments.editor_support.objective import geometry_loss
from visual_evidence_probe.experiments.local_repair import photometric


def loss_terms(pred,target,baseline,record):
    return geometry_loss(pred,target,baseline,record['region'],record['train'],
        weights=dict(local_normal=0.,protect_normal=.1,local_depth=1.,local_silhouette=.2,
            local_coverage=.2,protect_depth=1.,protect_silhouette=.2,protect_coverage=.2,protect=1.),
        depth_scale=PARAMS['depth_scale'])


def iou(pred,reference,domain=None):
    if domain is None:domain=torch.ones_like(pred)
    return (((pred&reference)&domain).sum((1,2))/((pred|reference)&domain).sum((1,2)).clamp_min(1)).cpu().tolist()


def link(path,source):
    if not path.exists():path.symlink_to(source)


def run_source(decoder,z,fit,k,w,context,baseline,record,name,original_mesh,index,seed):
    root=folder(index,seed);path=root/name;path.mkdir(parents=True,exist_ok=True)
    if (path/'complete.json').exists():
        from .config import read
        return read(path/'complete.json')
    target=record['targets'][name];count=record['counts'][name]
    active=record['active'];ids=torch.nonzero(active,as_tuple=False).flatten()
    features=z.feats.detach().clone();std=feature_std(z.device)
    original_structure=component_summary(original_mesh.vertices,original_mesh.faces)
    snapshots=[];curve=[]
    for endpoint in ('obj','pt'):
        link(path/('step_000.obj' if endpoint=='obj' else 'step_000_latent.pt'),root/('original.obj' if endpoint=='obj' else 'original_latent.pt'))
    _,gray,_,eroded=photometric.prepare_images(record['images'],record['masks'],PARAMS['side'])
    supports=torch.tensor(record['arrays']['validation_support'],device=k.device,dtype=torch.bool)
    heldout=record['heldout'][0]

    def photo(pred):
        with configured(record['train'],heldout):
            return photometric.validation_cost(pred,gray,eroded,k,w,supports) if int(supports.sum())>=8 else None
    baseline_validation=photo(baseline)
    observed_pixels=int(((record['region'][heldout]>.5)&record['observed'][heldout]).sum())
    baseline_terms=loss_terms(baseline,target,baseline,record)
    def row(step,pred,mesh,values):
        return dict(step=step,**{key:float(value) for key,value in values.items()},
            outside=visible_drift(pred,baseline,record['region']),
            iou_original=iou(pred['mask'],baseline['mask']),
            iou_observed=iou(pred['mask'],record['observed'],record['observation_valid']),
            validation_cost=baseline_validation if step==0 else photo(pred),
            structure=original_structure if step==0 else component_summary(mesh.vertices,mesh.faces))
    snapshots.append(row(0,baseline,original_mesh,baseline_terms))
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    if count and len(ids):
        delta=torch.nn.Parameter(torch.zeros((len(ids),features.shape[1]),device=z.device))
        optimizer=torch.optim.Adam([delta],lr=PARAMS['lr'])
        normalization=baseline_terms['loss'].clamp_min(1.e-4).detach()
        def current():return z.replace(features.index_add(0,ids,std*PARAMS['normalized_feature_cap']*torch.tanh(delta)))
        for step in range(1,PARAMS['steps']+1):
            optimizer.zero_grad(set_to_none=True)
            lat=current();_,_,_,pred=decode(decoder,lat,fit,k,w,context)
            values=loss_terms(pred,target,baseline,record)
            loss=values['loss']/normalization+PARAMS['feature_prior_weight']*torch.tanh(delta).square().mean()
            (loss*PARAMS['static_loss_scale']).backward()
            if delta.grad is None:raise ValueError('missing latent gradient')
            delta.grad.div_(PARAMS['static_loss_scale'])
            if not bool(torch.isfinite(delta.grad).all()):raise ValueError('nonfinite latent gradient')
            norm=float(delta.grad.norm());torch.nn.utils.clip_grad_norm_([delta],1.);optimizer.step()
            curve.append(dict(step=step,loss=float(loss.detach()),gradient_norm=norm,
                local_depth=float(values['local_depth'].detach()),protect=float(values['protect'].detach())))
            del lat,pred,loss,values
            if step in PARAMS['checkpoints']:
                with torch.no_grad():
                    lat=current()
                    if not torch.equal(lat.feats[~active],features[~active]):raise ValueError('frozen feature rows changed')
                    _,faces,world,pred=decode(decoder,lat,fit,k,w,context)
                    mesh=as_mesh(world,faces)
                    mesh.export(path/f'step_{step:03d}.obj',include_texture=False)
                    torch.save(dict(features=lat.feats.cpu(),coords=lat.coords.cpu()),path/f'step_{step:03d}_latent.pt')
                    snapshots.append(row(step,pred,mesh,loss_terms(pred,target,baseline,record)))
                write(path/'progress.json',dict(snapshots=snapshots,curve=curve))
                emit(index,seed,'optimizing',source=name,step=step,steps=PARAMS['steps'])
                del mesh,world,faces,pred,lat
        del delta,optimizer
    else:
        for step in PARAMS['checkpoints'][1:]:
            link(path/f'step_{step:03d}.obj',root/'original.obj')
            link(path/f'step_{step:03d}_latent.pt',root/'original_latent.pt')
    decisions=select(snapshots,original_structure,count,observed_pixels)
    for endpoint,step in (('proposal',PARAMS['steps']),('train_selected',decisions['own_training_step'])):
        link(path/f'{endpoint}.obj',path/f'step_{step:03d}.obj')
        link(path/f'{endpoint}_latent.pt',path/f'step_{step:03d}_latent.pt')
    result=dict(status='complete' if count and len(ids) else 'no_evidence',source=name,query_count=count,
        steps=PARAMS['steps'] if count and len(ids) else 0,elapsed_seconds=time.monotonic()-start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),snapshots=snapshots,curve=curve,
        decisions=decisions,active_rows=len(ids),total_rows=len(features),frozen_rows_exact=True,
        cuda_launch_blocking=os.environ.get('CUDA_LAUNCH_BLOCKING','0'),
        heldout_pixels=observed_pixels,uses_gt=False,model_weights_frozen=True,shared_policy_trajectory=True)
    write(path/'complete.json',result)
    gc.collect();torch.cuda.empty_cache()
    return result
