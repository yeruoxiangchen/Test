import copy
import gc
import time
import torch

from .config import OUT,PRIOR,PARAMS,folder,read,write,now,emit
from .data import link
from .anchors import prepare,surface_terms
from .objective import terms
from .trust import restore
from visual_evidence_probe.experiments.dense_evidence.gates import component_summary,select
from visual_evidence_probe.experiments.dense_evidence.optimize import iou
from visual_evidence_probe.experiments.dense_evidence.matching import configured
from visual_evidence_probe.experiments.editor_support.data import feature_std
from visual_evidence_probe.experiments.editor_support.optimize import decode,as_mesh,visible_drift
from visual_evidence_probe.experiments.local_repair import photometric


def reuse_reference(index,seed):
    root=folder(index,seed);path=root/'dense_reference';source=PRIOR/'objects'/f'{index:03d}'/str(seed)/'dense'
    path.mkdir(exist_ok=True)
    for step in PARAMS['checkpoints']:
        for name in (f'step_{step:03d}.obj',f'step_{step:03d}_latent.pt'):link(path/name,source/name)
    value=read(source/'complete.json')
    write(path/'complete.json',dict(value,arm='dense_reference',reused_round7=True,reference_seconds=value['elapsed_seconds'],elapsed_seconds=0.))
    seal(path,value)


def seal(path,value):
    for name,step in (('accepted',value['decisions']['observed_step']),('candidate',value['decisions']['own_training_step']),
            ('proposal',PARAMS['steps'] if value['steps'] else 0)):
        link(path/f'{name}.obj',path/f'step_{step:03d}.obj')
        link(path/f'{name}_latent.pt',path/f'step_{step:03d}_latent.pt')


def run(decoder,z,fit,k,w,context,baseline,record,mesh,index,seed,arm):
    path=folder(index,seed)/arm;path.mkdir(exist_ok=True)
    if (path/'complete.json').exists():return read(path/'complete.json')
    target=record['targets'][arm];count=record['counts'][arm]
    active=record['active'];ids=torch.nonzero(active,as_tuple=False).flatten()
    features=z.feats.detach().clone();std=feature_std(z.device)
    anchors=prepare(mesh,record['arrays']['selected_faces'],PARAMS['anchor_count'],PARAMS['anchor_random_seed']+index*1009+seed,z.device)
    torch.save({key:(value.cpu() if torch.is_tensor(value) else value) for key,value in anchors.items()},path/'anchors.pt')
    structure=component_summary(mesh.vertices,mesh.faces);heldout=record['heldout'][0]
    support=torch.tensor(record['arrays']['validation_support'],device=z.device,dtype=torch.bool)
    heldout_pixels=int(((record['region'][heldout]>.5)&record['observed'][heldout]).sum())
    def photo(pred):
        with configured(record['train'],heldout):
            return photometric.validation_cost(pred,record['gray'],record['eroded'],k,w,support) if int(support.sum())>=8 else None
    base_photo=photo(baseline)
    def snapshot(step,pred,current_mesh,loss,anchor):
        return dict(step=step,**{key:float(value) for key,value in loss.items()},
            outside=visible_drift(pred,baseline,record['region']),iou_original=iou(pred['mask'],baseline['mask']),
            iou_observed=iou(pred['mask'],record['observed'],record['observation_valid']),
            validation_cost=base_photo if step==0 else photo(pred),
            structure=structure if step==0 else component_summary(current_mesh.vertices,current_mesh.faces),
            anchors={key:float(value) for key,value in anchor.items()})
    with torch.no_grad():
        base_terms=terms(baseline,target,baseline,record)
        verts=torch.tensor(mesh.vertices,device=z.device,dtype=torch.float32);faces=torch.tensor(mesh.faces,device=z.device)
        initial_anchor=surface_terms(verts,faces,anchors,PARAMS['anchor_neighbors'],PARAMS['anchor_scale'])
        if float(initial_anchor['mean'])>1.e-6:raise ValueError('surface anchor baseline mismatch')
    snapshots=[snapshot(0,baseline,mesh,base_terms,initial_anchor)];curve=[];rollbacks=[]
    for name in ('step_000.obj','step_000_latent.pt'):link(path/name,folder(index,seed)/('original.obj' if name.endswith('.obj') else 'original_latent.pt'))
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    if count and len(ids):
        delta=torch.nn.Parameter(torch.zeros((len(ids),features.shape[1]),device=z.device));optimizer=torch.optim.Adam([delta],lr=PARAMS['lr'])
        def latent():return z.replace(features.index_add(0,ids,std*PARAMS['normalized_feature_cap']*torch.tanh(delta)))
        normalization=base_terms['loss'].clamp_min(1.e-4).detach()
        safe_delta=delta.detach().clone();safe_state=copy.deepcopy(optimizer.state_dict());safe_step=0
        for step in range(1,PARAMS['steps']+1):
            optimizer.zero_grad(set_to_none=True);current=latent()
            _,faces,world,pred=decode(decoder,current,fit,k,w,context)
            values=terms(pred,target,baseline,record)
            anchor=surface_terms(world,faces,anchors,PARAMS['anchor_neighbors'],PARAMS['anchor_scale'])
            loss=values['loss']/normalization+PARAMS['anchor_weight']*anchor['loss']+PARAMS['anchor_normal_weight']*anchor['normal_loss']
            loss+=PARAMS['feature_prior_weight']*torch.tanh(delta).square().mean()
            (loss*PARAMS['static_loss_scale']).backward()
            if delta.grad is None:raise ValueError('missing latent gradient')
            delta.grad.div_(PARAMS['static_loss_scale'])
            if not bool(torch.isfinite(delta.grad).all()):raise ValueError('nonfinite latent gradient')
            norm=float(delta.grad.norm());torch.nn.utils.clip_grad_norm_([delta],1.);optimizer.step()
            curve.append(dict(step=step,loss=float(loss.detach()),local_depth=float(values['local_depth'].detach()),
                gradient_norm=norm,anchor_mean=float(anchor['mean']),anchor_p95=float(anchor['p95']),learning_rate=optimizer.param_groups[0]['lr']))
            del current,world,faces,pred,loss,values,anchor
            if step%PARAMS['trust_interval']==0:
                with torch.no_grad():
                    current=latent();_,faces,world,pred=decode(decoder,current,fit,k,w,context)
                    a=surface_terms(world,faces,anchors,PARAMS['anchor_neighbors'],PARAMS['anchor_scale'])
                    # Only train-view exterior depth and original 3D anchors
                    # constrain optimization. Heldout RGB stays selection-only.
                    train=record['train'];outside=visible_drift({key:value[train] for key,value in pred.items()},
                        {key:value[train] for key,value in baseline.items()},record['region'][train])
                    safe=float(a['mean'])<=PARAMS['anchor_mean_limit'] and float(a['p95'])<=PARAMS['anchor_p95_limit']
                    safe &= outside['depth_mae'] is not None and outside['depth_mae']<=PARAMS['max_outside_depth_mae']
                    safe &= outside['depth_p95'] is not None and outside['depth_p95']<=PARAMS['max_outside_depth_p95']
                    if safe:
                        safe_delta=delta.detach().clone();safe_state=copy.deepcopy(optimizer.state_dict());safe_step=step
                    else:
                        restore(delta,optimizer,safe_delta,safe_state,PARAMS['min_learning_rate'])
                        rollbacks.append(dict(attempted_step=step,restored_group_step=safe_step,anchor_mean=float(a['mean']),anchor_p95=float(a['p95']),outside=outside))
                    del current,faces,world,pred,a
            if step in PARAMS['checkpoints']:
                with torch.no_grad():
                    current=latent()
                    if not torch.equal(current.feats[~active],features[~active]):raise ValueError('frozen latent rows changed')
                    _,faces,world,pred=decode(decoder,current,fit,k,w,context);edited=as_mesh(world,faces)
                    a=surface_terms(world,faces,anchors,PARAMS['anchor_neighbors'],PARAMS['anchor_scale'])
                    snapshots.append(snapshot(step,pred,edited,terms(pred,target,baseline,record),a))
                    edited.export(path/f'step_{step:03d}.obj',include_texture=False)
                    torch.save(dict(features=current.feats.cpu(),coords=current.coords.cpu()),path/f'step_{step:03d}_latent.pt')
                write(path/'progress.json',dict(curve=curve,snapshots=snapshots,rollbacks=rollbacks))
                emit(index,seed,'optimizing',arm=arm,step=step,rollbacks=len(rollbacks))
                del current,faces,world,pred,edited,a
        del optimizer,delta
    else:
        for step in PARAMS['checkpoints'][1:]:
            link(path/f'step_{step:03d}.obj',folder(index,seed)/'original.obj');link(path/f'step_{step:03d}_latent.pt',folder(index,seed)/'original_latent.pt')
    decisions=select(snapshots,structure,count,heldout_pixels)
    value=dict(time=now(),status='complete' if count and len(ids) else 'no_evidence',arm=arm,
        steps=PARAMS['steps'] if count and len(ids) else 0,query_count=count,elapsed_seconds=time.monotonic()-start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),decisions=decisions,curve=curve,snapshots=snapshots,
        rollbacks=rollbacks,heldout_pixels=heldout_pixels,frozen_rows_exact=True,uses_gt=False,
        model_weights_frozen=True,original_ss_fixed=True,anchor_kind='K-neighbor closest triangle surface, not nearest vertex')
    value['optimizer_snapshot_load_deepcopied']=True
    write(path/'complete.json',value);seal(path,value);gc.collect();torch.cuda.empty_cache();return value
