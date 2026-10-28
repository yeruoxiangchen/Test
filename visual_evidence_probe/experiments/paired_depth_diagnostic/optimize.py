import copy
import gc
import os
import time
import torch
from .config import OUT,PARAMS,folder,read,write,now,emit
from visual_evidence_probe.experiments.constrained_evidence.data import link
from visual_evidence_probe.experiments.constrained_evidence.anchors import prepare,surface_terms
from visual_evidence_probe.experiments.constrained_evidence.trust import restore
from visual_evidence_probe.experiments.constrained_evidence.objective import terms
from visual_evidence_probe.experiments.dense_evidence.gates import component_summary,select
from visual_evidence_probe.experiments.dense_evidence.optimize import iou
from visual_evidence_probe.experiments.dense_evidence.matching import configured
from visual_evidence_probe.experiments.editor_support.data import feature_std
from visual_evidence_probe.experiments.editor_support.optimize import decode,as_mesh,visible_drift
from visual_evidence_probe.experiments.local_repair import photometric


def run(decoder,z,fit,k,w,context,baseline,record,target,normalization,mesh,index,seed,arm):
    root=folder(index,seed);path=root/arm;path.mkdir(exist_ok=True)
    if (path/'complete.json').exists():return read(path/'complete.json')
    anchored=arm.endswith('_anchored');oracle=arm.startswith('oracle_')
    ids=torch.nonzero(record['active'],as_tuple=False).flatten();features=z.feats.detach().clone();std=feature_std(z.device)
    count=int(target['depth_valid'].sum());structure=component_summary(mesh.vertices,mesh.faces)
    heldout=record['heldout'][0];support=torch.tensor(record['arrays']['validation_support'],device=z.device,dtype=torch.bool)
    heldout_pixels=int(((record['region'][heldout]>.5)&record['observed'][heldout]).sum())
    anchors=None
    if anchored:
        anchors=prepare(mesh,record['arrays']['selected_faces'],PARAMS['anchor_count'],PARAMS['anchor_random_seed']+index*1009+seed,z.device)
        torch.save({key:(value.cpu() if torch.is_tensor(value) else value) for key,value in anchors.items()},path/'anchors.pt')
    def photo(pred):
        with configured(record['train'],heldout):
            return photometric.validation_cost(pred,record['gray'],record['eroded'],k,w,support) if int(support.sum())>=8 else None
    initial_photo=photo(baseline)
    def row(step,pred,current_mesh,values,anchor=None):
        return dict(step=step,**{key:float(value) for key,value in values.items()},
            outside=visible_drift(pred,baseline,record['region']),
            iou_original=iou(pred['mask'],baseline['mask']),iou_observed=iou(pred['mask'],record['observed'],record['observation_valid']),
            validation_cost=initial_photo if step==0 else photo(pred),
            structure=structure if step==0 else component_summary(current_mesh.vertices,current_mesh.faces),
            anchors={key:float(value) for key,value in anchor.items()} if anchor else None)
    with torch.no_grad():first=terms(baseline,target,baseline,record)
    snapshots=[row(0,baseline,mesh,first)];curve=[];rollbacks=[]
    for name in ('step_000.obj','step_000_latent.pt'):link(path/name,root/('original.obj' if name.endswith('.obj') else 'original_latent.pt'))
    torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    if count and len(ids):
        delta=torch.nn.Parameter(torch.zeros((len(ids),features.shape[1]),device=z.device));optimizer=torch.optim.Adam([delta],lr=PARAMS['lr'])
        def current():return z.replace(features.index_add(0,ids,std*PARAMS['normalized_feature_cap']*torch.tanh(delta)))
        safe_delta=delta.detach().clone();safe_state=copy.deepcopy(optimizer.state_dict());safe_step=0
        for step in range(1,PARAMS['steps']+1):
            optimizer.zero_grad(set_to_none=True);latent=current();_,faces,world,pred=decode(decoder,latent,fit,k,w,context)
            values=terms(pred,target,baseline,record)
            loss=values['loss']/normalization+PARAMS['feature_prior_weight']*torch.tanh(delta).square().mean()
            anchor=surface_terms(world,faces,anchors,PARAMS['anchor_neighbors'],PARAMS['anchor_scale']) if anchored else None
            if anchor:loss=loss+PARAMS['anchor_weight']*anchor['loss']+PARAMS['anchor_normal_weight']*anchor['normal_loss']
            (loss*PARAMS['static_loss_scale']).backward()
            if delta.grad is None:raise ValueError('missing latent gradient')
            delta.grad.div_(PARAMS['static_loss_scale'])
            if not bool(torch.isfinite(delta.grad).all()):raise ValueError('nonfinite latent gradient')
            norm=float(delta.grad.norm());torch.nn.utils.clip_grad_norm_([delta],1.);optimizer.step()
            curve.append(dict(step=step,loss=float(loss.detach()),local_depth=float(values['local_depth'].detach()),
                protect=float(values['protect'].detach()),gradient_norm=norm,learning_rate=optimizer.param_groups[0]['lr'],
                anchor_mean=float(anchor['mean']) if anchor else None))
            del latent,world,faces,pred,loss,values,anchor
            if anchored and step%PARAMS['trust_interval']==0:
                with torch.no_grad():
                    latent=current();_,faces,world,pred=decode(decoder,latent,fit,k,w,context)
                    anchor=surface_terms(world,faces,anchors,PARAMS['anchor_neighbors'],PARAMS['anchor_scale'])
                    train=record['train'];outside=visible_drift({key:v[train] for key,v in pred.items()},
                        {key:v[train] for key,v in baseline.items()},record['region'][train])
                    safe=float(anchor['mean'])<=PARAMS['anchor_mean_limit'] and float(anchor['p95'])<=PARAMS['anchor_p95_limit']
                    safe &= outside['depth_mae'] is not None and outside['depth_mae']<=PARAMS['max_outside_depth_mae']
                    safe &= outside['depth_p95'] is not None and outside['depth_p95']<=PARAMS['max_outside_depth_p95']
                    if safe:safe_delta=delta.detach().clone();safe_state=copy.deepcopy(optimizer.state_dict());safe_step=step
                    else:
                        restore(delta,optimizer,safe_delta,safe_state,PARAMS['min_learning_rate'])
                        rollbacks.append(dict(step=step,restored_group_step=safe_step,outside=outside,
                            anchor_mean=float(anchor['mean']),anchor_p95=float(anchor['p95'])))
                    del latent,world,faces,pred,anchor
            if step in PARAMS['checkpoints']:
                with torch.no_grad():
                    latent=current()
                    if not torch.equal(latent.coords,z.coords) or not torch.equal(latent.feats[~record['active']],features[~record['active']]):raise ValueError('frozen SS/rows changed')
                    _,faces,world,pred=decode(decoder,latent,fit,k,w,context);edited=as_mesh(world,faces)
                    anchor=surface_terms(world,faces,anchors,PARAMS['anchor_neighbors'],PARAMS['anchor_scale']) if anchored else None
                    snapshots.append(row(step,pred,edited,terms(pred,target,baseline,record),anchor))
                    edited.export(path/f'step_{step:03d}.obj',include_texture=False)
                    torch.save(dict(features=latent.feats.cpu(),coords=latent.coords.cpu()),path/f'step_{step:03d}_latent.pt')
                write(path/'progress.json',dict(curve=curve,snapshots=snapshots,rollbacks=rollbacks))
                emit(index,seed,'optimizing',arm=arm,step=step,rollbacks=len(rollbacks))
                del latent,world,faces,pred,edited,anchor
        del optimizer,delta
    else:
        for step in PARAMS['checkpoints'][1:]:
            link(path/f'step_{step:03d}.obj',root/'original.obj');link(path/f'step_{step:03d}_latent.pt',root/'original_latent.pt')
    decisions=select(snapshots,structure,count,heldout_pixels)
    decisions.update(input_only=False,privileged_diagnostic=True,gt_pruned_support=True,gt_depth_target=oracle)
    for decision in decisions['observed_decisions']:decision['uses_gt']=True
    for endpoint,step in (('accepted',decisions['observed_step']),('candidate',decisions['own_training_step']),('proposal',PARAMS['steps'] if count and len(ids) else 0)):
        link(path/f'{endpoint}.obj',path/f'step_{step:03d}.obj');link(path/f'{endpoint}_latent.pt',path/f'step_{step:03d}_latent.pt')
    value=dict(time=now(),arm=arm,status='complete' if count and len(ids) else 'no_common_support',
        steps=PARAMS['steps'] if count and len(ids) else 0,query_count=count,shared_normalization=float(normalization),
        elapsed_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        snapshots=snapshots,curve=curve,rollbacks=rollbacks,decisions=decisions,
        privileged_diagnostic=True,gt_support_used=True,gt_depth_in_gradient=oracle,
        heldout_rgb_in_gradient=False,novel_geometry_in_gradient=False,
        frozen_rows_exact=True,original_ss_fixed=True,model_weights_frozen=True,optimizer_backup_deepcopied=True,
        support_sha256=record['meta']['support_sha256'])
    value['cuda_launch_blocking']=os.environ.get('CUDA_LAUNCH_BLOCKING','0')
    write(path/'complete.json',value);gc.collect();torch.cuda.empty_cache();return value
