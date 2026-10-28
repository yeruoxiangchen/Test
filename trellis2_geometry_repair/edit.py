import argparse
import gc
import time

import numpy as np
import torch

from .config import OUT,PARAMS,folder,read,write,setup,snapshot,emit,now
from .runtime import load_model,NativePath,save_mesh,release
from .geometry import observations,render,transform,detached,mask_iou
from .evidence import build,photo_cost


def loss_terms(pred,target,baseline,record):
    from visual_evidence_probe.experiments.editor_support.objective import geometry_loss
    return geometry_loss(pred,target,baseline,record['region'],record['train'],depth_scale=.03,
        weights=dict(local_depth=1.,local_silhouette=.2,local_coverage=.2,local_normal=0.,
                     protect_depth=1.,protect_silhouette=.2,protect_normal=.1,protect_coverage=.2,protect=1.))


def structure(vertices,faces):
    from visual_evidence_probe.experiments.dense_evidence.gates import component_summary
    return component_summary(vertices.detach().cpu().numpy(),faces.detach().cpu().numpy())


def select(rows):
    base=rows[0];eligible=[];decisions=[]
    for row in rows[1:]:
        deltas=np.asarray(row['ious'])-np.asarray(base['ious'])
        outside=row['outside'];s=row['structure'];b=base['structure']
        mask_ok=bool(deltas.min()>=-PARAMS['mask_tolerance'])
        drift_ok=outside['mean'] is not None and outside['mean']<=PARAMS['max_outside_mean']
        drift_ok&=outside['p95'] is not None and outside['p95']<=PARAMS['max_outside_p95']
        drift_ok&=outside['lost_fraction']<=.005
        gain=(base['local_depth']-row['local_depth'])/max(base['local_depth'],1.e-8)
        photo_ok=row['photo']<=base['photo']+.002 if row['photo'] is not None and base['photo'] is not None else row['heldout_pixels']>=8
        topo_ok=s['large_component_count']<=b['large_component_count']
        topo_ok&=s['small_component_area_fraction']<=b['small_component_area_fraction']+.002
        topo_ok&=.8<=s['total_surface_area']/max(b['total_surface_area'],1.e-12)<=1.25
        okay=bool(mask_ok and drift_ok and photo_ok and topo_ok and row['queries']>=16 and gain>=.1)
        score=(row['photo']-base['photo'] if row['photo'] is not None and base['photo'] is not None else 0)-.5*float(deltas.mean())
        item=dict(step=row['step'],accepted=okay,mask_safe=mask_ok,outside_safe=bool(drift_ok),
                  photo_safe=bool(photo_ok),topology_safe=bool(topo_ok),depth_gain=float(gain),score=float(score))
        decisions.append(item)
        if okay:eligible.append(item)
    accepted=min(eligible,key=lambda r:(r['score'],-r['depth_gain'],r['step']))['step'] if eligible else 0
    candidate=min(rows,key=lambda r:(r['loss'],r['step']))['step']
    return dict(accepted_step=accepted,candidate_step=candidate,checks=decisions,gt_used=False)


def link(path,source):
    if path.exists():
        if path.resolve()!=source.resolve():raise RuntimeError('stale endpoint link')
    else:path.symlink_to(source)


def one(decoder,index,seed,context):
    root=folder(index,seed)
    if (root/'input_complete.json').exists():return
    saved=torch.load(root/'original_latent.pt',weights_only=False)
    features=saved['features'].cuda().float();coords=saved['coords'].cuda();std=saved['std'].cuda()
    matrix=torch.tensor(read(root/'fit.json')['matrix'],device='cuda:0',dtype=torch.float32)
    _,k,w,_,_=observations(index);branch=NativePath(decoder,features,coords)
    try:
        with torch.no_grad():
            v,f=branch(features)
            parity=float((v-branch.base_vertices).abs().max())
            if parity>2.e-7:raise RuntimeError(f'branch/native parity failed {parity}')
            baseline=detached(render(transform(v,matrix),f,k,w,context))
            record=build(index,seed,features,coords,matrix,baseline,f,transform(v,matrix))
        rows=torch.nonzero(record['active']).flatten();query_count=int(record['target']['depth_valid'].sum())
        snapshots=[];curve=[];torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        def snapshot(step,current):
            with torch.no_grad():
                v,f=branch.native(current);world=transform(v,matrix);pred=render(world,f,k,w,context)
                terms=loss_terms(pred,record['target'],baseline,record)
                common=pred['mask']&baseline['mask']&(record['region']<.5)
                distance=(pred['depth']-baseline['depth']).abs()[common]
                outside_original=baseline['mask']&(record['region']<.5)
                lost=float((~pred['mask'])[outside_original].float().mean()) if bool(outside_original.any()) else 0.
                row=dict(step=step,**{key:float(value) for key,value in terms.items()},
                    ious=mask_iou(pred['mask'],record['observed']),photo=photo_cost(pred,record),queries=query_count,
                    heldout_pixels=int(((record['region'][record['heldout'][0]]>.5)&record['observed'][record['heldout'][0]]).sum()),
                    outside=dict(mean=float(distance.mean()) if distance.numel() else None,
                                 p95=float(torch.quantile(distance,.95)) if distance.numel() else None,lost_fraction=lost),
                    structure=structure(world,f),native_decode=True,vertices=len(v),faces=len(f))
                snapshots.append(row);save_mesh(root/f'step_{step:03d}.obj',world,f)
                if not torch.equal(current[~record['active']],features[~record['active']]):raise RuntimeError('outside latent changed')
                torch.save(dict(features=current.cpu(),coords=coords.cpu(),space=saved['space'],resolution=512),root/f'step_{step:03d}_latent.pt')
                write(root/'progress.json',dict(snapshots=snapshots,curve=curve))
        snapshot(0,features)
        if len(rows) and query_count>=16 and not read(root/'fit.json')['low_registration_quality']:
            delta=torch.nn.Parameter(torch.zeros((len(rows),32),device='cuda:0'))
            optimizer=torch.optim.Adam([delta],lr=PARAMS['lr'])
            normalization=max(snapshots[0]['loss'],1.e-4)
            def current():return features.index_add(0,rows,std*PARAMS['feature_cap']*torch.tanh(delta))
            for step in range(1,PARAMS['steps']+1):
                optimizer.zero_grad(set_to_none=True);v,f=branch(current())
                pred=render(transform(v,matrix),f,k,w,context)
                terms=loss_terms(pred,record['target'],baseline,record)
                loss=terms['loss']/normalization+PARAMS['feature_prior']*torch.tanh(delta).square().mean()
                (loss*PARAMS['loss_scale']).backward();delta.grad.div_(PARAMS['loss_scale'])
                if not bool(torch.isfinite(delta.grad).all()):raise RuntimeError('nonfinite latent gradient')
                norm=float(delta.grad.norm());torch.nn.utils.clip_grad_norm_([delta],1.);optimizer.step()
                curve.append(dict(step=step,loss=float(loss.detach()),gradient_norm=norm))
                del loss,v,f,pred,terms
                if step in PARAMS['checkpoints']:
                    snapshot(step,current().detach());emit(index,seed,'editing',step=step)
            del optimizer,delta
        decisions=select(snapshots)
        for endpoint,step in [('accepted',decisions['accepted_step']),('candidate',decisions['candidate_step']),('proposal',snapshots[-1]['step'])]:
            link(root/f'{endpoint}.obj',root/f'step_{step:03d}.obj')
            link(root/f'{endpoint}_latent.pt',root/f'step_{step:03d}_latent.pt')
        value=dict(time=now(),index=index,seed=seed,decisions=decisions,snapshots=snapshots,curve=curve,
                   gt_used=False,weights_frozen=True,ss_fixed=True,outside_latent_rows_exact=True,
                   frozen_discrete_branch_for_gradients=False,native_mesh_at_every_optimization_step=True,
                   native_meshes_used_for_selection=True,
                   decoder_parity_max_error=parity,seconds=time.monotonic()-start,
                   peak_gib=torch.cuda.max_memory_allocated()/2**30,accepted_edit=decisions['accepted_step']>0,
                   status='edited' if curve else 'no_supported_edit')
        write(root/'input_complete.json',value);emit(index,seed,'input_complete',accepted=value['accepted_edit'])
    finally:branch.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--indices',default='5,6,10,11');p.add_argument('--seeds',default='42,43');a=p.parse_args()
    setup();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;snapshot('edit')
    if not read(OUT/'validation/smoke_complete.json')['passed']:raise RuntimeError('technical smoke required')
    import nvdiffrast.torch as dr
    context=dr.RasterizeCudaContext(device='cuda:0');decoder=load_model('shape_decoder',checkpoint=True)
    for index in map(int,a.indices.split(',')):
        for seed in map(int,a.seeds.split(',')):
            one(decoder,index,seed,context);release()


if __name__=='__main__':main()
