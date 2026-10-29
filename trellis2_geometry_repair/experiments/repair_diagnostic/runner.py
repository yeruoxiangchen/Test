import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .config import OUT,PARAMS,ARMS,case,original,read,write,now,setup,emit,digest,case_lock
from .data import load,oracle
from .objective import terms,summarize_errors,robust_depth
from .deformation import Graph
from ...runtime import load_model,decode_native,save_mesh
from ...geometry import transform,render,mask_iou
from ...edit import loss_terms as capped_terms
from ...evidence import photo_cost


def structure(vertices,faces):
    # Duplicate directed triangle edges are harmless for connectivity. Avoid
    # sorting/uniquing twelve million edges at every native checkpoint.
    v=vertices.detach().cpu().numpy();f=faces.detach().cpu().numpy().astype(np.int32,copy=False)
    rows=f.reshape(-1);cols=f[:,[1,2,0]].reshape(-1)
    graph=coo_matrix((np.ones(len(rows),np.bool_),(rows,cols)),shape=(len(v),len(v)))
    count,labels=connected_components(graph,directed=False)
    tri=v[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)*.5
    by_component=np.bincount(labels[f[:,0]],weights=area,minlength=count)
    total=float(area.sum());fractions=by_component[by_component>0]/max(total,1.e-12)
    return dict(total_surface_area=total,large_component_count=int((fractions>=.0005).sum()),
                small_component_area_fraction=float(fractions[fractions<.0005].sum()),
                component_count=len(fractions),vertex_count=len(v),face_count=len(f))


def initial_surface_parity(vertices,faces,baseline_vertices,baseline_faces):
    # OBJ loaders can discard unreferenced native dual-grid vertices. Compare
    # actual corresponding triangles (and their IDs), not total vertex counts.
    if len(faces)!=len(baseline_faces):raise RuntimeError('baseline face count differs')
    maximum=0.
    for start in range(0,len(faces),131072):
        a=vertices[faces[start:start+131072].long()]
        b=baseline_vertices[baseline_faces[start:start+131072].long()]
        maximum=max(maximum,float((a-b).abs().max()))
    if maximum>2.e-6:raise RuntimeError(f'editor initial triangle surface differs: {maximum}')
    return maximum


def select(rows):
    base=rows[0];decisions=[]
    for row in rows[1:]:
        delta=np.asarray(row['ious'])-np.asarray(base['ious']);outside=row['outside']
        gain=(base['local_depth']-row['local_depth'])/max(base['local_depth'],1.e-8)
        checks=dict(mask_safe=bool(delta.min()>=-PARAMS['mask_tolerance']),
              outside_safe=bool(outside['mean'] is not None and outside['mean']<=PARAMS['max_outside_mean']
                and outside['p95']<=PARAMS['max_outside_p95'] and outside['lost_fraction']<=.005),
              local_support_safe=row['depth_audit'].get('lost_fraction',1.)<=PARAMS['max_local_lost'],
              photo_safe=(row['photo']<=base['photo']+.002) if row['photo'] is not None and base['photo'] is not None else row['heldout_pixels']>=8,
              topology_safe=bool(row['structure']['large_component_count']<=base['structure']['large_component_count']
                and row['structure']['small_component_area_fraction']<=base['structure']['small_component_area_fraction']+.002
                and .8<=row['structure']['total_surface_area']/max(base['structure']['total_surface_area'],1.e-12)<=1.25),
              target_gain=gain>=PARAMS['min_target_gain'],enough_queries=row['depth_audit']['queries']>=16)
        score=(row['photo']-base['photo'] if row['photo'] is not None and base['photo'] is not None else 0)-.5*float(delta.mean())
        decisions.append(dict(step=row['step'],accepted=all(checks.values()),checks=checks,depth_gain=gain,score=score))
    eligible=[r for r in decisions if r['accepted']]
    accepted=min(eligible,key=lambda r:(r['score'],-r['depth_gain'],r['step']))['step'] if eligible else 0
    return dict(accepted_step=accepted,candidate_step=min(rows,key=lambda r:(r['loss'],r['step']))['step'],checks=decisions)


def link(dest,source):
    temp=dest.with_name(dest.name+f'.{os.getpid()}.tmp')
    temp.symlink_to(source.resolve());temp.replace(dest)


def input_audit(index,seed,data):
    path=case(index,seed)/'input_audit.json'
    if path.exists():return
    baseline=data['baseline'];stats=summarize_errors(baseline,data['target'],data['region'],data['train'])
    field=baseline['depth'].detach().clone().requires_grad_(True);pred=dict(baseline,depth=field)
    old=capped_terms(pred,data['target'],baseline,data)['local_depth']
    new=robust_depth(pred,data['target'],baseline,data['region'],data['train'])
    old_grad=torch.autograd.grad(old,field)[0];new_grad=torch.autograd.grad(new,field)[0]
    support=data['target']['depth_valid']&(data['region']>.5)
    by_view=[]
    for view in data['train']:
        valid=support[view]
        by_view.append(dict(view=view,queries=int(valid.sum()),
          baseline_depth_median=float(baseline['depth'][view][valid].median()) if bool(valid.any()) else None,
          target_depth_median=float(data['target']['depth'][view][valid].median()) if bool(valid.any()) else None))
    value=dict(time=now(),gt_used=False,depth=stats,by_view=by_view,
          old_capped_loss=float(old),new_robust_loss=float(new),
          old_depth_field_gradient_norm=float(old_grad.norm()),new_depth_field_gradient_norm=float(new_grad.norm()),
          old_zero_gradient_fraction=float((old_grad[support]==0).float().mean()),
          new_zero_gradient_fraction=float((new_grad[support]==0).float().mean()),
          active_rows=int(data['active'].sum()),total_rows=len(data['active']),
          active_fraction=float(data['active'].float().mean()),
          note='Depth-field gradients diagnose saturation; they do not prove native latent reachability.')
    write(path,value)


def one(index,seed,arm,context):
    root=case(index,seed)/arm;root.mkdir(parents=True,exist_ok=True)
    if (root/'complete.json').exists():return
    attempt=root/'attempts'/f'{now().replace(":","-")}_{os.getpid()}';attempt.mkdir(parents=True)
    write(attempt/'source_snapshot.json',read(OUT/'source_snapshot.json'))
    data=load(index,seed,context);input_audit(index,seed,data)
    is_oracle=arm.endswith('oracle');is_latent=arm.startswith('latent')
    target=data['target']
    if is_oracle:
        with torch.no_grad():target,_,_=oracle(index,seed,data,context)
    record=dict(data,target=target)
    decoder=None;graph=None;features=None;rows=[];curve=[]
    if is_latent:
        decoder=load_model('shape_decoder',checkpoint=True,binding_root=OUT)
        saved=torch.load(original(index,seed)/'original_latent.pt',weights_only=False)
        features=saved['features'].cuda().float();coords=saved['coords'].cuda();std=saved['std'].cuda()
        matrix=torch.tensor(read(original(index,seed)/'fit.json')['matrix'],device='cuda:0')
        active=torch.nonzero(data['active']).flatten()
        parameter=torch.nn.Parameter(features.new_zeros((len(active),32)))
        def current():return features.index_add(0,active,std*PARAMS['feature_cap']*torch.tanh(parameter))
        def forward():
            v,f=decode_native(decoder,current(),coords)
            return transform(v,matrix),f
        def state():
            z=current().detach()
            if not torch.equal(z[~data['active']],features[~data['active']]):raise RuntimeError('exterior latent changed')
            return dict(kind='latent',features=z.cpu(),coords=coords.cpu(),resolution=512,space=saved['space'])
        def restore(value):
            z=value['features'].cuda()
            v,f=decode_native(decoder,z,coords);return transform(v,matrix),f
        def prior():return PARAMS['feature_prior']*torch.tanh(parameter).square().mean()
        info=dict(active_rows=len(active),total_rows=len(features),active_fraction=len(active)/len(features),
                  ss_fixed=True,outside_latent_rows_exact=True,feature_cap=PARAMS['feature_cap'],native_decode_every_step=True)
        lr=PARAMS['latent_lr']
    else:
        graph=Graph(data['vertices'],data['faces'],data['baseline'],data['region'],data['train']);parameter=graph.delta
        def forward():return graph(),data['faces']
        def state():return dict(kind='mesh',delta=parameter.detach().cpu(),outside_vertices_exact=True)
        def restore(value):
            with torch.no_grad():parameter.copy_(value['delta'].cuda())
            return graph(),data['faces']
        prior=graph.regularizer;info=graph.info;lr=PARAMS['mesh_lr']
    optimizer=torch.optim.Adam([parameter],lr=lr)
    torch.cuda.reset_peak_memory_stats();start=time.monotonic();start_memory=torch.cuda.memory_allocated()
    def checkpoint(step):
        with torch.no_grad():
            v,f=forward();pred=render(v,f,data['k'],data['w'],context)
            values=terms(pred,target,data['baseline'],data['region'],data['train'])
            outside=data['baseline']['mask']&(data['region']<.5)
            common=outside&pred['mask'];distance=(pred['depth']-data['baseline']['depth']).abs()[common]
            row=dict(step=step,**{key:float(value) for key,value in values.items()},
                depth_audit=summarize_errors(pred,target,data['region'],data['train']),
                ious=mask_iou(pred['mask'],data['observed']),photo=photo_cost(pred,record),
                heldout_pixels=int(((data['region'][data['heldout'][0]]>.5)&data['observed'][data['heldout'][0]]).sum()),
                outside=dict(mean=float(distance.mean()) if distance.numel() else None,
                             p95=float(torch.quantile(distance,.95)) if distance.numel() else None,
                             lost_fraction=float((~pred['mask'])[outside].float().mean()) if bool(outside.any()) else 0.),
                structure=structure(v,f))
            if step==0:
                row['initial_surface_parity_max']=initial_surface_parity(v,f,data['vertices'],data['faces'])
            if graph is not None:
                exterior=torch.ones(len(v),device=v.device,dtype=torch.bool);exterior[graph.editable]=False
                if not torch.equal(v[exterior],data['vertices'][exterior]):raise RuntimeError('exterior mesh vertices changed')
            rows.append(row);torch.save(state(),attempt/f'step_{step:03d}.pt')
            write(attempt/'progress.json',dict(time=now(),snapshots=rows,curve=curve))
            emit(index,seed,arm,'checkpoint',step=step,raw_depth_mae=row['depth_audit'].get('raw_mae'))
    checkpoint(0);normalization=max(rows[0]['loss'],1.e-4)
    for step in range(1,PARAMS['steps']+1):
        optimizer.zero_grad(set_to_none=True);v,f=forward();pred=render(v,f,data['k'],data['w'],context)
        values=terms(pred,target,data['baseline'],data['region'],data['train'])
        loss=values['loss']/normalization+prior()
        (loss*PARAMS['loss_scale']).backward();parameter.grad.div_(PARAMS['loss_scale'])
        if not bool(torch.isfinite(parameter.grad).all()):raise RuntimeError('nonfinite edit gradient')
        grad=float(parameter.grad.norm());torch.nn.utils.clip_grad_norm_([parameter],1.);optimizer.step()
        curve.append(dict(step=step,loss=float(loss.detach()),gradient_norm=grad))
        del v,f,pred,values,loss
        if step in PARAMS['checkpoints']:checkpoint(step)
    decisions=select(rows);endpoints={};cached={0:original(index,seed)/'aligned_base.obj'}
    with torch.no_grad():
        for name,step in [('accepted',decisions['accepted_step']),('candidate',decisions['candidate_step']),('proposal',rows[-1]['step'])]:
            if step not in cached:
                v,f=restore(torch.load(attempt/f'step_{step:03d}.pt',weights_only=False))
                path=attempt/f'native_step_{step:03d}.obj';save_mesh(path,v,f);cached[step]=path
            link(root/f'{name}.obj',cached[step]);link(root/f'{name}_state.pt',attempt/f'step_{step:03d}.pt')
            endpoints[name]=dict(step=step,mesh=str(cached[step]),state=str(attempt/f'step_{step:03d}.pt'))
    result=dict(time=now(),index=index,seed=seed,uid=data['uid'],arm=arm,oracle=is_oracle,
          gt_used_in_training=is_oracle,input_selection_gt_used=False if not is_oracle else None,
          diagnostic_only=is_oracle,representation=info,decisions=decisions,snapshots=rows,curve=curve,
          endpoints=endpoints,seconds=time.monotonic()-start,
          peak_gib=torch.cuda.max_memory_allocated()/2**30,
          incremental_peak_gib=(torch.cuda.max_memory_allocated()-start_memory)/2**30,
          loss='uncapped Charbonnier with fixed baseline-dependent missing-ray costs',
          region_and_queries='frozen input-only native512 pilot decisions',attempt=str(attempt),
          source_archive=read(attempt/'source_snapshot.json')['archive'])
    write(root/'complete.json',result);emit(index,seed,arm,'complete',accepted=decisions['accepted_step']>0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--arm',choices=ARMS,required=True);a=p.parse_args();setup()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.65)
    import nvdiffrast.torch as dr
    with case_lock(a.index,a.seed):
        one(a.index,a.seed,a.arm,dr.RasterizeCudaContext(device='cuda:0'))


if __name__=='__main__':main()
