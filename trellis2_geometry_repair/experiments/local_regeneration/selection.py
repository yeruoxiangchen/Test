import argparse
from pathlib import Path
import time
import numpy as np
import torch
from scipy import ndimage

from .config import OUT,PARAMS,ARMS,case,source,read,write,save_tensor,setup,lock_case,emit,now
from .data import load
from .fields import decode_fields,to_mesh,compose,lookup
from .evidence import mask_cost
from .alignment import depth_cost,photo_score
from ...runtime import load_model,save_mesh,release
from ...geometry import transform,render,detached,mask_iou
from ..repair_diagnostic.runner import structure,initial_surface_parity,link
from visual_evidence_probe.experiments.editor_support.objective import world_from_depth


def native_points(world,matrix):return transform(world.reshape(-1,3),torch.linalg.inv(matrix)).reshape(world.shape)


def regional_targets(trusted,baseline,matrix,envelope,k,w):
    """Freeze the scoring rays before proposals, using only input geometry."""
    region=torch.as_tensor(envelope,device=matrix.device,dtype=torch.bool)
    def inside(points):
        cells=torch.floor((native_points(points,matrix)+.5)*32).long()
        return lookup(region,cells.reshape(-1,3)).bool().reshape(points.shape[:-1])
    target=world_from_depth(trusted['depth'],k,w)
    eligible=inside(target)|(inside(baseline['xyz'])&baseline['mask'])
    result=dict(trusted);result['valid']=trusted['valid']&eligible
    return result,dict(global_verified_pixels=int(trusted['valid'].sum()),
                       local_verified_pixels=int(result['valid'].sum()),
                       domain='fixed input-defined edit envelope',gt_used=False)


def outside_change(pred,base,matrix,roi):
    protected=~ndimage.binary_dilation(roi['envelope'],iterations=1)
    protected=torch.tensor(protected,device=matrix.device)
    def classify(points):
        c=torch.floor((native_points(points,matrix)+.5)*32).long().reshape(-1,3)
        return lookup(protected,c).bool().reshape(points.shape[:-1])
    old=classify(base['xyz'])&base['mask'];new=classify(pred['xyz'])&pred['mask']
    common=old&new;difference=(pred['depth']-base['depth']).abs()[common]
    return dict(mean=float(difference.mean()) if difference.numel() else 0.,
           p95=float(torch.quantile(difference,.95)) if difference.numel() else 0.,
           lost_fraction=float((old&~pred['mask']).sum()/old.sum().clamp_min(1)),
           checked_pixels=int(common.sum()),occluded_by_edit=int((old&pred['mask']&~new).sum()))


def row_metrics(v,f,data,matrix,roi,trusted,baseline,context,base_structure=None):
    pred=render(transform(v,matrix),f,data['k'],data['w'],context)
    mask=float(mask_cost(pred,data['observed'],data['train']))
    depth=float(depth_cost(pred,trusted,baseline,data['train']))
    result=dict(mask_cost=mask,depth_cost=depth,train_cost=mask+.05*depth,
       ious=mask_iou(pred['mask'],data['observed']),photo=photo_score(pred,data),
       outside=outside_change(pred,baseline,matrix,roi),
       structure=base_structure if base_structure is not None else structure(v,f))
    return result


def decision(row,base,verified_pixels,field_composite):
    change=np.asarray(row['ious'])-np.asarray(base['ious'])
    mask_gain=base['mask_cost']-row['mask_cost']
    depth_gain=(base['depth_cost']-row['depth_cost'])/max(base['depth_cost'],1.e-8)
    sufficient=(mask_gain>=PARAMS['min_mask_gain']) or (verified_pixels>=PARAMS['min_trusted_pixels'] and depth_gain>=PARAMS['min_depth_gain'])
    photo_safe=row['photo']<=base['photo']+PARAMS['photo_tolerance'] if row['photo'] is not None and base['photo'] is not None else True
    original=base['structure'];current=row['structure']
    checks=dict(mask_safe=bool(change.min()>=-PARAMS['mask_tolerance']),photo_safe=bool(photo_safe),
       observed_gain=bool(sufficient),training_nonregression=row['train_cost']<=base['train_cost']+1.e-5,
       exterior_safe=row['outside']['mean']<=.001 and row['outside']['p95']<=.004 and row['outside']['lost_fraction']<=.001,
       topology_safe=current['large_component_count']<=original['large_component_count']+1
          and current['small_component_area_fraction']<=original['small_component_area_fraction']+PARAMS['max_added_small_area']
          and .65<=current['total_surface_area']/max(original['total_surface_area'],1.e-12)<=PARAMS['max_surface_ratio'])
    if field_composite:checks['exterior_fields_exact']=row['field_checks']['exterior_fields_exact']
    return dict(accepted=all(checks.values()),checks=checks,mask_gain=mask_gain,depth_gain=depth_gain,
                score=row['train_cost']+.1*(row['photo'] if row['photo'] is not None else 0.))


def one(index,seed,context):
    root=case(index,seed)
    if (root/'input_complete.json').exists():return
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    data=load(index,seed,context);plan=torch.load(root/'plan.pt',weights_only=False)
    matrix=plan['matrix'].cuda();roi=plan['roi'];trusted={key:(value.cuda() if torch.is_tensor(value) else value) for key,value in plan['trusted'].items()}
    saved=torch.load(source(index,seed)/'original_latent.pt',weights_only=False)
    decoder=load_model('shape_decoder',binding_root=OUT)
    with torch.no_grad():
        bv,bf,base_fields=decode_fields(decoder,saved['features'].cuda(),saved['coords'].cuda())
        initial_surface_parity(bv,bf,data['native'],data['faces'])
        baseline=detached(render(transform(bv,matrix),bf,data['k'],data['w'],context))
        scoring_domain=dict(global_verified_pixels=int(trusted['valid'].sum()),
                            local_verified_pixels=int(trusted['valid'].sum()),domain='all verified input rays',gt_used=False)
        if PARAMS.get('local_depth_scoring',False):
            trusted,scoring_domain=regional_targets(trusted,baseline,matrix,roi['envelope'],data['k'],data['w'])
        save_tensor(root/'base_fields.pt',{key:value.cpu() for key,value in base_fields.items()})
        save_mesh(root/'registered_base.obj',transform(bv,matrix),bf)
        base=row_metrics(bv,bf,data,matrix,roi,trusted,baseline,context)
        base.update(name='base',kind='direct_native_decode',arm='base',field_checks=dict(exterior_fields_exact=True))
        rows=[];alpha=torch.tensor(roi['alpha'],device='cuda:0')
        candidates=read(root/'sampling_complete.json')['candidates']
        for config in candidates:
            payload=torch.load(config['path'],weights_only=False)
            try:
                v,f,raw=decode_fields(decoder,payload['features'].cuda(),payload['coords'].cuda())
                if len(f)==0:raise RuntimeError('empty native candidate')
                native=row_metrics(v,f,data,matrix,roi,trusted,baseline,context)
                native.update(name=config['name']+'_native',candidate=config['name'],arm=config['arm'],kind='direct_native_decode',path=config['path'])
                native['decision']=decision(native,base,int(trusted['valid'].sum()),False);rows.append(native)
                blended,checks=compose(base_fields,raw,alpha);v,f=to_mesh(blended)
                if len(f)==0:raise RuntimeError('empty field composite')
                merged=row_metrics(v,f,data,matrix,roi,trusted,baseline,context)
                merged.update(name=config['name']+'_composite',candidate=config['name'],arm=config['arm'],
                              kind='native_FDG_composite',path=config['path'],field_checks=checks)
                merged['decision']=decision(merged,base,int(trusted['valid'].sum()),True);rows.append(merged)
                emit(index,seed,'candidate_scored',name=config['name'],native_accepted=native['decision']['accepted'],composite_accepted=merged['decision']['accepted'])
            except RuntimeError as error:
                # A failed generated geometry is an explicit failed proposal;
                # numerical/input/backend failures remain fatal.
                if 'empty' not in str(error):raise
                rows.append(dict(name=config['name'],arm=config['arm'],invalid=True,error=str(error)))
            write(root/'scoring_progress.json',dict(time=now(),baseline=base,rows=rows,gt_used=False))
        selections={};export_cache={};export_root=root/'exports';export_root.mkdir(exist_ok=True)
        for arm in ARMS:
            choices=[r for r in rows if (arm=='adaptive' or r['arm']=='shape') and not r.get('invalid')]
            eligible=[r for r in choices if r['decision']['accepted']]
            accepted=min(eligible,key=lambda r:r['decision']['score']) if eligible else base
            candidate=min(choices,key=lambda r:r['decision']['score']) if choices else base
            arm_root=root/arm;arm_root.mkdir(exist_ok=True);exports={}
            for ep,choice in [('accepted',accepted),('candidate',candidate)]:
                if choice['name']=='base':
                    link(arm_root/f'{ep}.obj',root/'registered_base.obj')
                    exports[ep]=dict(name='base',kind='direct_native_decode',mesh=str(root/'registered_base.obj'),latent=str(source(index,seed)/'original_latent.pt'))
                    continue
                pair=choice['candidate']
                if pair not in export_cache:
                    payload=torch.load(choice['path'],weights_only=False)
                    v,f,raw=decode_fields(decoder,payload['features'].cuda(),payload['coords'].cuda())
                    native_mesh=export_root/f'{pair}_native.obj';save_mesh(native_mesh,transform(v,matrix),f)
                    raw,checks=compose(base_fields,raw,alpha);v,f=to_mesh(raw)
                    field_path=export_root/f'{pair}_fields.pt';save_tensor(field_path,{key:value.cpu() for key,value in raw.items()})
                    composite_mesh=export_root/f'{pair}_composite.obj';save_mesh(composite_mesh,transform(v,matrix),f)
                    export_cache[pair]=dict(latent=choice['path'],native_mesh=str(native_mesh),composite_mesh=str(composite_mesh),paired_composite_fields=str(field_path))
                item=dict(export_cache[pair],name=choice['name'],kind=choice['kind'])
                item['mesh']=item['native_mesh'] if choice['kind']=='direct_native_decode' else item['composite_mesh']
                item['field_state']=item['paired_composite_fields'] if choice['kind']=='native_FDG_composite' else None
                link(arm_root/f'{ep}.obj',Path(item['mesh']));exports[ep]=item
            selections[arm]=dict(accepted_edit=accepted['name']!='base',accepted=exports['accepted'],candidate=exports['candidate'],
                      candidates=len(choices),reason='input-only proposal decision' if choices else 'no reliable regional proposal')
        write(root/'input_complete.json',dict(time=now(),index=index,seed=seed,uid=data['uid'],baseline=base,rows=rows,
              selections=selections,matrix=matrix.cpu().tolist(),gt_used=False,field_composites_explicit=True,
              native_candidates_saved_as_latents=True,scope=read(root/'plan.json')['scope']))
        complete=read(root/'input_complete.json');complete.update(seconds=time.monotonic()-started,
              peak_gib=torch.cuda.max_memory_allocated()/2**30,scoring_domain=scoring_domain)
        write(root/'input_complete.json',complete)
        emit(index,seed,'input_complete',accepted={arm:v['accepted_edit'] for arm,v in selections.items()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    setup();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.7)
    import nvdiffrast.torch as dr
    with lock_case(a.index,a.seed),torch.no_grad():one(a.index,a.seed,dr.RasterizeCudaContext(device='cuda:0'))


if __name__=='__main__':main()
