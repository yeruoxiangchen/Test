import numpy as np
import torch

from .config import OUT,PARAMS,DIAGNOSTIC,case,previous,original,BASE,read,write,save,digest,now
from .backend import take_even
from ..local_regeneration.data import load as input_data
from ..local_regeneration.selection import regional_targets
from ..local_regeneration.fields import lookup
from ...geometry import transform,render,detached
from visual_evidence_probe.experiments.editor_support.objective import world_from_depth

def cpu(value):
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    return value

def device(value):
    if torch.is_tensor(value):return value.cuda()
    if isinstance(value,dict):return {k:device(v) for k,v in value.items()}
    if isinstance(value,list):return [device(v) for v in value]
    return value

def query_ids(valid,train,limit):
    flat=[];pixels=valid.shape[-1]*valid.shape[-2]
    for view in train:
        ids=torch.nonzero(valid[view].flatten()).flatten()+view*pixels
        flat.append(take_even(ids,max(1,limit//len(train))))
    return torch.cat(flat) if flat else torch.empty(0,device=valid.device,dtype=torch.long)

def load(index,seed,context):
    data=input_data(index,seed,context);root=previous(index,seed)
    plan=torch.load(root/'plan.pt',weights_only=False)
    matrix=plan['matrix'].cuda();envelope=torch.as_tensor(plan['roi']['envelope'],device='cuda:0')
    trusted=device(plan['trusted']);baseline=device(torch.load(root/'registered_render.pt',weights_only=False))
    trusted,domain=regional_targets(trusted,baseline,matrix,envelope,data['k'],data['w'])
    ids=query_ids(trusted['valid'],data['train'],PARAMS['max_queries'])
    inverse=torch.linalg.inv(matrix)
    native=transform(baseline['xyz'].reshape(-1,3),inverse)
    inside=lookup(envelope,torch.floor((native+.5)*32).long()).reshape(baseline['mask'].shape).bool()
    stable=trusted['valid']&baseline['mask']&((trusted['depth']-baseline['depth']).abs()<=.005)
    protected=baseline['mask']&data['observed']&(~inside|stable)
    data.update(matrix=matrix,envelope=envelope,baseline=baseline,trusted=trusted,
                query_ids=ids,protected_pixels=protected,domain=domain,roi=plan['roi'])
    return data

def record(data,reference,arm,index,seed):
    trusted={k:v.clone() if torch.is_tensor(v) else v for k,v in data['trusted'].items()}
    if arm in DIAGNOSTIC:
        sealed=read(case(index,seed)/'method_sealed.json')
        if digest(case(index,seed)/'method_complete.json')!=sealed['decision_sha256']:raise RuntimeError('input decision changed after sealing')
        diag=torch.load(case(index,seed)/'diagnostics/targets.pt',weights_only=False)
        trusted['valid'] &= diag['common'].cuda()
        if arm=='oracle':trusted['depth']=diag['depth'].cuda()
    xyz=world_from_depth(trusted['depth'],data['k'],data['w'])
    ids=data['query_ids'];ids=ids[trusted['valid'].flatten()[ids]]
    world=xyz.reshape(-1,3)[ids];points=transform(world,torch.linalg.inv(data['matrix']))
    confidence=data['trusted']['confidence'].flatten()[ids].clamp_min(.05)
    free=[]
    for offset in PARAMS['free_depth_offsets']:
        depth=(trusted['depth']-offset).clamp_min(.001)
        free.append(transform(world_from_depth(depth,data['k'],data['w']).reshape(-1,3)[ids],torch.linalg.inv(data['matrix'])))
    result=dict(data,trusted=trusted,points=points,point_weights=confidence,free_points=torch.cat(free),field_reference=reference)
    cells=torch.floor((points+.5)*32).long()
    meta=dict(arm=arm,diagnostic_only=arm in DIAGNOSTIC,gt_used=arm in DIAGNOSTIC,
        target_queries=int(trusted['valid'].sum()),surface_queries=len(points),query_ids_sha256=__import__('hashlib').sha256(ids.cpu().numpy().tobytes()).hexdigest(),
        target_outside_envelope=int((~lookup(data['envelope'],cells).bool()).sum()),
        protected_pixels=int(data['protected_pixels'].sum()),domain=data['domain'])
    return result,meta

def prepare_diagnostics(index,seed,data,context):
    root=case(index,seed);seal=root/'method_sealed.json'
    if not seal.exists():raise RuntimeError('input-only selections must be sealed before reading GT')
    if read(seal)['decision_sha256']!=digest(root/'method_complete.json'):raise RuntimeError('input seal mismatch')
    path=root/'diagnostics/targets.pt'
    if path.exists():return
    import trimesh
    desc=read(BASE/'data'/f'{index:03d}'/'evaluation_target.json')
    mesh=trimesh.load(desc['mesh'],force='mesh',process=False,skip_materials=True);mesh.apply_transform(np.asarray(desc['transform']))
    vertices=torch.tensor(np.asarray(mesh.vertices,np.float32),device='cuda:0');faces=torch.tensor(np.asarray(mesh.faces,np.int32),device='cuda:0')
    with torch.no_grad():gt=detached(render(vertices,faces,data['k'],data['w'],context))
    common=data['trusted']['valid']&gt['mask']
    save(path,dict(depth=gt['depth'].cpu(),common=common.cpu(),gt_used=True,diagnostic_only=True))
    write(root/'diagnostics/domain.json',dict(time=now(),input_queries=int(data['trusted']['valid'].sum()),paired_queries=int(common.sum()),
        same_domain_for_paired_input_and_oracle=True,actual_input_domain_unchanged=True,gt_used=True,diagnostic_only=True))
