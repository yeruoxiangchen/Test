import time

import numpy as np
import torch
import torch.nn.functional as F

from visual_evidence_probe.experiments.input_consistency.refine import transform, project_clip, silhouettes
from visual_evidence_probe.experiments.input_consistency.policy import select_checkpoint
from visual_evidence_probe.experiments.input_consistency.config import PARAMS as FIT
from visual_evidence_probe.render import render, mask_iou
from .config import folder, observation, binding, read, write, verify, emit


@torch.no_grad()
def buffers(mesh, k, w, context, side=128):
    import nvdiffrast.torch as dr
    device = torch.device('cuda:0')
    v = torch.tensor(np.asarray(mesh.vertices,np.float32),device=device)
    f = torch.tensor(np.asarray(mesh.faces,np.int32),device=device).contiguous()
    n = torch.tensor(np.array(mesh.vertex_normals,np.float32),device=device)
    k,w = k.to(device),w.to(device)
    clip = project_clip(v,k,w)
    rast,_ = dr.rasterize(context,clip,f,resolution=(side,side))
    light=F.normalize(v.new_tensor([.35,-.5,.8]),dim=0)
    color=(.28+.62*(n@light).clamp_min(0))[:,None]*v.new_tensor([.72,.78,.82])
    attrs = torch.cat((v,color),-1)[None].expand(len(k),-1,-1).contiguous()
    out,_ = dr.interpolate(attrs,rast,f)
    hit = rast[...,3]>0
    xyz = out[...,:3]
    z = torch.einsum('vhwc,vc->vhw',xyz,w[:,2,:3])+w[:,None,None,2,3]
    return dict(xyz=xyz*hit[...,None],depth=z*hit,mask=hit,rgb=out[...,3:]*hit[...,None],
                face_id=rast[...,3].long()-1)


def fit(index, seed, context):
    import trimesh
    r,k,w,_,obs = observation(index)
    path = folder(index)/'fitted'/f'{seed}.json'
    if path.exists():
        verify(read(path)['mesh']); return read(path)
    if index<4:
        source=r['existing_fitted'][str(seed)]
        result=dict(mesh=source,reused=True,uses_gt=False,elapsed_seconds=0.)
        write(path,result); return result
    source=folder(index)/'generated'/f'{seed}.obj'
    mesh=trimesh.load(source,force='mesh',process=False)
    v=torch.tensor(np.asarray(mesh.vertices,np.float32),device='cuda:0')
    faces=torch.tensor(np.asarray(mesh.faces,np.int32),device='cuda:0').contiguous()
    k,w=k.cuda(),w.cuda()
    target=F.interpolate(torch.tensor(obs,dtype=torch.float32,device='cuda:0')[:,None],size=(256,256),mode='area')[:,0]
    raw=torch.nn.Parameter(v.new_zeros(7))
    optimizer=torch.optim.Adam([raw],lr=FIT['lr'])
    snapshots=[]
    start=time.monotonic()
    def vertices():
        rot,t,s=transform(raw);return s*(v@rot.T)+t
    def snap(step):
        with torch.no_grad():
            hard=silhouettes(vertices(),faces,k,w,context,518,soft=False).cpu().numpy()
            rot,t,s=transform(raw)
            mat=np.eye(4);mat[:3,:3]=(s*rot).cpu().numpy();mat[:3,3]=t.cpu().numpy()
            snapshots.append(dict(step=step,ious=mask_iou(hard,obs).tolist(),transform=mat.tolist()))
            return hard
    hard=snap(0)
    _,legacy=render(mesh,k,w,context=context)
    if np.any(hard!=legacy):raise ValueError('fitter camera disagrees with frozen renderer')
    for step in range(1,101):
        optimizer.zero_grad(set_to_none=True)
        pred=silhouettes(vertices(),faces,k,w,context,256)
        intersect=(pred*target).sum((1,2));union=(pred+target-pred*target).sum((1,2)).clamp_min(1)
        loss=(1-intersect/union+.15*(pred*(1-target)).mean((1,2))+.08*(pred-target).abs().mean((1,2))).mean()
        loss.backward()
        if not torch.isfinite(raw.grad).all():raise ValueError('nonfinite fitting gradient')
        optimizer.step()
        if step%20==0:snap(step)
    chosen=select_checkpoint(snapshots,.005)
    mesh.apply_transform(np.asarray(chosen['transform']))
    path.parent.mkdir(parents=True,exist_ok=True)
    mesh.export(path.with_suffix('.obj'))
    record=dict(mesh=binding(path.with_suffix('.obj')),source=binding(source),uses_gt=False,reused=False,
                snapshots=snapshots,chosen=chosen,elapsed_seconds=time.monotonic()-start)
    write(path,record);emit(index,'fitted',seed=seed,before=snapshots[0]['ious'],after=chosen['ious'])
    return record
