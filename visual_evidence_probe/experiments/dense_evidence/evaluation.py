"""Evaluation-only geometry access and fast hard-depth rendering."""
import numpy as np
import torch

from .config import data_folder,folder,read


def load_target(index):
    import trimesh
    root=data_folder(index);rec=read(root/'observations.json')
    if rec['split']!='development':
        if not all((folder(index,seed)/'input_complete.json').exists() for seed in (42,43)):
            raise ValueError('validation GT cannot be loaded before both input-only decisions are sealed')
    description=read(root/'evaluation_target.json')
    cache=root/'evaluation_canonical.ply'
    source=cache if cache.exists() else description['mesh']
    mesh=trimesh.load(source,process=False,force='mesh',skip_materials=True)
    if not cache.exists():
        matrix=np.array(description['transform'])
        if not np.array_equal(matrix,np.eye(4)):mesh.apply_transform(matrix)
        if rec['split']=='validation':mesh.export(cache,encoding='binary')
    return mesh


@torch.no_grad()
def render_depth(mesh,k,w,context,side=256):
    import nvdiffrast.torch as dr
    from visual_evidence_probe.experiments.editor_support.objective import project_clip,world_from_depth
    v=torch.tensor(np.asarray(mesh.vertices,np.float32),device=k.device)
    f=torch.tensor(np.asarray(mesh.faces,np.int32),device=k.device)
    raster,_=dr.rasterize(context,project_clip(v,k,w),f,resolution=(side,side))
    z=(v[None]@w[:,2,:3,None]+w[:,None,2,3,None]).contiguous()
    depth,_=dr.interpolate(z,raster,f)
    hit=raster[...,3]>0;depth=depth[...,0]*hit
    return dict(depth=depth,mask=hit,xyz=world_from_depth(depth,k,w)*hit[...,None],
        face_id=raster[...,3].long()-1)
