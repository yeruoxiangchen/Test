"""Recorded raster-buffer fix over the unchanged frozen localization experiment."""
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ.setdefault('CUDA_HOME','/usr/local/cuda-12.6')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.6')
os.environ.setdefault('TORCH_EXTENSIONS_DIR','/home/zjr/Tracker/visual_evidence_probe/outputs/torch_extensions')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('SPCONV_ALGO','native')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')

import numpy as np
import torch
import torch.nn.functional as F

from visual_evidence_probe.experiments.localization import config,geometry,runner
from visual_evidence_probe.experiments.input_consistency.refine import project_clip
from visual_evidence_probe.common import binding,read,write,verify,lock_json,now


@torch.no_grad()
def buffers(mesh,k,w,context,side=128):
    import nvdiffrast.torch as dr
    device=torch.device('cuda:0')
    v=torch.tensor(np.asarray(mesh.vertices,np.float32),device=device)
    faces=torch.tensor(np.asarray(mesh.faces,np.int32),device=device).contiguous()
    normals=torch.tensor(np.array(mesh.vertex_normals,np.float32),device=device)
    k,w=k.to(device),w.to(device)
    clip=project_clip(v,k,w)
    rast,_=dr.rasterize(context,clip,faces,resolution=(side,side))
    cam_z=v[None]@w[:,2,:3,None]+w[:,None,2,3,None]
    z,_=dr.interpolate(cam_z.contiguous(),rast,faces)
    depth=z[...,0]
    axis=(torch.arange(side,device=device,dtype=torch.float32)+.5)*518/side-.5
    yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    pixel=torch.stack((xx,yy,torch.ones_like(xx)),-1)
    rays=torch.einsum('hwc,vdc->vhwd',pixel,torch.linalg.inv(k))
    camera=rays*depth[...,None]
    c2w=torch.linalg.inv(w)
    world=torch.einsum('vhwc,vdc->vhwd',camera,c2w[:,:3,:3])+c2w[:,None,None,:3,3]
    light=F.normalize(v.new_tensor([.35,-.5,.8]),dim=0)
    colors=(.28+.62*(normals@light).clamp_min(0))[:,None]*v.new_tensor([.72,.78,.82])
    rgb,_=dr.interpolate(colors[None].expand(len(k),-1,-1).contiguous(),rast,faces)
    hit=rast[...,3]>0
    return dict(xyz=world*hit[...,None],depth=depth*hit,mask=hit,rgb=rgb*hit[...,None],face_id=rast[...,3].long()-1)


def audit():
    import trimesh
    import nvdiffrast.torch as dr
    from visual_evidence_probe.runtime import setup
    from visual_evidence_probe.experiments.localization.scores import project
    setup();context=dr.RasterizeCudaContext(device=torch.device('cuda:0'))
    xy=np.array([[-.7,-.7],[.7,-.7],[.7,.7],[-.7,.7]])
    vertices=np.c_[xy,2+.2*xy[:,0]-.1*xy[:,1]]
    mesh=trimesh.Trimesh(vertices,[[0,1,2],[0,2,3]],process=False)
    k=torch.tensor([[[400.,0,258.5],[0,400.,258.5],[0,0,1.]]])
    w=torch.eye(4)[None]
    b=buffers(mesh,k,w,context,128)
    axis=(torch.arange(128,device='cuda:0')+.5)*518/128-.5
    yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    rays=torch.stack(((xx-258.5)/400,(yy-258.5)/400,torch.ones_like(xx)),-1)
    expected_depth=2/(rays@rays.new_tensor([-.2,.1,1.]))
    plane_error=float((b['depth'][0]-expected_depth)[b['mask'][0]].abs().max())
    if plane_error>5.e-5:raise ValueError(f'analytic plane depth failed {plane_error}')
    rec,k,w,_,_=config.observation(0)
    natural=trimesh.load(verify(read(config.folder(0)/'fitted/42.json')['mesh']),force='mesh',process=False)
    b=buffers(natural,k,w,context,128)
    expected=torch.stack((xx,yy),-1)
    maximum=0.
    for i in range(len(k)):
        uv,_=project(b['xyz'][i],k[i].cuda(),w[i].cuda())
        maximum=max(maximum,float((uv-expected)[b['mask'][i]].abs().max()))
    if maximum>2.e-3:raise ValueError(f'corrected projection failed {maximum}')
    result=dict(time=now(),analytic_tilted_plane_max_depth_error=plane_error,
                natural_mesh_max_projection_error_pixels=maximum,passed=True)
    write(config.OUT/'runtime/buffer_fix_audit.json',result);print(result,flush=True)


def register():
    config.check()
    receipt=config.OUT/'runtime/buffer_fix_audit.json'
    if not read(receipt)['passed']:raise ValueError('buffer audit missing')
    record=dict(format='localization_buffer_amendment.v1',base_protocol=binding(config.OUT/'protocol.json'),
        sources=[binding(__file__),binding(Path(__file__).with_name('localization_projection_audit.py'))],
        audit=binding(receipt),reason='interpolated XYZ did not reproject exactly for some small triangles',
        fix='interpolate camera-z, unproject exact pixel-center rays; same masks, cameras, weights, thresholds',
        original_sources_and_failed_preflight_preserved=True,gt_scores_seen=False)
    lock_json(config.OUT/'buffer_amendment.json',record)


def main():
    register()
    with patch.object(geometry,'buffers',buffers):runner.main()


if __name__=='__main__':main()
