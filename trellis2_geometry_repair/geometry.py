"""Camera/mesh utilities shared with the existing input-only geometry probe."""
import itertools
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .config import OUT,PARAMS,read


def observations(index,device='cuda:0'):
    rec=read(OUT/'data'/f'{index:03d}'/'observations.json')
    images=[Image.open(v['path']).convert('RGB') for v in rec['images']]
    masks=np.stack([np.asarray(Image.open(v['path']).convert('L'))>127 for v in rec['masks']])
    return rec,torch.tensor(rec['intrinsics'],device=device,dtype=torch.float32),\
        torch.tensor(rec['extrinsics'],device=device,dtype=torch.float32),images,masks


def transform(vertices,matrix):
    return vertices@matrix[:3,:3].T+matrix[:3,3]


def render(vertices,faces,k,w,context,side=256):
    # Generic differentiable rasterization only; no old model runtime/latent.
    from visual_evidence_probe.experiments.editor_support.objective import raster_tensor
    return raster_tensor(vertices.float(),faces,k,w,context,side)


def detached(pred):
    return {key:value.detach().clone() for key,value in pred.items() if torch.is_tensor(value)}


def cube_rotations():
    values=[]
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1.,1.),repeat=3):
            r=np.eye(3)[list(perm)]*np.asarray(signs)[:,None]
            if np.linalg.det(r)>.5:values.append(r)
    return np.stack(values).astype(np.float32)


def similarity(raw,initial):
    x,y,z=raw[:3].unbind();zero=x*0
    skew=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero)).reshape(3,3)
    rotation=torch.matrix_exp(skew)@initial
    scale=torch.exp(.5*torch.tanh(raw[6]));translation=.3*torch.tanh(raw[3:6])
    return rotation,scale,translation


def mask_iou(pred,target):
    return ((pred&target).sum((1,2))/(pred|target).sum((1,2)).clamp_min(1)).cpu().tolist()


def load_mesh(path,device='cuda:0'):
    import trimesh
    mesh=trimesh.load(path,force='mesh',process=False,skip_materials=True)
    return torch.tensor(np.asarray(mesh.vertices,np.float32),device=device),\
        torch.tensor(np.asarray(mesh.faces,np.int32),device=device)


def error(pred,target):
    value=(pred['depth']-target['depth']).abs().clamp_max(.1)
    value=torch.where(pred['mask'],value,torch.full_like(value,.1))
    return torch.where(target['mask'],value,torch.where(pred['mask'],torch.full_like(value,.1),torch.zeros_like(value)))


def novel_cameras(k,w,count=8):
    # Geometry-only camera placement; no GT geometry or image scores.
    centers=np.linalg.inv(w)[:,:3,3];directions=centers/np.linalg.norm(centers,axis=1,keepdims=True)
    candidates=[]
    for el in (-.35,.35,.8):
        for az in np.linspace(0,2*np.pi,48,endpoint=False)+.037:
            candidates.append([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    candidates=np.asarray(candidates);selected=[];reference=list(directions)
    for _ in range(count):
        similarity=candidates@np.asarray(reference).T
        idx=int(np.argmin(similarity.max(1)));p=candidates[idx];selected.append(p);reference.append(p)
        candidates=np.delete(candidates,idx,0)
    matrices=[];radius=float(np.median(np.linalg.norm(centers,axis=1)))
    for d in selected:
        center=d*radius;forward=-d
        right=np.cross(forward,np.array([0.,0.,1.]));right/=np.linalg.norm(right)
        down=np.cross(forward,right);r=np.stack([right,down,forward])
        m=np.eye(4);m[:3,:3]=r;m[:3,3]=-r@center;matrices.append(m)
    intr=np.repeat(np.array([[[650.,0.,258.5],[0.,650.,258.5],[0.,0.,1.]]]),count,axis=0)
    return intr.astype(np.float32),np.asarray(matrices,np.float32)
