"""Exact CV camera transport between DA3 processed images and the edit raster."""
import numpy as np
import torch
import torch.nn.functional as F


def sample_pixels(field,uv,image_size=518):
    grid=2*(uv+.5)/image_size-1
    return F.grid_sample(field[None],grid[None],align_corners=False,mode='bilinear',padding_mode='zeros')[0].permute(1,2,0)


def pixel_grid(side,device,dtype=torch.float32,image_size=518):
    axis=(torch.arange(side,device=device,dtype=dtype)+.5)*image_size/side-.5
    yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    return torch.stack((xx,yy,torch.ones_like(xx)),-1)


def project(points,k,w):
    camera=points@w[:3,:3].T+w[:3,3]
    h=camera@k.T
    return h[...,:2]/h[...,2:].clamp_min(1.e-8),camera[...,2]


def backproject(depth,k,w,image_size=518):
    pixels=pixel_grid(depth.shape[-1],depth.device,depth.dtype,image_size)
    ray=pixels@torch.linalg.inv(k).T
    camera=ray*depth[...,None]
    inverse=torch.linalg.inv(w)
    return camera@inverse[:3,:3].T+inverse[:3,3]


def remap_depth(depth,confidence,processed_k,input_k,side=256):
    """Sample the processed depth on exact input rays; never resize camera-z."""
    d=torch.as_tensor(depth,dtype=torch.float32)
    c=torch.ones_like(d) if confidence is None else torch.as_tensor(confidence,dtype=torch.float32)
    pk=torch.as_tensor(processed_k,dtype=torch.float32)
    ik=torch.as_tensor(input_k,dtype=torch.float32)
    pixels=pixel_grid(side,d.device)
    values,conf,domains=[],[],[]
    height,width=d.shape[-2:]
    for i in range(len(d)):
        rays=pixels@torch.linalg.inv(ik[i]).T
        location=rays@pk[i].T
        uv=location[...,:2]/location[...,2:]
        grid=torch.stack((2*(uv[...,0]+.5)/width-1,2*(uv[...,1]+.5)/height-1),-1)
        out=F.grid_sample(torch.stack((d[i],c[i]))[None],grid[None],align_corners=False,mode='bilinear')[0]
        valid=(uv[...,0]>=0)&(uv[...,0]<width)&(uv[...,1]>=0)&(uv[...,1]<height)
        values.append(out[0]);conf.append(out[1]);domains.append(valid)
    return torch.stack(values),torch.stack(conf),torch.stack(domains)


@torch.no_grad()
def consistency(depth,confidence,masks,k,w,*,quantile=.4,pixel_tolerance=2.,absolute_z=.02,relative_z=.015,min_references=1):
    """Fixed foreground/confidence and bidirectional ray consistency, no GT."""
    masks=F.avg_pool2d(masks[:,None].float(),3,stride=1,padding=1)[:,0]>.999
    finite=torch.isfinite(depth)&(depth>0)&torch.isfinite(confidence)
    eligible=masks&finite
    trusted=torch.zeros_like(eligible)
    thresholds=[]
    for i in range(len(depth)):
        values=confidence[i][eligible[i]]
        threshold=torch.quantile(values,quantile) if len(values) else confidence.new_tensor(float('inf'))
        thresholds.append(float(threshold) if torch.isfinite(threshold) else None)
        trusted[i]=eligible[i]&(confidence[i]>=threshold)
    support=torch.zeros_like(depth,dtype=torch.int32)
    roundtrip_error=torch.full_like(depth,float('inf'))
    expected=pixel_grid(depth.shape[-1],depth.device)[...,:2]
    for i in range(len(depth)):
        xyz=backproject(depth[i],k[i],w[i])
        for j in range(len(depth)):
            if i==j:continue
            uv,z=project(xyz,k[j],w[j])
            reference=sample_pixels(depth[j][None],uv)[...,0]
            ref_trust=sample_pixels(trusted[j][None].float(),uv)[...,0]>.99
            rays=torch.cat((uv,torch.ones_like(uv[...,:1])),-1)@torch.linalg.inv(k[j]).T
            camera=rays*reference[...,None]
            inverse=torch.linalg.inv(w[j]);world=camera@inverse[:3,:3].T+inverse[:3,3]
            back,_=project(world,k[i],w[i])
            error=torch.linalg.vector_norm(back-expected,dim=-1)*depth.shape[-1]/518
            tolerance=torch.maximum(torch.full_like(z,absolute_z),relative_z*z.abs())
            valid=trusted[i]&ref_trust&(z>0)&(uv>=0).all(-1)&(uv<518).all(-1)
            valid&=(reference-z).abs()<=tolerance
            valid&=error<=pixel_tolerance
            support[i]+=valid.int()
            roundtrip_error[i]=torch.minimum(roundtrip_error[i],torch.where(valid,error,torch.full_like(error,float('inf'))))
    valid=trusted&(support>=min_references)
    receipt=dict(foreground_pixels=eligible.sum((1,2)).cpu().tolist(),
        confidence_pixels=trusted.sum((1,2)).cpu().tolist(),consistent_pixels=valid.sum((1,2)).cpu().tolist(),
        confidence_thresholds=thresholds,uses_gt=False,
        roundtrip_mean_pixels=float(roundtrip_error[valid].mean()) if bool(valid.any()) else None)
    return valid,support,receipt
