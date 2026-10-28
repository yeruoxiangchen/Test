"""Input-only evidence scores; no GT geometry/depth is accepted here."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from .config import PARAMS

METHODS=('silhouette','boundary','direct_rgb','direct_dino','cross_rgb','cross_dino','shuffled_dino','fused_dino')


def sample_field(field, uv):
    # field [C,H,W]; uv in original full 518 pixel coordinates.
    grid=2*(uv+.5)/518-1
    return F.grid_sample(field[None],grid[None],align_corners=False,mode='bilinear',padding_mode='zeros')[0].permute(1,2,0)


def project(xyz,k,w):
    camera=xyz@w[:3,:3].T+w[:3,3]
    image=camera@k.T
    uv=image[...,:2]/image[...,2:].clamp_min(1.e-6)
    return uv,camera[...,2]


def cosine_distance(a,b):
    return ((1-(F.normalize(a.float(),dim=-1)*F.normalize(b.float(),dim=-1)).sum(-1))/2).clamp(0,1)


@torch.no_grad()
def compute(pred, raw_rgb, observed_mask, features, rendered_features, kraw, kcrop, w, index):
    device=pred['xyz'].device
    side=pred['mask'].shape[-1]
    view_count=len(features)
    original_rgb=torch.tensor(np.asarray(raw_rgb),device=device,dtype=torch.float32).permute(0,3,1,2)/255
    rgb=F.interpolate(original_rgb,size=(side,side),mode='area')
    obs=F.interpolate(torch.tensor(observed_mask,dtype=torch.float32,device=device)[:,None],
                      size=(side,side),mode='nearest')[:,0]>.5
    f=features.to(device).float().transpose(1,2).reshape(view_count,1024,37,37)
    rf=rendered_features.to(device).float().transpose(1,2).reshape_as(f)
    generator=torch.Generator().manual_seed(PARAMS['shuffle_seed']+index*131)
    perms=torch.stack([torch.randperm(1369,generator=generator) for _ in range(view_count)]).to(device)
    sf=torch.stack([features[j].to(device).float()[perms[j]].T.reshape(1024,37,37) for j in range(view_count)])
    kraw,kcrop,w=kraw.to(device),kcrop.to(device),w.to(device)
    axis=(torch.arange(side,device=device,dtype=torch.float32)+.5)*518/side-.5
    yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    pixels=torch.stack((xx,yy,torch.ones_like(xx)),-1)
    result={key:[] for key in METHODS}
    supports=[];domains=[];anchors=[]
    for i in range(view_count):
        affine=kcrop[i]@torch.linalg.inv(kraw[i])
        crop_uv=(pixels@affine.T)[...,:2]
        anchor=sample_field(f[i],crop_uv)
        reference_render=sample_field(rf[i],crop_uv)
        direct=cosine_distance(anchor,reference_render)
        direct_rgb=(rgb[i].permute(1,2,0)-pred['rgb'][i]).abs().mean(-1)
        sums=direct.new_zeros((3,side,side));support=direct.new_zeros((side,side))
        xyz=pred['xyz'][i]
        for j in range(view_count):
            if i==j:continue
            uv,depth=project(xyz,kraw[j],w[j])
            uv_feature,_=project(xyz,kcrop[j],w[j])
            ref_depth=sample_field(pred['depth'][j][None],uv)[...,0]
            ref_hit=sample_field(pred['mask'][j][None].float(),uv)[...,0]
            ref_obs=sample_field(obs[j][None].float(),uv)[...,0]
            valid=pred['mask'][i]&(depth>0)&(ref_hit>.99)&(ref_obs>.99)
            valid&=(depth-ref_depth).abs()<=PARAMS['self_visibility_tolerance']
            valid&=(uv>=0).all(-1)&(uv<518).all(-1)&(uv_feature>=0).all(-1)&(uv_feature<518).all(-1)
            ref_feat=sample_field(f[j],uv_feature)
            shuffled=sample_field(sf[j],uv_feature)
            ref_rgb=sample_field(rgb[j],uv)
            sums[0]+=cosine_distance(anchor,ref_feat)*valid
            sums[1]+=(rgb[i].permute(1,2,0)-ref_rgb).abs().mean(-1)*valid
            sums[2]+=cosine_distance(anchor,shuffled)*valid
            support+=valid
        cross=sums/support.clamp_min(1)[None]
        obs_np=obs[i].cpu().numpy();pm=pred['mask'][i].cpu().numpy()
        do=distance_transform_edt(obs_np);dp=distance_transform_edt(pm)
        boundary=np.exp(-np.minimum(do,dp)/PARAMS['boundary_decay_pixels'])
        domain=obs_np&pm&(do>PARAMS['erosion_pixels'])&(dp>PARAMS['erosion_pixels'])
        anchor_ok=(crop_uv>=0).all(-1)&(crop_uv<518).all(-1)
        maps=dict(silhouette=torch.logical_xor(obs[i],pred['mask'][i]).float(),
            boundary=torch.tensor(boundary,device=device),direct_rgb=direct_rgb,direct_dino=direct,
            cross_rgb=cross[1],cross_dino=cross[0],shuffled_dino=cross[2],fused_dino=(direct+cross[0])/2)
        for key,value in maps.items():result[key].append(value.float().cpu().numpy())
        supports.append(support.cpu().numpy());domains.append(domain);anchors.append(anchor_ok.cpu().numpy())
    result={key:np.stack(value) for key,value in result.items()}
    result.update(support=np.stack(supports),interior=np.stack(domains),anchor_valid=np.stack(anchors),
                  pred_mask=pred['mask'].cpu().numpy(),observed_mask=obs.cpu().numpy())
    result['common']=result['interior']&(result['support']>0)&result['anchor_valid']
    # Input-only fallback for pixels without cross-view support; do not call them reliable.
    result['fused_full']=np.where(result['support']>0,result['fused_dino'],result['direct_dino'])
    result['fused_full']=np.maximum(result['fused_full'],result['silhouette'])
    return result
