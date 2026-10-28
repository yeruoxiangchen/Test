"""Frozen training-view evidence weights; never consumes a heldout image."""
import torch

from .config import PARAMS
from visual_evidence_probe.experiments.local_repair import photometric as photo
from visual_evidence_probe.experiments.dense_evidence.camera import backproject,project


@torch.no_grad()
def build(target,sparse,baseline,gray,eroded,k,w,train):
    depth=target['depth'];valid=target['depth_valid'];weight=torch.zeros_like(depth)
    photo_supported=torch.zeros_like(valid);contradicted=torch.zeros_like(valid)
    diagnostics=[]
    for i in train:
        # NCC anchors keep their original depth and unit weight. New teacher
        # rays get weak weight when input texture cannot verify them.
        anchors=sparse['depth_valid'][i]
        count=int(anchors.sum())
        discrepancy=(depth[i]-sparse['depth'][i]).abs()[anchors]
        # Target keeps NCC values, so this must use the raw teacher separately
        # when called; the caller supplies raw_teacher_depth in target.
        raw=target.get('raw_teacher_depth',depth)
        discrepancy=(raw[i]-sparse['depth'][i]).abs()[anchors]
        agreement=torch.exp(-torch.median(discrepancy).square()/.03**2) if count>=16 else depth.new_tensor(1.)
        weight[i]=valid[i].float()*.10*agreement
        xyz=backproject(depth[i],k[i],w[i]);original=backproject(baseline['depth'][i],k[i],w[i])
        costs=[];improvements=[];supports=[]
        for j in train:
            if i==j:continue
            uv,z=project(xyz,k[j],w[j]);buv,bz=project(original,k[j],w[j])
            warped=photo.sample(gray[j][None],uv)[...,0];base_warp=photo.sample(gray[j][None],buv)[...,0]
            cost,sa,sb=photo.patch_cost(gray[i],warped);base_cost,_,_=photo.patch_cost(gray[i],base_warp)
            domain=valid[i]&eroded[i]&(z>0)&(bz>0)&baseline['mask'][i]
            domain&=(uv>=0).all(-1)&(uv<518).all(-1)&(buv>=0).all(-1)&(buv<518).all(-1)
            domain&=photo.sample(eroded[j][None].float(),uv)[...,0]>.99
            domain&=photo.sample(eroded[j][None].float(),buv)[...,0]>.99
            domain&=(sa>.02)&(sb>.02)
            costs.append(cost);improvements.append(base_cost-cost);supports.append(domain)
        support=torch.stack(supports);n=support.sum(0)
        cost=(torch.stack(costs)*support).sum(0)/n.clamp_min(1)
        gain=(torch.stack(improvements)*support).sum(0)/n.clamp_min(1)
        available=n>0;bad=available&((cost>PARAMS['reliable_photo_cost'])|(gain< -PARAMS['photo_tolerance']))
        verified=available&~bad
        confidence=torch.exp(-cost/.15)*torch.sigmoid((gain+PARAMS['reliable_photo_gain'])/.01)
        weight[i]=torch.where(verified,confidence*agreement,weight[i]);weight[i][bad]=0
        weight[i][anchors]=1.;weight[i]*=valid[i]
        photo_supported[i]=available;contradicted[i]=bad&~anchors
        diagnostics.append(dict(view=i,ncc_anchors=count,teacher_ncc_scale_agreement=float(agreement),
            photo_supported=int(available.sum()),teacher_contradictions=int((bad&~anchors).sum())))
    return weight,dict(views=diagnostics,uses_gt=False,heldout_rgb_used=False,
        low_texture_policy='weak0.1 weight, multiplied by per-view raw-teacher/NCC anchor agreement',
        all_ncc_anchors_retained=True)
