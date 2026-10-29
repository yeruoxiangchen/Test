"""Input-only photometric verification and connected object-space edit volumes."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .config import PARAMS,BASE,read
from ...geometry import transform
from visual_evidence_probe.experiments.local_repair import photometric as photo
from visual_evidence_probe.experiments.dense_evidence.matching import configured,propose
from visual_evidence_probe.experiments.editor_support.objective import world_from_depth


def grid_centers(device,grid=32):
    c=torch.cartesian_prod(*[torch.arange(grid,device=device)]*3)
    return c,(c.float()+.5)/grid-.5


def footprint_fraction(mask,corners,image_size=518):
    """Foreground fraction in a conservative projected voxel bounding box."""
    height,width=mask.shape
    scale=corners.new_tensor([width/image_size,height/image_size])
    pixel=(corners+.5)*scale-.5
    lo=torch.floor(pixel.amin(1)).long();hi=torch.ceil(pixel.amax(1)).long()
    framed=(hi[:,0]>=0)&(hi[:,1]>=0)&(lo[:,0]<width)&(lo[:,1]<height)
    lo[:,0].clamp_(0,width-1);hi[:,0].clamp_(0,width-1)
    lo[:,1].clamp_(0,height-1);hi[:,1].clamp_(0,height-1)
    integral=F.pad(mask.float().cumsum(0).cumsum(1),(1,0,1,0))
    count=integral[hi[:,1]+1,hi[:,0]+1]-integral[lo[:,1],hi[:,0]+1]-integral[hi[:,1]+1,lo[:,0]]+integral[lo[:,1],lo[:,0]]
    area=(hi[:,0]-lo[:,0]+1)*(hi[:,1]-lo[:,1]+1)
    return count/area.clamp_min(1)*framed


def mask_cost(pred,observed,views):
    a=pred['softmask'][views];b=observed[views].float()
    inter=(a*b).sum((1,2));union=(a+b-a*b).sum((1,2)).clamp_min(1)
    return (1-inter/union+.1*(a-b).abs().mean((1,2))).mean()


@torch.no_grad()
def patch_evidence(depth,gray,eroded,k,w,i,j):
    xyz=world_from_depth(depth,k,w)[i]
    uv,z=photo.project(xyz,k[j],w[j]);warped=photo.sample(gray[j][None],uv)[...,0]
    cost,sa,sb=photo.patch_cost(gray[i],warped,5)
    valid=eroded[i]&(depth[i]>0)&(z>0)&(uv>=0).all(-1)&(uv<518).all(-1)
    valid &= photo.sample(eroded[j][None].float(),uv)[...,0]>.99
    valid &= (sa>PARAMS['photo_min_std'])&(sb>PARAMS['photo_min_std'])
    centers=torch.linalg.inv(w)[:,:3,3]
    a=F.normalize(xyz-centers[i],dim=-1);b=F.normalize(xyz-centers[j],dim=-1)
    cosine=(a*b).sum(-1);valid &= (cosine<.985)&(cosine>-.5)
    return cost,valid,uv,z


@torch.no_grad()
def reliable_targets(index,seed,data,baseline):
    # Raw posed DA3 is reused, but every selected ray is re-verified from RGB.
    from ...config import LEGACY
    provider=read(LEGACY/'objects'/f'{index:03d}'/str(seed)/'dense_provider.json')
    train=data['train'];heldout=data['heldout'];device=baseline['depth'].device
    if provider['train_views']!=train or provider['heldout_views']!=heldout:raise RuntimeError('provider split changed')
    depths=baseline['depth'].clone();confidence=torch.zeros_like(baseline['mask'])
    with np.load(provider['path']+'/prediction.npz') as a:
        if not np.allclose(a['intrinsics'],data['k'][train].cpu().numpy(),atol=1.e-5):raise RuntimeError('provider K mismatch')
        if not np.allclose(a['extrinsics'],data['w'][train].cpu().numpy(),atol=1.e-5):raise RuntimeError('provider W mismatch')
        depths[train]=torch.tensor(a['depth'],device=device);confidence[train]=torch.tensor(a['valid'],device=device)
    target=torch.zeros_like(depths);valid=torch.zeros_like(confidence);scores=torch.zeros_like(depths)
    for i in train:
        costs=[];old_costs=[];supports=[]
        for j in train:
            if i==j:continue
            cost,allowed,uv,z=patch_evidence(depths,data['gray'],data['eroded'],data['k'],data['w'],i,j)
            old,old_valid,_,_=patch_evidence(baseline['depth'],data['gray'],data['eroded'],data['k'],data['w'],i,j)
            other=photo.sample(depths[j][None],uv)[...,0]
            allowed &= (other-z).abs()<=PARAMS['consistency_depth']
            allowed &= confidence[i]&data['observed'][i]
            old=torch.where(old_valid,old,torch.ones_like(old))
            costs.append(cost);old_costs.append(old);supports.append(allowed)
        count=torch.stack(supports).sum(0);support=torch.stack(supports)
        new=(torch.stack(costs)*support).sum(0)/count.clamp_min(1)
        old=(torch.stack(old_costs)*support).sum(0)/count.clamp_min(1)
        okay=(count>=1)&(new<=PARAMS['photo_max_cost'])&(old-new>=PARAMS['photo_min_gain'])
        valid[i]=okay;target[i]=depths[i];scores[i]=(old-new).clamp_min(0)*okay
    da3_count=int(valid.sum())
    ncc=propose(baseline,data['gray'],data['eroded'],data['k'],data['w'],train)
    ncc_count=0
    for row,i in enumerate(train):
        allowed=torch.tensor(ncc['valid'][row],device=device)
        std=(photo.pool(data['gray'][i].square(),5)-photo.pool(data['gray'][i],5).square()).clamp_min(0).sqrt()
        allowed &= std>PARAMS['photo_min_std']
        target[i][allowed]=baseline['depth'][i][allowed]+torch.tensor(ncc['delta'][row],device=device)[allowed]
        valid[i]|=allowed;scores[i][allowed]=torch.tensor(ncc['confidence'][row],device=device)[allowed];ncc_count+=int(allowed.sum())
    valid[heldout]=False
    return dict(depth=target,valid=valid,confidence=scores,da3_verified_pixels=da3_count,ncc_pixels=ncc_count,
                gt_used=False,old_mixed_targets_reused=False)


@torch.no_grad()
def project_cells(points,matrix,data,baseline):
    world=transform(points,matrix);inside=[];deficits=[]
    for i in data['train']:
        if PARAMS.get('voxel_aware',False):
            offset=torch.cartesian_prod(*[points.new_tensor([-.5,.5])]*3)/32
            corners=transform((points[:,None]+offset[None]).reshape(-1,3),matrix).reshape(-1,8,3)
            uv,z=photo.project(corners,data['k'][i],data['w'][i])
            foreground=footprint_fraction(data['observed'][i],uv)
            deficit=footprint_fraction(data['observed'][i]&~baseline['mask'][i],uv)
            framed=(z>.001).all(1)
            inside.append((foreground>0)&framed)
            deficits.append((deficit>=PARAMS['mask_deficit_fraction'])&framed)
            continue
        uv,z=photo.project(world[:,None],data['k'][i],data['w'][i]);uv=uv[:,0];z=z[:,0]
        foreground=photo.sample(data['observed'][i][None].float(),uv[:,None])[...,0][:,0]>.95
        seen=photo.sample(baseline['mask'][i][None].float(),uv[:,None])[...,0][:,0]>.5
        framed=(z>0)&(uv>=0).all(-1)&(uv<518).all(-1)
        inside.append(foreground&framed);deficits.append(foreground&~seen&framed)
    hull=torch.stack(inside).all(0);missing=torch.stack(deficits).any(0)&hull
    return hull,missing


def connected_core(weights,limit=1024):
    seeds=weights>0
    if not seeds.any():return np.zeros_like(seeds)
    expanded=ndimage.binary_dilation(seeds,structure=np.ones((3,3,3)))
    labels,count=ndimage.label(expanded)
    sums=np.bincount(labels.ravel(),weights=weights.ravel(),minlength=count+1);sums[0]=0
    selected=labels==int(sums.argmax())
    if int(selected.sum())>limit:
        peak=np.asarray(np.unravel_index(np.argmax(weights*selected),weights.shape))
        c=np.argwhere(selected);distance=((c-peak)**2).sum(1);order=np.argsort(distance,kind='stable')[:limit]
        selected[:]=False;selected[tuple(c[order].T)]=True
        labels,count=ndimage.label(selected);sums=np.bincount(labels.ravel(),weights=weights.ravel(),minlength=count+1);sums[0]=0
        selected=labels==int(sums.argmax())
    return selected


@torch.no_grad()
def volume(matrix,data,baseline,trusted,coords):
    device=coords.device;cells,centers=grid_centers(device)
    support=torch.zeros((32,32,32),device=device,dtype=torch.bool)
    c=coords[:,1:].long();support[c[:,0],c[:,1],c[:,2]]=True
    hull,missing=project_cells(centers,matrix,data,baseline)
    weights=np.zeros((32,32,32),np.float64);positive=np.zeros_like(weights,dtype=bool)
    votes=np.zeros_like(weights,dtype=np.int32)
    inverse=torch.linalg.inv(matrix)
    def add(points,values,positives=False):
        if not len(points):return
        native=transform(points,inverse);index=torch.floor((native+.5)*32).long()
        valid=((index>=0)&(index<32)).all(1);index=index[valid].cpu().numpy();v=values[valid].cpu().numpy()
        np.add.at(weights,tuple(index.T),v)
        if positives:
            positive[tuple(index.T)]=True;np.add.at(votes,tuple(index.T),1)
    discrepancy=(trusted['depth']-baseline['depth']).abs()
    verified=trusted['valid']&(discrepancy>=PARAMS['depth_floor'])
    target_points=world_from_depth(trusted['depth'],data['k'],data['w'])
    add(target_points[verified],discrepancy[verified],True)
    base_valid=verified&baseline['mask'];add(baseline['xyz'][base_valid],discrepancy[base_valid])
    # Connect source/target endpoints so a large correction is a 3D patch.
    for t in (.25,.5,.75):
        add((1-t)*baseline['xyz'][base_valid]+t*target_points[base_valid],discrepancy[base_valid]*.2)
    depth_weights=weights.copy()
    extra=baseline['mask']&~data['observed'];extra[data['heldout']]=False
    add(baseline['xyz'][extra],torch.full_like(baseline['depth'][extra],.025))
    near=ndimage.distance_transform_edt(~support.cpu().numpy())<=4
    missing_grid=missing.reshape(32,32,32).cpu().numpy()&near
    weights[missing_grid]+=.025
    chosen=depth_weights if PARAMS.get('voxel_aware',False) and int(verified.sum())>=PARAMS['min_trusted_pixels'] and depth_weights.any() else weights
    core=connected_core(chosen,PARAMS['max_core_cells'])
    distance=ndimage.distance_transform_edt(~core) if core.any() else np.full_like(weights,100.)
    alpha=np.clip(1-distance/PARAMS['buffer_cells'],0,1).astype(np.float32)
    alpha[core]=1.;envelope=alpha>0
    positive &= envelope & hull.reshape(32,32,32).cpu().numpy()
    if PARAMS.get('anchor_support',False):positive &= votes>=PARAMS['anchor_min_rays']
    update_region=core if PARAMS.get('anchor_support',False) else envelope
    outside_support=(positive|missing_grid)&~support.cpu().numpy()&update_region
    anchor_missing=positive&~support.cpu().numpy()&core
    result=dict(core=core,alpha=alpha,envelope=envelope,positive=positive,
                hull=hull.reshape(32,32,32).cpu().numpy(),missing=missing_grid,
                verified_pixels=int(verified.sum()),extra_pixels=int(extra.sum()),
                has_evidence=bool(core.any()),needs_ss=bool(outside_support.sum()>=8 or (PARAMS.get('anchor_support',False) and anchor_missing.any())),
                unsupported_positive_cells=int(outside_support.sum()),
                anchor_cells=int((positive&core).sum()),
                active_fraction=float(envelope[tuple(c.cpu().numpy().T)].mean()) if len(c) else 0.,
                gt_used=False)
    return result
