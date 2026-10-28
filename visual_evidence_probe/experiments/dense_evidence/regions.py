"""Seal input-only edit regions and fresh holdout-excluding sparse targets."""
import argparse
import gc

import numpy as np
import torch
import torch.nn.functional as F

from .config import OUT,ROUND6,PARAMS,data_folder,folder,registry,setup,register,read,write,digest,emit
from .data import observations,load_latent,fit_transform
from .matching import configured,propose
from visual_evidence_probe.experiments.geometry_evidence.metrics import render_mesh,face_region
from visual_evidence_probe.experiments.geometry_evidence.runner import load_obj
from visual_evidence_probe.experiments.editor_support.data import points_transform,active_tokens
from visual_evidence_probe.experiments.local_repair import photometric


def choose_holdout(w):
    centers=torch.linalg.inv(w)[:,:3,3]
    direction=F.normalize(centers,dim=-1)
    similarity=direction@direction.T
    similarity.fill_diagonal_(-1)
    return int(similarity.max(1).values.argmin())


@torch.no_grad()
def one(index,seed,context,device):
    root=folder(index,seed)
    if (root/'region_complete.json').exists():return
    if not (root/'baseline_complete.json').exists():raise FileNotFoundError(f'baseline not ready: {index}/{seed}')
    rec,k,w,images,masks=observations(index,device)
    z=load_latent(index,seed,device);fit=fit_transform(index,seed,device)
    mesh=load_obj(root/'original.obj');pred=render_mesh(mesh,k,w,context)
    rgb,gray,obs,eroded=photometric.prepare_images(images,masks,PARAMS['side'])
    if rec['split']=='development':
        with np.load(ROUND6/'objects'/f'{index:03d}'/str(seed)/'input_sealed.npz') as a:
            region=torch.tensor(a['region'],device=device,dtype=torch.float32)
            selected=torch.tensor(a['selected_faces'],device=device)
            prior_holdout=a['heldout_views'].tolist()
        heldout=int(prior_holdout[0]) if prior_holdout else choose_holdout(w)
        train=[i for i in range(len(k)) if i!=heldout]
        provenance='same round6 input-derived region; holdout chosen before new depth inference'
    else:
        from visual_evidence_probe.experiments.local_repair.deform import area_region
        from scipy.ndimage import distance_transform_edt
        heldout=choose_holdout(w);train=[i for i in range(len(k)) if i!=heldout]
        with configured(train):scores=photometric.localization_scores(pred,rgb,eroded,k,w)
        # Textureless/visibility-poor pixels retain an input-boundary fallback.
        for row,view in enumerate(train):
            boundary=np.exp(-distance_transform_edt(obs[view].cpu().numpy())/3.)*.1
            scores[row]=np.where(np.isfinite(scores[row]),scores[row],boundary)
            scores[row][~((pred['mask'][view]&obs[view]).cpu().numpy())]=np.nan
        selected_np,_=area_region(mesh,pred['face_id'][train].cpu().numpy(),scores,'automatic',PARAMS['random_seed']+index*1009+seed)
        selected=torch.tensor(selected_np,device=device)
        region=face_region(pred,selected)
        provenance='input-train-view cross-RGB residual plus observed-boundary fallback, top20% eligible face area'
    eligible=(region>.5)&pred['mask']
    points=pred['xyz'][eligible]
    if len(points):
        points=points[torch.linspace(0,len(points)-1,min(PARAMS['max_region_points'],len(points)),device=device).long()]
        native=points_transform(points,torch.linalg.inv(fit)).cpu().numpy()
        active=active_tokens(z,native)
    else:active=torch.zeros(len(z.feats),device=device,dtype=torch.bool)
    proposals=propose(pred,gray,eroded,k,w,train)
    ranked=[]
    for row,view in enumerate(train):
        valid=proposals['valid'][row]&(region[view].cpu().numpy()>.5)
        for y,x in np.argwhere(valid):ranked.append((float(proposals['confidence'][row,y,x]),view,int(y),int(x),row))
    ranked.sort(key=lambda q:(-q[0],q[1],q[2],q[3]));ranked=ranked[:PARAMS['max_sparse_constraints']]
    target_depth=np.zeros(pred['depth'].shape,np.float32);target_valid=np.zeros_like(target_depth,bool)
    pixels=[]
    for weight,view,y,x,row in ranked:
        delta=float(proposals['delta'][row,y,x])
        target_depth[view,y,x]=float(pred['depth'][view,y,x])+delta
        target_valid[view,y,x]=True;pixels.append([view,y,x])
    with configured(train,heldout):
        supports=photometric.validation_reference(pred,gray,eroded,k,w)
        supports &= region[heldout][None].bool()
        baseline_cost=photometric.validation_cost(pred,gray,eroded,k,w,supports)
    np.savez_compressed(root/'region_sealed.npz',region=region.cpu().numpy(),selected_faces=selected.cpu().numpy(),
        active_rows=active.cpu().numpy(),source_points=points.cpu().numpy(),train_views=np.array(train),
        heldout_views=np.array([heldout]),validation_support=supports.cpu().numpy())
    np.savez_compressed(root/'sparse_sealed.npz',depth=target_depth,valid=target_valid,pixel=np.asarray(pixels,np.int64).reshape(-1,3))
    write(root/'region_complete.json',dict(index=index,seed=seed,uid=rec['uid'],uses_gt=False,
        train_views=train,heldout_views=[heldout],provenance=provenance,
        region_sha256=digest(root/'region_sealed.npz'),sparse_sha256=digest(root/'sparse_sealed.npz'),
        sparse_queries=len(pixels),active_rows=int(active.sum()),baseline_validation_cost=baseline_cost,
        validation_pixels=int(supports.sum()),local_heldout_pixels=int((region[heldout]>.5).sum()),
        input_only_regions_and_proposals=True))
    emit(index,seed,'regions_sealed',queries=len(pixels),train=train,heldout=heldout)


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='0');p.add_argument('--indices',default=','.join(map(str,range(4,12))));p.add_argument('--seeds',default='42,43');a=p.parse_args()
    setup(a.gpu)
    from visual_evidence_probe.runtime import setup as runtime_setup
    import nvdiffrast.torch as dr
    device=runtime_setup();register('regions');context=dr.RasterizeCudaContext(device=device)
    for index in map(int,a.indices.split(',')):
        for seed in map(int,a.seeds.split(',')):
            one(index,seed,context,device);gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
