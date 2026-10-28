"""Seal new T2 edit regions from input-only DA3/NCC evidence."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from .config import OUT,LEGACY,PARAMS,folder,read,write,sha256,now
from .geometry import observations,detached,mask_iou


@torch.no_grad()
def build(index,seed,features,coords,matrix,baseline,faces,world_vertices):
    from visual_evidence_probe.experiments.local_repair import photometric as photo
    from visual_evidence_probe.experiments.dense_evidence.matching import configured,propose
    rec,k,w,images,masks=observations(index)
    root=folder(index,seed)
    provider=read(LEGACY/'objects'/f'{index:03d}'/str(seed)/'dense_provider.json')
    metadata=read(provider['path']+'/complete.json')
    if metadata['uses_gt']:raise RuntimeError('provider used GT')
    train=provider['train_views'];heldout=provider['heldout_views']
    if set(train)&set(heldout) or set(train)|set(heldout)!=set(range(len(k))):raise RuntimeError('invalid input split')
    hashes=[sha256(rec['images'][i]['path']) for i in train]
    if hashes!=metadata['input_image_sha256']:raise RuntimeError('cached teacher inputs differ')
    with np.load(provider['path']+'/prediction.npz') as a:
        depth,valid=a['depth'],a['valid']
        if not np.allclose(a['intrinsics'],k[train].cpu().numpy(),atol=1.e-5):raise RuntimeError('teacher intrinsics differ')
        if not np.allclose(a['extrinsics'],w[train].cpu().numpy(),atol=1.e-5):raise RuntimeError('teacher extrinsics differ')
    _,gray,observed,eroded=photo.prepare_images(images,masks,PARAMS['side'])
    target=detached(baseline);target['depth']=torch.zeros_like(baseline['depth'])
    target['depth_valid']=torch.zeros_like(baseline['mask'])
    for row,view in enumerate(train):
        target['depth'][view]=torch.tensor(depth[row],device='cuda:0')
        target['depth_valid'][view]=torch.tensor(valid[row],device='cuda:0')
    proposals=propose(baseline,gray,eroded,k,w,train)
    ncc_count=0
    for row,view in enumerate(train):
        valid_ncc=torch.tensor(proposals['valid'][row],device='cuda:0')
        target['depth'][view][valid_ncc]=baseline['depth'][view][valid_ncc]+torch.tensor(proposals['delta'][row],device='cuda:0')[valid_ncc]
        target['depth_valid'][view]|=valid_ncc;ncc_count+=int(valid_ncc.sum())
    target['depth_valid']&=observed&baseline['mask']
    target['depth_valid'][heldout]=False
    discrepancy=(target['depth']-baseline['depth']).abs()
    verified=target['depth_valid']&(discrepancy>=PARAMS['min_error'])
    score=torch.zeros(len(faces),device='cuda:0')
    face_ids=baseline['face_id'][verified];values=discrepancy[verified]
    if len(face_ids):score.scatter_reduce_(0,face_ids,values,reduce='amax',include_self=True)
    eligible=torch.nonzero(score>=PARAMS['min_error']).flatten()
    selected=torch.zeros(len(faces),device='cuda:0',dtype=torch.bool)
    if len(eligible):
        # Fixed top20% of eligible surface area, with an absolute error floor.
        # This is a pilot localization rule; no GT region is used.
        ranked=eligible[torch.argsort(score[eligible],descending=True,stable=True)]
        triangle=world_vertices[faces.long()]
        areas=torch.linalg.vector_norm(torch.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0],dim=-1),dim=-1)*.5
        cumulative=areas[ranked].cumsum(0);budget=areas[eligible].sum()*PARAMS['region_fraction']
        count=min(len(ranked),int((cumulative<=budget).sum())+1)
        selected[ranked[:max(1,count)]]=True
    region=(selected[baseline['face_id'].clamp_min(0)]&baseline['mask']).float()
    region=F.max_pool2d(region[:,None],5,stride=1,padding=2)[:,0]
    target['depth_valid']&=region>.5
    target['mask']=target['depth_valid'].clone();target['softmask']=observed.float()
    target['normal_valid']=torch.zeros_like(observed)
    points=baseline['xyz'][(region>.5)&baseline['mask']]
    if len(points):
        points=points[torch.linspace(0,len(points)-1,min(512,len(points)),device='cuda:0').long()]
        inverse=torch.linalg.inv(matrix);native=points@inverse[:3,:3].T+inverse[:3,3]
        centers=(coords[:,1:].float()+.5)/32-.5
        distance=cKDTree(native.cpu().numpy()).query(centers.cpu().numpy(),workers=2)[0]
        active=torch.tensor(distance<=PARAMS['active_radius'],device='cuda:0')
    else:active=torch.zeros(len(features),device='cuda:0',dtype=torch.bool)
    with configured(train,heldout[0]):
        supports=photo.validation_reference(baseline,gray,eroded,k,w)
        supports&=region[heldout[0]][None]>.5
        initial_photo=photo.validation_cost(baseline,gray,eroded,k,w,supports) if int(supports.sum())>=8 else None
    np.savez_compressed(root/'evidence_sealed.npz',region=region.cpu().numpy(),active=active.cpu().numpy(),
        selected_faces=selected.cpu().numpy(),train_views=np.array(train),heldout_views=np.array(heldout),
        target_depth=target['depth'].cpu().numpy(),target_valid=target['depth_valid'].cpu().numpy())
    write(root/'evidence.json',dict(time=now(),gt_used=False,provider=str(provider['path']),
          cached_provider_input_hashes_match=True,cameras_match=True,old_ncc_targets_reused=False,
          ncc_recomputed_on_t2=True,ncc_pixels=ncc_count,target_pixels=int(target['depth_valid'].sum()),
          selected_faces=int(selected.sum()),active_rows=int(active.sum()),total_rows=len(active),
          train_views=train,heldout_views=heldout,initial_photo=initial_photo))
    return dict(k=k,w=w,target=target,region=region,active=active,selected=selected,train=train,heldout=heldout,
                gray=gray,eroded=eroded,supports=supports,initial_photo=initial_photo,observed=observed)


def photo_cost(pred,record):
    from visual_evidence_probe.experiments.local_repair import photometric as photo
    from visual_evidence_probe.experiments.dense_evidence.matching import configured
    if int(record['supports'].sum())<8:return None
    with configured(record['train'],record['heldout'][0]):
        return photo.validation_cost(pred,record['gray'],record['eroded'],record['k'],record['w'],record['supports'])
