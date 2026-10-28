import numpy as np
import torch
from .config import OUT,PRIOR,PARAMS,folder,read,write,digest
from .reliability import build as reliable_weights
from visual_evidence_probe.experiments.dense_evidence.evidence import observed_domain,augment_depth
from visual_evidence_probe.experiments.editor_support.objective import freeze_target
from visual_evidence_probe.experiments.local_repair.photometric import prepare_images


def link(path,source):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():path.symlink_to(source)


@torch.no_grad()
def build(index,seed,baseline,rec,k,w,images,masks):
    source=PRIOR/'objects'/f'{index:03d}'/str(seed);root=folder(index,seed);root.mkdir(parents=True,exist_ok=True)
    for name in ('original.obj','original_latent.pt','fit.json'):link(root/name,source/name)
    with np.load(source/'region_sealed.npz') as data:arrays={key:data[key] for key in data.files}
    device=k.device;train=arrays['train_views'].tolist();heldout=arrays['heldout_views'].tolist()
    region=torch.tensor(arrays['region'],device=device);active=torch.tensor(arrays['active_rows'],device=device)
    observed,domain=observed_domain(rec,masks,device,PARAMS['side'])
    with np.load(source/'sparse_sealed.npz') as data:
        sparse=dict(depth=torch.tensor(data['depth'],device=device),depth_valid=torch.tensor(data['valid'],device=device))
    provider=read(source/'dense_provider.json')
    if provider['train_views']!=train or provider['heldout_views']!=heldout:raise ValueError('provider input split mismatch')
    target=freeze_target(baseline);target['depth']=torch.zeros_like(baseline['depth']);target['depth_valid']=torch.zeros_like(baseline['mask'])
    with np.load(provider['path']+'/prediction.npz') as data:
        for row,view in enumerate(data['train_views'].tolist()):
            target['depth'][view]=torch.tensor(data['depth'][row],device=device)
            target['depth_valid'][view]=torch.tensor(data['valid'][row],device=device)
    raw=target['depth'].clone()
    target['depth'],target['depth_valid']=augment_depth(target['depth'],target['depth_valid'],sparse['depth'],sparse['depth_valid'])
    target['depth_valid'] &= observed&domain;target['depth_valid'][heldout]=False
    target['depth'][~target['depth_valid']]=0
    target.update(mask=target['depth_valid'].clone(),softmask=observed.float(),silhouette_valid=domain,
        normal_valid=torch.zeros_like(baseline['mask']))
    reliable=freeze_target(target);reliable['raw_teacher_depth']=raw
    _,gray,_,eroded=prepare_images(images,masks,PARAMS['side'])
    weight,receipt=reliable_weights(reliable,sparse,baseline,gray,eroded,k,w,train)
    reliable['depth_weight']=weight
    reliable['depth_valid'] &= weight>0
    reliable['mask']=reliable['depth_valid'].clone();reliable['depth'][~reliable['depth_valid']]=0
    if bool(reliable['depth_valid'][heldout].any()) or bool(weight[heldout].any()):raise ValueError('heldout evidence leaked')
    counts=dict(dense_reference=int((target['depth_valid']&(region>.5))[train].sum()),
        anchored=int((target['depth_valid']&(region>.5))[train].sum()),
        anchored_reliable=int((reliable['depth_valid']&(region>.5))[train].sum()))
    np.savez_compressed(root/'evidence_sealed.npz',depth=target['depth'].cpu().numpy(),
        valid=target['depth_valid'].cpu().numpy(),reliable_valid=reliable['depth_valid'].cpu().numpy(),
        weights=weight.cpu().numpy(),region=arrays['region'],active_rows=arrays['active_rows'],
        train_views=arrays['train_views'],heldout_views=arrays['heldout_views'])
    write(root/'input_receipt.json',dict(index=index,seed=seed,uid=rec['uid'],split=rec['split'],
        counts=counts,source=str(source),uses_gt=False,reliability=receipt,
        source_region_sha256=digest(source/'region_sealed.npz'),source_sparse_sha256=digest(source/'sparse_sealed.npz'),
        anchors_and_regions_from_original_only=True,original_ss_fixed=True))
    return dict(region=region,active=active,train=train,heldout=heldout,arrays=arrays,
        targets=dict(dense_reference=target,anchored=target,anchored_reliable=reliable),counts=counts,
        observed=observed,observation_valid=domain,images=images,masks=masks,gray=gray,eroded=eroded)
