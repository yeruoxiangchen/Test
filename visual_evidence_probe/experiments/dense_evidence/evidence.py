import numpy as np
import torch
import torch.nn.functional as F

from .config import folder,read,write,digest
from visual_evidence_probe.experiments.editor_support.objective import freeze_target


def augment_depth(dense_depth,dense_valid,sparse_depth,sparse_valid):
    result=dense_depth.clone();valid=dense_valid.clone()
    result[sparse_valid]=sparse_depth[sparse_valid]
    valid |= sparse_valid
    return result,valid


def observed_domain(rec,masks,device,side):
    observed=F.interpolate(torch.tensor(masks,device=device,dtype=torch.float32)[:,None],size=(side,side),mode='nearest')[:,0]>.5
    if rec['kind']=='real_scene':
        # The inherited BOP full bounding rectangle is declared input assistance.
        # Its interior outside the VISIBLE mask remains unknown, not negative.
        with np.load(rec['observation_domain']) as a:domain=a['valid']
        valid=F.interpolate(torch.tensor(domain,device=device,dtype=torch.float32)[:,None],size=(side,side),mode='nearest')[:,0]>.5
        valid |= observed
    else:valid=torch.ones_like(observed)
    return observed,valid


def build(index,seed,baseline,rec,masks):
    root=folder(index,seed);device=baseline['depth'].device
    with np.load(root/'region_sealed.npz') as a:
        record={name:a[name] for name in a.files}
    region=torch.tensor(record['region'],device=device,dtype=torch.float32)
    active=torch.tensor(record['active_rows'],device=device,dtype=torch.bool)
    train=record['train_views'].tolist();heldout=record['heldout_views'].tolist()
    observed,domain=observed_domain(rec,masks,device,baseline['depth'].shape[-1])
    targets={}
    with np.load(root/'sparse_sealed.npz') as a:sparse_depth,sparse_valid=a['depth'],a['valid']
    sparse=freeze_target(baseline)
    sparse['depth']=torch.tensor(sparse_depth,device=device)
    sparse['depth_valid']=torch.tensor(sparse_valid,device=device)
    sparse['mask']=sparse['depth_valid'].clone()
    targets['sparse']=sparse
    provider=read(root/'dense_provider.json')
    if set(provider['train_views'])!=set(train) or set(provider['heldout_views'])!=set(heldout):raise ValueError('DA3 camera split differs from sealed input split')
    with np.load(provider['path']+'/prediction.npz') as a:
        depth,valid,source_views=a['depth'],a['valid'],a['train_views'].tolist()
    dense=freeze_target(baseline)
    dense['depth']=torch.zeros_like(baseline['depth']);dense['depth_valid']=torch.zeros_like(baseline['mask'])
    for row,view in enumerate(source_views):
        dense['depth'][view]=torch.tensor(depth[row],device=device)
        dense['depth_valid'][view]=torch.tensor(valid[row],device=device)
    dense['depth'],dense['depth_valid']=augment_depth(dense['depth'],dense['depth_valid'],
        sparse['depth'],sparse['depth_valid'])
    dense['mask']=observed.clone()
    targets['dense']=dense
    for name,target in targets.items():
        target['softmask']=observed.float()
        target['silhouette_valid']=domain
        target['normal_valid']=torch.zeros_like(baseline['mask'])
        target['depth_valid'] &= domain & observed
        target['depth_valid'][heldout]=False
        target['depth'][~target['depth_valid']]=0
        target['mask']=target['depth_valid'].clone()
        if bool(target['depth_valid'][heldout].any()):raise ValueError('heldout depth leaked into loss')
    counts={name:int((target['depth_valid']&(region>.5))[train].sum()) for name,target in targets.items()}
    write(root/'evidence_complete.json',dict(index=index,seed=seed,train_views=train,heldout_views=heldout,
        counts=counts,dense_keeps_sparse_anchors=True,gt_used=False,region_sha256=digest(root/'region_sealed.npz'),
        sparse_sha256=digest(root/'sparse_sealed.npz'),dense_provider=provider,
        observation_domain='visible foreground and annotated rectangle exterior' if rec['kind']=='real_scene' else 'complete clean-input foreground/background'))
    return dict(region=region,active=active,train=train,heldout=heldout,targets=targets,
        observed=observed,observation_valid=domain,counts=counts,arrays=record)
