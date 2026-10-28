import numpy as np
import torch
from .config import PARAMS,folder,read,write
from visual_evidence_probe.experiments.dense_evidence.evidence import observed_domain
from visual_evidence_probe.experiments.editor_support.objective import freeze_target
from visual_evidence_probe.experiments.local_repair.photometric import prepare_images


def base_record(index,seed,baseline,rec,k,w,images,masks):
    root=folder(index,seed);meta=read(root/'targets_complete.json')
    with np.load(root/'region_sealed.npz') as data:arrays={key:data[key] for key in data.files}
    observed,domain=observed_domain(rec,masks,k.device,PARAMS['side'])
    _,gray,_,eroded=prepare_images(images,masks,PARAMS['side'])
    return dict(region=torch.tensor(arrays['region'],device=k.device),
        active=torch.tensor(arrays['active_rows'],device=k.device),train=meta['train_views'],heldout=meta['heldout_views'],
        arrays=arrays,observed=observed,observation_valid=domain,gray=gray,eroded=eroded,
        images=images,masks=masks,meta=meta)


def target(index,seed,source,baseline,record):
    if source not in ('estimated','oracle'):raise ValueError(source)
    result=freeze_target(baseline)
    with np.load(folder(index,seed)/f'{source}_target.npz') as data:
        result['depth']=torch.tensor(data['depth'],device=baseline['depth'].device)
        result['depth_valid']=torch.tensor(data['valid'],device=baseline['depth'].device)
    if bool(result['depth_valid'][record['heldout']].any()):raise ValueError('heldout target leaked')
    if int(result['depth_valid'].sum())!=record['meta']['common_queries']:raise ValueError('support changed')
    result.update(mask=result['depth_valid'].clone(),softmask=record['observed'].float(),
        silhouette_valid=record['observation_valid'],normal_valid=torch.zeros_like(baseline['mask']))
    return result
