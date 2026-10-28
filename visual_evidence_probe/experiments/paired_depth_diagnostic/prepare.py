"""Seal paired targets in a separate process; no novel depth is exported."""
import hashlib
import numpy as np
import torch
from .config import OUT,ROUND7,ROUND8,PARAMS,folder,read,write,setup,now,register
from visual_evidence_probe.experiments.constrained_evidence.data import link


def matched_support(estimated_valid,gt_depth,gt_mask,region,train,heldout):
    if set(train)&set(heldout):raise ValueError('train/heldout overlap')
    valid=np.asarray(estimated_valid,bool)&np.asarray(gt_mask,bool)&np.isfinite(gt_depth)&(gt_depth>0)&(region>.5)
    train_mask=np.zeros(len(valid),bool);train_mask[train]=True
    valid &= train_mask[:,None,None]
    valid[heldout]=False
    return valid


def main():
    setup();register('prepare');registry=[]
    for index in PARAMS['indices']:
        rec=read(ROUND7/'data'/f'{index:03d}'/'observations.json')
        # This preparation process may read a cache containing novel tensors;
        # it exports INPUT camera-z only. The optimizer reads only sealed NPZs.
        cached=torch.load(ROUND7/'data'/f'{index:03d}'/'evaluation_targets.pt',map_location='cpu',weights_only=False)['input']
        gt_depth=cached['depth'].numpy();gt_mask=cached['mask'].numpy()
        for seed in PARAMS['seeds']:
            root=folder(index,seed);root.mkdir(parents=True,exist_ok=True)
            if (root/'targets_complete.json').exists():continue
            old=ROUND7/'objects'/f'{index:03d}'/str(seed)
            for name in ('original.obj','original_latent.pt','fit.json','region_sealed.npz','novel_region.npz'):
                link(root/name,old/name)
            with np.load(old/'region_sealed.npz') as data:
                region,train,heldout=data['region'],data['train_views'].tolist(),data['heldout_views'].tolist()
            with np.load(ROUND8/'objects'/f'{index:03d}'/str(seed)/'evidence_sealed.npz') as data:
                estimate,source_valid=data['depth'],data['valid']
            valid=matched_support(source_valid,gt_depth,gt_mask,region,train,heldout)
            if not np.isfinite(estimate[valid]).all() or not (estimate[valid]>0).all():raise ValueError('invalid estimate on common support')
            mask_hash=hashlib.sha256(valid.tobytes()).hexdigest()
            for source,depth in (('estimated',estimate),('oracle',gt_depth)):
                np.savez_compressed(root/f'{source}_target.npz',depth=np.where(valid,depth,0).astype(np.float32),valid=valid)
            meta=dict(time=now(),index=index,seed=seed,uid=rec['uid'],privileged_diagnostic=True,
                train_views=train,heldout_views=heldout,original_queries=int((source_valid&(region>.5))[train].sum()),
                common_queries=int(valid.sum()),removed_no_gt_intersection=int(((source_valid&(region>.5))[train]).sum()-valid.sum()),
                support_sha256=mask_hash,estimated_source='round8 unweighted NCC+DA3 (not reliability-weighted)',
                same_masks_all_arms=True,gt_support_used_even_in_estimated_arms=True,
                gt_input_depth_exported_only_for_oracle=True,novel_depth_exported=False)
            write(root/'targets_complete.json',meta);print(meta,flush=True)
        registry.append(dict(index=index,uid=rec['uid'],split='inspected_development_diagnostic'))
    write(OUT/'registry.json',dict(time=now(),rows=registry,parameters=PARAMS,privileged_diagnostic=True))


if __name__=='__main__':main()
