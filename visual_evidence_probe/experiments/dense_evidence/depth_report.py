"""Posthoc provider quality; GT never changes saved validity masks."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .config import OUT,ROUND6,PARAMS,folder,data_folder,setup,read,write
from .data import observations
from .evaluation import load_target,render_depth


def stats(depth,valid,gt,baseline,domain):
    selected=valid&domain
    if not selected.any():return dict(eligible=int(domain.sum()),valid=0,coverage=0.,before_mae=None,after_mae=None)
    before_raw=np.abs(baseline-gt)[selected];after_raw=np.abs(depth-gt)[selected]
    baseline_hit=baseline[selected]>0
    before=np.where(baseline_hit,np.minimum(before_raw,.1),.1)
    after=np.minimum(after_raw,.1)
    return dict(eligible=int(domain.sum()),valid=int(selected.sum()),coverage=float(selected.sum()/max(domain.sum(),1)),
        before_mae=float(before.mean()),after_mae=float(after.mean()),
        original_missing_fraction=float((~baseline_hit).mean()),
        before_raw_common_mae=float(before_raw[baseline_hit].mean()) if baseline_hit.any() else None,
        after_raw_mae=float(after_raw.mean()),
        improved_fraction=float((after<before).mean()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='0');p.add_argument('--indices',default='4,5,6,7,8,9,10,11');a=p.parse_args()
    setup(a.gpu)
    import nvdiffrast.torch as dr
    import trimesh
    device=torch.device('cuda:0');context=dr.RasterizeCudaContext(device=device)
    records=[]
    for index in map(int,a.indices.split(',')):
        rec,k,w,_,masks=observations(index,device)
        if rec['split']=='development':
            gt=torch.load(ROUND6/'objects'/f'{index:03d}'/'evaluation_targets.pt',map_location='cpu',weights_only=False)['input']
            gtd,gth=gt['depth'].numpy(),gt['mask'].numpy()
        else:
            target=load_target(index);gt=render_depth(target,k,w,context);gtd,gth=gt['depth'].cpu().numpy(),gt['mask'].cpu().numpy()
        observed=torch.nn.functional.interpolate(torch.tensor(masks,dtype=torch.float32)[:,None],size=(256,256),mode='nearest')[:,0].numpy()>.5
        for seed in PARAMS['seeds']:
            root=folder(index,seed)
            if not (root/'dense_provider.json').exists():continue
            provider=read(root/'dense_provider.json')
            with np.load(Path(provider['path'])/'prediction.npz') as z:
                depth,valid,train=z['depth'],z['valid'],z['train_views'].tolist()
            with np.load(root/'region_sealed.npz') as z:region=z['region'][train]>.5
            base_mesh=trimesh.load(root/'original.obj',process=False,force='mesh',skip_materials=True)
            baseline=render_depth(base_mesh,k,w,context)['depth'].cpu().numpy()[train]
            domain=gth[train]&observed[train]
            common=dict(index=index,seed=seed,uid=rec['uid'],split=rec['split'],
                metric_definition='clipped_fixed_missing_v2',
                whole=stats(depth,valid,gtd[train],baseline,domain),
                local=stats(depth,valid,gtd[train],baseline,domain&region),
                local_unfiltered=stats(depth,np.ones_like(valid),gtd[train],baseline,domain&region),
                method_targets_unchanged=True)
            records.append(common)
            print(common,flush=True)
    dest=OUT/'depth_quality.json'
    previous=read(dest)['records'] if dest.exists() else []
    merged={(r['index'],r['seed']):r for r in previous}
    merged.update({(r['index'],r['seed']):r for r in records})
    write(dest,dict(records=[merged[key] for key in sorted(merged)],posthoc_only=True,
        uses_gt_for_evaluation=True,changes_targets_or_selection=False,
        error_definition='fixed selected GT domain; missing original ray pays0.1; errors clipped0.1; raw common-ray error separate'))


if __name__=='__main__':main()
