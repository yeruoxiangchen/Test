"""Separate posthoc evaluator. No model or optimizer is imported here."""
import argparse
import gc
import numpy as np
import torch

from .config import OUT,PRIOR,ARMS,PARAMS,folder,read,write,setup,now,register,emit
from visual_evidence_probe.experiments.dense_evidence.data import observations
from visual_evidence_probe.experiments.dense_evidence.evaluation import load_target,render_depth
from visual_evidence_probe.experiments.geometry_evidence.runner import load_obj
from visual_evidence_probe.experiments.geometry_evidence.metrics import Evaluator


@torch.no_grad()
def one(index,seed,context,device):
    root=folder(index,seed)
    if (root/'complete.json').exists():return
    if not (root/'input_complete.json').exists():raise ValueError('all method decisions must be sealed before GT')
    source=PRIOR/'objects'/f'{index:03d}'/str(seed);rec,k,w,_,_=observations(index,device)
    cameras=read(PRIOR/'data'/f'{index:03d}'/'novel_cameras.json')
    nk=torch.tensor(cameras['intrinsics'],device=device,dtype=torch.float32);nw=torch.tensor(cameras['extrinsics'],device=device,dtype=torch.float32)
    target=load_target(index)
    cache=torch.load(PRIOR/'data'/f'{index:03d}'/'evaluation_targets.pt',map_location=device,weights_only=False)
    original=load_obj(root/'original.obj');baseline=render_depth(original,k,w,context);novel=render_depth(original,nk,nw,context)
    with np.load(source/'region_sealed.npz') as data:region=torch.tensor(data['region'],device=device)
    with np.load(source/'novel_region.npz') as data:novel_region=torch.tensor(data['region'],device=device)
    scorer=Evaluator(target,original,baseline,cache['input'],region,novel,cache['novel'],novel_region,k,w,nk,nw,context)
    original_metrics=read(source/'baseline_metrics.json');scored={(root/'original.obj').resolve():original_metrics}
    for arm in ARMS:
        path=root/arm;optimized=read(path/'complete.json');decisions=optimized['decisions']
        if arm=='dense_reference':
            metrics=read(source/'dense_observed/result.json')['metrics']
        else:
            metrics={}
            structures={r['step']:r['structure'] for r in optimized['snapshots']}
            for endpoint,step in (('accepted',decisions['observed_step']),('candidate',decisions['own_training_step']),
                    ('proposal',PARAMS['steps'] if optimized['steps'] else 0)):
                mesh_path=(path/f'{endpoint}.obj').resolve()
                if mesh_path not in scored:
                    mesh=load_obj(mesh_path);scored[mesh_path]=scorer(mesh,structure=structures.get(step));del mesh
                metrics[endpoint]=scored[mesh_path]
        result=dict(index=index,seed=seed,uid=rec['uid'],kind=rec['kind'],split='development',arm=arm,
            status=optimized['status'],accepted_edit=decisions['observed_step']>0,
            selected_step=decisions['observed_step'],candidate_step=decisions['own_training_step'],
            original_metrics=original_metrics,metrics=metrics,elapsed_seconds=optimized['elapsed_seconds'],
            peak_allocated_bytes=optimized['peak_allocated_bytes'],rollbacks=len(optimized.get('rollbacks',[])),
            source_seconds=optimized.get('reference_seconds'),reused_round7=optimized.get('reused_round7',False),
            same_original_frame=True,no_new_alignment=True,gt_only_after_selection=True,decisions=decisions)
        write(path/'result.json',result)
    write(root/'complete.json',dict(time=now(),index=index,seed=seed,arms=3,gt_only_after_selection=True))
    emit(index,seed,'evaluated_complete');del scorer,scored,cache,target,original,baseline,novel;gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='1');p.add_argument('--indices',default='4,5,6,7,8,9,10,11');p.add_argument('--seeds',default='42,43');a=p.parse_args();setup(a.gpu);register('evaluate')
    import nvdiffrast.torch as dr
    device=torch.device('cuda:0');context=dr.RasterizeCudaContext(device=device)
    for index in map(int,a.indices.split(',')):
        for seed in map(int,a.seeds.split(',')):one(index,seed,context,device)


if __name__=='__main__':main()
