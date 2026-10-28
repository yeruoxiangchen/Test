"""Evaluation-only novel GT and surface metrics after all4 trajectories seal."""
import argparse
import gc
import numpy as np
import torch
from .config import OUT,ROUND7,PARAMS,ARMS,folder,read,write,now,setup,register,emit
from visual_evidence_probe.experiments.dense_evidence.data import observations
from visual_evidence_probe.experiments.dense_evidence.evaluation import load_target,render_depth
from visual_evidence_probe.experiments.geometry_evidence.metrics import Evaluator
from visual_evidence_probe.experiments.geometry_evidence.runner import load_obj


@torch.no_grad()
def one(index,seed,context,device):
    root=folder(index,seed)
    if (root/'complete.json').exists():return
    if not (root/'optimization_complete.json').exists():raise ValueError('not all4 arms sealed')
    rec,k,w,_,_=observations(index,device);original=load_obj(root/'original.obj');target=load_target(index)
    camera=read(ROUND7/'data'/f'{index:03d}'/'novel_cameras.json')
    nk=torch.tensor(camera['intrinsics'],device=device,dtype=torch.float32);nw=torch.tensor(camera['extrinsics'],device=device,dtype=torch.float32)
    gt=torch.load(ROUND7/'data'/f'{index:03d}'/'evaluation_targets.pt',map_location=device,weights_only=False)
    baseline=render_depth(original,k,w,context);novel=render_depth(original,nk,nw,context)
    with np.load(root/'region_sealed.npz') as data:region=torch.tensor(data['region'],device=device)
    with np.load(root/'novel_region.npz') as data:novel_region=torch.tensor(data['region'],device=device)
    base_metrics=read(ROUND7/'objects'/f'{index:03d}'/str(seed)/'baseline_metrics.json')
    scorer=Evaluator(target,original,baseline,gt['input'],region,novel,gt['novel'],novel_region,k,w,nk,nw,context)
    scored={(root/'original.obj').resolve():base_metrics}
    with np.load(root/'estimated_target.npz') as data:estimated,valid=data['depth'],data['valid']
    with np.load(root/'oracle_target.npz') as data:truth=data['depth']
    bdepth=baseline['depth'].cpu().numpy();hit=baseline['mask'].cpu().numpy();common=valid&hit
    raw=(estimated-truth)[valid];before=np.where(hit,np.minimum(np.abs(bdepth-truth),.1),.1)[valid]
    important=common&(np.abs(truth-bdepth)>=1/256)
    quality=dict(index=index,seed=seed,pixels=int(valid.sum()),baseline_missing_fraction=float((~hit)[valid].mean()) if valid.any() else None,
        original_clipped_gt_mae=float(before.mean()) if len(before) else None,
        estimated_raw_gt_mae=float(np.abs(raw).mean()) if len(raw) else None,
        estimated_clipped_gt_mae=float(np.minimum(np.abs(raw),.1).mean()) if len(raw) else None,
        estimated_training_clipped_fraction=float((np.abs(estimated-bdepth)[common]>=.1).mean()) if common.any() else None,
        oracle_training_clipped_fraction=float((np.abs(truth-bdepth)[common]>=.1).mean()) if common.any() else None,
        estimated_correct_direction_fraction=float((((estimated-bdepth)*(truth-bdepth)>0)[important]).mean()) if important.any() else None,
        privileged_posthoc=True)
    write(root/'target_quality.json',quality)
    for arm in ARMS:
        path=root/arm;optimized=read(path/'complete.json');decisions=optimized['decisions'];structures={s['step']:s['structure'] for s in optimized['snapshots']}
        scores={}
        for endpoint,step in (('candidate',decisions['own_training_step']),('proposal',PARAMS['steps'] if optimized['steps'] else 0),('accepted',decisions['observed_step'])):
            mesh_path=(path/f'{endpoint}.obj').resolve()
            if mesh_path not in scored:
                mesh=load_obj(mesh_path);scored[mesh_path]=scorer(mesh,structure=structures.get(step));del mesh
            scores[endpoint]=scored[mesh_path]
        write(path/'result.json',dict(index=index,seed=seed,uid=rec['uid'],arm=arm,status=optimized['status'],
            original_metrics=base_metrics,metrics=scores,candidate_step=decisions['own_training_step'],selected_step=decisions['observed_step'],
            accepted_edit=decisions['observed_step']>0,elapsed_seconds=optimized['elapsed_seconds'],
            peak_allocated_bytes=optimized['peak_allocated_bytes'],rollbacks=len(optimized['rollbacks']),
            privileged_diagnostic=True,gt_support_used=True,gt_depth_in_gradient=arm.startswith('oracle_'),
            support_sha256=optimized['support_sha256'],shared_normalization=optimized['shared_normalization'],
            no_new_alignment=True,novel_gt_after_optimization_only=True))
    write(root/'complete.json',dict(time=now(),index=index,seed=seed,arms=4,privileged_diagnostic=True))
    emit(index,seed,'evaluated_complete');del scorer,scored,gt,target,original,baseline,novel;gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='0');p.add_argument('--indices',default='5,6,10,11');p.add_argument('--seeds',default='42,43');a=p.parse_args();setup(a.gpu);register('evaluate')
    import nvdiffrast.torch as dr
    device=torch.device('cuda:0');context=dr.RasterizeCudaContext(device=device)
    for index in map(int,a.indices.split(',')):
        for seed in map(int,a.seeds.split(',')):one(index,seed,context,device)


if __name__=='__main__':main()
