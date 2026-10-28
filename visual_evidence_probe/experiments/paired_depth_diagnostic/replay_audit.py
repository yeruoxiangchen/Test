"""Same-process repeated free-editor control for numerical trajectory variation."""
import gc
import os
import numpy as np
import torch
from .config import OUT,ROUND7,folder,read,write,setup,now,register
from .data import base_record,target
from .optimize import run
from visual_evidence_probe.experiments.dense_evidence.data import observations,load_latent,fit_transform
from visual_evidence_probe.experiments.dense_evidence.evaluation import load_target,render_depth
from visual_evidence_probe.experiments.constrained_evidence.objective import terms
from visual_evidence_probe.experiments.editor_support.data import load_decoder
from visual_evidence_probe.experiments.editor_support.objective import freeze_target
from visual_evidence_probe.experiments.editor_support.optimize import decode
from visual_evidence_probe.experiments.editor_support.forward_policy import FixedForwardNative
from visual_evidence_probe.experiments.geometry_evidence.runner import load_obj
from visual_evidence_probe.experiments.geometry_evidence.metrics import Evaluator


def check(gpu='0'):
    path=OUT/'numerical_replay_audit.json'
    if path.exists():return read(path)
    setup(gpu);register('numerical_replay_audit')
    from visual_evidence_probe.runtime import setup as runtime_setup
    import nvdiffrast.torch as dr
    device=runtime_setup();context=dr.RasterizeCudaContext(device=device);decoder,_=load_decoder(device)
    index,seed=5,42;root=folder(index,seed);rec,k,w,images,masks=observations(index,device)
    z=load_latent(index,seed,device);fit=fit_transform(index,seed,device);mesh=load_obj(root/'original.obj')
    with FixedForwardNative():
        with torch.no_grad():
            _,_,_,pred=decode(decoder,z,fit,k,w,context);baseline=freeze_target(pred)
            record=base_record(index,seed,baseline,rec,k,w,images,masks)
            estimate=target(index,seed,'estimated',baseline,record)
            normalization=terms(baseline,estimate,baseline,record)['loss'].clamp_min(1.e-4).detach()
        for arm in ('numerical_repeat_1','numerical_repeat_2'):
            run(decoder,z,fit,k,w,context,baseline,record,estimate,normalization,mesh,index,seed,arm)
    del decoder,z,estimate,record,pred;gc.collect();torch.cuda.empty_cache()
    camera=read(ROUND7/'data/005/novel_cameras.json')
    nk=torch.tensor(camera['intrinsics'],device=device,dtype=torch.float32);nw=torch.tensor(camera['extrinsics'],device=device,dtype=torch.float32)
    gt=load_target(index);cache=torch.load(ROUND7/'data/005/evaluation_targets.pt',map_location=device,weights_only=False)
    baseline=render_depth(mesh,k,w,context);novel=render_depth(mesh,nk,nw,context)
    with np.load(root/'region_sealed.npz') as data:region=torch.tensor(data['region'],device=device)
    with np.load(root/'novel_region.npz') as data:novel_region=torch.tensor(data['region'],device=device)
    scorer=Evaluator(gt,mesh,baseline,cache['input'],region,novel,cache['novel'],novel_region,k,w,nk,nw,context)
    results={'original_run':read(root/'estimated_free/result.json')['metrics']['proposal']};latents={}
    for label,arm in (('original_run','estimated_free'),('repeat_1','numerical_repeat_1'),('repeat_2','numerical_repeat_2')):
        latents[label]=torch.load(root/arm/'step_080_latent.pt',map_location='cpu',weights_only=False)
        if label!='original_run':results[label]=scorer(load_obj(root/arm/'proposal.obj'))
    pairs={}
    for left,right in (('repeat_1','repeat_2'),('original_run','repeat_1'),('original_run','repeat_2')):
        a,b=latents[left],latents[right];diff=a['features']-b['features']
        pairs[f'{left}_versus_{right}']=dict(coords_exact=torch.equal(a['coords'],b['coords']),
            features_bitwise_equal=torch.equal(a['features'],b['features']),max_feature_difference=float(diff.abs().max()),
            feature_rms_difference=float(diff.square().mean().sqrt()),
            absolute_cd_difference=abs(results[left]['surface']['chamfer_l1']-results[right]['surface']['chamfer_l1']),
            absolute_novel_local_depth_difference=abs(results[left]['novel']['local_target_depth_mae']-results[right]['novel']['local_target_depth_mae']))
    value=dict(time=now(),index=index,seed=seed,complete=True,privileged_diagnostic=True,
        same_process_repeats=True,hyperparameters_unchanged=True,roles='2 additional identical estimated_free controls; not new method arms',
        execution=dict(repeats_cuda_launch_blocking=os.environ.get('CUDA_LAUNCH_BLOCKING','0'),
            original_cuda_launch_blocking=read(root/'estimated_free/complete.json').get('cuda_launch_blocking','not recorded for original smoke'),
            note='repeat1/2 share execution mode; comparison to original also includes execution-mode/run differences'),
        results=results,comparisons=pairs,
        limitation='one case only; measures numerical/run variability, not a universal noise bound')
    write(path,value);del scorer,cache,gt;gc.collect();torch.cuda.empty_cache();return value
