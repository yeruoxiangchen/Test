import argparse
import gc
import traceback
import numpy as np
import torch
from .config import OUT,PARAMS,ARMS,folder,read,write,now,emit,setup,register
from .data import base_record,target
from .optimize import run
from visual_evidence_probe.experiments.dense_evidence.data import observations,load_latent,fit_transform
from visual_evidence_probe.experiments.constrained_evidence.objective import terms
from visual_evidence_probe.experiments.editor_support.data import load_decoder
from visual_evidence_probe.experiments.editor_support.objective import freeze_target
from visual_evidence_probe.experiments.editor_support.optimize import decode
from visual_evidence_probe.experiments.geometry_evidence.runner import load_obj


def one(decoder,index,seed,context,device):
    root=folder(index,seed)
    if (root/'optimization_complete.json').exists():return
    rec,k,w,images,masks=observations(index,device);z=load_latent(index,seed,device);fit=fit_transform(index,seed,device)
    original=load_obj(root/'original.obj')
    with torch.no_grad():
        _,faces,world,pred=decode(decoder,z,fit,k,w,context);baseline=freeze_target(pred)
        error=float(np.abs(world.cpu().numpy()-original.vertices).max())
        if error>4.e-7 or not np.array_equal(faces.cpu().numpy(),original.faces):raise ValueError('decoder parity changed')
        record=base_record(index,seed,baseline,rec,k,w,images,masks)
        estimate=target(index,seed,'estimated',baseline,record)
        normalization=terms(baseline,estimate,baseline,record)['loss'].clamp_min(1.e-4).detach()
        write(root/'decoder_parity.json',dict(passed=True,max_vertex_error=error))
    del pred,world,faces
    for arm in ARMS:
        source='oracle' if arm.startswith('oracle_') else 'estimated'
        current_target=estimate if source=='estimated' else target(index,seed,source,baseline,record)
        run(decoder,z,fit,k,w,context,baseline,record,current_target,normalization,original,index,seed,arm)
        del current_target;gc.collect();torch.cuda.empty_cache()
    write(root/'optimization_complete.json',dict(time=now(),index=index,seed=seed,arms=4,
        privileged_diagnostic=True,same_support_all_arms=True,shared_normalization=float(normalization),
        novel_gt_loaded=False))
    emit(index,seed,'optimization_complete');del original,baseline,z,estimate,record;gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='0');p.add_argument('--indices',default='5,6,10,11');p.add_argument('--seeds',default='42,43');a=p.parse_args();setup(a.gpu);register('run')
    from visual_evidence_probe.runtime import setup as runtime_setup
    from visual_evidence_probe.experiments.editor_support.forward_policy import FixedForwardNative
    import nvdiffrast.torch as dr
    device=runtime_setup();decoder,binding=load_decoder(device);write(OUT/'decoder.json',binding);context=dr.RasterizeCudaContext(device=device)
    with FixedForwardNative() as policy:
        for index in map(int,a.indices.split(',')):
            for seed in map(int,a.seeds.split(',')):
                try:one(decoder,index,seed,context,device)
                except BaseException as error:
                    write(folder(index,seed)/f'failure_{now().replace(":","-")}.json',dict(error=str(error),traceback=traceback.format_exc()));raise
        write(OUT/'kernel_policy.json',policy.manifest())


if __name__=='__main__':main()
