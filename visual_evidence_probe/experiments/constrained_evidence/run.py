import argparse
import gc
import traceback
import numpy as np
import torch

from .config import OUT,PRIOR,PARAMS,folder,read,write,setup,register,now,emit
from .data import build
from .optimize import run,reuse_reference
from visual_evidence_probe.experiments.dense_evidence.data import observations,load_latent,fit_transform
from visual_evidence_probe.experiments.editor_support.data import load_decoder
from visual_evidence_probe.experiments.editor_support.objective import freeze_target
from visual_evidence_probe.experiments.editor_support.optimize import decode
from visual_evidence_probe.experiments.geometry_evidence.runner import load_obj


def one(decoder,index,seed,context,device):
    root=folder(index,seed)
    if (root/'input_complete.json').exists():return
    rec,k,w,images,masks=observations(index,device);z=load_latent(index,seed,device);fit=fit_transform(index,seed,device)
    original=load_obj(PRIOR/'objects'/f'{index:03d}'/str(seed)/'original.obj')
    with torch.no_grad():
        _,faces,world,pred=decode(decoder,z,fit,k,w,context);baseline=freeze_target(pred)
        if not np.array_equal(faces.cpu().numpy(),original.faces):raise ValueError('original decoder faces mismatch')
        error=float(np.abs(world.cpu().numpy()-original.vertices).max())
        if error>4.e-7:raise ValueError('original decoder vertex mismatch')
        record=build(index,seed,baseline,rec,k,w,images,masks)
        write(root/'decoder_parity.json',dict(passed=True,max_vertex_error=error))
    del pred,world,faces
    reuse_reference(index,seed)
    for arm in ('anchored','anchored_reliable'):run(decoder,z,fit,k,w,context,baseline,record,original,index,seed,arm)
    write(root/'input_complete.json',dict(time=now(),index=index,seed=seed,uses_gt=False,arms=3))
    emit(index,seed,'input_complete')
    del original,record,baseline,z;gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='1');p.add_argument('--indices',default='4,5,6,7,8,9,10,11');p.add_argument('--seeds',default='42,43');a=p.parse_args();setup(a.gpu);register('run')
    from visual_evidence_probe.runtime import setup as runtime_setup
    from visual_evidence_probe.experiments.editor_support.forward_policy import FixedForwardNative
    import nvdiffrast.torch as dr
    device=runtime_setup();decoder,binding=load_decoder(device);write(OUT/'decoder.json',binding)
    context=dr.RasterizeCudaContext(device=device)
    with FixedForwardNative() as policy:
        for index in map(int,a.indices.split(',')):
            for seed in map(int,a.seeds.split(',')):
                try:one(decoder,index,seed,context,device)
                except BaseException as error:
                    write(folder(index,seed)/f'failure_{now().replace(":","-")}.json',dict(error=str(error),traceback=traceback.format_exc()));raise
        write(OUT/'kernel_policy.json',policy.manifest())


if __name__=='__main__':main()
