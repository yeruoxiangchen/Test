import argparse
import gc
import time
import traceback

import numpy as np
import torch

from .config import OUT,PARAMS,folder,registry,setup,read,write,emit,register,now
from .data import observations,load_latent,fit_transform
from .evidence import build
from .optimize import run_source,link
from visual_evidence_probe.experiments.editor_support.data import load_decoder
from visual_evidence_probe.experiments.editor_support.optimize import decode,as_mesh
from visual_evidence_probe.experiments.editor_support.objective import freeze_target
from visual_evidence_probe.experiments.geometry_evidence.runner import load_obj


def one(decoder,index,seed,context,device):
    root=folder(index,seed)
    if (root/'input_complete.json').exists():return
    start=time.monotonic()
    rec,k,w,images,masks=observations(index,device)
    z=load_latent(index,seed,device);fit=fit_transform(index,seed,device)
    with torch.no_grad():
        vertices,faces,world,pred=decode(decoder,z,fit,k,w,context)
        baseline=freeze_target(pred)
        original=load_obj(root/'original.obj')
        same_shape=tuple(original.vertices.shape)==tuple(world.shape) and tuple(original.faces.shape)==tuple(faces.shape)
        same_faces=same_shape and np.array_equal(original.faces,faces.cpu().numpy())
        error=float(np.max(np.abs(original.vertices-world.cpu().numpy()))) if same_shape else None
        passed=bool(same_faces and error<4.e-7)
        write(root/'decoder_parity.json',dict(passed=passed,same_faces=bool(same_faces),max_vertex_error=error))
        if not passed:raise ValueError(f'original decoder parity failed {index}/{seed}: {error}')
        record=build(index,seed,baseline,rec,masks)
        record.update(images=images,masks=masks)
    del vertices,faces,world,pred
    sources={}
    for name in ('sparse','dense'):
        local=record['region']>.5
        sparse,dense=record['targets']['sparse'],record['targets']['dense']
        same_target=torch.equal(sparse['depth_valid']&local,dense['depth_valid']&local)
        same_target &= torch.equal(sparse['depth']*(sparse['depth_valid']&local),dense['depth']*(dense['depth_valid']&local))
        if name=='dense' and same_target and 'sparse' in sources:
            path=root/name;path.mkdir(exist_ok=True)
            for source_path in (root/'sparse').iterdir():
                if source_path.name!='complete.json' and source_path.is_file():link(path/source_path.name,source_path)
            result=dict(sources['sparse'],source='dense',reused_sparse_trajectory=True,
                elapsed_seconds=0.,effective_optimization_seconds=sources['sparse']['elapsed_seconds'])
            write(path/'complete.json',result)
        else:
            result=run_source(decoder,z,fit,k,w,context,baseline,record,name,original,index,seed)
        sources[name]=result
        for policy,key in (('legacy','legacy_step'),('observed','observed_step')):
            arm=root/f'{name}_{policy}';arm.mkdir(exist_ok=True)
            step=result['decisions'][key]
            for filename,source in (('accepted.obj',root/name/f'step_{step:03d}.obj'),
                    ('accepted_latent.pt',root/name/f'step_{step:03d}_latent.pt'),
                    ('candidate.obj',root/name/'train_selected.obj'),('candidate_latent.pt',root/name/'train_selected_latent.pt'),
                    ('proposal.obj',root/name/'proposal.obj')):link(arm/filename,source)
            write(arm/'selection.json',dict(index=index,seed=seed,uid=rec['uid'],split=rec['split'],kind=rec['kind'],
                source=name,policy=policy,selected_step=step,accepted_edit=step>0,
                candidate_step=result['decisions']['own_training_step'],input_only=True,
                same_trajectory_for_policies=True,query_count=result['query_count']))
    write(root/'input_complete.json',dict(time=now(),index=index,seed=seed,uid=rec['uid'],split=rec['split'],
        gt_used=False,sources=list(sources),arms=4,elapsed_seconds=time.monotonic()-start))
    emit(index,seed,'input_complete',seconds=time.monotonic()-start)


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='0');p.add_argument('--indices',default='4,5,6,7,8,9,10,11');p.add_argument('--seeds',default='42,43');a=p.parse_args()
    setup(a.gpu)
    from visual_evidence_probe.runtime import setup as runtime_setup
    from visual_evidence_probe.experiments.editor_support.forward_policy import FixedForwardNative
    import nvdiffrast.torch as dr
    device=runtime_setup();register('run');context=dr.RasterizeCudaContext(device=device)
    decoder,binding=load_decoder(device);write(OUT/f'decoder_{a.gpu}.json',binding)
    with FixedForwardNative() as policy:
        for index in map(int,a.indices.split(',')):
            for seed in map(int,a.seeds.split(',')):
                try:one(decoder,index,seed,context,device)
                except Exception as error:
                    write(folder(index,seed)/f'failure_{now().replace(":","-")}.json',dict(error=str(error),traceback=traceback.format_exc()))
                    raise
                gc.collect();torch.cuda.empty_cache()
        write(OUT/f'editor_policy_{a.gpu}.json',policy.manifest())


if __name__=='__main__':main()
