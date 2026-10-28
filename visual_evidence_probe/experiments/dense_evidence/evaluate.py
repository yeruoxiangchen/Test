"""Score only sealed input-only decisions, including every fallback."""
import argparse
import gc
from pathlib import Path

import numpy as np
import torch

from .config import OUT,PARAMS,ARMS,folder,data_folder,registry,setup,read,write,emit,register,now
from .data import observations
from .evaluation import load_target,render_depth
from .gates import component_summary
from visual_evidence_probe.experiments.geometry_evidence.targets import novel_cameras
from visual_evidence_probe.experiments.geometry_evidence.metrics import Evaluator,face_region
from visual_evidence_probe.experiments.geometry_evidence.runner import load_obj


@torch.no_grad()
def one(index,seeds,context,device):
    if not all((folder(index,seed)/'input_complete.json').exists() for seed in seeds):
        raise ValueError(f'input-only decisions incomplete for {index}')
    rec,k,w,_,_=observations(index,device)
    target=load_target(index)
    nk_np,nw_np,camera_receipt=novel_cameras(k.cpu().numpy(),w.cpu().numpy(),8)
    nk=torch.tensor(nk_np,device=device);nw=torch.tensor(nw_np,device=device)
    write(data_folder(index)/'novel_cameras.json',dict(intrinsics=nk_np.tolist(),extrinsics=nw_np.tolist(),**camera_receipt))
    gt=render_depth(target,k,w,context);ngt=render_depth(target,nk,nw,context)
    torch.save(dict(input={key:gt[key].cpu() for key in ('depth','mask','xyz')},
        novel={key:ngt[key].cpu() for key in ('depth','mask','xyz')}),data_folder(index)/'evaluation_targets.pt')
    for seed in seeds:
        root=folder(index,seed)
        if (root/'complete.json').exists():continue
        original=load_obj(root/'original.obj')
        baseline=render_depth(original,k,w,context);novel=render_depth(original,nk,nw,context)
        with np.load(root/'region_sealed.npz') as a:
            region=torch.tensor(a['region'],device=device);selected=torch.tensor(a['selected_faces'],device=device)
        novel_region=face_region(novel,selected)
        np.savez_compressed(root/'novel_region.npz',region=novel_region.cpu().numpy())
        scorer=Evaluator(target,original,baseline,gt,region,novel,ngt,novel_region,k,w,nk,nw,context)
        structure=component_summary(original.vertices,original.faces)
        base_metrics=scorer(original,original=True,structure=structure)
        write(root/'baseline_metrics.json',base_metrics)
        cache={(root/'original.obj').resolve():base_metrics}
        for arm_name in ARMS:
            arm=root/arm_name;selection=read(arm/'selection.json')
            source=read(root/selection['source']/'complete.json')
            source_path=root/selection['source']
            steps={r['step']:r['structure'] for r in source['snapshots']}
            scores={}
            for endpoint,filename,step in (
                ('accepted','accepted.obj',selection['selected_step']),
                ('candidate','candidate.obj',selection['candidate_step']),
                ('proposal','proposal.obj',PARAMS['steps'] if source['steps'] else 0)):
                path=(arm/filename).resolve()
                if path not in cache:
                    mesh=load_obj(path)
                    cache[path]=scorer(mesh,structure=steps.get(step))
                    del mesh
                scores[endpoint]=cache[path]
            result=dict(**selection,status=source['status'],original_metrics=base_metrics,metrics=scores,
                source_elapsed_seconds=source['elapsed_seconds'],peak_allocated_bytes=source['peak_allocated_bytes'],
                reused_sparse_trajectory=source.get('reused_sparse_trajectory',False),
                original_frame_preserved=True,gt_used_only_after_selection=True,
                selected_checkpoint_diagnostic=source['decisions'])
            write(arm/'result.json',result)
        write(root/'complete.json',dict(time=now(),index=index,seed=seed,uid=rec['uid'],split=rec['split'],
            arms=4,no_new_alignment=True,gt_used_for_evaluation_only=True))
        emit(index,seed,'evaluated_complete')
        del scorer,cache,original,baseline,novel;gc.collect();torch.cuda.empty_cache()
    del target,gt,ngt;gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='0');p.add_argument('--indices',default='4,5,6,7,8,9,10,11');p.add_argument('--seeds',default='42,43');a=p.parse_args()
    setup(a.gpu)
    from visual_evidence_probe.runtime import setup as runtime_setup
    import nvdiffrast.torch as dr
    device=runtime_setup();register('evaluate');context=dr.RasterizeCudaContext(device=device)
    for index in map(int,a.indices.split(',')):one(index,list(map(int,a.seeds.split(','))),context,device)


if __name__=='__main__':main()
