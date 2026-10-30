"""Standalone inference contract test, with dataset/GT file reads denied."""
import argparse
import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from . import common as c


def worker(output,checkpoint,ss_checkpoint):
    import torch
    from .data import RepairDataset,intervention
    from .vendor_pose.runtime import denormalize_shape_latent
    out=Path(output).resolve()
    if not out.is_relative_to(c.ROOT/'outputs'):
        raise ValueError('Inference test output must remain under project storage')
    out.mkdir(parents=True,exist_ok=True)
    ds=RepairDataset(c.DATA/'overfit.json','shape')
    sample=ds[0]
    pair=intervention(sample,'shape',0,'cuda')
    shutil.copyfile(sample['row']['pose_file'],out/'pose.pt')
    np.savez_compressed(out/'reference_ss.npz',z=sample['pair_pack']['pairs'][0]['bad_ss'].numpy())
    keep=pair['present']
    raw=denormalize_shape_latent(pair['bad'])
    np.savez_compressed(out/'reference_shape.npz',coords=raw.coords[keep,1:].cpu().numpy(),
        feats=raw.feats[keep].float().cpu().numpy())
    c.write(out/'roi.json',pair['bounds64'].cpu().tolist())
    del ds,sample,pair,raw
    gc.collect()
    torch.cuda.empty_cache()
    args=['--pose',out/'pose.pt','--ss-latent',out/'reference_ss.npz',
          '--shape-latent',out/'reference_shape.npz','--roi',out/'roi.json',
          '--ss-checkpoint',ss_checkpoint,'--shape-checkpoint',checkpoint,'--output',out/'result']
    subprocess.run([c.PYTHON,'-B','-m','pose_conditioned_geometry_repair.inference_smoke',
        '--child',*map(str,args)],env=c.environment(),cwd=c.REPO,check=True)
    from .vendor_pose.data import load_shape_target,load_ss_target
    coords,feats=load_shape_target(out/'result/shape_repaired.npz','inference-smoke')
    load_ss_target(out/'result/ss_repaired.npz','inference-smoke')
    result=c.read(out/'result/inference.json')
    assert result['outside_features_max_abs']==0 and result['outside_occupancy_changes']==0
    assert result['gt_accessed'] is False
    assert len(coords)==len(feats) and (out/'result/repaired.obj').stat().st_size>0
    c.write(out/'verified.json',{'time':c.now(),'passed':True,'native_npz_readback':True,
        'dataset_and_gt_reads_denied_in_inference_child':True,'shape_tokens':len(coords),
        'result':result,'scope':'Runtime contract test on a synthetic fixture; not a reconstruction score'})


def child(args):
    from . import infer
    forbidden=[c.DATA.resolve(),c.SOURCE.resolve(),(c.ROOT/'outputs/round1_20261029').resolve()]
    # Checkpoints in V1 are allowed; evaluation meshes/datasets are not.
    allowed=[Path(args[args.index('--ss-checkpoint')+1]).resolve(),
             Path(args[args.index('--shape-checkpoint')+1]).resolve()]
    def hook(event,values):
        if event=='open' and values and isinstance(values[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(values[0])).resolve()
            if any(p.is_relative_to(root) for root in forbidden) and p not in allowed:
                raise RuntimeError('Inference attempted a dataset/GT read: '+str(p))
    sys.addaudithook(hook)
    sys.argv=['infer',*args]
    infer.main()


def main():
    if '--child' in sys.argv[1:]:
        child(sys.argv[sys.argv.index('--child')+1:])
        return
    p=argparse.ArgumentParser()
    p.add_argument('--worker',action='store_true')
    p.add_argument('--output',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--ss-checkpoint',required=True)
    args=p.parse_args()
    if args.worker:
        worker(args.output,args.checkpoint,args.ss_checkpoint)
    else:
        from . import runner
        c.OUT=c.ROOT/'outputs/bridge_v2_20261030'
        runner.command('standalone_inference_smoke','inference_smoke',[
            '--worker','--output',args.output,'--checkpoint',args.checkpoint,
            '--ss-checkpoint',args.ss_checkpoint],gpu=True,required=[Path(args.output)/'verified.json'])


if __name__=='__main__':
    main()
