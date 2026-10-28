"""Stress the unchanged deterministic decoder with the largest Dev64 GT."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from pose_aligned_layered_reconstruction.frozen_slat import load_frozen_slat30k
from pose_aligned_layered_reconstruction.prepare_official_experiment import read_json, atomic_json


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output_dir',required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    device=torch.device('cuda:0')
    torch.cuda.set_device(device)
    root=Path(__file__).resolve().parents[1]
    row=read_json(root/'outputs/round1_20260909/data_v2/dev64.json')['samples'][15]
    runtime=load_frozen_slat30k(device,evaluation_uids=[row['uid']])
    runtime['model'].cpu()
    torch.cuda.empty_cache()
    from trellis.modules import sparse as sp
    with np.load(row['source']['slat']) as data:
        c=np.asarray(data['coords'],np.int32)
        feats=torch.from_numpy(np.asarray(data['feats'],np.float32)).to(device)
    coords=torch.from_numpy(np.concatenate((np.zeros((len(c),1),np.int32),c),axis=1)).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    start=time.time()
    mesh=runtime['decoder'](sp.SparseTensor(coords=coords,feats=feats))[0].to_trimesh(transform_pose=False)
    report=dict(passed=bool(len(mesh.faces) and np.isfinite(mesh.vertices).all()), index=15, uid=row['uid'],
        input_points=len(c), vertices=len(mesh.vertices), faces=len(mesh.faces), seconds=time.time()-start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device), mode='unchanged_deterministic_decoder_flow_CPU_offload',
        binding=runtime['binding'])
    atomic_json(Path(args.output_dir)/'report.json',report)
    mesh.export(Path(args.output_dir)/'target_00015.ply')
    print(report,flush=True)


if __name__=='__main__':
    main()
