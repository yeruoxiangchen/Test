"""Whole fixed new-object cohort: input-bbox raster-resolution control."""
import argparse
from pathlib import Path
from unittest.mock import patch

from . import localization_corrected as corrected

import numpy as np
import torch
from PIL import Image

from visual_evidence_probe.experiments.localization import config,runner,geometry,visuals
from visual_evidence_probe.experiments.localization.metrics import evaluate
from visual_evidence_probe.experiments.localization.report import aggregate,paired,full_summary
from visual_evidence_probe.common import binding,read,write,verify,lock_json,now

BASE=config.OUT
OUT=BASE/'input_crop_resolution_control'


def folder(index):return OUT/'objects'/f'{index:03d}'


def observation(index):
    r=read(folder(index)/'observations.json')
    images=[Image.open(verify(b)).convert('RGB') for b in r['images']]
    masks=np.stack([np.asarray(Image.open(verify(b)).convert('L'))>127 for b in r['masks']])
    return r,torch.tensor(r['intrinsics'],dtype=torch.float32),torch.tensor(r['extrinsics'],dtype=torch.float32),images,masks


def prepare():
    from ar_ss_flow.shared_object_preprocessing import prepare_shared_object_arrays
    p=dict(format='localization_input_crop_control.v1',source=binding(__file__),
        base_protocol=binding(BASE/'protocol.json'),buffer_fix=binding(BASE/'buffer_amendment.json'),
        scope='post-hoc resolution control on all eight fixed new objects; not selected by AP',
        indices=list(range(4,12)),params=config.PARAMS,
        reason='full-image128 with two-pixel erosion leaves some thin objects almost no interior',
        change='raster and score on existing input-only crop cameras; same four RGB inputs, original DINO cache, meshes, thresholds',
        no_inference_or_fitting_rerun=True,no_gt_crop=True)
    lock_json(OUT/'protocol.json',p)
    for index in range(4,12):
        rec,k,w,images,masks=config.observation(index)
        prepared=prepare_shared_object_arrays([np.asarray(im) for im in images],[m.astype(np.uint8)*255 for m in masks],
            resolution=518,foreground_margin=1.1,alpha_threshold=.8)
        cached=torch.load(BASE/'objects'/f'{index:03d}'/'features.pt',map_location='cpu',weights_only=False)
        ck=torch.from_numpy(prepared.source_to_feature_affines)@k
        if not torch.equal(ck,cached['intrinsics']):raise ValueError('crop camera differs from frozen model inputs')
        folder(index).mkdir(parents=True,exist_ok=True)
        ib=[];mb=[]
        for j,(im,mask) in enumerate(zip(prepared.images,prepared.masks)):
            ip=folder(index)/f'{j:02d}.png';mp=folder(index)/f'{j:02d}_mask.png'
            im.save(ip);Image.fromarray((mask>.8).astype(np.uint8)*255).save(mp)
            ib.append(binding(ip));mb.append(binding(mp))
        for name in ('features.pt','features.json'):
            dest=folder(index)/name
            if not dest.exists():dest.symlink_to((BASE/'objects'/f'{index:03d}'/name).resolve())
        lock_json(folder(index)/'observations.json',{**rec,'images':ib,'masks':mb,'intrinsics':ck.tolist(),
            'crop':prepared.geometry_record(),'source':binding(BASE/'objects'/f'{index:03d}'/'observations.json')})


def run():
    import trimesh
    import nvdiffrast.torch as dr
    from pose_aligned_layered_reconstruction.paired_encoder import build_frozen_dino_encoder
    from visual_evidence_probe.runtime import setup
    from visual_evidence_probe.render import render,mask_iou
    config.check();prepare();device=setup();context=dr.RasterizeCudaContext(device=device)
    pipeline=build_frozen_dino_encoder(device=device)
    sealed=[]
    with patch.object(runner,'folder',folder),patch.object(runner,'observation',observation),patch.object(geometry,'buffers',corrected.buffers):
        for i in range(4,12):
            for seed in ('42','43'):
                mesh=read(BASE/'objects'/f'{i:03d}'/f'fitted/{seed}.json')['mesh']
                path=runner.make_scores(i,seed,mesh,pipeline,context)
                sealed.append(binding(path.with_suffix('.json')))
    lock_json(OUT/'sealed_input_scores.json',dict(records=sealed,uses_gt=False))
    records=[]
    with patch.object(visuals,'folder',folder):
        for i in range(4,12):
            rec,k,w,images,masks=observation(i)
            gt=runner.target_mesh(i)
            _,gm=render(gt,k,w,context=context)
            iou=mask_iou(gm,masks)
            write(folder(i)/'camera_audit.json',dict(ious=iou.tolist(),passed=bool((iou>=.9).all())))
            if (iou<.9).any():raise ValueError(f'crop camera failure {i}: {iou}')
            b=corrected.buffers(gt,k,w,context,128)
            depth=b['depth'].cpu().numpy();mask=b['mask'].cpu().numpy()
            for seed in ('42','43'):
                path=folder(i)/f'scores/{seed}.npz'
                with np.load(path) as a:maps={k:a[k] for k in a.files}
                values={str(t):evaluate(maps,maps['pred_depth'],depth,mask,t) for t in config.PARAMS['depth_thresholds']}
                record=dict(index=i,uid=rec['uid'],split='omni_test',key=seed,success=True,control=False,metrics=values)
                write(folder(i)/f'metrics/{seed}.json',record);records.append(record)
                error=(maps['pred_mask']^mask)|((maps['pred_mask']&mask)&(np.abs(maps['pred_depth']-depth)>.02))
                meta=read(path.with_suffix('.json'));mesh=trimesh.load(verify(meta['mesh']),force='mesh',process=False)
                visuals.visualize(i,seed,maps,depth,mask,error,images,mesh,k,w,context)
                print(dict(time=now(),stage='crop_evaluated',index=i,seed=seed,coverage=values['0.02']['common_coverage']),flush=True)
    summary=aggregate(records)
    comparison={f'{c}_minus_{b}':paired(summary,c,b) for c in ('direct_dino','cross_dino','fused_dino') for b in ('boundary','shuffled_dino')}
    write(OUT/'summary.json',dict(time=now(),records=16,summaries=summary,full_visible_summary=full_summary(records),
        paired_comparisons=comparison,all_successful=True,posthoc_control=True,parameters_unchanged=True))
    print(OUT/'summary.json',flush=True)


def main():
    import os
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='0');a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    run()


if __name__=='__main__':main()
