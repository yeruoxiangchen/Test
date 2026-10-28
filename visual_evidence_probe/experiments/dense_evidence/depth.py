"""Frozen posed DA3 inference; the heldout image never enters this module."""
import argparse
import gc
import time
from pathlib import Path
from types import MethodType

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .config import OUT,PARAMS,data_folder,folder,setup,registry,read,write,digest,register
from .camera import remap_depth,consistency
from .da3_setup import MODEL


def predict(model,images,k,w):
    prediction=model.inference(images,extrinsics=w.copy(),intrinsics=k.copy(),
        align_to_input_ext_scale=True,process_res=PARAMS['depth_resolution'],
        process_res_method='upper_bound_resize',infer_gs=False,export_dir=None)
    pose_error=float(np.max(np.abs(prediction.extrinsics-w[:,:3])))
    if pose_error>1.e-5:raise ValueError('DA3 returned poses do not preserve provided input frame')
    depth,confidence,domain=remap_depth(prediction.depth,prediction.conf,prediction.intrinsics,k,PARAMS['side'])
    if not bool(torch.isfinite(depth).all()) or not bool((depth>0).any()):raise ValueError('invalid DA3 depth')
    return prediction,depth,confidence,domain,dict(input_pose_max_error=pose_error,
        processed_shape=list(prediction.depth.shape),processed_intrinsics=prediction.intrinsics.tolist(),
        provider_alignment=model._last_input_alignment)


def install_alignment_receipt(model):
    original=model._align_to_input_extrinsics_intrinsics
    def capture(self,extrinsics,intrinsics,prediction,align_to_input_ext_scale=True,**kwargs):
        before=prediction.depth.copy();raw_poses=prediction.extrinsics.copy()
        output=original(extrinsics,intrinsics,prediction,align_to_input_ext_scale,**kwargs)
        ratio=np.median(before[np.isfinite(before)&(output.depth>0)]/output.depth[np.isfinite(before)&(output.depth>0)])
        self._last_input_alignment=dict(raw_predicted_poses=raw_poses.tolist(),
            raw_to_output_depth_divisor=float(ratio),uses_gt_mesh=False,
            source='official API camera-trajectory alignment to provided input poses')
        return output
    model._align_to_input_extrinsics_intrinsics=MethodType(capture,model)


def run_index(model,index,seeds,scale_check):
    rec=read(data_folder(index)/'observations.json')
    for seed in seeds:
        root=folder(index,seed)
        if not (root/'region_complete.json').exists():raise FileNotFoundError(f'region not ready: {index}/{seed}')
        meta=read(root/'region_complete.json');train=meta['train_views'];heldout=meta['heldout_views']
        if set(train)&set(heldout):raise ValueError('holdout leaked into DA3 input')
        key='train_'+'_'.join(map(str,train))
        output=OUT/'depths'/f'{index:03d}'/key
        if (output/'complete.json').exists():
            write(root/'dense_provider.json',dict(path=str(output),train_views=train,heldout_views=heldout,uses_gt=False))
            continue
        output.mkdir(parents=True,exist_ok=True)
        provider_images=rec.get('raw_images') if rec['kind']=='real_scene' else rec['images']
        images=[Image.open(provider_images[i]['path']).convert('RGB') for i in train]
        masks=np.stack([np.asarray(Image.open(rec['masks'][i]['path']).convert('L'))>127 for i in train])
        k=np.asarray(rec['intrinsics'],np.float32)[train];w=np.asarray(rec['extrinsics'],np.float32)[train]
        torch.manual_seed(PARAMS['random_seed']+index)
        torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        prediction,depth,confidence,domain,receipt=predict(model,images,k,w)
        observed=F.interpolate(torch.tensor(masks,dtype=torch.float32)[:,None],size=depth.shape[-2:],mode='nearest')[:,0]>.5
        valid,support,filtering=consistency(depth,confidence,observed,torch.from_numpy(k),torch.from_numpy(w),
            quantile=PARAMS['dense_confidence_quantile'],pixel_tolerance=PARAMS['dense_reprojection_pixels'],
            absolute_z=PARAMS['dense_absolute_z_tolerance'],relative_z=PARAMS['dense_relative_z_tolerance'],
            min_references=PARAMS['dense_min_references'])
        valid &= domain
        np.savez_compressed(output/'prediction.npz',depth=depth.numpy(),confidence=confidence.numpy(),
            valid=valid.numpy(),support=support.numpy(),processed_depth=prediction.depth,
            processed_confidence=prediction.conf,processed_intrinsics=prediction.intrinsics,
            intrinsics=k,extrinsics=w,train_views=np.array(train))
        if scale_check:
            scaled=w.copy();scaled[:,:3,3]*=2
            _,double_depth,_,_,_=predict(model,images,k,scaled)
            relative=(double_depth-2*depth).abs()/(2*depth).clamp_min(1.e-6)
            error=float(relative[observed&domain].mean())
            write(output/'scale_equivariance.json',dict(multiplier=2.,relative_error=error,
                passed=error<.01,no_gt_used=True))
            if error>=.01:raise ValueError(f'camera-scale transport failed: {error}')
            scale_check=False
        value=dict(index=index,uid=rec['uid'],train_views=train,heldout_views=heldout,
            input_image_paths=[provider_images[i]['path'] for i in train],
            input_image_sha256=[digest(provider_images[i]['path']) for i in train],
            provider='DA3-LARGE-1.1',weights=read(OUT/'da3_weights.json'),uses_gt=False,
            input_background='raw_scene' if rec['kind']=='real_scene' else 'same_cropped_black_foreground',
            filtering=filtering,camera=receipt,elapsed_seconds=time.monotonic()-start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),prediction_sha256=digest(output/'prediction.npz'))
        write(output/'complete.json',value)
        write(root/'dense_provider.json',dict(path=str(output),train_views=train,heldout_views=heldout,uses_gt=False))
        print(dict(stage='dense_prediction_complete',index=index,seed=seed,train=train,
            valid_pixels=int(valid.sum()),seconds=value['elapsed_seconds']),flush=True)
        del prediction,depth,confidence,valid,support;gc.collect();torch.cuda.empty_cache()
    return scale_check


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='0');p.add_argument('--indices',default='4,5,6,7,8,9,10,11');p.add_argument('--seeds',default='42,43');p.add_argument('--scale-check',action='store_true');a=p.parse_args()
    setup(a.gpu);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from depth_anything_3.api import DepthAnything3
    register('depth')
    model=DepthAnything3.from_pretrained(str(MODEL)).eval().requires_grad_(False).to('cuda:0')
    install_alignment_receipt(model)
    checking=a.scale_check
    for index in map(int,a.indices.split(',')):
        checking=run_index(model,index,list(map(int,a.seeds.split(','))),checking)


if __name__=='__main__':main()
