import argparse
import gc
import os
import time
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from visual_evidence_probe import runtime
from visual_evidence_probe.render import render, mask_iou
from .config import OUT, PARAMS, SEEDS, folder, prepare, check, observation, binding, read, write, verify, lock_json, emit, now


def cleanup():
    gc.collect();torch.cuda.empty_cache()


@torch.no_grad()
def encode_inputs(indices,device):
    from ar_ss_flow.shared_object_preprocessing import prepare_shared_object_arrays
    from pose_aligned_layered_reconstruction.paired_encoder import build_frozen_dino_encoder,_encode_patch_tokens
    pipeline=build_frozen_dino_encoder(device=device)
    for i in indices:
        path=folder(i)/'features.pt'
        if path.exists():verify(read(path.with_suffix('.json'))['features']);continue
        rec,k,w,images,masks=observation(i)
        prepared=prepare_shared_object_arrays([np.asarray(im) for im in images],
            [m.astype(np.uint8)*255 for m in masks],resolution=518,foreground_margin=1.1,alpha_threshold=.8)
        ck=torch.from_numpy(prepared.source_to_feature_affines)@k
        tokens=torch.cat([_encode_patch_tokens(pipeline,prepared.images[j:j+2]).half().cpu()
                          for j in range(0,len(images),2)])
        for j,im in enumerate(prepared.images):
            p=folder(i)/'inputs'/f'{j:02d}.png';p.parent.mkdir(parents=True,exist_ok=True);im.save(p)
        torch.save(dict(features=tokens,intrinsics=ck),path)
        write(path.with_suffix('.json'),dict(features=binding(path),crop=prepared.geometry_record(),
              dino_precision='FP32',cache_precision='FP16',uses_gt=False))
        emit(i,'encoded',views=len(images))
    del pipeline;cleanup()


def input_for(index):
    from pose_aligned_layered_reconstruction.dino_context import build_masked_dino_contexts
    rec,k,w,_,_=observation(index)
    p=folder(index)/'features.pt'
    cached=torch.load(verify(read(p.with_suffix('.json'))['features']),map_location='cpu',weights_only=False)
    features=cached['features'].float()
    return dict(intrinsics=cached['intrinsics'],extrinsics=w,image_size=(518,518),grid_transform='identity',
                extrinsics_type='w2c',camera_forward_sign=1.,**build_masked_dino_contexts(features),visual_patch_features=features)


@torch.no_grad()
def generate(indices,device,gpu):
    from pose_aligned_layered_reconstruction.experiments.hssd_operator_audit_20261025.kernel_policy import FixedNative
    new=[i for i in indices if i>=4]
    if not new:return
    with FixedNative() as policy:
        ss=runtime.load_ss(device)
        lock_json(OUT/'runtime'/f'{gpu}_ss.json',ss['binding'])
        for i in new:
            sample=input_for(i)
            for seed in SEEDS:
                path=folder(i)/'generated'/f'{seed}_ss.npz';path.parent.mkdir(parents=True,exist_ok=True)
                if not path.exists():
                    start=time.monotonic();runtime.sample_ss(ss,sample,i,seed,path)
                    write(path.with_suffix('.json'),dict(coords=binding(path),elapsed_seconds=time.monotonic()-start))
                emit(i,'ss_complete',seed=seed)
        del ss;cleanup()
        slat=runtime.load_slat(device)
        lock_json(OUT/'runtime'/f'{gpu}_slat.json',slat['binding'])
        for i in new:
            sample=input_for(i)
            for seed in SEEDS:
                path=folder(i)/'generated'/f'{seed}.obj'
                if path.exists():verify(read(path.with_suffix('.json'))['mesh']);continue
                cp=path.with_name(f'{seed}_ss.npz')
                with np.load(verify(read(cp.with_suffix('.json'))['coords'])) as arrays:coords=arrays['coords']
                start=time.monotonic();_,cfg=runtime.mesh_from_support(slat,sample,coords,i,seed,path)
                write(path.with_suffix('.json'),dict(mesh=binding(path),cfg=cfg,elapsed_seconds=time.monotonic()-start,uses_gt=False))
                emit(i,'mesh_complete',seed=seed);torch.cuda.empty_cache()
        del slat;cleanup()
        write(OUT/'runtime'/f'{gpu}_kernel.json',policy.manifest())


@torch.no_grad()
def make_scores(index,key,mesh_path,pipeline,context):
    import trimesh
    from pose_aligned_layered_reconstruction.paired_encoder import _encode_patch_tokens
    from .geometry import buffers
    from .scores import compute
    path=folder(index)/'scores'/f'{key}.npz'
    if path.exists():verify(read(path.with_suffix('.json'))['maps']);return path
    rec,k,w,images,masks=observation(index)
    p=folder(index)/'features.pt'
    cache=torch.load(verify(read(p.with_suffix('.json'))['features']),map_location='cpu',weights_only=False)
    mesh=trimesh.load(verify(mesh_path),force='mesh',process=False)
    start=time.monotonic()
    cropped,_=render(mesh,cache['intrinsics'],w,context=context)
    rf=torch.cat([_encode_patch_tokens(pipeline,cropped[j:j+2]).half().cpu() for j in range(0,len(images),2)])
    pred=buffers(mesh,k,w,context,PARAMS['side'])
    maps=compute(pred,[np.asarray(im)*m[...,None] for im,m in zip(images,masks)],masks,
                  cache['features'],rf,k,cache['intrinsics'],w,index)
    # Pixel-center / perspective interpolation check without ground truth.
    from .scores import project
    maximum=0.
    axis=(torch.arange(PARAMS['side'],device='cuda:0')+.5)*518/PARAMS['side']-.5
    yy,xx=torch.meshgrid(axis,axis,indexing='ij');expected=torch.stack((xx,yy),-1)
    for j in range(len(k)):
        uv,_=project(pred['xyz'][j],k[j].cuda(),w[j].cuda())
        if pred['mask'][j].any():maximum=max(maximum,float((uv-expected)[pred['mask'][j]].abs().max()))
    if maximum>2.e-3:raise ValueError(f'buffer projection mismatch {maximum}')
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**maps,pred_depth=pred['depth'].cpu().numpy(),
                        pred_xyz=pred['xyz'].cpu().numpy(),face_id=pred['face_id'].cpu().numpy())
    write(path.with_suffix('.json'),dict(index=index,key=key,maps=binding(path),mesh=mesh_path,
          input_only=True,uses_gt_for_score=False,projection_error_pixels=maximum,
          elapsed_seconds=time.monotonic()-start,common_coverage=float(maps['common'].sum()/max(maps['interior'].sum(),1))))
    emit(index,'scores_sealed',key=key,seconds=time.monotonic()-start)
    return path


def target_mesh(index):
    import trimesh
    r=read(folder(index)/'evaluation_target.json')
    mesh=trimesh.load(verify(r['mesh']),force='mesh',process=False,skip_materials=True)
    mesh.apply_transform(np.asarray(r['transform']))
    if not np.isfinite(mesh.vertices).all():raise ValueError('invalid target')
    return mesh


@torch.no_grad()
def evaluate_index(index,pipeline,context):
    import trimesh
    from scipy.ndimage import distance_transform_edt
    from .geometry import buffers
    from .metrics import evaluate
    from .visuals import visualize,paint_mesh
    from pose_aligned_layered_reconstruction.mesh_metrics import surface_metrics
    rec,k,w,images,masks=observation(index)
    # Natural input-only score artifacts have already been sealed at worker level.
    gt=target_mesh(index)
    gt_path=folder(index)/'target_canonical.ply'
    if not gt_path.exists():gt.export(gt_path)
    _,gt_full_masks=render(gt,k,w,context=context)
    ious=mask_iou(gt_full_masks,masks)
    write(folder(index)/'camera_audit.json',dict(ious=ious.tolist(),passed=bool((ious>=.90).all()),uses_gt=True))
    if (ious<.90).any():raise ValueError(f'GT camera check failed {index}: {ious}')
    gb=buffers(gt,k,w,context,PARAMS['side'])
    depth=gb['depth'].cpu().numpy();mask=gb['mask'].cpu().numpy()
    keys=[str(s) for s in SEEDS]
    if index>=4:
        keys+=['perfect_geometry','local_defect']
        perfect=binding(gt_path)
        make_scores(index,'perfect_geometry',perfect,pipeline,context)
        defect_path=folder(index)/'local_defect.obj'
        if not defect_path.exists():
            pos=np.unravel_index(np.argmax(distance_transform_edt(mask[0])),mask[0].shape)
            center=gb['xyz'][0,pos[0],pos[1]].cpu().numpy()
            defect=gt.copy()
            dist=np.linalg.norm(defect.vertices-center,axis=1)
            weight=np.exp(-.5*(dist/PARAMS['defect_sigma'])**2)
            defect.vertices+=PARAMS['defect_displacement']*weight[:,None]*np.asarray(defect.vertex_normals)
            defect.export(defect_path)
            write(defect_path.with_suffix('.json'),dict(oracle_constructed_control=True,center=center.tolist(),
                  sigma=PARAMS['defect_sigma'],maximum_displacement=PARAMS['defect_displacement']))
        make_scores(index,'local_defect',binding(defect_path),pipeline,context)
    for key in keys:
        path=folder(index)/'scores'/f'{key}.npz'
        meta=read(path.with_suffix('.json'))
        with np.load(verify(meta['maps'])) as a:maps={name:a[name] for name in a.files}
        result=folder(index)/'metrics'/f'{key}.json'
        if result.exists() and (folder(index)/'figures'/f'{key}.jpg').exists() and (folder(index)/'heat_meshes'/f'{key}_input_scores.ply').exists():continue
        values={str(t):evaluate(maps,maps['pred_depth'],depth,mask,t) for t in PARAMS['depth_thresholds']}
        errors=(maps['pred_mask']^mask)|((maps['pred_mask']&mask)&(np.abs(maps['pred_depth']-depth)>.02))
        if key=='perfect_geometry' and np.max(np.abs(maps['pred_depth']-depth))>1.e-5:
            raise ValueError('perfect geometry is not a zero-depth-error control')
        labels=folder(index)/'labels'/f'{key}.npz';labels.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(labels,depth=depth,mask=mask,error=errors)
        predicted=trimesh.load(verify(meta['mesh']),force='mesh',process=False)
        surface=surface_metrics(predicted,gt,count=50000,seed=20261026) if key.isdecimal() else None
        write(result,dict(index=index,uid=rec['uid'],split=rec['split'],key=key,success=True,
            control=not key.isdecimal(),uses_gt_for_scores=False,uses_gt_for_labels=True,
            metrics=values,surface=surface,maps=meta['maps'],mesh=meta['mesh'],labels=binding(labels)))
        visualize(index,key,maps,depth,mask,errors,images,predicted,k,w,context)
        paint_mesh(predicted,maps,folder(index)/'heat_meshes'/f'{key}_input_scores.ply')
        emit(index,'evaluated',key=key,coverage=values['0.02']['common_coverage'])
    emit(index,'complete',natural_records=2,controls=2 if index>=4 else 0)


def worker(indices,gpu):
    import nvdiffrast.torch as dr
    from pose_aligned_layered_reconstruction.paired_encoder import build_frozen_dino_encoder
    from .geometry import fit
    check();device=runtime.setup();start=time.monotonic()
    context=dr.RasterizeCudaContext(device=device)
    encode_inputs(indices,device);generate(indices,device,gpu)
    for i in indices:
        for seed in SEEDS:fit(i,seed,context)
    pipeline=build_frozen_dino_encoder(device=device)
    artifacts=[]
    for i in indices:
        for seed in SEEDS:
            mesh=read(folder(i)/'fitted'/f'{seed}.json')['mesh']
            path=make_scores(i,str(seed),mesh,pipeline,context)
            artifacts.append(binding(path.with_suffix('.json')))
    lock_json(OUT/'runtime'/f'{gpu}_natural_scores_sealed.json',dict(records=artifacts,uses_gt=False))
    for i in indices:evaluate_index(i,pipeline,context)
    del pipeline;cleanup();check()
    write(OUT/'runtime'/f'{gpu}_complete.json',dict(time=now(),indices=indices,elapsed_seconds=time.monotonic()-start))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','report'])
    p.add_argument('--indices',default='0,1,2,3,4,5,6,7,8,9,10,11');p.add_argument('--gpu',default='0')
    a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    if a.action=='prepare':prepare()
    elif a.action=='report':
        from .report import report
        report()
    else:
        indices=[int(v) for v in a.indices.split(',')]
        try:worker(indices,a.gpu)
        except BaseException as e:
            write(OUT/'runtime'/f'{a.gpu}_failed.json',dict(time=now(),error=str(e),indices=indices));raise


if __name__=='__main__':main()
