"""Repair native T2 inputs without any GT or evaluation dataset dependency."""
from __future__ import annotations
import argparse
import gc
from pathlib import Path
import time
import numpy as np
import torch
from . import common as c
from .corruptions import roi_mask,remap_reference,ss_edit_mask
from .flow import sample_flow,sample_bridge
from .evaluate import load_model
from .vendor_pose.data import load_pose_payload,load_ss_target,load_shape_target
from .vendor_pose.runtime import decode_shape_latent,export_geometry_mesh,denormalize_shape_latent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pose',required=True,help='Native DINOv3 features and calibrated cameras')
    p.add_argument('--ss-latent',required=True,help='Corrupted native SS npz with z[8,16,16,16]')
    p.add_argument('--shape-latent',required=True,help='Corrupted native Shape npz, denormalized feats[N,32]')
    p.add_argument('--roi',required=True,help='JSON [[lo_x,lo_y,lo_z],[hi_x,hi_y,hi_z]], grid64')
    p.add_argument('--ss-checkpoint',required=True)
    p.add_argument('--shape-checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,default=12)
    p.add_argument('--validate-only',action='store_true',help='Validate input schemas on CPU without loading flow models')
    args=p.parse_args()
    if args.steps<1:
        raise ValueError('Sampling steps must be positive')
    out=Path(args.output).resolve()
    if not out.is_relative_to(c.ROOT):
        raise ValueError('Inference output must be under '+str(c.ROOT))
    out.mkdir(parents=True,exist_ok=True)
    pose_metadata=torch.load(args.pose,map_location='cpu',weights_only=False)
    uid=str(pose_metadata.get('uid','inference'))
    del pose_metadata
    sample=load_pose_payload(Path(args.pose),uid)
    bounds=torch.tensor(c.read(args.roi),dtype=torch.int32)
    if bounds.shape!=(2,3) or (bounds[0]<0).any() or (bounds[1]>64).any() or (bounds[1]<=bounds[0]).any():
        raise ValueError('ROI must be a nonempty AABB inside grid64')
    zss=load_ss_target(Path(args.ss_latent),uid)[None]
    refcoords,ref_feats=load_shape_target(Path(args.shape_latent),uid)
    if args.validate_only:
        c.write(out/'input_validation.json',{'time':c.now(),'passed':True,'uid':uid,
            'views':len(sample['dino_features']),'shape_tokens':len(refcoords),'ss_shape':list(zss.shape),
            'gt_accessed':False,'models_loaded':False})
        return
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    from trellis2 import models
    from trellis2.modules import sparse as sp
    start=time.time()
    bounds=bounds.cuda()
    zss=zss.cuda()
    refcoords=refcoords.int().cuda()
    refcoords=torch.cat([torch.zeros_like(refcoords[:,:1]),refcoords],-1)
    ref_feats=ref_feats.cuda()
    ssmodel=load_model('ss','repair',args.ss_checkpoint)
    ssdecoder=models.from_pretrained(c.MODEL_PATHS['ss_decoder']).eval().cuda()
    bad_occ=(ssdecoder(zss)>0)[0,0]
    pair={'bad':zss,'mask':ss_edit_mask(bounds,'cuda'),
          'present':torch.ones((1,1,16,16,16),device='cuda',dtype=torch.bool)}
    with torch.autocast('cuda',dtype=torch.bfloat16):
        sampler=sample_bridge if getattr(ssmodel,'repair_objective','flow')=='bridge' else sample_flow
        repaired=sampler(ssmodel,sample,pair,seed=args.seed,steps=args.steps)
    raw=(ssdecoder(repaired)>0)[0,0]
    axes=torch.arange(64,device='cuda')
    xyz=torch.stack(torch.meshgrid(axes,axes,axes,indexing='ij'),-1)
    region=roi_mask(xyz.reshape(-1,3),bounds,64).reshape(64,64,64)
    final_occ=torch.where(region,raw,bad_occ)
    q=torch.unique(torch.argwhere(final_occ)//2,dim=0).int()
    q=torch.cat([torch.zeros_like(q[:,:1]),q],-1)
    outside=~roi_mask(refcoords,bounds,32)
    q=torch.unique(torch.cat([q[roi_mask(q,bounds,32)],refcoords[outside]]),dim=0)
    if not 0<len(q)<=8192:
        raise ValueError('Repaired SS exceeds the native Shape512 support contract')
    ssencoder=models.from_pretrained(c.MODEL_PATHS['ss_encoder']).eval().cuda()
    composed_z=ssencoder(final_occ[None,None].float(),sample_posterior=False)
    np.savez_compressed(out/'ss_repaired.npz',z=composed_z[0].float().cpu().numpy(),
                         sampled_z=repaired[0].float().cpu().numpy(),
                         occupancy64=torch.argwhere(final_occ).cpu().numpy())
    del ssencoder,composed_z
    del ssmodel,ssdecoder
    gc.collect()
    torch.cuda.empty_cache()
    shape_model=load_model('shape','repair',args.shape_checkpoint)
    feats,present=remap_reference(refcoords,ref_feats,q)
    ref=sp.SparseTensor(feats,q)
    edit=roi_mask(q,bounds,32)
    if not present[~edit].all():
        raise ValueError('Exterior input support was lost')
    pair={'bad':ref,'mask':edit,'present':present}
    with torch.autocast('cuda',dtype=torch.bfloat16):
        sampler=sample_bridge if getattr(shape_model,'repair_objective','flow')=='bridge' else sample_flow
        shape=sampler(shape_model,sample,pair,seed=args.seed,steps=args.steps)
    np.savez_compressed(out/'shape_repaired.npz',coords=shape.coords[:,1:].cpu().numpy(),
                         normalized_feats=shape.feats.float().cpu().numpy(),
                         feats=denormalize_shape_latent(shape).feats.float().cpu().numpy())
    del shape_model
    gc.collect()
    torch.cuda.empty_cache()
    decoder=models.from_pretrained(c.MODEL_PATHS['shape_decoder']).eval().cuda()
    decoder.set_resolution(512)
    decoder.low_vram=True
    mesh=decode_shape_latent(shape,decoder)
    export_geometry_mesh(mesh,out/'repaired.obj')
    c.write(out/'inference.json',{'time':c.now(),'args':vars(args),'gt_accessed':False,
        'elapsed_seconds':time.time()-start,'shape_tokens':len(q),
        'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
        'outside_features_max_abs':float((shape.feats[~edit]-ref.feats[~edit]).abs().max()) if (~edit).any() else 0.,
        'outside_occupancy_changes':int((final_occ[~region]!=bad_occ[~region]).sum()),
        'localization':'caller-provided ROI; no automatic error localization in this entry point'})


if __name__=='__main__':
    main()
