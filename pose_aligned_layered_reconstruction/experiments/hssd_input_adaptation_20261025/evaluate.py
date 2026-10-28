"""Frozen SS/SLat sampling with synchronized variable views and explicit reuse."""
import argparse
import numpy as np
import torch
from .common import *
from ...v2.contracts import validate_sample
from ...v2.dataset import select_views
from ...v2.layer_predictor import GeometryConfig
from ...v2.ss import build_ss
from ...native_ss import LayeredSSCFGFlow
from ...checkpoints import DEFAULT_PARENT_SS30K
from ...pretrained import DEFAULT_PRETRAINED,load_trellis_model
from ...frozen_slat import load_frozen_slat30k,master_noise,sparse_noise,sample_mesh_latent
from ...evaluate_ss_occupancy import occupancy_metrics
from ...mesh_metrics import surface_metrics,structure_metrics
from ..hssd_cell_query_20261023.model import build,ResidualCFG,observations
from ..hssd_operator_audit_20261025.kernel_policy import FixedNative
from ..round2_v2.memory_efficient_mesh import install_memory_efficient_mesh
from ..hssd_shared_scene_20261025.evaluate import partitioned_metrics

def jobs(arm=None,smoke=False):
    if smoke:
        objs=plan()['plan']['objects']
        return [('full_all','yxc30k',objs[0],42),('crop518_visible','yxc30k',objs[3],42)]
    return [j for j in new_jobs() if arm is None or j[1]==arm]

def dest_for(variant,arm,obj,seed):
    if variant=='full_all':return OUT/'smoke_replay'/arm/obj['uid']/str(seed)
    return case_dir(variant,arm,obj['uid'],seed)

def load_input(obj,variant,cache):
    domain=source_domain(variant);key=domain,obj['uid']
    if key not in cache:
        path=input_path(obj,variant)
        if domain!='full':
            receipt=read(path.parent/'receipt.json')
            if receipt['plan_identity']!=plan()['identity']:raise ValueError('Wrong feature cache plan')
            verify(receipt['images']);verify(receipt['sample']);verify(receipt['encoder'])
        value=torch.load(path,map_location='cpu',weights_only=True)
        validate_sample(value,require_labels=False,require_ss_target=True)
        if value['object_uid']!=obj['uid']:raise ValueError('Object/input mismatch')
        for b in value['source_files']:verify(b)
        cache[key]=value,binding(path)
    value,bound=cache[key]
    selected=view_indices(obj,variant)
    sample=select_views(observations(value),torch.tensor(selected,dtype=torch.long))
    if len(sample['intrinsics'])!=len(selected):raise ValueError('Context/view mismatch')
    return value,sample,{'sample':bound,'selected_indices':selected,'selected_frames':[FRAMES[i] for i in selected]}

def load_ss(arm,device):
    if arm=='yxc30k':
        sampler,model,bindings=build_ss(GeometryConfig(),device,str(DEFAULT_PARENT_SS30K),
            str(DEFAULT_PRETRAINED),False,condition_mode='no_layer')
        pretrained=str(DEFAULT_PRETRAINED)
    else:
        cp=torch.load(CELL/'train/occluded_appearance_control/last.pt',map_location='cpu',weights_only=False)
        if cp['contract']['arm']!='appearance_control' or cp['step']!=cp['contract']['steps']:
            raise ValueError('Expected completed appearance checkpoint')
        sampler,model,saved,bindings=build(cp['contract'],device)
        if cp['bindings']!=bindings:raise ValueError('Appearance base changed')
        model.residual.load_state_dict(cp['residual'],strict=True);pretrained=saved['pretrained']
    model.eval().requires_grad_(False)
    decoder,db=load_trellis_model(pretrained,'ss_dec_conv3d_16l8_fp16')
    decoder=decoder.to(device).eval().requires_grad_(False)
    expected=read(SHARED/'evaluations'/arm/plan()['plan']['objects'][0]['uid']/'42/ss.json')
    if bindings!=expected['model_binding'] or db!=expected['decoder_binding']:raise ValueError('SS model differs from reused baseline')
    return sampler,model,decoder,bindings,db

@torch.no_grad()
def ss(arm,smoke=False):
    torch.set_num_threads(2);device=torch.device('cuda:0');ident=freeze_runtime();cache={};runtime=None
    for variant,arm,obj,seed in jobs(arm,smoke):
        source,sample,inputs=load_input(obj,variant,cache)
        dest=dest_for(variant,arm,obj,seed);dest.mkdir(parents=True,exist_ok=True)
        path=dest/'ss.json'
        if path.exists():
            old=read(path)
            if old['identity']!=ident or old['inputs']!=inputs:raise ValueError('Changed completed SS case')
            verify(old['coords']);continue
        if runtime is None:
            event('load_ss',arm=arm);runtime=load_ss(arm,device)
        sampler,model,decoder,bindings,db=runtime
        positive=sample['stock_condition'].to(device)
        event('sample_ss',variant=variant,arm=arm,uid=obj['uid'],seed=seed,views=inputs['selected_frames'])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if arm=='yxc30k':wrapper=LayeredSSCFGFlow(model,positive,sample,use_layer_branch=False)
            else:wrapper=ResidualCFG(model,positive,model.prepare(sample,device),enabled=True)
            gen=torch.Generator(device=device).manual_seed(seed*1000003+obj['original_index']*1009)
            noise=torch.randn((1,8,16,16,16),generator=gen,device=device)
            latent=sampler.sample(wrapper,noise,cond=positive,neg_cond=torch.zeros_like(positive),
                steps=25,cfg_strength=5.,cfg_interval=[.5,1.],rescale_t=3.,verbose=False).samples
            logits=decoder(latent.to(next(decoder.parameters()).dtype))
        if not torch.isfinite(latent).all() or not torch.isfinite(logits).all():raise ValueError('Nonfinite SS; target retained as failure')
        coords=torch.argwhere(logits>0)[:,[0,2,3,4]].cpu().numpy().astype(np.int32)
        if not len(coords):raise ValueError('Empty SS; target retained as failure')
        if variant=='full_all':
            old=read(SHARED/'evaluations'/arm/obj['uid']/str(seed)/'ss.json')
            with np.load(verify(old['coords'])) as data:
                if not np.array_equal(coords,data['coords']):raise ValueError('Baseline SS replay differs; stop before interpreting input changes')
        np.savez_compressed(dest/'ss.npz',coords=coords)
        write(path,{'identity':ident,'variant':variant,'arm':arm,'object_uid':obj['uid'],'seed':seed,
            'inputs':inputs,'coords':binding(dest/'ss.npz'),'model_binding':bindings,'decoder_binding':db,
            'occupancy':occupancy_metrics(coords,source['target_coords'].numpy()),'success':True})
    event('ss_complete',arm=arm,smoke=smoke)

@torch.no_grad()
def mesh(smoke=False):
    import trimesh
    torch.set_num_threads(2);device=torch.device('cuda:0');ident=freeze_runtime();cache={};runtime=None
    p=plan()['plan'];original=read(SHARED/'plan.json')['plan'];install_memory_efficient_mesh()
    with FixedNative() as fixed:
        for variant,arm,obj,seed in jobs(smoke=smoke):
            source,sample,inputs=load_input(obj,variant,cache)
            dest=dest_for(variant,arm,obj,seed);path=dest/'mesh.json'
            if path.exists():
                r=read(path)
                if r['identity']!=ident or r['inputs']!=inputs:raise ValueError('Changed completed mesh')
                verify(r['mesh']);verify(r['world_mesh']);continue
            support=read(dest/'ss.json')
            if support['identity']!=ident or support['inputs']!=inputs:raise ValueError('SS/input mismatch')
            if runtime is None:
                event('load_slat');runtime=load_frozen_slat30k(device,evaluation_uids=[o['uid'] for o in p['objects']])
                expected=read(SHARED/'evaluations/yxc30k'/p['objects'][0]['uid']/'42/mesh.json')['slat_binding']
                if runtime['binding']!=expected:raise ValueError('SLat model/protocol differs from reused baseline')
            event('sample_mesh',variant=variant,arm=arm,uid=obj['uid'],seed=seed,views=inputs['selected_frames'])
            with np.load(verify(support['coords'])) as data:coords=data['coords']
            initial=sparse_noise(coords,master_noise(seed,obj['original_index'],device))
            latent,calls=sample_mesh_latent(runtime,sample,initial)
            if calls!={'positive_calls':25,'negative_calls':19}:raise ValueError('Incomplete SLat sampling')
            output=runtime['decoder'](latent)[0].to_trimesh(transform_pose=False)
            if not len(output.faces) or not np.isfinite(output.vertices).all():raise ValueError('Invalid mesh; target retained as failure')
            output.export(dest/'mesh_canonical.ply');output=trimesh.load(dest/'mesh_canonical.ply',process=False,force='mesh')
            target=trimesh.load(verify(obj['mesh']),process=False,force='mesh')
            world=output.copy();world.apply_transform(np.linalg.inv(obj['canonical_from_world']));world.export(dest/'mesh_world.ply')
            surface=surface_metrics(output,target)
            if variant=='full_all':
                old=read(SHARED/'evaluations'/arm/obj['uid']/str(seed)/'mesh.json')
                reference=trimesh.load(verify(old['mesh']),process=False,force='mesh')
                if not np.array_equal(output.vertices,reference.vertices) or not np.array_equal(output.faces,reference.faces):
                    raise ValueError('Baseline mesh replay differs; do not mix numerical protocols')
                if surface!=old['surface']:raise ValueError('Baseline metric replay differs')
            write(path,{'identity':ident,'variant':variant,'arm':arm,'object_uid':obj['uid'],'seed':seed,'inputs':inputs,
                'success':True,'fitted_transform':False,'ss':binding(dest/'ss.json'),'mesh':binding(dest/'mesh_canonical.ply'),
                'world_mesh':binding(dest/'mesh_world.ply'),'gt':obj['mesh'],'canonical_from_world':obj['canonical_from_world'],
                'surface':surface,'structure':structure_metrics(output),'partitioned':partitioned_metrics(output,target,obj,original),
                'slat_binding':runtime['binding'],'kernel_policy':fixed.manifest(),'calls':calls,
                'gt_partition_scope':'fixed original four shared views for every variant'})
            del initial,latent,output,target,world
    if smoke:
        write(OUT/'smoke.json',{'time':now(),'identity':ident,'passed':True,'baseline_replay_exact':True,
            'cropped_two_view_case_complete':True,'cases':[binding(dest_for(v,a,o,s)/'mesh.json') for v,a,o,s in jobs(smoke=True)]})
    event('mesh_complete',smoke=smoke)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['ss','mesh'])
    parser.add_argument('--arm',choices=ARMS,default='yxc30k');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.action=='ss':ss(args.arm,args.smoke)
    else:mesh(args.smoke)
