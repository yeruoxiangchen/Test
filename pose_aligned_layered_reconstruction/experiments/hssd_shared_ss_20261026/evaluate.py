"""Matched SS samples and frozen SLat meshes, with resumable bound latents."""
import argparse
import gc
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from .common import (ARMS, STEPS, OUT, SHARED, VARIANT, CALLS, binding, verify,
    read, write, plan, policy, new_jobs, logical_cases, case_dir, freeze_runtime,
    check_smoke, event, now, train_dir)
from .data import load_case
from .model import build, ResidualCFG
from ...pretrained import load_trellis_model
from ...frozen_slat import (load_frozen_slat30k, master_noise, sparse_noise,
    sample_mesh_latent, configure_determinism, bootstrap_external_paths)
from ...evaluate_ss_occupancy import occupancy_metrics
from ...mesh_metrics import surface_metrics, structure_metrics
from ..hssd_shared_scene_20261025.evaluate import partitioned_metrics
from ..hssd_input_adaptation_int32_20261025.tiled_native import LargeInputNative
from ..round2_v2.memory_efficient_mesh import install_memory_efficient_mesh


def tensor_binding(value):
    value = value.detach().contiguous().cpu()
    return {'shape':list(value.shape),'dtype':str(value.dtype),
        'sha256':hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def verify_nested(value):
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:
            verify(value)
        else:
            for child in value.values(): verify_nested(child)
    elif isinstance(value,list):
        for child in value: verify_nested(child)


def object_row(case):
    return next(obj for obj in plan()['plan']['objects'] if obj['uid']==case['uid'])


def training_binding(arm):
    folder = train_dir(arm)
    complete = read(folder/'complete.json')
    run = read(folder/'run.json')
    expected = policy(arm)
    if complete.get('step') != STEPS or complete.get('contract') != expected:
        raise ValueError('Require complete matched training for '+arm)
    if run.get('contract')!=expected or run.get('data_identity')!=complete.get('data_identity'):
        raise ValueError('Completed training data/run metadata differs: '+arm)
    checkpoint=binding(folder/'last.pt')
    if complete.get('checkpoint')!=checkpoint:
        raise ValueError('Completed training checkpoint changed: '+arm)
    return {'checkpoint':checkpoint,'completion':binding(folder/'complete.json'),'run':binding(folder/'run.json'),
        'contract':expected}


def load_ss_runtime(arm,device,trained=True):
    contract = policy(arm)
    trained_model = training_binding(arm) if trained else None
    cp = torch.load(verify(trained_model['checkpoint']),map_location='cpu',weights_only=False) if trained else None
    if cp and (cp['contract']!=contract or cp['step']!=STEPS):
        raise ValueError('Trained SS checkpoint cursor/contract differs')
    sampler,model,saved,bindings = build(contract,device,checkpointing=False)
    if cp:
        if cp['bindings']!=bindings:
            raise ValueError('Trained SS frozen-parent binding differs')
        model.residual.load_state_dict(cp['residual'],strict=True)
    model.eval().requires_grad_(False)
    decoder,decoder_binding = load_trellis_model(saved['pretrained'],'ss_dec_conv3d_16l8_fp16')
    decoder = decoder.to(device).eval().requires_grad_(False)
    refcase = next(c for c in logical_cases() if c['arm']==arm and not c['is_reference'])
    reference = read(verify(refcase['condition_record']))
    if decoder_binding!=reference['decoder_binding']:
        raise ValueError('SS decoder differs from archived input baseline')
    return {'sampler':sampler,'model':model,'decoder':decoder,'model_binding':bindings,
        'decoder_binding':decoder_binding,'trained_model':trained_model}


@torch.no_grad()
def sample_support(runtime,obj,case,sample,evidence):
    device = next(runtime['model'].parameters()).device
    positive = sample['stock_condition'].to(device)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        prepared = runtime['model'].prepare(sample,device,evidence=evidence)
        wrapper = ResidualCFG(runtime['model'],positive,prepared)
        generator = torch.Generator(device=device).manual_seed(case['seed']*1000003+obj['original_index']*1009)
        noise = torch.randn((1,8,16,16,16),generator=generator,device=device)
        latent = runtime['sampler'].sample(wrapper,noise,cond=positive,neg_cond=torch.zeros_like(positive),
            steps=25,cfg_strength=5.,cfg_interval=[.5,1.],rescale_t=3.,verbose=False).samples
        logits = runtime['decoder'](latent.to(next(runtime['decoder'].parameters()).dtype))
    calls = {'positive_calls':wrapper.positive_calls,'negative_calls':wrapper.negative_calls}
    if calls!=CALLS or not torch.isfinite(latent).all() or not torch.isfinite(logits).all():
        raise ValueError('Incomplete or nonfinite SS generation')
    coords = torch.argwhere(logits>0)[:,[0,2,3,4]].cpu().numpy().astype(np.int32)
    if not len(coords):
        raise ValueError('Empty SS; retain this case as a failure')
    return coords,{'calls':calls,'initial_noise':tensor_binding(noise),
        'latent':tensor_binding(latent),'logits':tensor_binding(logits),
        'ss_master_seed':case['seed']*1000003+obj['original_index']*1009}


def validate_ss_record(row,case,identity,inputs=None,evidence_binding=None,check_evidence=False):
    if case not in logical_cases() or case['is_reference']:
        raise ValueError('Only planned new SS results are accepted')
    condition = read(verify(case['condition_record']))
    trained_model=training_binding(case['arm'])
    run=read(verify(trained_model['run']))
    training_rows=[r for r in run['data_identity'] if r['uid']==case['uid']]
    if len(training_rows)!=1 or training_rows[0]['original_index']!=case['original_index']:
        raise ValueError('SS object does not uniquely match its training data identity')
    training_row=training_rows[0]
    if training_row['inputs']!=condition['inputs']:
        raise ValueError('Training image observations differ from archived condition')
    if case['arm']=='appearance_control':
        expected_evidence=None
    else:
        expected_evidence=binding(OUT/'evidence'/case['uid']/'receipt.json')
    if training_row['evidence_binding']!=expected_evidence:
        raise ValueError('Training evidence does not match the planned object and arm')
    expected = {'identity':identity,'case':case,'object_uid':case['uid'],'seed':case['seed'],
        'arm':case['arm'],'variant':VARIANT,'inputs':condition['inputs'],
        'condition_record':case['condition_record'],'success':True,'calls':CALLS,
        'trained_model':trained_model,'model_binding':run['bindings'],
        'evidence':training_row['evidence_binding']}
    for key,value in expected.items():
        if row.get(key)!=value:
            raise ValueError('SS record contract differs: '+key)
    if inputs is not None and inputs!=condition['inputs']:
        raise ValueError('Actual image observations differ from archived condition')
    if check_evidence and row.get('evidence')!=evidence_binding:
        raise ValueError('Actual SS geometry evidence differs from saved record')
    if case['arm']=='appearance_control' and row.get('evidence') is not None:
        raise ValueError('Appearance control must not consume the geometry cache')
    if case['arm']=='geometry_observed' and (not isinstance(row.get('evidence'),dict)
            or not {'path','sha256'}.issubset(row['evidence'])):
        raise ValueError('Geometry arm must bind its actual observation evidence')
    verify(row['coords']); verify_nested(row.get('evidence'))
    if row['decoder_binding']!=condition['decoder_binding']:
        raise ValueError('SS decoder identity differs')
    with np.load(verify(row['coords'])) as data:
        coords=data['coords']
    if (coords.dtype!=np.int32 or coords.ndim!=2 or coords.shape[1]!=4 or not len(coords)
            or np.any(coords[:,0]!=0) or np.any(coords[:,1:]<0) or np.any(coords[:,1:]>=64)
            or not np.array_equal(np.unique(coords,axis=0),coords)
            or len(coords)!=row['occupancy']['predicted_voxels']):
        raise ValueError('SS support is invalid or differs from recorded count')
    return row


@torch.no_grad()
def ss(arm):
    configure_determinism(); bootstrap_external_paths()
    torch.set_num_threads(2)
    identity=freeze_runtime(); check_smoke()
    device=torch.device('cuda:0'); cache={}; runtime=None
    for case in new_jobs(arm):
        obj=object_row(case)
        source,sample,inputs,evidence,evidence_binding=load_case(obj,arm,cache)
        dest=case_dir(case);dest.mkdir(parents=True,exist_ok=True); path=dest/'ss.json'
        if path.exists():
            validate_ss_record(read(path),case,identity,inputs,evidence_binding,True)
            continue
        if runtime is None:
            event('load_trained_ss',arm=arm)
            runtime=load_ss_runtime(arm,device)
        event('sample_ss',arm=arm,uid=case['uid'],seed=case['seed'])
        coords,sampling=sample_support(runtime,obj,case,sample,evidence)
        np.savez_compressed(dest/'ss.npz',coords=coords)
        row={'identity':identity,'case':case,'object_uid':case['uid'],'seed':case['seed'],
            'arm':arm,'variant':VARIANT,'inputs':inputs,'condition_record':case['condition_record'],
            'evidence':evidence_binding,'coords':binding(dest/'ss.npz'),
            'model_binding':runtime['model_binding'],'decoder_binding':runtime['decoder_binding'],
            'trained_model':runtime['trained_model'],'occupancy':occupancy_metrics(coords,source['target_coords'].numpy()),
            'calls':sampling['calls'],'sampling':sampling,'success':True}
        validate_ss_record(row,case,identity,inputs,evidence_binding,True)
        write(path,row)
        event('ss_case_complete',arm=arm,uid=case['uid'],seed=case['seed'],support=len(coords))
    event('ss_complete',arm=arm,samples=12)


@torch.no_grad()
def sample_or_resume(dest,contract,runtime,sample,initial):
    from trellis.modules import sparse as sp
    path=dest/'sampled_slat_receipt.json'
    if path.exists():
        row=read(path)
        if row['contract']!=contract:
            raise ValueError('Changed cached SLat support/observations/model/noise')
        saved=torch.load(verify(row['latent']),map_location='cpu',weights_only=True)
        latent=sp.SparseTensor(coords=saved['coords'].to(initial.device),feats=saved['feats'].to(initial.device))
        calls=row['calls']
        event('reuse_sampled_slat',case=contract['case'])
    else:
        # The frozen inference function applies its own explicit image allowlist.
        latent,calls=sample_mesh_latent(runtime,sample,initial)
        if calls!=CALLS or not torch.equal(latent.coords,initial.coords) or not torch.isfinite(latent.feats).all():
            raise ValueError('Invalid generated SLat')
        temp=dest/f'sampled_slat.{os.getpid()}.tmp'
        torch.save({'coords':latent.coords.detach().cpu(),'feats':latent.feats.detach().cpu()},temp)
        temp.replace(dest/'sampled_slat.pt')
        write(path,{'time':now(),'contract':contract,'calls':calls,'latent':binding(dest/'sampled_slat.pt')})
    if (calls!=CALLS or not torch.equal(latent.coords,initial.coords)
            or latent.feats.shape!=initial.feats.shape or not torch.isfinite(latent.feats).all()):
        raise ValueError('Cached SLat differs or is nonfinite')
    return latent,calls,binding(path)


@torch.no_grad()
def mesh_case(case,obj,ss_row,ss_binding,sample,inputs,dest,identity,runtime,policy_native):
    import trimesh
    with np.load(verify(ss_row['coords'])) as data:coords=data['coords']
    initial=sparse_noise(coords,master_noise(case['seed'],obj['original_index'],torch.device('cuda:0')))
    contract={'identity':identity,'case':case,'ss':ss_binding,'inputs':inputs,
        'slat_binding':runtime['binding'],'original_index':obj['original_index'],
        'master_seed':case['seed']*2000003+obj['original_index']*2017+7919,
        'initial_coords':tensor_binding(initial.coords),'initial_noise':tensor_binding(initial.feats)}
    event('sample_mesh',arm=case['arm'],uid=case['uid'],seed=case['seed'],support=len(coords))
    tiled_start=len(policy_native.tiled_calls)
    latent,calls,receipt=sample_or_resume(dest,contract,runtime,sample,initial)
    event('decode_mesh',arm=case['arm'],uid=case['uid'],seed=case['seed'])
    mesh=runtime['decoder'](latent)[0].to_trimesh(transform_pose=False)
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError('Invalid mesh; retain this case as failure')
    mesh.export(dest/'mesh_canonical.ply')
    mesh=trimesh.load(dest/'mesh_canonical.ply',process=False,force='mesh')
    target=trimesh.load(verify(obj['mesh']),process=False,force='mesh')
    world=mesh.copy();world.apply_transform(np.linalg.inv(obj['canonical_from_world']))
    world.export(dest/'mesh_world.ply')
    saved=trimesh.load(dest/'mesh_world.ply',process=False,force='mesh')
    error=float(np.max(np.abs(saved.vertices-world.vertices)))
    if error>1e-6 or not np.array_equal(saved.faces,mesh.faces):
        raise ValueError('Canonical/world export differs')
    shared=read(verify(plan()['plan']['shared_plan']))['plan']
    row={'identity':identity,'case':case,'object_uid':case['uid'],'seed':case['seed'],'arm':case['arm'],
        'variant':VARIANT,'inputs':inputs,'condition_record':case['condition_record'],'ss':ss_binding,
        'evidence':ss_row.get('evidence'),'trained_model':ss_row.get('trained_model'),
        'success':True,'fitted_transform':False,'canonical_from_world':obj['canonical_from_world'],'gt':obj['mesh'],
        'mesh':binding(dest/'mesh_canonical.ply'),'world_mesh':binding(dest/'mesh_world.ply'),
        'surface':surface_metrics(mesh,target),'structure':structure_metrics(mesh),
        'partitioned':partitioned_metrics(mesh,target,obj,shared),'slat_binding':runtime['binding'],
        'calls':calls,'sampled_slat_receipt':receipt,'world_export_max_abs_error':error,
        'kernel_policy':policy_native.manifest(),'native_input_row_tiling':policy_native.tiled_calls[tiled_start:],
        'runtime_amendment':plan()['plan']['runtime_amendment'],
        'gt_partition_scope':'fixed original four shared views for every variant'}
    return row


def validate_mesh_record(row,case,identity,ss_row=None,ss_binding=None):
    if case not in logical_cases() or case['is_reference']:
        raise ValueError('Only planned new meshes accepted')
    ss_path=case_dir(case)/'ss.json'
    ss_row=validate_ss_record(read(ss_path),case,identity) if ss_row is None else ss_row
    ss_binding=binding(ss_path) if ss_binding is None else ss_binding
    obj=object_row(case);p=plan()['plan']
    expected={'identity':identity,'case':case,'object_uid':case['uid'],'seed':case['seed'],'arm':case['arm'],
        'variant':VARIANT,'inputs':ss_row['inputs'],'condition_record':case['condition_record'],'ss':ss_binding,
        'evidence':ss_row.get('evidence'),'trained_model':ss_row['trained_model'],
        'success':True,'fitted_transform':False,'canonical_from_world':obj['canonical_from_world'],
        'gt':obj['mesh'],'slat_binding':p['slat_binding'],'calls':CALLS}
    for key,value in expected.items():
        if row.get(key)!=value:raise ValueError('Mesh contract differs: '+key)
    for key in ['mesh','world_mesh','sampled_slat_receipt']:verify(row[key])
    receipt=read(verify(row['sampled_slat_receipt']));verify(receipt['latent'])
    contract=receipt['contract']
    required={'identity':identity,'case':case,'ss':ss_binding,'inputs':ss_row['inputs'],
        'slat_binding':p['slat_binding'],'original_index':obj['original_index'],
        'master_seed':case['seed']*2000003+obj['original_index']*2017+7919}
    for key,value in required.items():
        if contract.get(key)!=value:raise ValueError('Latent contract differs: '+key)
    n=ss_row['occupancy']['predicted_voxels']
    for key,shape,dtype in [('initial_coords',[n,4],'torch.int32'),('initial_noise',[n,8],'torch.float32')]:
        b=contract[key]
        if b['shape']!=shape or b['dtype']!=dtype or len(b['sha256'])!=64:
            raise ValueError('Latent initial-noise/coordinate receipt invalid')
    if receipt['calls']!=CALLS:raise ValueError('Incomplete SLat call receipt')
    return row


@torch.no_grad()
def mesh(arm):
    configure_determinism();bootstrap_external_paths();torch.set_num_threads(2)
    identity=freeze_runtime();check_smoke();install_memory_efficient_mesh()
    cache={};runtime=None
    with LargeInputNative() as native:
        for case in new_jobs(arm):
            obj=object_row(case)
            _,sample,inputs,_,ev=load_case(obj,arm,cache)
            dest=case_dir(case);sp=dest/'ss.json';mp=dest/'mesh.json'
            ss_row=validate_ss_record(read(sp),case,identity,inputs,ev,True);sb=binding(sp)
            if mp.exists():validate_mesh_record(read(mp),case,identity,ss_row,sb);continue
            if runtime is None:
                event('load_frozen_slat',arm=arm)
                runtime=load_frozen_slat30k(torch.device('cuda:0'),evaluation_uids=plan()['plan']['evaluation_uid_context'])
                if runtime['binding']!=plan()['plan']['slat_binding']:raise ValueError('Frozen SLat changed')
            row=mesh_case(case,obj,ss_row,sb,sample,inputs,dest,identity,runtime,native)
            validate_mesh_record(row,case,identity,ss_row,sb);write(mp,row)
            event('mesh_case_complete',arm=arm,uid=case['uid'],seed=case['seed'])
    event('mesh_complete',arm=arm,samples=12)


@torch.no_grad()
def smoke_replay():
    """Exact old table SS + mesh replay through the new zero adapter path."""
    import trimesh
    configure_determinism();bootstrap_external_paths();torch.set_num_threads(2)
    identity=freeze_runtime();install_memory_efficient_mesh()
    case=next(c for c in new_jobs(ARMS[0]) if c['uid'].endswith('i075') and c['seed']==42)
    obj=object_row(case);old_ss=read(verify(case['condition_record']))
    frozen=next(c for c in logical_cases() if c['is_reference'] and c['uid']==case['uid'] and c['seed']==42)
    old_mesh=read(verify(frozen['reference_mesh']))
    _,sample,inputs,evidence,eb=load_case(obj,ARMS[0],{})
    dest=OUT/'smoke/replay'/case['uid']/str(case['seed']);dest.mkdir(parents=True,exist_ok=True)
    receipt_path=dest/'replay.json'
    if receipt_path.exists():
        receipt=read(receipt_path)
        if receipt['identity']!=identity or not receipt['passed']:raise ValueError('Changed or failed replay receipt')
        verify(receipt['mesh_record']);verify(receipt['ss_record'])
        cached=read(receipt['mesh_record']['path'])
        for key in ['mesh','world_mesh','sampled_slat_receipt']:verify(cached[key])
        verify(read(cached['sampled_slat_receipt']['path'])['latent'])
        verify(read(receipt['ss_record']['path'])['coords'])
        return binding(receipt_path)
    runtime=load_ss_runtime(ARMS[0],torch.device('cuda:0'),trained=False)
    coords,sampling=sample_support(runtime,obj,case,sample,evidence)
    with np.load(verify(old_ss['coords'])) as data:
        if not np.array_equal(coords,data['coords']):raise ValueError('Zero-adapter SS replay differs')
    np.savez_compressed(dest/'ss.npz',coords=coords)
    replay_ss={**old_ss,'coords':binding(dest/'ss.npz'),'inputs':inputs,'calls':sampling['calls'],
        'evidence':eb,'trained_model':None,'sampling':sampling}
    write(dest/'ss.json',replay_ss)
    del runtime
    gc.collect()
    torch.cuda.empty_cache()
    slat=load_frozen_slat30k(torch.device('cuda:0'),evaluation_uids=plan()['plan']['evaluation_uid_context'])
    if slat['binding']!=plan()['plan']['slat_binding']:raise ValueError('Replay SLat binding differs')
    with LargeInputNative() as native:
        row=mesh_case(case,obj,replay_ss,binding(dest/'ss.json'),sample,inputs,dest,identity,slat,native)
    for key in ['mesh','world_mesh']:
        a=trimesh.load(verify(row[key]),process=False,force='mesh');b=trimesh.load(verify(old_mesh[key]),process=False,force='mesh')
        if not np.array_equal(a.vertices,b.vertices) or not np.array_equal(a.faces,b.faces):
            raise ValueError('Zero-adapter full mesh replay differs: '+key)
    for key in ['surface','structure','partitioned','slat_binding','calls','canonical_from_world','fitted_transform','inputs']:
        if row[key]!=old_mesh[key]:raise ValueError('Replay field differs: '+key)
    write(dest/'mesh.json',row)
    write(receipt_path,{'time':now(),'identity':identity,'passed':True,'ss_exact':True,'mesh_exact':True,
        'case':case,'reference_ss':case['condition_record'],'reference_mesh':frozen['reference_mesh'],
        'ss_record':binding(dest/'ss.json'),'mesh_record':binding(dest/'mesh.json')})
    event('zero_adapter_replay_complete',ss_exact=True,mesh_exact=True)
    return binding(receipt_path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['ss','mesh','smoke-replay'])
    p.add_argument('--arm',choices=ARMS,default=ARMS[0]);a=p.parse_args()
    if a.action=='smoke-replay':smoke_replay()
    elif a.action=='ss':ss(a.arm)
    else:mesh(a.arm)
