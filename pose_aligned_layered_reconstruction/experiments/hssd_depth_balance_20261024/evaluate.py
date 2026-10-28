"""Paired SS16 evaluation; separate bounded mesh4 evaluation with archived references."""
import argparse
import json
import shutil
import numpy as np
import torch
from ...v2.dataset import SceneDataset, select_views
from ...pretrained import load_trellis_model
from ...evaluate_ss_occupancy import occupancy_metrics
from ...frozen_slat import load_frozen_slat30k, master_noise, sparse_noise, sample_mesh_latent
from ...mesh_metrics import surface_metrics, structure_metrics, aggregate_mesh_records
from ..round2_v2.memory_efficient_mesh import install_memory_efficient_mesh
from .common import ARMS, REFERENCES, OUT, CELL, CAUSAL, SEEDS, INDICES, FORMAT, SS_STEPS, freeze, policy, read, binding, verify, write, locked, check_record
from .model import build, ResidualCFG, observations

ALL_ARMS=(*REFERENCES,*ARMS)


def root_for(arm):
    return OUT/'evaluations'/arm


def identity(arm):
    value={'protocol':freeze(),'arm':arm}
    if arm in ARMS:
        value.update(contract=policy('ss',arm),checkpoint=binding(OUT/'train'/f'ss_{arm}'/'last.pt'))
    else:
        path=CELL/'evaluations'/f'occluded_{arm}'
        value.update(reference_identity=binding(path/'identity.json'),reference_complete=binding(path/'ss_complete.json'))
        old=read(path/'identity.json')
        if old!=read(path/'ss_complete.json'):raise ValueError('Reference SS incomplete')
        if 'checkpoint' in old:verify(old['checkpoint'])
    return value


def check_ss_complete(arm):
    root=root_for(arm); ident=identity(arm)
    if read(root/'ss_complete.json')!=ident or read(root/'identity.json')!=ident:
        raise ValueError('Changed/incomplete SS identity')
    rows=[]
    for index in range(16):
        for seed in SEEDS:
            row=read(root/f'{index:03d}'/f'{seed}_ss.json')
            check_record(row,index,seed,arm)
            rows.append(row)
    return rows


def reference_ss(arm):
    ident=identity(arm); root=root_for(arm)
    locked(root/'identity.json',ident)
    for index in range(16):
        for seed in SEEDS:
            src=CELL/'evaluations'/f'occluded_{arm}'/f'{index:03d}'/f'{seed}_ss.json'
            row=read(src);check_record(row,index,seed,arm)
            locked(root/f'{index:03d}'/f'{seed}_ss.json',{**row,'reference_record':binding(src)})
    locked(root/'ss_complete.json',ident)


@torch.no_grad()
def sample_ss(arm):
    if arm in REFERENCES:
        return reference_ss(arm)
    torch.set_num_threads(2)
    device=torch.device('cuda:0')
    ident=identity(arm);root=root_for(arm);contract=ident['contract']
    locked(root/'identity.json',ident)
    if (root/'ss_complete.json').exists():
        check_ss_complete(arm);return
    cp=torch.load(verify(ident['checkpoint']),map_location='cpu',weights_only=False)
    if cp['format']!=FORMAT or cp['contract']!=contract or cp['step']!=SS_STEPS:
        raise ValueError('Require completed new SS training')
    sampler,model,saved,bindings=build(contract,device)
    if cp['bindings']!=bindings:raise ValueError('Changed frozen weights')
    model.residual.load_state_dict(cp['state'],strict=True)
    model.eval().requires_grad_(False)
    decoder,db=load_trellis_model(saved['pretrained'],'ss_dec_conv3d_16l8_fp16')
    decoder=decoder.to(device).eval()
    locked(root/'ss_decoder_binding.json',db)
    dataset=SceneDataset(verify(contract['manifest']),require_labels=False,require_ss_target=True)
    for index in range(16):
        source=dataset[index]
        sample=select_views(observations(source),torch.arange(8))
        with torch.autocast('cuda',dtype=torch.bfloat16):prepared=model.prepare(sample,device)
        folder=root/f'{index:03d}';folder.mkdir(exist_ok=True)
        for seed in SEEDS:
            dest=folder/f'{seed}_ss.json'
            if dest.exists():check_record(read(dest),index,seed,arm);continue
            positive=sample['stock_condition'].to(device)
            wrapper=ResidualCFG(model,positive,prepared)
            generator=torch.Generator(device=device).manual_seed(seed*1000003+index*1009)
            noise=torch.randn((1,8,16,16,16),device=device,generator=generator)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z=sampler.sample(wrapper,noise,cond=positive,neg_cond=torch.zeros_like(positive),
                    steps=25,cfg_strength=5.,cfg_interval=[.5,1.],rescale_t=3.,verbose=False).samples
                logits=decoder(z.to(next(decoder.parameters()).dtype))
            finite=bool(torch.isfinite(z).all() and torch.isfinite(logits).all())
            coords=torch.argwhere(logits>0)[:,[0,2,3,4]].cpu().numpy().astype(np.int32) if finite else np.empty((0,4),np.int32)
            coord_path=folder/f'{seed}_ss.npz';np.savez_compressed(coord_path,coords=coords)
            row={'uid':source['uid'],'object_uid':source['object_uid'],'index':index,'seed':seed,'arm':arm,
                'gt':source['target_mesh_binding'],'coords':binding(coord_path),'success':bool(finite and len(coords)),
                'positive_calls':wrapper.positive_calls,'negative_calls':wrapper.negative_calls}
            if row['success']:row['occupancy']=occupancy_metrics(coords,source['target_coords'].numpy())
            else:row['failure']='empty or nonfinite SS'
            write(dest,row);print(json.dumps({'arm':arm,'index':index,'seed':seed,'success':row['success']}),flush=True)
    locked(root/'ss_complete.json',ident)


def summarize_ss():
    rows={a:check_ss_complete(a) for a in ALL_ARMS}
    comparisons={}
    for candidate,control in [('balanced','uniform'),('uniform','appearance_control'),('balanced','appearance_control'),('oracle_point','appearance_control')]:
        rr=[{**r,'mode':mode,'surface':{k:r['occupancy'][k] for k in ['iou','precision','recall','f1']} if r['success'] else {}}
            for arm,mode in [(candidate,'candidate'),(control,'parent')] for r in rows[arm]]
        comparisons[candidate+'_vs_'+control]=(aggregate_mesh_records(rr,[r['uid'] for r in rows[candidate][::3]],SEEDS)
            if all(r['success'] for r in rr) else {'failures':[r for r in rr if not r['success']]})
    write(OUT/'ss_summary.json',{'scope':'same16 training objects; old references reused; no automatic expansion',
          'records':rows,'comparisons':comparisons,'new_ss_count':96,'reused_ss_count':144})


@torch.no_grad()
def decode_mesh(arm, limit=None):
    import trimesh
    torch.set_num_threads(2)
    check_ss_complete(arm)
    ident=identity(arm);root=OUT/'mesh'/arm
    locked(root/'identity.json',ident)
    dataset=SceneDataset(verify(ident['protocol']['manifest']),require_labels=False)
    runtime=None;added=0
    for index in INDICES:
        sample=select_views(dataset[index],torch.arange(8))
        folder=root/f'{index:03d}';folder.mkdir(exist_ok=True)
        for seed in SEEDS:
            dest=folder/f'{seed}.json'
            if dest.exists():check_record(read(dest),index,seed,arm);continue
            if arm in REFERENCES:
                src=CAUSAL/'mesh'/arm/f'{index:03d}'/f'{seed}.json'
                row=read(src);check_record(row,index,seed,arm)
                # Same original global index, fixed SLat, and paired coordinatewise noise.
                original_ss=read(row['source_ss']['path'])
                current_ss=read(root_for(arm)/f'{index:03d}'/f'{seed}_ss.json')
                if original_ss['coords']!=current_ss['coords'] or row['fitted_transform'] is not False:
                    raise ValueError('Reference mesh has different support/frame')
                row={**row,'reference_record':binding(src)}
            else:
                spath=root_for(arm)/f'{index:03d}'/f'{seed}_ss.json'
                ss=read(spath);check_record(ss,index,seed,arm)
                gt_source=verify(sample['target_mesh_binding']);gt_path=folder/'gt.ply'
                if not gt_path.exists():shutil.copyfile(gt_source,gt_path)
                if binding(gt_path)['sha256']!=binding(gt_source)['sha256']:raise ValueError('Changed GT copy')
                row={'uid':sample['uid'],'object_uid':sample['object_uid'],'index':index,'seed':seed,'arm':arm,
                    'gt':binding(gt_path),'source_ss':binding(spath),'success':False}
                if not ss['success']:row['failure']=ss['failure']
                else:
                    if runtime is None:
                        install_memory_efficient_mesh();runtime=load_frozen_slat30k(torch.device('cuda:0'))
                        locked(root/'slat_binding.json',runtime['binding'])
                        for reference in REFERENCES:
                            if runtime['binding']!=read(CAUSAL/'mesh'/reference/'slat_binding.json'):
                                raise ValueError('SLat differs from archived references')
                    with np.load(verify(ss['coords'])) as data:coords=data['coords']
                    initial=sparse_noise(coords,master_noise(seed,index,'cuda:0'))
                    z,_=sample_mesh_latent(runtime,sample,initial)
                    mesh=runtime['decoder'](z)[0].to_trimesh(transform_pose=False)
                    if len(mesh.faces) and np.isfinite(mesh.vertices).all():
                        path=folder/f'{seed}.ply';mesh.export(path)
                        mesh=trimesh.load(path,force='mesh',process=False)
                        target=trimesh.load(gt_path,force='mesh',process=False)
                        row.update(success=True,mesh=binding(path),surface=surface_metrics(mesh,target),structure=structure_metrics(mesh),
                            native_bbox_min=np.asarray(mesh.vertices).min(0).tolist(),native_bbox_max=np.asarray(mesh.vertices).max(0).tolist(),
                            gt_bbox_min=np.asarray(target.vertices).min(0).tolist(),gt_bbox_max=np.asarray(target.vertices).max(0).tolist(),fitted_transform=False)
                    else:row['failure']='empty or nonfinite mesh'
                    del initial,z,mesh
            write(dest,row);added+=1
            print(json.dumps({'arm':arm,'index':index,'seed':seed,'success':row['success']}),flush=True)
            if limit is not None and added>=limit:return
    locked(root/'complete.json',ident)


def summarize_mesh():
    rows={}
    for arm in ALL_ARMS:
        root=OUT/'mesh'/arm
        if read(root/'complete.json')!=identity(arm):raise ValueError('Incomplete mesh')
        rows[arm]=[]
        for i in INDICES:
            for seed in SEEDS:
                row=read(root/f'{i:03d}'/f'{seed}.json');check_record(row,i,seed,arm);rows[arm].append(row)
    comparisons={}
    for candidate,control in [('balanced','uniform'),('uniform','appearance_control'),('balanced','appearance_control'),('oracle_point','appearance_control')]:
        rr=[{**r,'mode':mode} for arm,mode in [(candidate,'candidate'),(control,'parent')] for r in rows[arm]]
        comparisons[candidate+'_vs_'+control]=(aggregate_mesh_records(rr,[r['uid'] for r in rows[candidate][::3]],SEEDS)
            if all(r['success'] for r in rr) else {'failures':[r for r in rr if not r['success']]})
    from .visuals import galleries
    images=galleries(rows)
    write(OUT/'mesh_summary.json',{'scope':'selected4 training diagnostic, not population or heldout performance',
          'records':rows,'comparisons':comparisons,'galleries':images,'new_mesh_count':24,'reused_mesh_count':36})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['ss','ss-summary','mesh','mesh-summary'])
    p.add_argument('--arm',choices=ALL_ARMS);p.add_argument('--limit',type=int);a=p.parse_args()
    if a.phase=='ss-summary':summarize_ss()
    elif a.phase=='mesh-summary':summarize_mesh()
    elif a.arm:
        sample_ss(a.arm) if a.phase=='ss' else decode_mesh(a.arm,a.limit)
    else:p.error('--arm required')
