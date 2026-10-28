"""Frozen paired complete SS sampling under component interventions."""
import argparse
import json
import numpy as np
import torch
from ...v2.dataset import SceneDataset, select_views
from ...pretrained import load_trellis_model
from ...evaluate_ss_occupancy import occupancy_metrics
from ...mesh_metrics import aggregate_mesh_records
from ..hssd_cell_query_20261023.common import policy as cell_policy, FORMAT
from ..hssd_cell_query_20261023.model import build, ResidualCFG
from .common import OUT, CELL, PILOT, INDICES, SEEDS, ADAPTERS, VARIANTS, read, binding, verify, write, freeze, locked_identity, check_record
from .interventions import prepare_variants


def load_model(adapter, device):
    contract=cell_policy('occluded',adapter)
    path=CELL/'train'/f'occluded_{adapter}'/'last.pt'
    cp=torch.load(path,map_location='cpu',weights_only=False)
    assert cp['format']==FORMAT and cp['contract']==contract and cp['step']==200
    sampler,model,saved,bound=build(contract,device)
    assert cp['bindings']==bound
    model.residual.load_state_dict(cp['residual'],strict=True)
    return sampler,model.eval().requires_grad_(False),saved


@torch.no_grad()
def sample(adapter, limit=None):
    torch.set_num_threads(2)
    root=OUT/'ss'/adapter
    identity={'protocol':freeze(),'adapter':adapter,'checkpoint':binding(CELL/'train'/f'occluded_{adapter}'/'last.pt')}
    locked_identity(root/'identity.json',identity)
    dataset=SceneDataset(PILOT/'occluded16.json',require_labels=True,require_ss_target=True)
    expected=[(i,v,s) for i in INDICES for v in VARIANTS for s in SEEDS]
    pending=[]
    for i,v,s in expected:
        path=root/f'{i:03d}'/v/f'{s}.json'
        if path.exists():check_record(read(path),dataset.rows[i]['uid'],s)
        else:pending.append((i,v,s))
    if not pending:
        write(root/'complete.json',identity)
        return
    device=torch.device('cuda:0')
    sampler,model,saved=load_model(adapter,device)
    decoder,decoder_binding=load_trellis_model(saved['pretrained'],'ss_dec_conv3d_16l8_fp16')
    decoder=decoder.to(device).eval().requires_grad_(False)
    locked_identity(root/'decoder.json',decoder_binding)
    added=0
    for index in INDICES:
        if not any(i==index for i,_,_ in pending):continue
        source=select_views(dataset[index],torch.arange(8))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            prepared,native,parity=prepare_variants(model,source,device)
        locked_identity(root/f'{index:03d}'/'parity.json',parity)
        for variant in VARIANTS:
            for seed in SEEDS:
                if (index,variant,seed) not in pending:continue
                positive=source['stock_condition'].to(device)
                wrapper=ResidualCFG(model,positive,prepared[variant])
                gen=torch.Generator(device=device).manual_seed(seed*1000003+index*1009)
                noise=torch.randn((1,8,16,16,16),device=device,generator=gen)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    z=sampler.sample(wrapper,noise,cond=positive,neg_cond=torch.zeros_like(positive),
                        steps=25,cfg_strength=5.,cfg_interval=[.5,1.],rescale_t=3.,verbose=False).samples
                    logits=decoder(z.to(next(decoder.parameters()).dtype))
                finite=bool(torch.isfinite(z).all() and torch.isfinite(logits).all())
                coords=torch.argwhere(logits>0)[:,[0,2,3,4]].cpu().numpy().astype(np.int32) if finite else np.empty((0,4),np.int32)
                row={'uid':source['uid'],'object_uid':source['object_uid'],'index':index,'seed':seed,'adapter':adapter,'variant':variant,
                    'gt':source['target_mesh_binding'],'success':finite and len(coords)>0,
                    'positive_calls':wrapper.positive_calls,'negative_calls':wrapper.negative_calls}
                if variant==parity['native_variant']:
                    previous=read(CELL/'evaluations'/f'occluded_{adapter}'/f'{index:03d}'/f'{seed}_ss.json')
                    check_record(previous,source['uid'],seed)
                    with np.load(verify(previous['coords'])) as data:
                        if not np.array_equal(coords,data['coords']):
                            raise ValueError('Native SS replay differs from frozen CellQuery result; review before interpreting interventions')
                    row['native_replay_exact']=True
                    row['native_reference']=binding(CELL/'evaluations'/f'occluded_{adapter}'/f'{index:03d}'/f'{seed}_ss.json')
                if row['success']:row['occupancy']=occupancy_metrics(coords,source['target_coords'].numpy())
                else:row['failure']='empty or nonfinite SS support'
                folder=root/f'{index:03d}'/variant
                folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(folder/f'{seed}.npz',coords=coords)
                row['coords']=binding(folder/f'{seed}.npz')
                write(folder/f'{seed}.json',row)
                added+=1
                print(json.dumps({'adapter':adapter,'variant':variant,'index':index,'seed':seed,'success':row['success'],'iou':row.get('occupancy',{}).get('iou')}),flush=True)
                if limit is not None and added>=limit:return
    write(root/'complete.json',identity)


def summarize():
    result={'scope':'4 post-hoc diagnostic training objects, fixed before intervention sampling; no heldout or automatic success claim','adapters':{}}
    dataset=SceneDataset(PILOT/'occluded16.json',require_labels=False)
    for adapter in ADAPTERS:
        root=OUT/'ss'/adapter
        assert read(root/'complete.json')==read(root/'identity.json')
        records={}
        for variant in VARIANTS:
            rows=[]
            for i in INDICES:
                for seed in SEEDS:
                    r=read(root/f'{i:03d}'/variant/f'{seed}.json')
                    check_record(r,dataset.rows[i]['uid'],seed)
                    rows.append(r)
            records[variant]=rows
        contrasts={}
        for variant in VARIANTS[1:]:
            rows=[{**r,'mode':mode,'surface':{k:r['occupancy'][k] for k in ['iou','precision','recall','f1']} if r['success'] else {}}
                for key,mode in [('predicted','parent'),(variant,'candidate')] for r in records[key]]
            contrasts[variant+'_vs_predicted']=(aggregate_mesh_records(rows,[dataset.rows[i]['uid'] for i in INDICES],SEEDS)
                if all(r['success'] for r in rows) else {'failures':[r for r in rows if not r['success']]})
            if 'all_meshes_successful' in contrasts[variant+'_vs_predicted']:
                contrasts[variant+'_vs_predicted']['all_ss_successful']=contrasts[variant+'_vs_predicted'].pop('all_meshes_successful')
        result['adapters'][adapter]={'records':records,'comparisons':contrasts}
    write(OUT/'ss_summary.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['sample','summary']);p.add_argument('--adapter',choices=ADAPTERS);p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.action=='summary':summarize()
    elif a.adapter:sample(a.adapter,a.limit)
    else:p.error('--adapter required')
