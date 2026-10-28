"""Preserve a pre-training import failure and verify its numerical-no-op repair."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[3]))
MODULE='pose_aligned_layered_reconstruction.experiments.hssd_depth_confidence_20261026'
c=importlib.import_module(MODULE+'.common')
r=importlib.import_module(MODULE+'.runner')
ARCHIVE=c.OUT/'repairs/hash_import_20261026'


def prepare():
    with r.open_lock():
        state=c.read(c.OUT/'status.json')
        if state['status']!='failed' or state['stage']!='smoke':raise ValueError('Expected stopped pre-training smoke failure')
        if (c.OUT/'train').exists() or (c.OUT/'smoke').exists() or (c.OUT/'evaluations').exists():
            raise ValueError('Refuse reset after model artifacts have been created')
        runtime=c.read(c.OUT/'evaluation_identity.json')
        expected=next(v for v in runtime['sources'] if v['path']==str(c.HERE/'common.py'))
        source=(c.HERE/'common.py').read_bytes()
        if hashlib.sha256(source).hexdigest()!=expected['sha256']:raise ValueError('Archive original source before editing')
        if ARCHIVE.exists():raise ValueError('Repair archive already exists; inspect instead of overwriting')
        ARCHIVE.mkdir(parents=True)
        (ARCHIVE/'common_before.py').write_bytes(source)
        for name in ('evaluation_identity.json','plan.json','preflight.json','status.json','progress.json'):
            (ARCHIVE/name).write_bytes((c.OUT/name).read_bytes())
        log=Path(state['log']);(ARCHIVE/'failed_smoke.log').write_bytes(log.read_bytes())
        (c.OUT/'evaluation_identity.json').rename(ARCHIVE/'evaluation_identity_original.json')
        c.write(ARCHIVE/'repair.json',{'time':c.now(),'old_runtime_identity':c.digest(runtime),
            'old_source':expected,'old_plan_identity':c.plan()['identity'],
            'reason':'tensor_hash aliased from a standalone-imported prior common; relative import lacks package',
            'scope':'change hash helper import resolution only; no formal training/sampling/zero gate artifacts existed',
            'all_old_files_preserved':True,'old_smoke_passed':False})
        print(json.dumps({'archived':str(ARCHIVE),'old_runtime_identity':c.digest(runtime)}))


def verify_repair():
    import torch
    original=importlib.import_module('pose_aligned_layered_reconstruction.experiments.hssd_cell_query_20261023.common').tensor_hash
    cases=[]
    for dtype in (torch.float32,torch.bfloat16,torch.float16,torch.int32,torch.bool):
        for shape in ((),(7,),(3,4),(0,)):
            value=torch.arange(1 if not shape else __import__('math').prod(shape)).reshape(shape).to(dtype)
            state={'probe':value}
            if c.tensor_hash(state)!=original(state):raise ValueError('Hash changed')
            cases.append({'dtype':str(dtype),'shape':list(shape)})
    state={'x':torch.arange(12).reshape(3,4).T,'y':torch.tensor([1.],requires_grad=True)}
    if c.tensor_hash(state)!=original(state):raise ValueError('Hash changed on noncontiguous/grad tensors')
    current=c.freeze_runtime();old=c.read(ARCHIVE/'evaluation_identity.json');new=c.read(c.OUT/'evaluation_identity.json')
    old_sources={v['path']:v['sha256'] for v in old['sources']};new_sources={v['path']:v['sha256'] for v in new['sources']}
    changed=[p for p in old_sources if old_sources[p]!=new_sources.get(p)]
    if changed!=[str(c.HERE/'common.py')] or set(old_sources)!=set(new_sources):raise ValueError('Unrelated runtime sources changed')
    if any(old[k]!=new[k] for k in old if k!='sources'):raise ValueError('Numerical/input contract changed')
    if torch.cuda.is_initialized():raise ValueError('CPU check initialized CUDA')
    c.write(ARCHIVE/'verification.json',{'time':c.now(),'passed':True,'new_runtime_identity':current,
        'old_runtime_identity':c.digest(old),'same_plan':True,'only_source_changed':changed,
        'hash_parity_cases':len(cases)+1,'cuda_initialized':False,'scope':'unchanged existing hash implementation; import resolution repaired'})
    print(json.dumps({'passed':True,'hash_parity_cases':len(cases)+1,'new_runtime_identity':current}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','verify']);args=parser.parse_args()
    r.configure_environment()
    if args.action=='prepare':prepare()
    else:verify_repair()
