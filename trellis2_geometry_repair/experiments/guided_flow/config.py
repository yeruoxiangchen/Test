from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib
import os

from ...config import ROOT,STORE,MODELS,OUT as BASE,read,write,now,T2_ROOT

PROFILE=os.environ.get('T2_GUIDED_PROFILE','fixed')
if PROFILE not in ('initial','fixed'):raise ValueError('unknown guided-flow profile')
OUT=STORE/('outputs/guided_flow_v3_20261029' if PROFILE=='initial' else 'outputs/guided_flow_v3_fixed_20261029')
V2=STORE/'outputs/local_regeneration_v2_20261029'
METHOD=('prior','budget_prior','input')
DIAGNOSTIC=('paired_input','oracle')
ARMS=METHOD+DIAGNOSTIC
PARAMS=dict(indices=[5,6,10,11],seeds=[42,43],side=256,resolution=512,
    support='ss_t065_n101',flow_steps=12,sigma_min=1.e-5,
    proposal_seeds=[101,202],bank_seeds=[101,202,303,404,505,606,707,808],
    guide_steps=[9,10,11],guide_lr=.08,guide_cap=.25,backtracking=[1.,.5],
    max_queries=512,field_anchors=4096,subdivision_anchors=512,min_queries=16,
    point_scale=.02,free_depth_offsets=[.02,.05],loss_scale=512.,offload_flow=True,
    weights=dict(depth=1.,silhouette=.3,surface=.25,subdivision=.15,free=.1,
                 protect=1.,field_protect=.05,prior=.01))
if PROFILE=='fixed':PARAMS.update(objective_fixed_support=True,missing_margin=1.)

def case(i,s):return OUT/'objects'/f'{i:03d}'/str(s)
def previous(i,s):return V2/'objects'/f'{i:03d}'/str(s)
def original(i,s):return BASE/'objects'/f'{i:03d}'/str(s)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def setup():
    import sys
    sys.dont_write_bytecode=True
    if str(T2_ROOT) not in sys.path:sys.path.insert(0,str(T2_ROOT))
    OUT.mkdir(parents=True,exist_ok=True)

def save(path,value):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp');torch.save(value,temp);temp.replace(path)

@contextmanager
def lock_case(i,s):
    p=case(i,s);p.mkdir(parents=True,exist_ok=True)
    with (p/'worker.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)

def emit(i,s,stage,**kwargs):
    value=dict(time=now(),index=i,seed=s,stage=stage,**kwargs)
    write(case(i,s)/'status.json',value);print(value,flush=True)

def snapshot():
    paths=sorted(Path(__file__).parent.glob('*.py'))+sorted(Path(__file__).parent.glob('*.md'))
    paths += [ROOT/'run_guided_flow.sh',ROOT/'runtime.py',ROOT/'compat.py',ROOT/'geometry.py']
    paths += sorted((ROOT/'experiments/local_regeneration').glob('*.py'))
    records=[dict(path=str(p),sha256=digest(p)) for p in paths]
    identity=hashlib.sha256(''.join(r['sha256'] for r in records).encode()).hexdigest()
    dest=OUT/'source_archives'/identity
    for p in paths:
        target=dest/p.relative_to(ROOT.parent);target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.write_bytes(p.read_bytes())
    write(OUT/'source_snapshot.json',dict(time=now(),params=PARAMS,archive=str(dest),sources=records))
