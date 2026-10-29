from pathlib import Path
from contextlib import contextmanager
import fcntl
import hashlib
import os

from ...config import ROOT,STORE,MODELS,OUT as BASE,read,write,now

PROFILE=os.environ.get('T2_LOCAL_PROFILE','v2')
if PROFILE not in ('v1','v2'):raise ValueError('unknown local regeneration profile')
OUT=STORE/('outputs/local_regeneration_20261029' if PROFILE=='v1' else 'outputs/local_regeneration_v2_20261029')
DIAGNOSTIC=STORE/'outputs/repair_diagnostic_20261028'
ARMS=('shape','adaptive')
PARAMS=dict(indices=[5,6,10,11],seeds=[42,43],side=256,grid=32,
  shape_times=[.35,.75],proposal_seeds=[101,202],ss_times=[.65,1.],
  flow_steps=12,sigma_min=1.e-5,buffer_cells=2,max_core_cells=1024,
  depth_floor=.005,photo_max_cost=.30,photo_min_gain=.015,photo_min_std=.02,
  consistency_depth=.02,min_trusted_pixels=16,mask_tolerance=.002,
  registration_steps=30,registration_lr=.025,registration_finalists=2,
  ss_guidance_steps=6,ss_guidance_lr=.06,ss_guidance_cap=1.,
  min_mask_gain=.001,min_depth_gain=.05,photo_tolerance=.002,
  max_added_small_area=.003,max_surface_ratio=1.35)
if PROFILE=='v2':
    PARAMS.update(profile='voxel_aware_anchored_v2',voxel_aware=True,anchor_support=True,
                  anchor_min_rays=3,mask_deficit_fraction=.05,local_depth_scoring=True)


def case(i,s):return OUT/'objects'/f'{i:03d}'/str(s)
def source(i,s):return BASE/'objects'/f'{i:03d}'/str(s)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup():
    import sys
    from ...config import T2_ROOT
    sys.dont_write_bytecode=True
    if str(T2_ROOT) not in sys.path:sys.path.insert(0,str(T2_ROOT))
    OUT.mkdir(parents=True,exist_ok=True)


@contextmanager
def lock_case(i,s):
    root=case(i,s);root.mkdir(parents=True,exist_ok=True)
    with (root/'worker.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def snapshot():
    paths=sorted(Path(__file__).parent.glob('*.py'))+sorted(Path(__file__).parent.glob('*.md'))
    paths += [ROOT/'run_local_regeneration.sh',ROOT/'runtime.py',ROOT/'geometry.py',ROOT/'config.py']
    records=[dict(path=str(p),sha256=digest(p)) for p in paths]
    identity=hashlib.sha256(''.join(r['sha256'] for r in records).encode()).hexdigest()
    archive=OUT/'source_archives'/identity
    for p in paths:
        dest=archive/p.relative_to(ROOT.parent);dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():dest.write_bytes(p.read_bytes())
    write(OUT/'source_snapshot.json',dict(time=now(),params=PARAMS,sources=records,archive=str(archive)))


def emit(i,s,stage,**kwargs):
    value=dict(time=now(),index=i,seed=s,stage=stage,**kwargs)
    write(case(i,s)/'status.json',value);print(value,flush=True)


def save_tensor(path,value):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    torch.save(value,temporary);temporary.replace(path)
