"""Storage-isolated, resumable native T2 scene preparation."""
import hashlib
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = Path('/data/zjr/pose_constrained_trellis2')
OUT = ROOT / 'outputs/hssd_scene_20261028'
DATA = ROOT / 'datasets/hssd_scene_native_v1'
SOURCE = Path('/data/zjr/pose_aligned_layered_reconstruction/datasets/hssd_generalization_v1')
PYTHON = '/home/zjr/anaconda3/envs/trellis2/bin/python'
MODULE = 'pose_constrained_trellis2.experiments.hssd_scene_20261028'
sys.path.insert(0,str(ROOT/'packages'))
sys.path.insert(0,str(REPO/'TRELLIS2'))

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()
def write(p,x):
    p=Path(p)
    if not p.resolve().is_relative_to(ROOT): raise ValueError('Writes must stay in project /data')
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f'.{os.getpid()}.tmp')
    t.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)
def immutable(p,x):
    if Path(p).exists():
        if read(p)!=x: raise ValueError('Frozen identity changed: '+str(p))
    else: write(p,x)
def environment():
    env=dict(os.environ)
    for key,suffix in {'TMPDIR':'tmp','TORCH_HOME':'cache/torch','HF_HOME':'cache/huggingface',
        'HF_HUB_CACHE':'cache/huggingface/hub','TORCH_EXTENSIONS_DIR':'cache/extensions',
        'MPLCONFIGDIR':'cache/matplotlib','NUMBA_CACHE_DIR':'cache/numba','XDG_CACHE_HOME':'cache/xdg',
        'PIP_CACHE_DIR':'cache/pip','TRITON_CACHE_DIR':'cache/triton','CUDA_CACHE_PATH':'cache/cuda',
        'FLEX_GEMM_CACHE_DIR':'cache/flex_gemm'}.items():
        p=ROOT/suffix;p.mkdir(parents=True,exist_ok=True);env[key]=str(p)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        FLEX_GEMM_AUTOTUNE_CACHE_PATH=str(ROOT/'cache/flex_gemm/autotune_cache.json'),
        PYTHONPATH=str(ROOT/'packages')+':'+str(REPO)+':'+str(REPO/'TRELLIS2'))
    return env
def models(): return read(OUT/'models.json')
def plan():
    d=read(OUT/'plan.json')
    if digest(d['plan'])!=d['identity']: raise ValueError('Plan identity mismatch')
    return d
