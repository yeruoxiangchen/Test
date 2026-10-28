"""Data-volume policy and small, immutable experiment receipts."""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = Path('/data/zjr/pose_aligned_layered_reconstruction')
OUT = DATA / 'outputs/hssd_generalization_20261027'
DATASET = DATA / 'datasets/hssd_generalization_v1'
RAW = DATASET / 'raw'
LEGACY = ROOT / 'datasets/scene_occlusion_pilot_20261020'
PYTHON = '/home/zjr/anaconda3/envs/reconviagen/bin/python'
BLENDER = '/tmp/blender-3.0.1-linux-x64/blender'
MODULE = 'pose_aligned_layered_reconstruction.experiments.hssd_generalization_20261027'
REPO = 'hssd/hssd-hab'
REVISION = '4369cb9876214c7fbebcf552eb532380e4d287e4'
SEED = 20261027
SCENE_COUNTS = {'train': 16, 'val': 4, 'test': 4}
TARGETS_PER_SCENE = 8
CANDIDATES_PER_SCENE = 32
DEVELOPMENT_SCENES = {'102343992', '102344022', '102344049'}
CATEGORIES = {'chair', 'table', 'bench', 'stool', 'couch', 'bed', 'cabinet',
              'chest_of_drawers', 'wardrobe', 'shelves', 'stand', 'filing_cabinet'}


def now(): return datetime.now(timezone.utc).isoformat()
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def rank(value): return hashlib.sha256((str(SEED) + ':' + str(value)).encode()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def data_path(path):
    path = Path(path).absolute()
    if not path.is_relative_to(DATA) or not path.resolve().is_relative_to(DATA.resolve()):
        raise ValueError('Dataset/output/cache writes must be below ' + str(DATA))
    return path


def write(path, value):
    path = data_path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def locked(path, value):
    path = data_path(path)
    if path.exists():
        if read(path) != value: raise ValueError('Immutable receipt changed: ' + str(path))
    else: write(path, value)


def sha(path):
    path = Path(path)
    if path.stat().st_size > 1024 * 1024 and not path.resolve().is_relative_to(DATA):
        raise ValueError('Do not rehash large home inputs: ' + str(path))
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''): h.update(chunk)
    return h.hexdigest()


def binding(path):
    path = Path(path).absolute()
    return {'path': str(path), 'sha256': sha(path)}


def verify(row):
    path = Path(row['path'])
    if sha(path) != row['sha256']: raise ValueError('Changed source: ' + str(path))
    return path


def environment():
    env = dict(os.environ)
    for key, suffix in {'TMPDIR': 'tmp', 'TORCH_HOME': 'cache/torch', 'HF_HOME': 'cache/huggingface',
                        'HF_HUB_CACHE': 'cache/huggingface/hub', 'TORCH_EXTENSIONS_DIR': 'cache/torch_extensions',
                        'MPLCONFIGDIR': 'cache/matplotlib', 'NUMBA_CACHE_DIR': 'cache/numba',
                        'XDG_CACHE_HOME': 'cache/xdg', 'BLENDER_USER_CONFIG': 'cache/blender/config',
                        'BLENDER_USER_SCRIPTS': 'cache/blender/scripts', 'BLENDER_USER_DATAFILES': 'cache/blender/datafiles'}.items():
        path = DATA / suffix; data_path(path).mkdir(parents=True, exist_ok=True); env[key] = str(path)
    env.update(PYTHONDONTWRITEBYTECODE='1', MKL_THREADING_LAYER='GNU', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return env


def event(stage, **fields):
    row = {'time': now(), 'stage': stage, **fields}; write(OUT / 'progress.json', row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def plan():
    row = read(OUT / 'candidate_plan.json')
    if row['identity'] != digest(row['plan']): raise ValueError('Candidate plan changed')
    return row
