"""Task-local storage, provenance and runtime configuration."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
ROOT = Path('/data/zjr/pose_conditioned_geometry_repair')
DATA = ROOT / 'datasets/native_objaverse_v1'
OUT = ROOT / 'outputs/round1_20261029'
PYTHON = '/home/zjr/anaconda3/envs/trellis2/bin/python'
SOURCE = Path('/data/zjr/yxc_genrecon2_objaverse5k_targets_v1')
MODEL_PATHS = {
    'ss_flow': '/data/zjr/trellis2_geometry_repair/models/ckpts/ss_flow_img_dit_1_3B_64_bf16',
    'shape_flow': '/data/zjr/trellis2_geometry_repair/models/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16',
    'ss_encoder': '/data/zjr/pose_constrained_trellis2/models/ss_enc_conv3d_16l8_fp16',
    'ss_decoder': '/data/zjr/trellis2_geometry_repair/models/ckpts/ss_dec_conv3d_16l8_fp16',
    'shape_encoder': '/data/zjr/models/microsoft_TRELLIS.2-4B_shape_vae/ckpts/shape_enc_next_dc_f16c32_fp16',
    'shape_decoder': '/data/zjr/models/microsoft_TRELLIS.2-4B_shape_vae/ckpts/shape_dec_next_dc_f16c32_fp16',
    'dino': '/data/zjr/pose_constrained_trellis2/models/dinov3',
}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value, immutable=False):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Runtime writes must remain under ' + str(ROOT))
    if immutable and path.exists():
        if read(path) != value:
            raise ValueError('Immutable record changed: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def environment():
    result = dict(os.environ)
    for key, folder in {
        'TMPDIR': 'tmp', 'TORCH_HOME': 'cache/torch', 'HF_HOME': 'cache/huggingface',
        'HF_HUB_CACHE': 'cache/huggingface/hub', 'TORCH_EXTENSIONS_DIR': 'cache/extensions',
        'MPLCONFIGDIR': 'cache/matplotlib', 'NUMBA_CACHE_DIR': 'cache/numba',
        'XDG_CACHE_HOME': 'cache/xdg', 'TRITON_CACHE_DIR': 'cache/triton',
        'CUDA_CACHE_PATH': 'cache/cuda', 'PIP_CACHE_DIR': 'cache/pip',
    }.items():
        p = ROOT / folder
        p.mkdir(parents=True, exist_ok=True)
        result[key] = str(p)
    (ROOT / 'cache/flex_gemm').mkdir(parents=True, exist_ok=True)
    result.update(PYTHONDONTWRITEBYTECODE='1', TRELLIS2_ROOT=str(REPO / 'TRELLIS2'),
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                  PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                  FLEX_GEMM_AUTOTUNE_CACHE_PATH=str(ROOT / 'cache/flex_gemm/autotune_cache.json'),
                  PYTHONPATH=':'.join([str(REPO), str(REPO / 'TRELLIS2'),
                                      '/data/zjr/pose_constrained_trellis2/packages']))
    return result


def source_identity():
    files = sorted(p for p in HERE.rglob('*') if p.is_file() and
                   p.suffix in {'.py', '.sh', '.md', '.json'} and '__pycache__' not in p.parts)
    value = {str(p.relative_to(REPO)): sha(p) for p in files}
    return {'identity': digest(value), 'files': value}


@lru_cache(maxsize=1)
def plan():
    p = read(DATA / 'plan.json')
    if digest(p['plan']) != p['identity']:
        raise ValueError('Selection plan identity changed')
    return p


def sample_rows():
    candidate=DATA/'formal_candidates.json'
    if candidate.exists():
        value=read(candidate)
        if digest(value['plan'])!=value['identity']:
            raise ValueError('Formal candidate pool changed')
        return value['plan']['samples']
    return plan()['plan']['samples']
