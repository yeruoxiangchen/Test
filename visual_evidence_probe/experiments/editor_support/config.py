import os
from pathlib import Path
import sys

from visual_evidence_probe.experiments.local_repair.config import read, write, digest, now

STORE = Path('/data/zjr/visual_evidence_probe')
OUT = STORE / 'outputs/editor_support_diagnosis_20261026'
BASE = STORE / 'outputs/visible_error_localization_20261026'
PRIOR = STORE / 'outputs/local_slat_repair_20261026'
INPUT = STORE / 'outputs/local_geometry_repair_cropped_20261026'
PARAMS = dict(recovery_indices=[4, 6], indices=list(range(4, 12)), seeds=[42, 43],
    side=256, evaluation_side=256, recovery_steps=80, formal_steps=40,
    recovery_checkpoints=[0, 20, 40, 80], formal_checkpoints=[0, 10, 20, 40],
    lr=.03, normalized_feature_cap=.12, perturb_normalized_rms=.035,
    perturb_normalized_max=.075, active_radius_voxels=2.,
    max_constraints=512, max_added_tokens=128, max_added_fraction=.20,
    initialization_steps=20, initialization_lr=.03, initialization_feature_cap=.5,
    initialize_mask_regression=.005, initialize_depth_mae=.25 / 256,
    initialize_depth_p95=1 / 256, initialize_normal_degrees=2.,
    local_depth_scale=.03, feature_prior_weight=.0001,
    protect_weight=1., proxy_field_weight=.2,
    validation_min_pixels=8, validation_min_gain=.0001,
    max_silhouette_regression=.005, random_seed=20261026,
    recovery_min_depth_reduction=.50, recovery_min_success_cases=3)


def setup(gpu=None):
    if gpu is not None: os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cache = STORE / 'cache'
    values = dict(TMPDIR=STORE / 'tmp', XDG_CACHE_HOME=cache, TORCH_HOME=cache / 'torch',
        HF_HOME=cache / 'huggingface', HF_HUB_CACHE=cache / 'huggingface/hub',
        TORCH_EXTENSIONS_DIR=cache / 'torch_extensions', MPLCONFIGDIR=cache / 'matplotlib',
        TRITON_CACHE_DIR=cache / 'triton', NUMBA_CACHE_DIR=cache / 'numba',
        CUDA_CACHE_PATH=cache / 'cuda', CUMM_CACHE_PATH=cache / 'cumm',
        TORCHINDUCTOR_CACHE_DIR=cache / 'torchinductor', CUDA_HOME='/usr/local/cuda-12.6',
        TORCH_CUDA_ARCH_LIST='8.6', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        SPCONV_ALGO='native', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        MKL_NUM_THREADS='2', MAX_JOBS='2', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    for key, value in values.items(): os.environ[key] = str(value)
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    sys.dont_write_bytecode = True
    for folder in ('tmp', 'cache/torch_extensions', 'cache/matplotlib'): (STORE / folder).mkdir(parents=True, exist_ok=True)


def register(stage):
    sources = sorted(Path(__file__).parent.glob('*.py')) + [Path(__file__).with_name('PROTOCOL.md')]
    value = dict(format='editor_support_diagnosis.v1', params=PARAMS, stage=stage,
        sources=[dict(path=str(p), sha256=digest(p)) for p in sources if p.exists()],
        original_model_binding=read(PRIOR / 'model_binding.json'),
        scope='Exploratory reuse of inspected eight Omni scanned objects with four rendered input views.',
        recovery='Four decoder-generated reachable targets; target latent is never optimized against or exposed to optimizer.',
        factorial='original vs locally expanded SS crossed with corner-SDF proxy vs full decoded mesh geometry supervision.',
        gt_policy='No scan GT in target construction, initialization, gradients, checkpoint selection or SS expansion; GT only final evaluation.',
        support_policy='Expansion plus baseline-preserving initialization is a joint intervention; all failures and step0 drift retained.',
        validation='Input-only visibility chooses heldout view; constraints from other views only; no validation support means unable.',
        no_model_training=True, physical_io_root=str(STORE))
    p = OUT / 'source_snapshots' / f'{stage}_{now().replace(":", "-")}.json'
    write(p, value)
    return value


def folder(index, seed, phase='factorial'):
    return OUT / phase / f'{index:03d}' / str(seed)


def emit(stage, **kw):
    r = dict(time=now(), stage=stage, **kw)
    write(OUT / 'status.json', r)
    print(r, flush=True)
