import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

STORE = Path('/data/zjr/visual_evidence_probe')
BASE = STORE / 'outputs/visible_error_localization_20261026'
OUT = STORE / 'outputs/local_geometry_repair_20261026'
PARAMS = dict(indices=list(range(4, 12)), seeds=[42, 43], train_views=[0, 1, 2],
    validation_view=3, side=256, patch=5, depth_offsets=[round(v * .005, 6) for v in range(-12, 13)],
    min_baseline_degrees=5., max_baseline_degrees=85., visibility_tolerance=.015,
    candidate_visibility_tolerance=.08, min_texture_std=.02, min_ncc_gain=.02,
    max_ncc_cost=.30, pair_depth_agreement=.015, min_support_views=2,
    region_fraction=.20, max_constraints=2048, max_displacement=.04,
    smooth_weight=1., ridge_weight=.05, amplitudes=[0., .25, .5, 1.],
    max_silhouette_regression=.005, validation_min_gain=.0001,
    metrics_side=128, metric_threshold=.02, surface_samples=50000,
    random_seed=20261026)
METHODS = ('automatic', 'random', 'oracle_region', 'oracle_direction')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def physical(path):
    path = Path(path)
    logical = Path('/home/zjr/Tracker/visual_evidence_probe/outputs')
    try:
        return STORE / 'outputs' / path.relative_to(logical)
    except ValueError:
        return path


def folder(index, seed=None):
    p = OUT / 'objects' / f'{index:03d}'
    return p if seed is None else p / str(seed)


def emit(index, seed, stage, **kwargs):
    record = dict(time=now(), index=index, seed=seed, stage=stage, **kwargs)
    write(folder(index, seed) / 'status.json', record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def setup():
    cache = STORE / 'cache'
    env = dict(TMPDIR=STORE / 'tmp', TORCH_HOME=cache / 'torch', HF_HOME=cache / 'huggingface',
        HF_HUB_CACHE=cache / 'huggingface/hub', TORCH_EXTENSIONS_DIR=cache / 'torch_extensions',
        MPLCONFIGDIR=cache / 'matplotlib', XDG_CACHE_HOME=cache, CUDA_HOME='/usr/local/cuda-12.6',
        TORCH_CUDA_ARCH_LIST='8.6', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
        SPCONV_ALGO='native', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    for key, value in env.items():
        os.environ[key] = str(value)
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    os.environ['MAX_JOBS'] = '2'
    sys.dont_write_bytecode = True
    for p in (STORE / 'tmp', cache / 'torch_extensions', cache / 'matplotlib'):
        p.mkdir(parents=True, exist_ok=True)


def protocol():
    sources = sorted(Path(__file__).parent.glob('*.py')) + [Path(__file__).with_name('PROTOCOL.md')]
    value = dict(format='local_geometry_repair.v1', params=PARAMS, methods=list(METHODS),
        sources=[dict(path=str(p), sha256=digest(p)) for p in sources],
        base_protocol=dict(path=str(BASE / 'protocol.json'), sha256=digest(BASE / 'protocol.json')),
        scope='Exploratory reuse of all eight Omni objects already inspected in round3; not independent validation.',
        input_policy='Four views used by original frozen generation/global fit; views0:3 estimate local corrections; view3 selects amplitudes only.',
        gt_policy='No GT in automatic/random proposals or selection. Oracle region/direction are privileged diagnostics, not deployable methods.',
        representation='Fixed-topology bounded vertex deformation first; separate frozen SLat feature optimization with same constraints.',
        acceptance='Fixed held-out-from-edit-estimation input patch NCC + per-view silhouette no-regression; GT evaluation only.',
        unknown='No texture/multi-view support means no proposal. Failures and zero-edit cases retained.',
        physical_io_root=str(STORE))
    p = OUT / 'protocol.json'
    if p.exists() and read(p) != value:
        raise ValueError('repair protocol/source changed; register a separate amendment')
    if not p.exists():
        write(p, value)
    return value
