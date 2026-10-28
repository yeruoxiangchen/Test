import copy
import os
import sys
from pathlib import Path

from visual_evidence_probe.experiments.local_repair.config import read, write, digest, now

STORE = Path('/data/zjr/visual_evidence_probe')
OUT = STORE / 'outputs/failure_repair_20261028'
PRIOR = STORE / 'outputs/dense_evidence_20261027'
OMNI = Path('/data/zjr/omniobject3d_reconviagen_style_omni200_20cat_upright_random4_20260824_v2')
PARAMS = dict(indices=list(range(100, 116)), seeds=[42, 43], objects=16,
              selection_seed=20261028, bootstrap_seed=20261028,
              bad_depth_threshold=1 / 256, material_cd_relative_gain=.01,
              material_bad_depth_relative_gain=.05, input_depth_scale=.03)


def folder(index, seed=None):
    p = OUT / 'objects' / f'{index:03d}'
    return p if seed is None else p / str(seed)


def data_folder(index):
    return OUT / 'data' / f'{index:03d}'


def registry():
    return read(OUT / 'registry.json')


def setup(gpu=None):
    from visual_evidence_probe.experiments.editor_support.config import setup as configure
    configure(gpu)


def register(stage):
    experiment_root = Path(__file__).parents[1]
    sources = []
    for name in ('failure_repair', 'dense_evidence', 'editor_support', 'geometry_evidence', 'local_repair'):
        sources.extend((experiment_root / name).glob('*.py'))
    sources += [Path(__file__).with_name('PROTOCOL.md'), Path(__file__).parents[2] / 'run_failure_repair.sh']
    value = dict(time=now(), stage=stage, params=PARAMS,
                 sources=[dict(path=str(p), sha256=digest(p)) for p in sorted(set(sources)) if p.exists()],
                 method='frozen round7 dense evidence + observed-policy complete pipeline',
                 model='original30K SS/SLat, fixed decoder; per-object final-SLat editing',
                 gt_policy='all input-only decisions for both seeds sealed before GT evaluation',
                 physical_io_root=str(STORE), output=str(OUT),
                 cuda_launch_blocking=os.environ.get('CUDA_LAUNCH_BLOCKING', '0'))
    write(OUT / 'source_snapshots' / f'{stage}_{now().replace(":", "-")}.json', value)
    return value


def bind_engine():
    """Bind inherited helpers to round10 in this child process only.

    No old source/config/result is edited. Bind before importing stage modules;
    reject late binding so copied OUT constants cannot write to a prior round.
    """
    from visual_evidence_probe.experiments.dense_evidence import config as engine
    prefix = 'visual_evidence_probe.experiments.dense_evidence.'
    unexpected = [name for name in sys.modules if name.startswith(prefix) and name != prefix + 'config']
    if unexpected and engine.OUT != OUT:
        raise RuntimeError(f'engine must be bound before helper imports: {unexpected}')
    engine.OUT = OUT
    engine.PARAMS = copy.deepcopy(engine.PARAMS)
    engine.PARAMS.update(dev_indices=[], validation_indices=PARAMS['indices'], real_indices=[],
                         validation_object_count=PARAMS['objects'])
    engine.ARMS = ('full',)
    engine.register = register
    if engine.folder(100, 42) != folder(100, 42):
        raise RuntimeError('output binding mismatch')
    return engine
