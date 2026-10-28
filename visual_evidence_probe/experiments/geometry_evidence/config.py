from pathlib import Path

from visual_evidence_probe.experiments.editor_support.config import setup
from visual_evidence_probe.experiments.local_repair.config import read, write, digest, now

STORE = Path('/data/zjr/visual_evidence_probe')
OUT = STORE / 'outputs/geometry_evidence_20261027'
BASE = STORE / 'outputs/visible_error_localization_20261026'
PREVIOUS = STORE / 'outputs/editor_support_diagnosis_20261026'
OLD_INPUT = STORE / 'outputs/local_geometry_repair_cropped_20261026'
ARMS = ('oracle_dense_normal', 'oracle_dense_depth', 'oracle_sparse', 'input_sparse')
PARAMS = dict(indices=list(range(4, 12)), seeds=[42, 43], side=256,
    steps=80, checkpoints=[0, 20, 40, 80], normalized_feature_cap=.5,
    lr=.03, static_loss_scale=1024., local_depth_scale=.03,
    feature_prior_weight=.0001, active_radius_voxels=2.,
    region_radius=.045, region_dilation=2, max_region_points=512,
    novel_views=8, novel_focal=650., surface_samples=50000,
    random_seed=20261027, missing_depth=.1, max_silhouette_regression=.005,
    max_outside_depth_mae=.25 / 256, max_outside_depth_p95=1 / 256,
    validation_min_gain=.0001)


def folder(index, seed=None):
    path = OUT / 'objects' / f'{index:03d}'
    return path if seed is None else path / str(seed)


def emit(index, seed, stage, **kw):
    value = dict(time=now(), index=index, seed=seed, stage=stage, **kw)
    write(folder(index, seed) / 'status.json', value)
    print(value, flush=True)


def register(stage):
    sources = sorted(Path(__file__).parent.glob('*.py')) + [Path(__file__).with_name('PROTOCOL.md'),
        Path(__file__).parents[2] / 'run_geometry_evidence.sh']
    dependencies = [Path(__file__).parents[1] / 'editor_support' / name for name in
        ('data.py', 'mesh_backend.py', 'forward_policy.py', 'objective.py', 'optimize.py', 'pool.py')]
    record = dict(format='geometry_evidence.v1', time=now(), stage=stage,
        params=PARAMS, arms=list(ARMS),
        sources=[dict(path=str(p), sha256=digest(p)) for p in sources + dependencies],
        weights='frozen original 30K latent endpoints and frozen stock mesh decoder',
        scope='same eight inspected Omni objects, two seeds; exploratory development cohort',
        gt_policy='input targets/regions/active rows sealed before GT; oracle arms explicitly privileged; input arm cannot access GT',
        selection='best own training objective among fixed checkpoints; separate input-only deployment gate; novel views never select',
        difference_from_round5='80 steps, feature cap0.5 training std, source-region editable rows shared across all arms; no SS expansion',
        physical_io_root=str(STORE))
    write(OUT / 'source_snapshots' / f'{stage}_{now().replace(":", "-")}.json', record)
    return record
