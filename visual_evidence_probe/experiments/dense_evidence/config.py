from pathlib import Path
import os

from visual_evidence_probe.experiments.editor_support.config import setup
from visual_evidence_probe.experiments.local_repair.config import read, write, digest, now

STORE = Path('/data/zjr/visual_evidence_probe')
OUT = STORE / 'outputs/dense_evidence_20261027'
ROUND6 = STORE / 'outputs/geometry_evidence_20261027'
LOCALIZATION = STORE / 'outputs/visible_error_localization_20261026'
OMNI = Path('/data/zjr/omniobject3d_reconviagen_style_omni200_20cat_upright_random4_20260824_v2')
ARMS = ('sparse_legacy', 'sparse_observed', 'dense_legacy', 'dense_observed')
PARAMS = dict(side=256, seeds=[42, 43], steps=80, checkpoints=[0, 20, 40, 80],
    normalized_feature_cap=.5, lr=.03, static_loss_scale=1024., depth_scale=.03,
    feature_prior_weight=.0001, local_depth_weight=1., local_mask_weight=.2,
    local_coverage_weight=.2, protect_weight=1., protect_normal_weight=.1,
    region_fraction=.20, region_dilation=2, active_radius_voxels=2.,
    max_region_points=512, max_sparse_constraints=512,
    depth_resolution=504, dense_confidence_quantile=.40, dense_min_references=1,
    dense_keep_sparse_anchors=True,
    dense_reprojection_pixels=2., dense_relative_z_tolerance=.015,
    dense_absolute_z_tolerance=.02, min_valid_depth_pixels=16,
    max_outside_depth_mae=.25/256, max_outside_depth_p95=1/256,
    observed_mask_tolerance=.002, mask_min_gain=.0005, validation_min_gain=.0001,
    max_added_small_component_area_fraction=.002,
    small_component_area_fraction=.0005, min_validation_pixels=8,
    metric_samples=50000, novel_views=8, random_seed=20261027,
    validation_object_count=16, dev_indices=list(range(4, 12)), validation_indices=list(range(12, 28)),
    real_indices=[40, 41, 42])


def data_folder(index): return OUT / 'data' / f'{index:03d}'


def folder(index, seed=None):
    path = OUT / 'objects' / f'{index:03d}'
    return path if seed is None else path / str(seed)


def registry(): return read(OUT / 'registry.json')


def emit(index, seed, stage, **kw):
    value = dict(time=now(), index=index, seed=seed, stage=stage, **kw)
    write(folder(index, seed) / 'status.json', value)
    print(value, flush=True)


def register(stage):
    files = sorted(Path(__file__).parent.glob('*.py')) + sorted(Path(__file__).parent.glob('*.md')) + [
        Path(__file__).parents[2] / 'run_dense_evidence.sh']
    value = dict(format='dense_evidence.v1', time=now(), stage=stage, params=PARAMS,
        sources=[dict(path=str(p), sha256=digest(p)) for p in files if p.exists()],
        weights='frozen original30K SS/SLat and frozen stock mesh decoder; separate frozen DA3-LARGE-1.1',
        gt_policy='all candidates and selections sealed before GT is loaded; no GT depth or geometry in inference',
        execution=dict(cuda_launch_blocking=os.environ.get('CUDA_LAUNCH_BLOCKING','0')),
        physical_io_root=str(STORE), output=str(OUT))
    write(OUT / 'source_snapshots' / f'{stage}_{now().replace(":", "-")}.json', value)
    return value
