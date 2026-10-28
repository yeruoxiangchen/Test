"""Query-level head substitution using the unchanged production feature builder.

After validity-weighted interpolation, substitute one sampled component at a time.
This is a diagnostic intervention, not an inference-capable model. The temporary
patch is scoped to one serial feature call and always restores the function.
"""
from unittest.mock import patch
import torch
from ... import layer_matching
from ...v2.conditioning import query_geometry
from ...projection import canonical_query_points
from ..hssd_full_diagnostic_20261021.diagnose import parts, replace_parts, oracle_prediction
from ..hssd_cell_query_20261023.model import observations
from .common import VARIANTS


@torch.no_grad()
def features(sample, prediction, oracle, variant, points=None):
    if variant not in VARIANTS:
        raise ValueError(variant)
    if prediction.depths.shape != oracle.depths.shape:
        raise ValueError('Require equal native prediction/GT rasters')
    if points is None:
        points = canonical_query_points(16, sample['grid_transform'], device=prediction.depths.device)
    pp, projection = parts(sample, prediction, points)
    gp, _ = parts(sample, oracle, points)
    values, tau = replace_parts(pp, gp, variant)
    with patch.object(layer_matching, '_sample_prediction', return_value=values):
        result = query_geometry(sample, prediction, points, absolute_tolerance=tau, relative_tolerance=0.)
    return result.features


@torch.no_grad()
def prepare_variants(model, sample, device):
    # Original native preparation is retained as an independent parity check.
    native = model.prepare(sample, device)
    pred = model.base.predict_layers(observations(sample), device=device, view_indices=None, source='predicted')
    gt = oracle_prediction(sample, device)
    rows = {v: {**native, 'features': features(sample, pred, gt, v)} for v in VARIANTS}
    native_variant = 'predicted' if model.arm == 'predicted_point' else 'oracle'
    error = float((rows[native_variant]['features'] - native['features']).abs().max())
    if error > 1.e-6:
        raise ValueError(f'Production 12-channel feature parity failed: {error}')
    return rows, native, {'native_variant': native_variant, 'feature_max_abs_error': error}
