"""Existing-checkpoint geometry diagnostics. No training or mesh sampling."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from ...contracts import ground_truth_prediction
from ...layer_matching import _sample_prediction
from ...projection import canonical_query_points, resize_sampling_grid
from ...v2.conditioning import query_geometry
from ...v2.contracts import LayeredPredictionV2
from ...v2.dataset import SceneDataset
from ...v2.layer_predictor import GeometryConfig, RayLayerPredictor
from ...v2.losses import resized_labels
from ...v2.ray_geometry import CameraBundle

PACKAGE = Path(__file__).resolve().parents[2]
PILOT = PACKAGE / 'outputs/hssd_pilot_20261020'
OUT = PACKAGE / 'outputs/hssd_full_diagnostic_20261021'
VARIANTS = ('predicted', 'gt_depth', 'gt_existence', 'gt_ownership', 'confidence_one',
            'gt_depth_confidence_one', 'gt_depth_existence_ownership', 'oracle',
            'tolerance_005', 'tolerance_010')


def bind(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return {'path': str(path), 'sha256': h.hexdigest()}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def scalar_stats(value):
    value = value.detach().float().reshape(-1)
    if not value.numel():
        return {'count': 0, 'mean': None}
    if not torch.isfinite(value).all():
        raise ValueError('Nonfinite diagnostic value')
    q = torch.quantile(value, value.new_tensor([.1, .5, .9]))
    return {'count': value.numel(), 'mean': float(value.mean()), 'p10': float(q[0]),
            'median': float(q[1]), 'p90': float(q[2]),
            'fraction_gt_001': float((value > .01).float().mean()),
            'fraction_gt_01': float((value > .1).float().mean())}


def parts(sample, prediction, points):
    cameras = CameraBundle.from_sample(sample, points.device)
    projection = cameras.project(points, prediction.spatial_size[1])
    grid = resize_sampling_grid(projection['image_grid'], image_size=cameras.image_size,
                                map_size=prediction.spatial_size)
    return _sample_prediction(prediction, grid), projection


def replace_parts(pred, gt, variant):
    """Pointwise interventions after interpolation; never mutate model predictions.

    GT depth is substituted only where a GT layer exists (sampled existence>.5),
    avoiding arbitrary oracle invalid-suffix depths. Other heads remain predicted.
    These interventions diagnose support, and are not deployable inference modes.
    """
    d, e, o, c = pred
    gd, ge, go, gc = gt
    gd_existing = torch.where(ge > .5, gd, d)
    tau = .02
    if variant in ('gt_depth', 'gt_depth_confidence_one', 'gt_depth_existence_ownership'):
        d = gd_existing
    if variant in ('gt_existence', 'gt_depth_existence_ownership'):
        e = ge
    if variant in ('gt_ownership', 'gt_depth_existence_ownership'):
        o = go
    if variant in ('confidence_one', 'gt_depth_confidence_one'):
        c = torch.ones_like(c)
    if variant == 'oracle':
        d, e, o, c = gt
    if variant == 'tolerance_005':
        tau = .05
    if variant == 'tolerance_010':
        tau = .1
    if variant not in VARIANTS:
        raise ValueError(variant)
    return (d, e, o, c), tau


def support_from_parts(values, projection, tolerance):
    d, e, o, c = values
    proximity = torch.exp(-(projection['camera_depth'][..., None] - d).abs() / tolerance)
    return (e * c * proximity * o).amax(-1) * projection['valid'].float()


def load_prediction(path, device):
    d = torch.load(path, map_location=device, weights_only=True)
    return LayeredPredictionV2(**d).validate()


def oracle_prediction(sample, device):
    p = ground_truth_prediction(sample).to(device)
    return LayeredPredictionV2(p.depths, p.stop_logits, p.target_logits, p.confidence_logits).validate()


def query_sets(sample, device):
    from scipy.spatial import cKDTree
    coords = sample['target_coords'][:, -3:].long().to(device)
    coarse = torch.unique(coords // 4, dim=0)
    shell = torch.zeros(4096, dtype=torch.bool, device=device)
    shell[coarse[:, 0] * 256 + coarse[:, 1] * 16 + coarse[:, 2]] = True
    p16 = canonical_query_points(16, sample['grid_transform'], device=device)
    p64 = (coords.float() + .5) / 64 - .5
    if sample['grid_transform'] == 'pixal3d_rotation':
        p64 = p64 @ p64.new_tensor(((1., 0., 0.), (0., 0., -1.), (0., 1., 0.))).T
    distances = cKDTree(p64.cpu().numpy()).query(p16.cpu().numpy())[0]
    far = torch.as_tensor(distances > .1, device=device)
    return p16, p64, shell, far


def depth_metrics(sample, pred):
    gt, target, count, valid, mask = resized_labels(sample, pred)
    rank = torch.arange(pred.layer_count, device=gt.device)[None, :, None, None]
    exists = rank < count[:, None]
    target_all = exists & (target > .5)
    selected = target_all & valid
    non_target = exists & (target <= .5)
    preceding = torch.cat((torch.zeros_like(non_target[:, :1]), non_target[:, :-1]), 1).cumsum(1) > 0
    error = (pred.depths - gt).abs()
    output = {'target_supervised_fraction': float((target_all & mask).sum() / target_all.sum().clamp_min(1))}
    for label, selection in [('all_target', selected), ('front_target', selected & (rank == 0)),
                             ('behind_non_target', selected & preceding)]:
        e = error[selection]
        output[label] = scalar_stats(e)
        if len(e):
            output[label].update(fraction_lt_002=float((e < .02).float().mean()),
                                 fraction_lt_005=float((e < .05).float().mean()))
    output['target_confidence'] = scalar_stats(torch.sigmoid(pred.confidence_logits)[selected])
    output['target_confidence_training_target'] = scalar_stats(torch.exp(-error[selected] / .05))
    output['target_ownership'] = scalar_stats(torch.sigmoid(pred.target_logits)[selected])
    output['target_existence'] = scalar_stats(pred.valid_probabilities()[selected])
    return output


@torch.no_grad()
def geometry(args):
    device = torch.device(args.device)
    torch.set_num_threads(2)
    source_paths = [Path(__file__), PACKAGE / 'layer_matching.py', PACKAGE / 'projection.py',
                    PACKAGE / 'v2/layer_predictor.py', PACKAGE / 'v2/conditioning.py',
                    PACKAGE / 'v2/multiview_evidence.py', PACKAGE / 'v2/ray_geometry.py',
                    PACKAGE / 'v2/losses.py', PACKAGE / 'contracts.py', PACKAGE / 'v2/contracts.py',
                    PACKAGE / 'v2/dataset.py']
    identity = {'version': 1, 'sources': [bind(p) for p in source_paths],
                'variants': list(VARIANTS), 'device_type': device.type, 'dtype': 'float32',
                'far_from_surface_threshold': .1, 'aggregation': 'equal object weight',
                'scope': 'All original16 per domain, fixed existing checkpoints, diagnostics only'}
    for domain in args.domains:
        dataset = SceneDataset(PILOT / f'{domain}16.json')
        checkpoint_path = PILOT / f'{domain}_lari/checkpoints/last.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config = GeometryConfig(**checkpoint['geometry_config'])
        model = RayLayerPredictor(config).to(device).eval()
        model.load_state_dict(checkpoint['model_state'], strict=True)
        domain_identity = {**identity, 'manifest': bind(dataset.path), 'checkpoint': bind(checkpoint_path)}
        for index in range(args.start, min(args.end, len(dataset))):
            started = time.time()
            path = OUT / 'geometry' / domain / f'{index:03d}.json'
            if path.exists():
                old = json.loads(path.read_text())
                if old['identity'] != domain_identity:
                    raise ValueError('Existing diagnostic has different sources/settings')
                if bind(old['prediction']['path']) != old['prediction']:
                    raise ValueError('Changed prediction cache')
                continue
            sample = dataset[index]
            cams = CameraBundle.from_sample(sample, device)
            pred = model(raw_patch_features=sample['raw_visual_patch_features'].to(device).float(),
                masked_patch_features=sample['masked_visual_patch_features'].to(device).float(),
                visible_target_mask=sample['visible_target_mask'].to(device).float(),
                intrinsics=cams.intrinsics, extrinsics=cams.w2c, extrinsics_type='w2c',
                camera_forward_sign=cams.forward_sign, image_size=cams.image_size)
            oracle = oracle_prediction(sample, device)
            if pred.depths.shape != oracle.depths.shape:
                raise ValueError('Interventions require equal native GT/prediction rasters')
            p16, p64, shell, far = query_sets(sample, device)
            records = {name: {} for name in VARIANTS}
            parity = {}
            for set_name, points in [('ss16', p16), ('gt64', p64)]:
                pp, projection = parts(sample, pred, points)
                gp, _ = parts(sample, oracle, points)
                for variant in VARIANTS:
                    values, tau = replace_parts(pp, gp, variant)
                    support = support_from_parts(values, projection, tau)
                    max_view = support.amax(0)
                    rows = {'all': scalar_stats(max_view), 'mean_over_views': float(support.mean())}
                    if set_name == 'ss16':
                        rows.update(surface_cells=scalar_stats(max_view[shell]),
                                    other_cells=scalar_stats(max_view[~shell]),
                                    far_from_surface=scalar_stats(max_view[far]))
                    records[variant][set_name] = rows
                    if variant in ('predicted', 'oracle'):
                        actual = query_geometry(sample, pred if variant == 'predicted' else oracle,
                                                points, absolute_tolerance=.02, relative_tolerance=0.).target_support
                        error = float((support - actual).abs().max())
                        if error > 1.e-6:
                            raise ValueError('Diagnostic support differs from production matcher')
                        parity[set_name + '_' + variant] = error
            pred_path = OUT / 'predictions' / domain / f'{index:03d}.pt'
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({key: getattr(pred, key).cpu() for key in
                        ('depths', 'stop_logits', 'target_logits', 'confidence_logits')}, pred_path)
            result = {'identity': domain_identity, 'index': index, 'uid': sample['uid'],
                'sample': dataset.rows[index], 'prediction': bind(pred_path),
                'depth': depth_metrics(sample, pred), 'variants': records,
                'production_matcher_max_abs_error': parity, 'elapsed_seconds': time.time() - started}
            write(path, result)
            print(json.dumps({'domain': domain, 'index': index, 'elapsed': result['elapsed_seconds'],
                  'depth_mae': result['depth']['all_target']['mean'],
                  'surface_coverage': records['predicted']['gt64']['all']['fraction_gt_01']}), flush=True)
        del model, checkpoint
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def summarize(args):
    report = {'scope': 'Post-hoc geometry interventions; not mesh quality or deployable inference', 'domains': {}}
    for domain in ('clean', 'occluded'):
        paths = sorted((OUT / 'geometry' / domain).glob('*.json'))
        rows = [json.loads(p.read_text()) for p in paths]
        if len(rows) != 16 or [r['index'] for r in rows] != list(range(16)):
            raise ValueError('Require complete original16 cohort before summary')
        if any(r['identity'] != rows[0]['identity'] for r in rows):
            raise ValueError('Mixed diagnostic identities')
        def mean(values):
            valid = [v for v in values if v is not None]
            return sum(valid) / len(valid) if valid else None
        summary = {'objects': len(rows), 'target_depth_mae': mean([r['depth']['all_target']['mean'] for r in rows]),
            'target_error_lt_002': mean([r['depth']['all_target']['fraction_lt_002'] for r in rows]),
            'target_confidence': mean([r['depth']['target_confidence']['mean'] for r in rows]),
            'target_supervised_fraction': mean([r['depth']['target_supervised_fraction'] for r in rows]),
            'variants': {}, 'source_records': [bind(p) for p in paths]}
        for variant in VARIANTS:
            summary['variants'][variant] = {}
            for name, set_name, part in [('surface64', 'gt64', 'all'), ('surface16', 'ss16', 'surface_cells'),
                                       ('far16', 'ss16', 'far_from_surface')]:
                summary['variants'][variant][name] = {key: mean([r['variants'][variant][set_name][part].get(key) for r in rows])
                                                     for key in ('mean', 'fraction_gt_001', 'fraction_gt_01')}
        report['domains'][domain] = summary
    write(OUT / 'geometry_summary.json', report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['geometry', 'summarize'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--domains', nargs='+', choices=['clean', 'occluded'], default=['clean', 'occluded'])
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=16)
    args = parser.parse_args()
    if not 0 <= args.start < args.end <= 16:
        parser.error('require 0 <= start < end <= 16')
    geometry(args) if args.action == 'geometry' else summarize(args)


if __name__ == '__main__':
    main()
