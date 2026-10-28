"""Read-only mesh audit: native scoring and GT-assisted proper SE(3) diagnosis.

The saved transform uses column vectors: p_target = R @ p_prediction + t.
This is an oracle diagnosis, never an inference-time pose correction. Scale,
reflection, component filtering, mesh repair, and deformation are forbidden.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / 'pose_aligned_layered_reconstruction'
SOURCE = PACKAGE / 'outputs/round2_v2_diagnostics_20261018/mesh'
OUT = PACKAGE / 'outputs/round2_v2_mesh_visual_audit_20261018/alignment'
LEGACY = ROOT / 'experiments/rigid_aligned_mesh_eval_20260911.py'
_spec = importlib.util.spec_from_file_location('legacy_rigid_mesh_audit', LEGACY)
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)
proper_axis_transforms = _legacy.proper_axis_transforms
kabsch = _legacy.kabsch
rotation_angle_degrees = _legacy.rotation_angle_degrees
from pose_aligned_layered_reconstruction.mesh_metrics import sample_surface

VERSION = 'palr.round2.mesh_pose_audit.v1'
MODES = ('parent', 'v1', 'raw')
THRESHOLDS = (.01, .02, .05)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def binding(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def distances(source, target):
    forward, fi = cKDTree(target).query(source, workers=1)
    reverse, ri = cKDTree(source).query(target, workers=1)
    return forward, reverse, fi, ri


def point_metrics(p, pn, g, gn):
    pd, gd, pi, gi = distances(p, g)
    result = dict(pred_to_gt_mean=float(pd.mean()), gt_to_pred_mean=float(gd.mean()),
                  chamfer_l1=float(.5 * (pd.mean() + gd.mean())),
                  chamfer_l2=float(.5 * ((pd ** 2).mean() + (gd ** 2).mean())),
                  normal_consistency=float(.5 * (np.abs((pn * gn[pi]).sum(1)).mean()
                                                 + np.abs((gn * pn[gi]).sum(1)).mean())))
    for threshold in THRESHOLDS:
        key = str(float(threshold)).replace('.', 'p')
        precision, recall = float((pd < threshold).mean()), float((gd < threshold).mean())
        result.update({f'precision_{key}': precision, f'recall_{key}': recall,
                       f'fscore_{key}': 2 * precision * recall / (precision + recall)
                       if precision + recall else 0.})
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError('nonfinite surface metrics')
    return result


def pose_checks(transform):
    transform = np.asarray(transform, np.float64)
    rotation = transform[:3, :3]
    determinant = float(np.linalg.det(rotation))
    singular = np.linalg.svd(rotation, compute_uv=False)
    orthogonality = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('invalid transform')
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1.e-10, rtol=0):
        raise ValueError('invalid homogeneous row')
    if abs(determinant - 1.) > 1.e-9 or np.max(np.abs(singular - 1.)) > 1.e-9 or orthogonality > 1.e-9:
        raise ValueError('transform is not proper SE(3): reflection/scale forbidden')
    return dict(detR=determinant, singular_values=singular.tolist(),
                orthogonality_error=orthogonality, scale_fitted=False, reflection_allowed=False,
                rotation_degrees=rotation_angle_degrees(transform),
                translation_norm=float(np.linalg.norm(transform[:3, 3])))


def symmetric_icp(source, target, initial_rotation, iterations=35):
    """Legacy bidirectional NN / Kabsch ICP, retaining its best L1 iterate.

Kabsch minimizes paired squared distances; its update need not decrease the L1
selection objective. Retaining the best iterate avoids losing a better pose.
    """
    row_rotation = np.asarray(initial_rotation, np.float64).T
    translation = target.mean(0) - source.mean(0) @ row_rotation
    target_tree = cKDTree(target)
    best = None
    previous = np.inf
    for iteration in range(iterations + 1):
        transformed = source @ row_rotation + translation
        fd, fi = target_tree.query(transformed, workers=1)
        rd, ri = cKDTree(transformed).query(target, workers=1)
        score = float(.5 * (fd.mean() + rd.mean()))
        if best is None or score < best[2]:
            best = (row_rotation.copy(), translation.copy(), score, iteration)
        if iteration == iterations or (iteration > 0 and previous - score < 1.e-7):
            break
        previous = score
        row_rotation, translation = kabsch(
            np.concatenate((source, source[ri])), np.concatenate((target[fi], target)))
    rotation, translation, score, best_iteration = best
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation.T, translation
    pose_checks(transform)
    return dict(matrix_predicted_to_target=transform.tolist(), registration_chamfer=score,
                best_iteration=best_iteration, iterations_executed=iteration)


def fit_candidates(source, target):
    hypotheses = proper_axis_transforms(source, target)
    candidates = []
    for index, rotation in enumerate(hypotheses):
        candidate = symmetric_icp(source, target, rotation)
        candidate['initialization_index'] = index
        candidates.append(candidate)
    candidates.sort(key=lambda c: (c['registration_chamfer'], c['initialization_index']))
    return candidates


def cases_and_protocol():
    plan_file = SOURCE / 'plan.json'
    original_plan = json.loads(plan_file.read_text())
    expected = {(r['index'], c['seed'], c['mode']): r
                for r in original_plan['plan']['rows'] for c in r['cases']}
    cases, seen = [], set()
    for path in sorted((SOURCE / 'records').glob('*.json')):
        record = json.loads(path.read_text())
        key = (record['index'], record['seed'], record['mode'])
        if key in seen or key not in expected or not record['success']:
            raise ValueError(f'bad/mixed/duplicate source record {path}')
        seen.add(key)
        if record['identity_hash'] != original_plan['identity_hash']:
            raise ValueError(f'original plan binding mismatch {path}')
        row = expected[key]
        if any(record[k] != row[k] for k in ('object_index', 'object_uid', 'domain', 'uid')):
            raise ValueError(f'object identity mismatch {path}')
        target_meta_file = SOURCE / 'targets' / f"{record['object_index']:03d}.json"
        target_meta = json.loads(target_meta_file.read_text())
        if target_meta['identity_hash'] != original_plan['identity_hash']:
            raise ValueError('target plan binding mismatch')
        for bound in (record['mesh'], target_meta['mesh']):
            if sha(bound['path']) != bound['sha256']:
                raise ValueError(f"source mesh changed: {bound['path']}")
        cases.append({k: record[k] for k in ('index', 'object_index', 'object_uid', 'uid', 'domain', 'seed', 'mode')}
                     | dict(id=path.stem, source_record=binding(path), mesh=record['mesh'],
                            target=target_meta['mesh'], expected_native=record['surface'],
                            surface_seed=record['seed'] * 1009 + record['object_index'] * 9173,
                            registration_seed=20261018 + record['object_index'] * 1009 + record['seed'] * 17))
    if seen != set(expected) or len(cases) != 384:
        raise ValueError(f'incomplete input: {len(cases)} / 384')
    protocol = dict(format=VERSION, source_plan=binding(plan_file),
                    code=[binding(__file__), binding(LEGACY), binding(PACKAGE / 'mesh_metrics.py')],
                    cases=[{k: c[k] for k in ('id', 'source_record', 'mesh', 'target')} for c in cases],
                    registration_points=4096, scoring_points=20000, max_iterations=35,
                    initialization='identity rotation with centroid translation plus 24 proper PCA-axis rotations',
                    final_candidates='3 lowest-registration-CD poses plus untouched identity; select only by 20K Chamfer-L1',
                    registration_samples='independent from scoring samples; target uses registration_seed+1',
                    scoring_samples='same deterministic surface points and seed as original native metrics; transform points/normals rigidly',
                    registration_seed='20261018 + object_index*1009 + seed*17, same across modes and input domains',
                    source_meshes_unchanged=True, oracle_gt_assisted=True,
                    scale_fitted=False, reflection_allowed=False, component_filtering=False,
                    primary_metric='original native-frame metrics remain primary; aligned scores diagnose shape vs pose',
                    symmetry_caveat='fitted angle is not ground-truth pose error; symmetric shapes can have equivalent large rotations')
    protocol['protocol_hash'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return cases, protocol


def evaluate_case(case, protocol_hash):
    start = time.time()
    destination = OUT / 'records' / (case['id'] + '.json')
    if destination.exists():
        cached = json.loads(destination.read_text())
        if cached['protocol_hash'] != protocol_hash or any(cached[k] != case[k] for k in ('mesh', 'target', 'source_record')):
            raise ValueError('resume binding mismatch')
        pose_checks(cached['alignment']['matrix_predicted_to_target'])
        return cached
    predicted = trimesh.load(case['mesh']['path'], process=False, force='mesh')
    target = trimesh.load(case['target']['path'], process=False, force='mesh')
    p, pn = sample_surface(predicted, 20000, case['surface_seed'])
    g, gn = sample_surface(target, 20000, case['surface_seed'])
    native = point_metrics(p, pn, g, gn)
    error = max(abs(native[k] - case['expected_native'][k]) for k in native)
    if error > 2.e-12:
        raise ValueError(f"native reproduction mismatch {case['id']}: {error}")
    source_points, _ = sample_surface(predicted, 4096, case['registration_seed'])
    target_points, _ = sample_surface(target, 4096, case['registration_seed'] + 1)
    candidates = fit_candidates(source_points, target_points)
    identity = np.eye(4)
    selected, aligned = dict(matrix_predicted_to_target=identity.tolist(), initialization_index=-1,
                            registration_chamfer=None), dict(native)
    scored = []
    for candidate in candidates[:3]:
        transform = np.asarray(candidate['matrix_predicted_to_target'])
        metrics = point_metrics(p @ transform[:3, :3].T + transform[:3, 3],
                                pn @ transform[:3, :3].T, g, gn)
        scored.append(dict(**candidate, surface_chamfer=metrics['chamfer_l1']))
        if metrics['chamfer_l1'] < aligned['chamfer_l1'] - 1.e-14:
            selected, aligned = candidate, metrics
    alignment = dict(**selected, **pose_checks(selected['matrix_predicted_to_target']),
                     initial_hypotheses=len(candidates), selected_identity_on_surface_metric=selected['initialization_index'] == -1,
                     scored_candidates=scored, final_selection_metric='chamfer_l1')
    result = {k: v for k, v in case.items() if k != 'expected_native'} | dict(
        format=VERSION, protocol_hash=protocol_hash, success=True, oracle_gt_assisted=True,
        native=native, se3_aligned=aligned, alignment=alignment,
        native_reproduction_max_abs_error=error, elapsed_seconds=time.time() - start)
    dump(destination, result)
    return result


def paired(records, left_mode, right_mode, field, metric):
    grouped = {}
    for record in records:
        grouped.setdefault(record['object_uid'], {}).setdefault((record['index'], record['seed']), {})[record['mode']] = record
    values = {}
    sign = -1. if metric.startswith('chamfer') else 1.
    for uid, cases in sorted(grouped.items()):
        values[uid] = float(np.mean([sign * (m[left_mode][field][metric] - m[right_mode][field][metric])
                                    for _, m in sorted(cases.items())]))
    gains = np.array(list(values.values()))
    rng = np.random.default_rng(20261018)
    bootstrap = gains[rng.integers(len(gains), size=(5000, len(gains)))].mean(1)
    return dict(mean_improvement=float(gains.mean()), object_count=len(gains), positive_means_better=True,
                wins=int((gains > 1.e-12).sum()), ties=int((np.abs(gains) <= 1.e-12).sum()),
                losses=int((gains < -1.e-12).sum()), object_bootstrap_ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
                per_object=values)


def aggregate(cases, protocol):
    records = []
    for case in cases:
        path = OUT / 'records' / (case['id'] + '.json')
        record = json.loads(path.read_text())
        if not record['success'] or record['protocol_hash'] != protocol['protocol_hash']:
            raise ValueError('incomplete or mismatched aligned records')
        for bound in (record['source_record'], record['mesh'], record['target']):
            if sha(bound['path']) != bound['sha256']:
                raise ValueError('source changed during alignment')
        pose_checks(record['alignment']['matrix_predicted_to_target'])
        if record['se3_aligned']['chamfer_l1'] > record['native']['chamfer_l1'] + 1.e-12:
            raise ValueError('identity safeguard failure')
        records.append(record)
    result = dict(format=VERSION, protocol_hash=protocol['protocol_hash'], record_count=len(records),
                  all_meshes_successful=True, all_source_hashes_unchanged=True, all_proper_SE3=True,
                  native_reproduction_max_abs_error=max(r['native_reproduction_max_abs_error'] for r in records), groups={})
    metrics = ('chamfer_l1', 'fscore_0p02', 'precision_0p02', 'recall_0p02', 'normal_consistency')
    for name, subset in [('all', records), ('clean', [r for r in records if r['domain'] == 'legacy_clean']),
                         ('occluded', [r for r in records if r['domain'] != 'legacy_clean'])]:
        group = dict(record_count=len(subset), summary={}, paired={}, pose={})
        for field in ('native', 'se3_aligned'):
            group['summary'][field] = {mode: {m: float(np.mean([r[field][m] for r in subset if r['mode'] == mode]))
                                             for m in metrics} for mode in MODES}
            group['paired'][field] = {f'{left}_minus_{right}': {m: paired(subset, left, right, field, m)
                                                               for m in ('chamfer_l1', 'fscore_0p02')}
                                      for left, right in [('raw', 'parent'), ('v1', 'parent'), ('raw', 'v1')]}
        for mode in MODES:
            selected = [r for r in subset if r['mode'] == mode]
            angles = np.array([r['alignment']['rotation_degrees'] for r in selected])
            translations = np.array([r['alignment']['translation_norm'] for r in selected])
            group['pose'][mode] = dict(rotation_degrees_mean=float(angles.mean()), rotation_degrees_median=float(np.median(angles)),
                                       rotation_degrees_p90=float(np.quantile(angles, .9)), rotation_degrees_max=float(angles.max()),
                                       rotations_above_10_degrees=int((angles > 10).sum()), rotations_above_45_degrees=int((angles > 45).sum()),
                                       translation_mean=float(translations.mean()), translation_median=float(np.median(translations)),
                                       translation_p90=float(np.quantile(translations, .9)), translation_max=float(translations.max()),
                                       identity_selected=sum(r['alignment']['selected_identity_on_surface_metric'] for r in selected),
                                       mean_cd_alignment_gain=float(np.mean([r['native']['chamfer_l1'] - r['se3_aligned']['chamfer_l1'] for r in selected])))
        native_gain = group['paired']['native']['raw_minus_parent']['chamfer_l1']['mean_improvement']
        aligned_gain = group['paired']['se3_aligned']['raw_minus_parent']['chamfer_l1']['mean_improvement']
        group['aligned_to_native_cd_gain_ratio'] = aligned_gain / native_gain if native_gain else None
        result['groups'][name] = group
    result['largest_rotation_cases'] = [dict(id=r['id'], mode=r['mode'], object_index=r['object_index'],
                                           rotation_degrees=r['alignment']['rotation_degrees'], translation_norm=r['alignment']['translation_norm'],
                                           native_cd=r['native']['chamfer_l1'], aligned_cd=r['se3_aligned']['chamfer_l1'])
                                        for r in sorted(records, key=lambda r: r['alignment']['rotation_degrees'], reverse=True)[:24]]
    dump(OUT / 'aggregate.json', result)
    dump(OUT / 'complete.json', dict(record_count=len(records), protocol_hash=protocol['protocol_hash'], aggregate=binding(OUT / 'aggregate.json')))
    return result


def self_test():
    rng = np.random.default_rng(20261018)
    source = rng.normal(size=(4096, 3)) * [.43, .17, .09]
    source[:700] += [.37, .21, -.12]
    known = np.eye(4)
    known[:3, :3] = Rotation.from_euler('xyz', [31., -47., 83.], degrees=True).as_matrix()
    known[:3, 3] = [.23, -.17, .08]
    target = source @ known[:3, :3].T + known[:3, 3]
    fitted = fit_candidates(source, target)[0]
    recovered = np.asarray(fitted['matrix_predicted_to_target'])
    rotation_error = float(np.degrees(Rotation.from_matrix(recovered[:3, :3] @ known[:3, :3].T).magnitude()))
    translation_error = float(np.linalg.norm(recovered[:3, 3] - known[:3, 3]))
    if rotation_error > 1.e-6 or translation_error > 1.e-8 or fitted['registration_chamfer'] > 1.e-8:
        raise AssertionError('known SE3 recovery failed')
    reflection = np.eye(4); reflection[0, 0] = -1
    scale = np.eye(4); scale[:3, :3] *= 1.2
    forbidden = {}
    for name, transform in [('reflection', reflection), ('scale', scale)]:
        try:
            pose_checks(transform)
        except ValueError:
            forbidden[name] = 'rejected'
        else:
            raise AssertionError(f'{name} accepted')
    mirrored = source * [-1, 1, 1]
    rotation, translation = kabsch(source, mirrored)
    reflection_residual = float(np.sqrt(np.mean(np.sum((source @ rotation + translation - mirrored) ** 2, axis=1))))
    scaled = source * 1.2
    scale_rotation, scale_translation = kabsch(source, scaled)
    scale_residual = float(np.sqrt(np.mean(np.sum((source @ scale_rotation + scale_translation - scaled) ** 2, axis=1))))
    if np.linalg.det(rotation) < .999999 or reflection_residual < .01 or scale_residual < .01:
        raise AssertionError('Kabsch removed forbidden reflection/scale')
    report = dict(passed=True, known_transform=known.tolist(), recovered_transform=recovered.tolist(),
                  rotation_error_degrees=rotation_error, translation_error=translation_error,
                  registration_chamfer=fitted['registration_chamfer'], transform_checks=pose_checks(recovered),
                  forbidden_transform_validation=forbidden, reflection_fit_residual=reflection_residual,
                  scale_fit_residual=scale_residual)
    dump(OUT / 'self_test.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--limit', type=int, default=0, help='Benchmark first N records; no full aggregate.')
    parser.add_argument('--priority-objects', type=int, nargs='*', default=[], help='Scheduling only; full cohort and metrics stay identical.')
    parser.add_argument('--aggregate-only', action='store_true')
    parser.add_argument('--self-test-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.self_test_only:
        print(json.dumps(self_test()), flush=True)
        return
    cases, protocol = cases_and_protocol()
    protocol_file = OUT / 'protocol.json'
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        raise ValueError('existing protocol changed')
    dump(protocol_file, protocol)
    if not args.aggregate_only:
        self_test()
        selected = cases[:args.limit] if args.limit else cases
        priority = {obj: rank for rank, obj in enumerate(args.priority_objects)}
        selected = sorted(selected, key=lambda case: (priority.get(case['object_index'], len(priority)), case['id']))
        start = time.time()
        def report(n, record):
            progress = dict(completed=n, scheduled=len(selected), id=record['id'], elapsed_seconds=time.time() - start,
                            native_cd=record['native']['chamfer_l1'], aligned_cd=record['se3_aligned']['chamfer_l1'],
                            rotation_degrees=record['alignment']['rotation_degrees'], case_seconds=record['elapsed_seconds'])
            dump(OUT / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
        if args.workers == 1:
            for n, case in enumerate(selected, 1):
                report(n, evaluate_case(case, protocol['protocol_hash']))
        else:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as executor:
                futures = [executor.submit(evaluate_case, case, protocol['protocol_hash']) for case in selected]
                for n, future in enumerate(as_completed(futures), 1):
                    report(n, future.result())
    if not args.limit:
        result = aggregate(cases, protocol)
        print(json.dumps(dict(complete=True, records=result['record_count'], summary=result['groups']['all']['summary'])), flush=True)


if __name__ == '__main__':
    main()
