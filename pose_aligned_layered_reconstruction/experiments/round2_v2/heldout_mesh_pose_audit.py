"""Proper SE(3) oracle audit of the frozen Dev64 / strong16 mesh cohort.

Run CPU registration while generation writes atomic source records. The protocol
is fixed by the complete source plan, never by which records happen to exist.
Aggregation is a separate command and refuses incomplete cohorts. Original PLY
files are read only. The saved transform is column-vector p_target = R p + t.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import ast
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import itertools
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
SOURCE = PACKAGE / 'outputs/round2_v2_heldout_mesh_20261018'
OUT = SOURCE / 'alignment'
FROZEN = Path(__file__).with_name('mesh_pose_audit.py')
LEGACY = ROOT / 'experiments/rigid_aligned_mesh_eval_20260911.py'
METRICS = PACKAGE / 'mesh_metrics.py'
VERSION = 'palr.round2.heldout_mesh_pose_audit.v1'
MODES = ('parent', 'raw')
THRESHOLDS = (.01, .02, .05)
_HASH_CACHE = {}


def sha(path):
    path = Path(path).resolve()
    before = path.stat()
    signature = (before.st_size, before.st_mtime_ns)
    cached = _HASH_CACHE.get(str(path))
    if cached is not None and cached[0] == signature:
        return cached[1]
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    after = path.stat()
    if signature != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'source changed while hashing: {path}')
    _HASH_CACHE[str(path)] = (signature, digest.hexdigest())
    return digest.hexdigest()


def binding(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def verify(bound):
    if sha(bound['path']) != bound['sha256']:
        raise ValueError(f"frozen source changed: {bound['path']}")


def load_numerical_functions():
    """Reuse exact frozen numerical functions without importing Torch pipelines.

    Extraction is limited to named top-level functions and all source files are
    SHA-bound by protocol.json. No implementation or algorithm is rewritten.
    """
    namespace = dict(np=np, cKDTree=cKDTree, Rotation=Rotation, itertools=itertools,
                     THRESHOLDS=THRESHOLDS, OUT=OUT, dump=dump)
    groups = [(LEGACY, {'proper_axis_transforms', 'kabsch', 'rotation_angle_degrees'}),
              (METRICS, {'sample_surface'}),
              (FROZEN, {'distances', 'point_metrics', 'pose_checks', 'symmetric_icp',
                        'fit_candidates', 'self_test'})]
    for path, wanted in groups:
        nodes = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in wanted]
        if {node.name for node in nodes} != wanted:
            raise ValueError(f'frozen numerical functions missing: {path}')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


_NUMERICAL = load_numerical_functions()
sample_surface = _NUMERICAL['sample_surface']
point_metrics = _NUMERICAL['point_metrics']
pose_checks = _NUMERICAL['pose_checks']
fit_candidates = _NUMERICAL['fit_candidates']


def expected_cases():
    source_plan = json.loads((SOURCE / 'plan.json').read_text())
    rows = source_plan['plan']['rows']
    expected, identities = {}, {}
    for row in rows:
        index = int(row['index'])
        domain = 'clean' if index < 64 else 'strong_occluded'
        object_index = index if index < 64 else index - 64
        if row['domain'] != domain or row['object_index'] != object_index:
            raise ValueError(f'wrong domain/object index: {index}')
        if object_index in identities and identities[object_index] != row['object_uid']:
            raise ValueError('clean/strong object identity mismatch')
        identities[object_index] = row['object_uid']
        for case in row['cases']:
            seed, mode = int(case['seed']), case['mode']
            identity = f'{index:03d}_{seed}_{mode}'
            if identity in expected:
                raise ValueError('duplicate planned case')
            expected[identity] = {k: row[k] for k in ('index', 'object_index', 'object_uid', 'uid', 'domain')}
            expected[identity].update(id=identity, seed=seed, mode=mode,
                                      target_mesh=row['target_mesh'], identity_hash=source_plan['identity_hash'])
    canonical = {f'{index:03d}_{seed}_{mode}' for index in range(80) for seed in (42, 43) for mode in MODES}
    if set(expected) != canonical or len(identities) != 64 or len(rows) != 80:
        raise ValueError('cohort must be all Dev64 and fixed first16 strong, two seeds and two modes')
    return expected


def prepare_protocol():
    cases = expected_cases()
    protocol = dict(
        format=VERSION, source_plan=binding(SOURCE / 'plan.json'),
        code=[binding(path) for path in (__file__, FROZEN, LEGACY, METRICS)],
        expected_cases=[cases[key] for key in sorted(cases)], expected_record_count=320,
        domain_object_counts={'clean': 64, 'strong_occluded': 16}, modes=list(MODES), seeds=[42, 43],
        registration_points=4096, scoring_points=20000, max_iterations=35,
        initialization='identity rotation with centroid translation plus 24 proper PCA-axis rotations',
        final_candidates='3 lowest-registration-CD poses plus untouched identity; select once by independent 20K Chamfer-L1',
        registration_seed='20261018 + object_index*1009 + seed*17; target registration seed + 1',
        surface_seed='seed*1009 + object_index*9173 (not row index)',
        scoring_samples='same deterministic surface points and seed as native metrics; rigidly transform points/normals',
        numerical_functions='exact named functions extracted from frozen previous audit and native metric source',
        source_record_binding='per-record SHA saved on completion; full protocol fixed before incremental consumption',
        source_meshes_unchanged=True, oracle_gt_assisted=True, scale_fitted=False, reflection_allowed=False,
        component_filtering=False, deformation=False, primary_metric='native-frame surface metrics',
        threshold_units='canonical normalized sparse-grid length; not physical meters',
        caveats=['alignment is GT-assisted diagnosis, not inference-time correction',
                 'fitted rotation is not ground-truth pose error; symmetric objects can have equivalent rotations',
                 'proper SE3 leaves size error in residual geometry',
                 'finite multistart ICP does not guarantee a globally optimal rigid registration',
                 'Dev64 is a development cohort previously used for diagnostics, not an untouched final test set'],
        aggregation='clean64 and strong16 separately; object-cluster bootstrap after averaging two inference seeds',
        bootstrap=dict(samples=5000, seed=20261018, uid_order='lexicographic', confidence=.95))
    protocol['protocol_hash'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    path = OUT / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('existing protocol differs; immutable audit cannot be silently resumed')
    dump(path, protocol)
    return cases, protocol


def load_case(identity, expected):
    source_file = SOURCE / 'records' / (identity + '.json')
    bound = binding(source_file)
    record = json.loads(source_file.read_text())
    if record.get('success') is not True:
        raise ValueError(f'unsuccessful source record: {identity}')
    for key in ('index', 'object_index', 'object_uid', 'uid', 'domain', 'seed', 'mode', 'identity_hash', 'target_mesh'):
        if record[key] != expected[key]:
            raise ValueError(f'planned source mismatch {identity}: {key}')
    for mesh in (record['mesh'], record['target_mesh']):
        verify(mesh)
    surface_seed = record['seed'] * 1009 + record['object_index'] * 9173
    if 'surface_seed' in record and record['surface_seed'] != surface_seed:
        raise ValueError('source scoring seed mismatch')
    return {k: expected[k] for k in ('id', 'index', 'object_index', 'object_uid', 'uid', 'domain', 'seed', 'mode')} | dict(
        source_record=bound, mesh=record['mesh'], target=record['target_mesh'], target_mesh=record['target_mesh'],
        expected_native=record['surface'], surface_seed=surface_seed,
        registration_seed=20261018 + record['object_index'] * 1009 + record['seed'] * 17)


def evaluate_case(identity, expected, protocol_hash):
    started = time.time()
    case = load_case(identity, expected)
    destination = OUT / 'records' / (identity + '.json')
    if destination.exists():
        cached = json.loads(destination.read_text())
        if cached['protocol_hash'] != protocol_hash or any(cached[key] != case[key] for key in ('mesh', 'target', 'source_record')):
            raise ValueError('resume binding mismatch')
        pose_checks(cached['alignment']['matrix_predicted_to_target'])
        return cached
    predicted = trimesh.load(case['mesh']['path'], process=False, force='mesh')
    target = trimesh.load(case['target']['path'], process=False, force='mesh')
    p, pn = sample_surface(predicted, 20000, case['surface_seed'])
    g, gn = sample_surface(target, 20000, case['surface_seed'])
    native = point_metrics(p, pn, g, gn)
    if set(native) != set(case['expected_native']):
        raise ValueError('native metric schema mismatch')
    error = max(abs(native[key] - case['expected_native'][key]) for key in native)
    if error > 2.e-12:
        raise ValueError(f'native reproduction mismatch {identity}: {error}')
    source_points, _ = sample_surface(predicted, 4096, case['registration_seed'])
    target_points, _ = sample_surface(target, 4096, case['registration_seed'] + 1)
    candidates = fit_candidates(source_points, target_points)
    selected = dict(matrix_predicted_to_target=np.eye(4).tolist(), initialization_index=-1,
                    registration_chamfer=None)
    aligned, scored = dict(native), []
    for candidate in candidates[:3]:
        transform = np.asarray(candidate['matrix_predicted_to_target'])
        metrics = point_metrics(p @ transform[:3, :3].T + transform[:3, 3], pn @ transform[:3, :3].T, g, gn)
        scored.append(dict(**candidate, surface_chamfer=metrics['chamfer_l1']))
        if metrics['chamfer_l1'] < aligned['chamfer_l1'] - 1.e-14:
            selected, aligned = candidate, metrics
    alignment = dict(**selected, **pose_checks(selected['matrix_predicted_to_target']),
                     initial_hypotheses=len(candidates), selected_identity_on_surface_metric=selected['initialization_index'] == -1,
                     scored_candidates=scored, final_selection_metric='chamfer_l1')
    result = {key: value for key, value in case.items() if key != 'expected_native'} | dict(
        format=VERSION, protocol_hash=protocol_hash, success=True, oracle_gt_assisted=True,
        native=native, se3_aligned=aligned, alignment=alignment,
        native_reproduction_max_abs_error=error, elapsed_seconds=time.time() - started)
    dump(destination, result)
    return result


def bootstrap(values):
    ordered = {uid: float(values[uid]) for uid in sorted(values)}
    effects = np.array(list(ordered.values()))
    if not len(effects) or not np.isfinite(effects).all():
        raise ValueError('invalid per-object effects')
    rng = np.random.default_rng(20261018)
    samples = effects[rng.integers(len(effects), size=(5000, len(effects)))].mean(1)
    return dict(mean_improvement=float(effects.mean()), median_improvement=float(np.median(effects)),
                object_count=len(effects), positive_means_better=True, wins=int((effects > 1.e-12).sum()),
                ties=int((np.abs(effects) <= 1.e-12).sum()), losses=int((effects < -1.e-12).sum()),
                object_bootstrap_ci95=np.quantile(samples, [.025, .975]).tolist(), per_object=ordered)


def paired(records, field, metric):
    grouped = {}
    for record in records:
        key = (record['object_uid'], record['seed'])
        if record['mode'] in grouped.setdefault(key, {}):
            raise ValueError('paired aggregate contains duplicate object/seed/mode')
        grouped[key][record['mode']] = record
    uids = sorted({uid for uid, _ in grouped})
    if set(grouped) != {(uid, seed) for uid in uids for seed in (42, 43)}:
        raise ValueError('paired aggregate has incomplete seed coverage')
    sign = -1 if metric.startswith('chamfer') else 1
    per_object, by_seed = {}, {'42': [], '43': []}
    for uid in uids:
        gains = []
        for seed in (42, 43):
            pair = grouped[uid, seed]
            if set(pair) != set(MODES):
                raise ValueError('paired aggregate missing model')
            gain = sign * (pair['raw'][field][metric] - pair['parent'][field][metric])
            gains.append(gain)
            by_seed[str(seed)].append(gain)
        per_object[uid] = float(np.mean(gains))
    return bootstrap(per_object) | dict(by_seed={key: float(np.mean(value)) for key, value in by_seed.items()})


def group_summary(records):
    metrics = ('chamfer_l1', 'fscore_0p02', 'precision_0p02', 'recall_0p02', 'normal_consistency')
    result = dict(record_count=len(records), object_count=len({r['object_uid'] for r in records}), summary={}, paired={}, pose={})
    for field in ('native', 'se3_aligned'):
        result['summary'][field] = {mode: {key: float(np.mean([r[field][key] for r in records if r['mode'] == mode]))
                                         for key in metrics} for mode in MODES}
        result['paired'][field] = {'raw_minus_parent': {key: paired(records, field, key)
                                                       for key in ('chamfer_l1', 'fscore_0p02')}}
    for mode in MODES:
        selected = [r for r in records if r['mode'] == mode]
        angles = np.array([r['alignment']['rotation_degrees'] for r in selected])
        translations = np.array([r['alignment']['translation_norm'] for r in selected])
        result['pose'][mode] = dict(
            rotation_degrees_mean=float(angles.mean()), rotation_degrees_median=float(np.median(angles)),
            rotation_degrees_p90=float(np.quantile(angles, .9)), rotation_degrees_max=float(angles.max()),
            rotations_above_10_degrees=int((angles > 10).sum()), rotations_above_45_degrees=int((angles > 45).sum()),
            translation_mean=float(translations.mean()), translation_median=float(np.median(translations)),
            translation_p90=float(np.quantile(translations, .9)), translation_max=float(translations.max()),
            identity_selected=sum(r['alignment']['selected_identity_on_surface_metric'] for r in selected),
            mean_cd_alignment_gain=float(np.mean([r['native']['chamfer_l1'] - r['se3_aligned']['chamfer_l1'] for r in selected])))
    native = result['paired']['native']['raw_minus_parent']['chamfer_l1']['mean_improvement']
    aligned = result['paired']['se3_aligned']['raw_minus_parent']['chamfer_l1']['mean_improvement']
    result['aligned_to_native_cd_gain_ratio'] = aligned / native if native else None
    return result


def aggregate(cases, protocol):
    started = time.time()
    for bound in [protocol['source_plan'], *protocol['code']]:
        verify(bound)
    actual = {path.stem for path in (OUT / 'records').glob('*.json')}
    if actual != set(cases):
        raise ValueError(f'aggregation requires exact 320 records, currently {len(actual)}')
    records = []
    for identity, expected in sorted(cases.items()):
        case = load_case(identity, expected)
        record = json.loads((OUT / 'records' / (identity + '.json')).read_text())
        if record['id'] != identity or record['protocol_hash'] != protocol['protocol_hash'] or not record['success']:
            raise ValueError('audit identity/protocol mismatch')
        for key in ('mesh', 'target', 'source_record', 'index', 'object_index', 'object_uid', 'domain', 'mode', 'seed'):
            if record[key] != case[key]:
                raise ValueError(f'audit source mismatch {identity}: {key}')
        for bound in (record['mesh'], record['target'], record['source_record']):
            verify(bound)
        pose_checks(record['alignment']['matrix_predicted_to_target'])
        if record['native'] != case['expected_native']:
            error = max(abs(record['native'][key] - case['expected_native'][key]) for key in record['native'])
            if error > 2.e-12:
                raise ValueError('native metrics changed')
        if record['se3_aligned']['chamfer_l1'] > record['native']['chamfer_l1'] + 1.e-12:
            raise ValueError('identity safeguard failure')
        records.append(record)
    groups = {'clean': [r for r in records if r['domain'] == 'clean'],
              'strong_occluded': [r for r in records if r['domain'] == 'strong_occluded'],
              'matched16_clean': [r for r in records if r['domain'] == 'clean' and r['object_index'] < 16]}
    summary = {key: group_summary(value) for key, value in groups.items()}
    difference = {}
    for field in ('native', 'se3_aligned'):
        difference[field] = {}
        for metric in ('chamfer_l1', 'fscore_0p02'):
            clean = summary['matched16_clean']['paired'][field]['raw_minus_parent'][metric]['per_object']
            strong = summary['strong_occluded']['paired'][field]['raw_minus_parent'][metric]['per_object']
            if set(clean) != set(strong):
                raise ValueError('matched clean/strong UID mismatch')
            difference[field][metric] = bootstrap({uid: strong[uid] - clean[uid] for uid in clean}) | dict(
                interpretation='positive means the V2 relative advantage over parent is larger under strong occlusion')
    result = dict(format=VERSION, protocol_hash=protocol['protocol_hash'], record_count=len(records),
                  all_meshes_successful=True, all_source_hashes_unchanged=True, all_proper_SE3=True,
                  native_reproduction_max_abs_error=max(r['native_reproduction_max_abs_error'] for r in records),
                  groups=summary, matched16_strong_minus_clean_relative_gain=difference,
                  caveats=protocol['caveats'])
    result['largest_rotation_cases'] = [dict(id=r['id'], mode=r['mode'], domain=r['domain'], object_index=r['object_index'],
                                           rotation_degrees=r['alignment']['rotation_degrees'], translation_norm=r['alignment']['translation_norm'],
                                           native_cd=r['native']['chamfer_l1'], aligned_cd=r['se3_aligned']['chamfer_l1'])
                                       for r in sorted(records, key=lambda r: r['alignment']['rotation_degrees'], reverse=True)[:24]]
    for path, (signature, _) in _HASH_CACHE.items():
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != signature:
            raise ValueError(f'source changed during aggregation: {path}')
    verification = dict(passed=True, record_count=len(records), source_hashes_unique_files=len(_HASH_CACHE),
                        source_hashes_all_match=True, source_stats_unchanged=True,
                        native_reproduction_max_abs_error=result['native_reproduction_max_abs_error'],
                        max_detR_abs_error=max(abs(r['alignment']['detR'] - 1) for r in records),
                        max_singular_value_abs_error=max(abs(v - 1) for r in records for v in r['alignment']['singular_values']),
                        max_orthogonality_error=max(r['alignment']['orthogonality_error'] for r in records),
                        identity_selected=sum(r['alignment']['selected_identity_on_surface_metric'] for r in records),
                        elapsed_seconds=time.time() - started)
    dump(OUT / 'aggregate.json', result)
    dump(OUT / 'verification.json', verification)
    dump(OUT / 'complete.json', dict(record_count=len(records), protocol_hash=protocol['protocol_hash'],
                                    aggregate=binding(OUT / 'aggregate.json'), verification=binding(OUT / 'verification.json')))
    return result


def run(cases, protocol, workers, watch, max_cases):
    done, submitted = set(), set()
    started = time.time()
    scheduled_count = len(cases) if max_cases == 0 else min(max_cases, len(cases))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        pending = {}
        while True:
            ready = [key for key in sorted(cases) if key not in submitted and (SOURCE / 'records' / (key + '.json')).exists()]
            for identity in ready[:max(0, min(workers * 2 - len(pending), scheduled_count - len(submitted)))]:
                future = executor.submit(evaluate_case, identity, cases[identity], protocol['protocol_hash'])
                pending[future] = identity
                submitted.add(identity)
            if pending:
                finished, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                for future in finished:
                    identity = pending.pop(future)
                    record = future.result()
                    done.add(identity)
                    progress = dict(completed=len(done), expected=len(cases), scheduled_limit=scheduled_count,
                                    id=identity, elapsed_seconds=time.time() - started,
                                    native_cd=record['native']['chamfer_l1'], aligned_cd=record['se3_aligned']['chamfer_l1'],
                                    rotation_degrees=record['alignment']['rotation_degrees'], case_seconds=record['elapsed_seconds'])
                    dump(OUT / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
            if len(done) == scheduled_count:
                break
            if not pending and not ready:
                if not watch:
                    break
                time.sleep(10)
    print(json.dumps(dict(stage='registration_finished', completed=len(done), expected=len(cases),
                          aggregation='run the separate aggregate command after all 320 source records exist')), flush=True)


def self_test():
    report = _NUMERICAL['self_test']()
    synthetic = []
    for uid, improvements in [('a', (.01, .03)), ('b', (-.01, .01))]:
        for seed, improvement in zip((42, 43), improvements):
            for mode, cd in [('parent', .1), ('raw', .1 - improvement)]:
                synthetic.append(dict(object_uid=uid, seed=seed, mode=mode,
                                      native={'chamfer_l1': cd}))
    result = paired(synthetic, 'native', 'chamfer_l1')
    if result['object_count'] != 2 or abs(result['mean_improvement'] - .01) > 1.e-12:
        raise AssertionError('paired bootstrap must average seeds within each independent object')
    if result['wins'] != 1 or result['ties'] != 1:
        raise AssertionError('paired metric direction/count test failed')
    try:
        paired(synthetic[:-1], 'native', 'chamfer_l1')
    except ValueError:
        pass
    else:
        raise AssertionError('incomplete model pair was accepted')
    report['object_cluster_pairing'] = dict(passed=True, seed_average_before_bootstrap=True,
                                           incomplete_pairs_rejected=True, example=result)
    report['numerical_sources'] = [binding(path) for path in (FROZEN, LEGACY, METRICS)]
    dump(OUT / 'self_test.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'self-test', 'run', 'aggregate'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--watch', action='store_true', help='Consume atomic records until all 320 exist and finish.')
    parser.add_argument('--max-cases', type=int, default=0, help='Benchmark only; full aggregation remains separate.')
    args = parser.parse_args()
    if args.command == 'self-test':
        print(json.dumps(self_test()), flush=True)
        return
    cases, protocol = prepare_protocol()
    if args.command == 'prepare':
        print(json.dumps(dict(prepared=True, expected=len(cases), protocol_hash=protocol['protocol_hash'])), flush=True)
    elif args.command == 'run':
        if args.workers < 1 or args.workers > 8 or args.max_cases < 0:
            raise ValueError('use 1 to 8 CPU workers, each with one numerical thread')
        run(cases, protocol, args.workers, args.watch, args.max_cases)
    else:
        result = aggregate(cases, protocol)
        print(json.dumps(dict(complete=True, records=result['record_count'],
                              groups={key: value['summary'] for key, value in result['groups'].items()})), flush=True)


if __name__ == '__main__':
    main()
