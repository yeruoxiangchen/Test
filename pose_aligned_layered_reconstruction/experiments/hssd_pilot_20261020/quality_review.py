"""CPU-only, cohort-wide review of surface-sampling sensitivity; keep the original gate."""
import importlib.util
import json
import math
from pathlib import Path

if __package__:
    from .common import OUT, PACKAGE, binding, event, now, object_dir, plan, verify, write_json
else:
    from common import OUT, PACKAGE, binding, event, now, object_dir, plan, verify, write_json


SETTINGS = {
    'version': 'hssd16_sampling_review.v1',
    'point_count_per_mesh': 100000,
    'seeds': [42, 43, 44],
    'chamfer_l1_max': .015,
    'fscore_0p02_min': .85,
    'rule': 'every seed for every original object must pass',
    'alignment': 'native canonical frame; no ICP, centering, scaling or rotation fit',
}
HARD_CHECKS = {
    'coordinate', 'ss_roundtrip', 'target_projection', 'input_projection', 'finite_mesh',
}
REPORT = OUT / 'data_quality_review.json'
DETAILS = OUT / 'diagnostics/data_gate_review_20261020'
METRICS_SOURCE = PACKAGE / 'mesh_metrics.py'


def record_passes(record):
    if set(record['hard_checks']) != HARD_CHECKS:
        raise ValueError('Incomplete non-statistical data checks')
    samples = record['samples']
    if [sample['seed'] for sample in samples] != SETTINGS['seeds']:
        raise ValueError('Review must include all fixed seeds exactly once in order')
    if not all(value is True for value in record['hard_checks'].values()):
        return False
    for sample in samples:
        if sample['count'] != SETTINGS['point_count_per_mesh']:
            raise ValueError('Review sample count changed')
        values = sample['surface']
        if not values or not all(math.isfinite(value) for value in values.values()):
            return False
        if (values['chamfer_l1'] > SETTINGS['chamfer_l1_max']
                or values['fscore_0p02'] < SETTINGS['fscore_0p02_min']):
            return False
    return True


def validate_review():
    report = json.loads(REPORT.read_text())
    if report['settings'] != SETTINGS:
        raise ValueError('Surface-review protocol changed')
    for key in ['reviewer_source', 'metrics_source', 'plan', 'original_gate']:
        verify(report[key])
    expected = [row['uid'] for row in plan()['plan']['objects']]
    if [record['uid'] for record in report['objects']] != expected:
        raise ValueError('Review omitted, reordered or substituted original targets')
    decisions = []
    for record in report['objects']:
        for artifact in record['inputs']:
            verify(artifact)
        decisions.append(record_passes(record))
    passed = bool(decisions) and all(decisions)
    if report['passed'] is not passed:
        raise ValueError('Review summary disagrees with individual checks')
    return report


def build():
    if REPORT.exists():
        result = validate_review()
        event(stage='sampling_review_reused', passed=result['passed'], report=str(REPORT))
        return result
    import trimesh
    spec = importlib.util.spec_from_file_location('hssd_review_metrics', METRICS_SOURCE)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    original = json.loads((OUT / 'data_gate.json').read_text())
    originals = {record['uid']: record for record in original['objects']}
    cohort = plan()['plan']['objects']
    if len(cohort) != 16 or set(originals) != {row['uid'] for row in cohort}:
        raise ValueError('Review requires all original 16 objects and original gate results')
    records = []
    for index, row in enumerate(cohort):
        root = object_dir(row)
        audit = json.loads((root / 'latent_audit.json').read_text())
        visual = json.loads((root / 'visual_audit.json').read_text())
        projection = json.loads((root / 'input_projection_audit.json').read_text())
        inputs = [audit[key] for key in ['raw_mesh', 'decoded_mesh', 'ss_target', 'slat_target']]
        inputs += [binding(root / name) for name in
                   ['latent_audit.json', 'visual_audit.json', 'input_projection_audit.json']]
        for artifact in inputs:
            verify(artifact)
        predicted = trimesh.load(audit['decoded_mesh']['path'], force='mesh', process=False)
        target = trimesh.load(audit['raw_mesh']['path'], force='mesh', process=False)
        record = {
            'uid': row['uid'], 'inputs': inputs, 'original_20k_seed42': originals[row['uid']],
            'hard_checks': {
                'coordinate': bool(audit['coordinate']['passed']
                                   and audit['coordinate']['max_symmetric_vertex_distance'] < 3.e-5),
                'ss_roundtrip': bool(audit['ss']['passed'] and audit['ss']['iou'] >= .90),
                'target_projection': bool(visual['min_source_silhouette_iou'] >= .95),
                'input_projection': bool(projection['passed'] and len(projection['views']) == 16
                                         and all(view['silhouette_iou'] >= .95 for view in projection['views'])),
                'finite_mesh': bool(audit['finite_and_nonempty']),
            },
            'samples': [],
        }
        for seed in SETTINGS['seeds']:
            surface = metrics.surface_metrics(predicted, target,
                                              count=SETTINGS['point_count_per_mesh'], seed=seed)
            record['samples'].append({'seed': seed, 'count': SETTINGS['point_count_per_mesh'],
                                      'surface': surface})
        record['passed'] = record_passes(record)
        records.append(record)
        write_json(DETAILS / 'objects' / (row['uid'] + '.json'), record)
        event(stage='sampling_review', object=index+1, total=16, uid=row['uid'],
              passed=record['passed'],
              min_f02=min(sample['surface']['fscore_0p02'] for sample in record['samples']),
              max_cd=max(sample['surface']['chamfer_l1'] for sample in record['samples']))
    result = {
        'created_at': now(), 'settings': SETTINGS, 'passed': all(record['passed'] for record in records),
        'reviewer_source': binding(Path(__file__)), 'metrics_source': binding(METRICS_SOURCE),
        'plan': binding(OUT / 'plan.json'), 'original_gate': binding(OUT / 'data_gate.json'),
        'original_gate_passed': original['passed'], 'objects': records,
        'scope': 'Explicit post-failure engineering protocol amendment before formal training; '
                 'original 20K seed42 result remains failed. No target exclusion or geometry change. '
                 'This qualifies the 16-object learning diagnostic, not dataset expansion or paper claims.',
        'training_and_evaluation': 'Unchanged; paired parent/candidate mesh evaluations still use '
                                   'the original native-frame metric protocol.',
    }
    write_json(REPORT, result)
    validate_review()
    event(stage='sampling_review_complete', passed=result['passed'], report=str(REPORT))
    return result


if __name__ == '__main__':
    raise SystemExit(0 if build()['passed'] else 1)
