"""Finalize the fixed Dev64/strong16 mesh experiment from complete evidence.

No model inference or mesh changes. Recompute paired object statistics from
individual records independently of both aggregators, then bind final evidence.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[2]
OUT = PACKAGE / 'outputs/round2_v2_heldout_mesh_20261018'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def binding(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def independent_effect(records, field, metric):
    uids = sorted({r['object_uid'] for r in records})
    lookup = {(r['object_uid'], r['seed'], r['mode']): r for r in records}
    expected = {(u, s, m) for u in uids for s in (42, 43) for m in ('parent', 'raw')}
    if len(lookup) != len(records) or set(lookup) != expected:
        raise ValueError('incomplete or duplicate object pairs')
    values = np.array([[[lookup[u, s, m][field][metric] for m in ('parent', 'raw')]
                        for s in (42, 43)] for u in uids])
    sign = -1 if metric == 'chamfer_l1' else 1
    gains = sign * (values[:, :, 1] - values[:, :, 0]).mean(axis=1)
    rng = np.random.default_rng(20261018)
    boot = gains[rng.integers(len(uids), size=(5000, len(uids)))].mean(axis=1)
    return {'object_count': len(uids), 'parent_mean': float(values[:, :, 0].mean()),
            'v2_mean': float(values[:, :, 1].mean()), 'mean_improvement': float(gains.mean()),
            'median_improvement': float(np.median(gains)),
            'leave_one_object_out_mean_range': [float(x) for x in (
                ((gains.sum()-gains)/(len(gains)-1)).min(),
                ((gains.sum()-gains)/(len(gains)-1)).max())],
            'ci95': np.quantile(boot, [.025, .975]).tolist(),
            'wins': int((gains > 1e-12).sum()), 'losses': int((gains < -1e-12).sum()),
            'ties': int((np.abs(gains) <= 1e-12).sum()),
            'relative_reduction_percent': float(100 * gains.mean() / values[:, :, 0].mean())
                if metric == 'chamfer_l1' else None}


def plot_effects(stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    labels = ['Clean64 / native', 'Clean64 / SE(3)', 'Strong16 / native', 'Strong16 / SE(3)']
    locations = [(d, f) for d in ('clean', 'strong_occluded') for f in ('native', 'se3_aligned')]
    colors = ['#17679c', '#73a9c9', '#b85e12', '#dfa166']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, scale, title in zip(axes, ('chamfer_l1', 'fscore_0p02'), (1, 100),
                                       ('Chamfer reduction (normalized grid units)', 'F-score@0.02 gain (percentage points)')):
        for y, ((domain, field), color) in enumerate(zip(locations, colors)):
            item = stats[domain][field][metric]
            center = item['mean_improvement'] * scale
            lower, upper = np.asarray(item['ci95']) * scale
            ax.errorbar(center, y, xerr=[[max(0, center-lower)], [max(0, upper-center)]],
                        fmt='o', color=color, capsize=4, linewidth=2)
        ax.axvline(0, color='#777777', lw=1, linestyle='--')
        ax.set_yticks(range(4), labels)
        ax.invert_yaxis()
        ax.set_xlabel(title)
        ax.grid(axis='x', alpha=.2)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('V2 vs parent on Dev64 / Strong16: paired object effects', fontsize=13)
    fig.text(.5, .015, 'Two inference seeds averaged per object; 95% object bootstrap CI. Development set, official SLat VAE proxy GT.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .94))
    dest = OUT / 'figures/generalization_effects'
    dest.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.pdf'):
        fig.savefig(dest.with_suffix(suffix), dpi=180)
    plt.close(fig)


def main():
    plan = read(OUT / 'plan.json')
    native = read(OUT / 'aggregate.json')
    aligned = read(OUT / 'alignment/aggregate.json')
    registration = read(OUT / 'alignment/verification.json')
    visuals = read(OUT / 'visual_report_manifest.json')
    visual_integrity = read(OUT / 'visual_integrity.json')
    gallery = read(OUT / 'gallery_validation.json')
    repeatability = read(OUT / 'runtime_repeatability/compare.json')
    identical_support = read(OUT / 'identical_support_runtime_audit.json')
    robustness = read(OUT / 'object_gain_robustness.json')
    if (identical_support['identity_hash'] != plan['identity_hash']
        or identical_support['control_case_count'] != 6 or identical_support['formal_records_modified']
        or robustness['aggregate_sha256'] != binding(OUT / 'aggregate.json')['sha256']
        or not robustness['diagnostic_only_no_objects_removed_from_main_analysis']):
        raise ValueError('runtime-control or outlier-sensitivity evidence mismatch')
    if (repeatability['identity_hash'] != plan['identity_hash'] or repeatability['new_cases'] != 4
        or not repeatability['source_records_and_original_meshes_unchanged']):
        raise ValueError('runtime repeatability audit is incomplete')
    scheduler = read(OUT / 'scheduler_status.json')
    repair_paths = list((OUT / 'repairs').glob('*_status.json'))
    repairs = [read(path) for path in repair_paths]
    repaired_ranges = {tuple(item['range']) for item in repairs if item['state'] == 'complete' and item['exit_status'] == 0
                       and item['identity_hash'] == plan['identity_hash']}
    unresolved_ranges = [item['range'] for item in scheduler['failures'] if tuple(item['range']) not in repaired_ranges]
    if (native['identity_hash'] != plan['identity_hash'] or native['record_count'] != 320
        or aligned['record_count'] != 320 or not registration['passed']
        or not visuals['complete'] or not visual_integrity['passed'] or not gallery['passed']
        or scheduler['state'] not in ('complete', 'worker_failed') or scheduler['active'] or scheduler['queue']
        or unresolved_ranges or any(item['state'] != 'complete' for item in repairs)):
        raise ValueError('finalization requires complete, verified generation/alignment/visualization')
    runtime_paths = list((OUT / 'workers').glob('*_binding.json'))
    if not runtime_paths:
        raise ValueError('no recorded generation runtime bindings')
    for path in runtime_paths:
        runtime = read(path)
        if (runtime['identity_hash'] != plan['identity_hash'] or runtime['evaluation_uid_count'] != 64
            or not runtime['slat_training_disjoint'] or runtime['weights'] != 'ema'
            or runtime['sha256'] != plan['plan']['training_audit']['frozen_slat30k']['checkpoint']['sha256']):
            raise ValueError('generation runtime or training-disjointness binding mismatch')
    expected = {f'{r["index"]:03d}_{c["seed"]}_{c["mode"]}' for r in plan['plan']['rows'] for c in r['cases']}
    paths = list((OUT / 'alignment/records').glob('*.json'))
    if {p.stem for p in paths} != expected or len(paths) != 320:
        raise ValueError('wrong final alignment cohort')
    records = [read(path) for path in paths]
    groups = {'clean': [r for r in records if r['domain'] == 'clean'],
              'strong_occluded': [r for r in records if r['domain'] == 'strong_occluded'],
              'matched16_clean': [r for r in records if r['domain'] == 'clean' and r['object_index'] < 16]}
    stats, maximum_error = {}, 0.0
    for domain, rows in groups.items():
        stats[domain] = {}
        for field in ('native', 'se3_aligned'):
            stats[domain][field] = {}
            for metric in ('chamfer_l1', 'fscore_0p02'):
                item = independent_effect(rows, field, metric)
                reference = aligned['groups'][domain]['paired'][field]['raw_minus_parent'][metric]
                errors = [abs(item['mean_improvement'] - reference['mean_improvement']),
                          max(abs(a-b) for a, b in zip(item['ci95'], reference['object_bootstrap_ci95'])),
                          abs(item['parent_mean'] - aligned['groups'][domain]['summary'][field]['parent'][metric]),
                          abs(item['v2_mean'] - aligned['groups'][domain]['summary'][field]['raw'][metric])]
                if (item['wins'], item['ties'], item['losses']) != (reference['wins'], reference['ties'], reference['losses']):
                    raise ValueError('independent win/tie/loss mismatch')
                if field == 'native':
                    source = native['same16_clean'] if domain == 'matched16_clean' else native['cohorts'][domain]
                    reference_native = source['paired'][metric]
                    errors += [abs(item['mean_improvement'] - reference_native['mean_improvement']),
                               max(abs(a-b) for a, b in zip(item['ci95'], reference_native['object_bootstrap_ci95']))]
                maximum_error = max(maximum_error, *errors)
                stats[domain][field][metric] = item
    if maximum_error > 2e-12:
        raise ValueError(f'independent statistics mismatch: {maximum_error}')
    check = {'passed': True, 'independent_pairing_and_bootstrap': True,
             'max_absolute_aggregate_difference': maximum_error,
             'cohorts': stats, 'scope': plan['plan']['cohort_policy']}
    save(OUT / 'independent_statistics.json', check)
    with (OUT / 'metrics_summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['cohort', 'frame', 'metric', 'objects', 'parent', 'v2', 'improvement', 'ci95_low', 'ci95_high', 'wins', 'ties', 'losses'])
        for domain, fields in stats.items():
            for field, metrics in fields.items():
                for metric, item in metrics.items():
                    writer.writerow([domain, field, metric, item['object_count'], item['parent_mean'], item['v2_mean'],
                                     item['mean_improvement'], *item['ci95'], item['wins'], item['ties'], item['losses']])
    plot_effects(stats)
    evidence = [OUT / p for p in ('plan.json', 'input_audit.json', 'aggregate.json', 'statistics_self_test.json',
        'alignment/protocol.json', 'alignment/self_test.json', 'alignment/aggregate.json', 'alignment/verification.json',
        'alignment/complete.json', 'visual_report_manifest.json', 'visual_integrity.json', 'gallery_validation.json',
        'independent_statistics.json', 'metrics_summary.csv', 'figures/generalization_effects.png',
        'figures/generalization_effects.pdf', 'run_journal.jsonl', 'reuse_identity_audit.json', 'scheduler_status.json')]
    evidence += [Path(__file__), Path(__file__).with_name('run_heldout_mesh_workers.py'), *runtime_paths]
    evidence += sorted((OUT / 'failures').glob('*.json'))
    evidence += sorted((OUT / 'scheduler_history').glob('*'))
    evidence += repair_paths
    evidence += [OUT / 'runtime_repeatability' / p for p in ('protocol.json', 'compare.json', 'run_status.json')]
    evidence += [Path(__file__).with_name('repeatability_audit.py')]
    evidence += [OUT / p for p in ('identical_support_runtime_audit.json',
                                  'audit_identical_support.py', 'object_gain_robustness.json')]
    support_sensitivity = identical_support['secondary_sensitivity']
    original_clean = support_sensitivity['original_clean']['paired']
    zero_effect_clean = support_sensitivity['zero_method_effect_on_identical_support_clean']['paired']
    report = {'status': 'complete', 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
              'identity_hash': plan['identity_hash'], 'comparison': 'V2 SS step2000 raw vs parent, frozen SLat30K EMA',
              'record_count': 320, 'generated_meshes': native['source_counts']['generated'],
              'scheduler_recovery_history': scheduler.get('failed_attempts', []) + scheduler['failures'],
              'coordinator_exit_state': scheduler['state'],
              'independent_range_repairs': repairs,
              'all_planned_cases_recovered_and_complete': True,
              'runtime_repeatability': {'additional_control_meshes': 4,
                  'mesh_bit_exact_controls': sum(c['repeat_vs_original']['mesh_sha256_equal'] for c in repeatability['comparisons'].values()),
                  'max_observed_cd_difference': max(abs(c['repeat_vs_original']['metric_delta_right_minus_left']['chamfer_l1']) for c in repeatability['comparisons'].values()),
                  'max_observed_fscore_0p02_difference': max(abs(c['repeat_vs_original']['metric_delta_right_minus_left']['fscore_0p02']) for c in repeatability['comparisons'].values()),
                  'interpretation': repeatability['interpretation'],
                  'comparison_report': binding(OUT / 'runtime_repeatability/compare.json')},
              'identical_support_controls': {
                  'case_count': identical_support['control_case_count'],
                  'maximum_observed_absolute_delta': identical_support['maximum_observed_absolute_delta'],
                  'interpretation': identical_support['interpretation'],
                  'sensitivity_only_formal_records_unchanged': True,
                  'clean_mean_gain_with_control_effect_set_to_zero': {
                      metric: {'formal': original_clean[metric]['mean_improvement'],
                               'sensitivity': zero_effect_clean[metric]['mean_improvement']}
                      for metric in ('chamfer_l1', 'fscore_0p02')},
                  'report': binding(OUT / 'identical_support_runtime_audit.json')},
              'object_gain_robustness': binding(OUT / 'object_gain_robustness.json'),
              'reused_clean_parent_meshes': native['source_counts']['reused'], 'gt_meshes': 64,
              'results': stats, 'same16_strong_minus_clean_relative_gain': aligned['matched16_strong_minus_clean_relative_gain'],
              'pose_correction_diagnostics': {key: value['pose'] for key, value in aligned['groups'].items()},
              'verification': {'generation_all_successful': True, 'proper_se3': registration,
                               'visuals': visual_integrity, 'independent_statistics_max_abs_difference': maximum_error,
                               'generation_runtime_bindings_checked': len(runtime_paths),
                               'gallery': gallery,
                               'failed_generation_attempts': native['failed_attempt_count']},
              'visualization': {key: visuals[key] for key in ('native_render_images', 'aligned_render_images',
                               'overlay_images', 'atlas_pages', 'figure_pairs_png_pdf', 'representative_indices')},
              'gallery': str(OUT / 'index.html'),
              'limitations': ['Dev64 was already used for development; not an untouched final test set.',
                  'Training disjointness covers audited checkpoint data_identity UIDs, not all public backbone pretraining data.',
                  'GT is the official SLat VAE proxy, not the original high-resolution source mesh.',
                  'Strong occlusion contains 16 preselected objects and synthetic corruptions.',
                  'Object bootstrap covers fixed trained weights and two inference seeds, not training-run variation.',
                  'Observed floating-point inference variation is documented separately; four reruns do not bound all other cases.',
                  'SE(3) is GT-assisted diagnosis. No scale, reflection, nonrigid warp, or component filtering.',
                  'Fitted rotation is affected by object symmetry and shape; it is not ground-truth pose error.'],
              'evidence': [binding(path) for path in evidence]}
    save(OUT / 'final_report.json', report)
    print(json.dumps({'status': report['status'], 'results': stats}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
