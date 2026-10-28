"""Validate and summarize the completed, immutable HSSD full-branch diagnostics."""
import hashlib
import json
import os
from pathlib import Path
from statistics import mean

PACKAGE = Path(__file__).resolve().parents[2]
OUT = PACKAGE / 'outputs/hssd_full_diagnostic_20261021'
PILOT = PACKAGE / 'outputs/hssd_pilot_20261020'


def binding(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def read_cohort(folder):
    paths = sorted(folder.glob('*.json'))
    rows = [json.loads(p.read_text()) for p in paths]
    if len(rows) != 16 or [r['index'] for r in rows] != list(range(16)):
        raise ValueError(f'Incomplete original16 cohort: {folder}')
    if len({r['uid'] for r in rows}) != 16:
        raise ValueError(f'Duplicate object: {folder}')
    identities = [{k: v for k, v in r['identity'].items() if k != 'geometry_record'} for r in rows]
    if any(i != identities[0] for i in identities):
        raise ValueError(f'Mixed diagnostic identities: {folder}')
    return rows, paths


def validate_binding(expected, checked):
    if expected['path'] in checked:
        actual = checked[expected['path']]
    else:
        actual = binding(expected['path'])['sha256']
        checked[expected['path']] = actual
    if actual != expected['sha256']:
        raise ValueError(f"Changed bound artifact: {expected['path']}")


def make_report():
    checked = {}
    frozen = {}
    for name in ['launch_sources.json', 'frozen_protocol.json', 'smoke.json']:
        receipt = json.loads((PILOT / name).read_text())
        for source in receipt['sources']:
            validate_binding(source, checked)
        frozen[name] = {'verified_sources': len(receipt['sources']), 'receipt': binding(PILOT / name)}
    report = {'scope': 'Existing checkpoints; diagnostics only. No training, optimizer updates or mesh sampling.',
        'aggregation': 'Equal object weight; forwards on fixed original indices 0,1,2,3 only, two times each.',
        'limitations': ['Geometry uses FP32 cached LaRI; SS conditions/forwards use bf16 autocast.',
            'Surface queries are GT64 occupied voxel centers / their SS16 cell centers, not continuous mesh vertices.',
            'Support is the maximum over 8 views; coverage threshold 0.1 is diagnostic, not a revised fit gate.',
            'Far16 is >0.1 from every GT64 surface voxel center; responses are leakage proxies, not exact mesh FPs.',
            'Oracle/head/tolerance/support interventions change inference distributions and are not deployable scores.',
            'Fixed-noise velocity MSE measures neither complete CFG sampling nor final mesh quality.',
            'Hidden-before-injection RMS is reconstructed by subtraction after the addition, so its ratio is approximate under finite precision.',
            'Full and no_layer differ in trainable input information; this is not appearance plus/minus geometry.',
            'The separate occluded SLat failure is not resolved by these SS probes.'],
        'frozen_source_verification': frozen, 'geometry_summary': binding(OUT / 'geometry_summary.json'),
        'summary_source': binding(Path(__file__)), 'domains': {}}
    geometry_summary = json.loads((OUT / 'geometry_summary.json').read_text())
    for domain in ['clean', 'occluded']:
        geometry, gp = read_cohort(OUT / 'geometry' / domain)
        for row in geometry:
            for b in row['identity']['sources'] + [row['identity']['checkpoint'], row['identity']['manifest'], row['prediction']]:
                validate_binding(b, checked)
        for b in geometry_summary['domains'][domain]['source_records']:
            validate_binding(b, checked)
        arms = {}
        cp = []
        for arm in ['full', 'no_layer']:
            rows, paths = read_cohort(OUT / 'conditions' / domain / arm)
            cp.extend(paths)
            if [r['uid'] for r in rows] != [r['uid'] for r in geometry]:
                raise ValueError('Geometry/condition object mismatch')
            for row in rows:
                identity = row['identity']
                for b in identity['sources'] + [identity['checkpoint'], identity['manifest'], identity['geometry_record']]:
                    validate_binding(b, checked)
                if identity['forward_indices'] != [0, 1, 2, 3] or identity['times'] != [.25, .75]:
                    raise ValueError('Unexpected forward scope')
                if list(row['variants']) != identity['variants']:
                    raise ValueError('Missing/reordered variants')
                for v in row['variants'].values():
                    if [f['time'] for f in v['forwards']] != ([.25, .75] if row['index'] < 4 else []):
                        raise ValueError('Missing/extra fixed-noise forwards')
            arms[arm] = rows
        parent_differences = []
        for a, b in zip(arms['full'][:4], arms['no_layer'][:4]):
            for fa, fb in zip(a['variants']['parent']['forwards'], b['variants']['parent']['forwards']):
                parent_differences.append(abs(fa['flow_loss'] - fb['flow_loss']))
        if max(parent_differences) != 0.:
            raise ValueError('Paired parent fixed-noise losses differ across arms')
        section = {'geometry': {k: v for k, v in geometry_summary['domains'][domain].items() if k != 'source_records'},
            'parent_loss_max_abs_difference_between_arms': max(parent_differences),
            'production_support_max_abs_error': max(max(r['production_matcher_max_abs_error'].values()) for r in geometry),
            'geometry_records': [binding(p) for p in gp], 'condition_records': [binding(p) for p in cp],
            'variants': {}}
        for key in ['target_confidence_training_target', 'target_ownership', 'target_existence', 'front_target', 'behind_non_target']:
            values = [r['depth'][key]['mean'] for r in geometry if r['depth'][key]['mean'] is not None]
            section['geometry'][key + '_mean'] = mean(values) if values else None
        for arm, rows in arms.items():
            for variant in rows[0]['variants']:
                if arm == 'no_layer' and variant == 'parent':
                    continue
                conditions = [r['variants'][variant]['condition'] for r in rows]
                all_forwards = [f for r in rows for f in r['variants'][variant]['forwards']]
                block_keys = ['parent_appearance_injection_rms', 'layer_injection_rms',
                              'layer_weight_times_condition_rms', 'layer_bias_rms']
                result = {'objects_conditions': len(rows), 'objects_forwards': 4,
                    'layer_condition_rms': mean(c['actual_layer_rms'] for c in conditions),
                    'blocks_mean': {k: mean(mean(b[k] for b in c['blocks']) for c in conditions) for k in block_keys},
                    'flow_loss_mean': mean(f['flow_loss'] for f in all_forwards),
                    'flow_loss_by_time': {str(t): mean(f['flow_loss'] for f in all_forwards if f['time'] == t) for t in [.25, .75]},
                    'output_delta_from_parent_rms': mean(f['output_delta_from_parent_rms'] for f in all_forwards),
                    'layer_to_hidden_ratio_mean': mean(mean(b['layer_to_hidden_ratio'] for b in f['block_ratios']) for f in all_forwards),
                    'per_object': []}
                for row in rows[:4]:
                    loss = mean(f['flow_loss'] for f in row['variants'][variant]['forwards'])
                    parent_loss = mean(f['flow_loss'] for f in row['variants']['parent']['forwards'])
                    full_loss = mean(f['flow_loss'] for f in arms['full'][row['index']]['variants']['full']['forwards'])
                    result['per_object'].append({'index': row['index'], 'uid': row['uid'], 'flow_loss': loss,
                        'gain_over_parent': parent_loss - loss, 'gain_over_full': full_loss - loss})
                result['objects_better_than_parent'] = sum(r['gain_over_parent'] > 0 for r in result['per_object'])
                result['objects_better_than_full'] = sum(r['gain_over_full'] > 0 for r in result['per_object'])
                result['flow_loss_gain_over_parent'] = mean(r['gain_over_parent'] for r in result['per_object'])
                result['flow_loss_gain_over_full'] = mean(r['gain_over_full'] for r in result['per_object'])
                section['variants'][variant] = result
        report['domains'][domain] = section
    report['verified_artifact_count'] = len(checked)
    report['counts'] = {'geometry_records': 32, 'condition_records': 64, 'ss_forwards': 176}
    return report


def plot(report):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/palr-hssd-diagnostic-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    colors = {'clean': '#2878b5', 'occluded': '#e07a25'}
    variants = ['predicted', 'gt_depth', 'gt_ownership', 'confidence_one', 'tolerance_005', 'oracle']
    labels = ['Predicted', 'GT depth', 'GT owner', 'Conf = 1', 'Tau = .05', 'GT oracle']
    x = np.arange(len(variants))
    for offset, domain in [(-.19, 'clean'), (.19, 'occluded')]:
        s = report['domains'][domain]['geometry']['variants']
        axes[0, 0].bar(x + offset, [100 * s[v]['surface64']['fraction_gt_01'] for v in variants], .36,
                       label=domain, color=colors[domain])
        axes[0, 1].bar(x + offset, [100 * s[v]['far16']['fraction_gt_01'] for v in variants], .36,
                       label=domain, color=colors[domain])
    for ax in axes[0]:
        ax.set_xticks(x, labels, rotation=20)
        ax.set_ylabel('Support > 0.1 (%)')
    axes[0, 0].set_title('GT64 surface coverage | all 16 objects, equal weight')
    axes[0, 1].set_title('Far16 leakage proxy | distance from GT64 surface > 0.1')
    axes[0, 0].legend(frameon=False)
    injection_labels = ['Clean full', 'Clean no-layer', 'Occluded full', 'Occluded no-layer']
    pairs = [(d, v) for d in ['clean', 'occluded'] for v in ['full', 'no_layer']]
    signal = [report['domains'][d]['variants'][v]['blocks_mean']['layer_weight_times_condition_rms'] for d, v in pairs]
    bias = [report['domains'][d]['variants'][v]['blocks_mean']['layer_bias_rms'] for d, v in pairs]
    axes[1, 0].bar(np.arange(4) - .18, signal, .35, label='W x condition', color='#2878b5')
    axes[1, 0].bar(np.arange(4) + .18, bias, .35, label='bias', color='#8e8e8e')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xticks(np.arange(4), injection_labels, rotation=20)
    axes[1, 0].set_ylabel('Mean block RMS (log scale)')
    axes[1, 0].set_title('Actual injection | all 16 objects / 24 blocks')
    axes[1, 0].legend(frameon=False)
    fv = ['full', 'view_gate_only', 'injection_only', 'oracle', 'confidence_one', 'tolerance_005', 'no_layer']
    fl = ['Full', 'Gate only', 'Inject only', 'Oracle', 'Conf = 1', 'Tau = .05', 'No-layer']
    for domain in ['clean', 'occluded']:
        axes[1, 1].plot(np.arange(len(fv)), [1000 * report['domains'][domain]['variants'][v]['flow_loss_gain_over_parent'] for v in fv],
                        'o-', label=domain, color=colors[domain])
    axes[1, 1].axhline(0, color='gray', lw=.8)
    axes[1, 1].set_xticks(np.arange(len(fv)), fl, rotation=20)
    axes[1, 1].set_ylabel('Parent MSE - intervention MSE (x 1e-3)')
    axes[1, 1].set_title('Fixed-noise forwards | first 4 objects, t=.25/.75')
    axes[1, 1].legend(frameon=False)
    fig.suptitle('HSSD full-branch diagnostics: existing checkpoints, no training or mesh sampling', fontsize=14)
    folder = OUT / 'figures'
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / 'diagnostic_overview.png', dpi=180)
    fig.savefig(folder / 'diagnostic_overview.pdf')
    plt.close(fig)


def main():
    report = make_report()
    write(OUT / 'report.json', report)
    plot(report)
    print(json.dumps({'counts': report['counts'], 'verified_artifacts': report['verified_artifact_count'],
        'report': str(OUT / 'report.json')}, indent=2))


if __name__ == '__main__':
    main()
