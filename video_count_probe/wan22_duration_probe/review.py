from __future__ import annotations

import html
from pathlib import Path
import random

from video_count_probe.core import atomic_json, checked_plan, file_sha, read_json, writer_lock
from video_count_probe.wan22_action_probe.review import make_sheet, records
from video_count_probe.wan22_squat_count.review import tri_and, validate_annotation
from .protocol import CONDITIONS, scoring_plan, verify_reference
from .runtime import verify_noise

METRICS = ('exact_count', 'count_and_stop', 'last_second_rest', 'initial_upright')
CRITERION = ('Record each visible upright -> lowered hips/bent knees -> upright cycle. '
             'Record initial/terminal partial attempts separately. Do not count low-position bobbing as another cycle. '
             'Judge the final second separately; allow small hand motion/breathing. Unknown values stay null.')


def score(label, condition):
    result = {k: None for k in (*METRICS, 'observed_complete_count', 'stopped_after_target',
                               'target_completion_seconds', 'completed_before_final_second',
                               'initial_partial', 'extra_partial_after_target', 'cycle_times_seconds')}
    if not label.get('reviewed'):
        return result
    fps = condition['config']['fps']
    target = condition['target_count']
    cycles, partials = label.get('complete_cycles'), label.get('partial_attempts')
    known = label['count_judgeable']
    count = len(cycles) if known else None
    exact = count == target if known else None
    stopped = completion = deadline = extra = None
    if known and count < target:
        stopped = deadline = False
    elif known:
        end = cycles[target - 1]['end_frame']
        completion = end / fps
        deadline = end <= condition['config']['frames'] - 1 - fps
        extra = any(p['start_frame'] >= end for p in partials) if partials is not None else None
        stopped = False if count > target or extra is True else True if extra is False else None
    return {**result, 'observed_complete_count': count, 'exact_count': exact,
            'stopped_after_target': stopped, 'count_and_stop': tri_and([exact, stopped]),
            'last_second_rest': label.get('final_second_upright_still'),
            'initial_upright': label.get('initial_upright'), 'target_completion_seconds': completion,
            'completed_before_final_second': deadline, 'extra_partial_after_target': extra,
            'initial_partial': any(p['start_frame'] == 0 for p in partials) if partials is not None else None,
            'cycle_times_seconds': [{k: v / fps for k, v in event.items()} for event in cycles] if known else None}


def aggregate(rows):
    result = {'expected': len(rows), 'generated': sum(r['generated'] for r in rows),
              'reviewed': sum(r['reviewed'] for r in rows)}
    for metric in METRICS:
        values = [r[metric] for r in rows]
        yes, no = sum(x is True for x in values), sum(x is False for x in values)
        result[metric] = {'true': yes, 'false': no, 'unknown_or_unreviewed': len(values) - yes - no}
    return result


def paired(rows, first, second):
    groups = {condition: {r['seed']: r for r in rows if r['condition'] == condition} for condition in (first, second)}
    seeds = sorted(groups[first].keys() & groups[second].keys())
    result = {'first': first, 'second': second, 'expected_seed_pairs': len(seeds),
              'noise_relation': 'identical_for_verified_generated_B' if first == 'a5' else 'different_shapes_seed_matched_only'}
    for metric in ('exact_count', 'count_and_stop'):
        diffs = [int(groups[second][seed][metric]) - int(groups[first][seed][metric]) for seed in seeds
                 if groups[first][seed][metric] is not None and groups[second][seed][metric] is not None]
        result[metric] = {'decided_pairs': len(diffs), 'second_wins': diffs.count(1),
                          'first_wins': diffs.count(-1), 'ties': diffs.count(0),
                          'mean_second_minus_first': sum(diffs) / len(diffs) if diffs else None}
    return result


def checked_records(run, plan, reference_records):
    completed = records(run, plan)
    for record in completed.values():
        spec = plan['conditions'][record['job']['condition']]
        if record.get('config') != spec['config'] or record['prompt'] != spec['prompt']:
            raise ValueError('Record configuration/prompt does not match condition')
        if any(record[k] != spec['config'][k] for k in ('frames', 'fps', 'height', 'width')):
            raise ValueError('Record geometry differs from plan')
        verify_noise(record['job'], record['initial_latent_sha256'], reference_records)
    return completed


def report(root, mode, annotations=None, write=True):
    run = Path(root) / mode
    plan = checked_plan(run / 'plan.json')
    _, _, reference_records, reference_labels, _ = verify_reference(plan)
    completed = checked_records(run, plan, reference_records)
    source = Path(annotations) if annotations else run / 'annotations.json'
    data = read_json(source) if source.exists() else None
    if annotations and data is None:
        raise FileNotFoundError(source)
    labels = {}
    if data:
        if data['plan_hash'] != plan['plan_hash'] or data.get('rater_type') not in ('ai', 'human') or not data.get('rater', '').strip():
            raise ValueError('Annotation plan binding/reviewer provenance required')
        for row in data['items']:
            jid = row['job_id']
            if jid in labels or jid not in completed:
                raise ValueError('Duplicate/unknown/ungenerated annotation')
            record = completed[jid]
            validate_annotation(row, record, scoring_plan(plan['conditions'][record['job']['condition']]))
            labels[jid] = row
    rows = []
    if plan['reference_conditions_in_report']:
        for seed, record in reference_records.items():
            label = reference_labels[seed]
            rows.append({'id': record['job']['id'], 'condition': 'a5', 'seed': seed, 'generated': True,
                         'reviewed': label['reviewed'], 'provenance': 'reused_reference', **score(label, plan['conditions']['a5'])})
    for job in plan['jobs']:
        label = labels.get(job['id'], {})
        rows.append({**job, 'generated': job['id'] in completed, 'reviewed': label.get('reviewed', False),
                     'provenance': 'new_generation', **score(label, plan['conditions'][job['condition']])})
    new_rows = [r for r in rows if r['provenance'] == 'new_generation']
    errors = [read_json(run / 'errors' / (j['id'] + '.json')) for j in plan['jobs']
              if j['id'] not in completed and (run / 'errors' / (j['id'] + '.json')).exists()]
    decided = sum(r['reviewed'] and all(r[k] is not None for k in ('exact_count', 'count_and_stop', 'last_second_rest')) for r in new_rows)
    result = {'plan_hash': plan['plan_hash'], 'scientific_evaluation': plan['scientific_evaluation'],
              'generated_new': len(completed), 'expected_new': len(plan['jobs']),
              'reused_reference': len(rows) - len(new_rows), 'reviewed_new': sum(r['reviewed'] for r in new_rows),
              'decided_new': decided, 'execution_errors': sum(e.get('state') != 'interrupted' for e in errors),
              'interrupted_jobs': sum(e.get('state') == 'interrupted' for e in errors),
              'assessment_complete': len(completed) == decided == len(plan['jobs']) and
                  all(r['exact_count'] is not None and r['count_and_stop'] is not None for r in rows),
              'reference_provenance': plan['reference'],
              'new_review_provenance': {'rater': data['rater'], 'rater_type': data['rater_type'],
                                        'path': str(source), 'sha256': file_sha(source)} if data else None,
              'verified_a_b_noise_pairs': sum(r['job']['condition'] == 'b5' for r in completed.values()),
              'conditions': {c: aggregate([r for r in rows if r['condition'] == c]) for c in CONDITIONS
                             if any(r['condition'] == c for r in rows)},
              'paired': {'a5_vs_b5': paired(rows, 'a5', 'b5'), 'b5_vs_c8': paired(rows, 'b5', 'c8')},
              'per_video': rows,
              'resources': {c: {'peak_allocated_mib': max((r['peak_allocated_mib'] for r in completed.values() if r['job']['condition'] == c), default=None),
                                 'peak_reserved_mib': max((r['peak_reserved_mib'] for r in completed.values() if r['job']['condition'] == c), default=None),
                                 'generation_seconds': [r['generation_seconds'] for r in completed.values() if r['job']['condition'] == c]}
                            for c in ('b5', 'c8')},
              'limitations': plan['limits'] + ['last_second_rest is diagnostic, not an extra clock requirement on B/C. No quality claim before explicit annotation.']}
    if write:
        with writer_lock(run / '.writer.lock'):
            atomic_json(run / 'summary.json', result)
    print(f'generated_new={len(completed)}/{len(plan["jobs"])} reused_A={result["reused_reference"]} '
          f'reviewed_new={result["reviewed_new"]} errors={result["execution_errors"]} '
          f'assessment_complete={result["assessment_complete"]}; {run / "summary.json"}', flush=True)
    return result


def review(root, mode):
    run = Path(root) / mode
    with writer_lock(run / '.writer.lock'):
        plan = checked_plan(run / 'plan.json')
        _, _, reference_records, _, _ = verify_reference(plan)
        completed = checked_records(run, plan, reference_records)
        jobs = list(plan['jobs'])
        random.Random(62941).shuffle(jobs)
        blind, private, labels, cards = [], [], [], []
        for index, job in enumerate(jobs, 1):
            if job['id'] not in completed:
                continue
            record = completed[job['id']]
            config = plan['conditions'][job['condition']]['config']
            sheet = run / 'review_frames' / f'{index:03d}.jpg'
            sidecar = sheet.with_suffix('.json')
            binding = {'video_sha256': record['video_sha256'], 'plan_hash': plan['plan_hash']}
            if not sheet.exists() or not sidecar.exists() or read_json(sidecar) != binding:
                make_sheet(run / record['video'], sheet, config,
                           f'Clip {index:03d} | squat | 8 fps overview, original 24 fps | condition/seed hidden')
                atomic_json(sidecar, binding)
            blind.append({'blind_id': index, 'sheet': str(sheet), 'criterion': CRITERION,
                          'last_frame': config['frames'] - 1, 'last_second_start': config['frames'] - 1 - config['fps']})
            private.append({'blind_id': index, **job, 'video_sha256': record['video_sha256'], 'video': record['video']})
            labels.append({'blind_id': index, 'job_id': job['id'], 'video_sha256': record['video_sha256'],
                           'reviewed': False, 'count_judgeable': None, 'complete_cycles': None, 'partial_attempts': None,
                           'initial_upright': None, 'final_second_upright_still': None, 'notes': ''})
            cards.append(f'<article><h2>Clip {index:03d}</h2><video controls preload="metadata" width="768" src="{record["video"]}"></video>'
                         f'<p>24 fps; last second: frames {config["frames"] - 1 - config["fps"]}..{config["frames"] - 1}. '
                         f'<a href="review_frames/{index:03d}.jpg">Time-labelled frames</a></p></article>')
        atomic_json(run / 'review_frames/blind_manifest.json', {'plan_hash': plan['plan_hash'], 'items': blind,
                    'limits': 'Condition/seed labels hidden, but duration remains visible. Not fully blind to length; disclose reviewer provenance.'})
        atomic_json(run / 'review_frames/private_mapping.json', {'plan_hash': plan['plan_hash'], 'items': private})
        atomic_json(run / 'annotations_template.json', {'plan_hash': plan['plan_hash'], 'rater': '', 'rater_type': 'ai', 'items': labels})
        document = ('<!doctype html><meta charset="utf-8"><title>Squat duration diagnostic review</title>'
                    '<style>body{font:16px/1.6 sans-serif;max-width:950px;margin:24px auto}video{max-width:100%}article{margin:30px 0}</style>'
                    '<h1>New videos only; reference labels preserved separately</h1><p>' + html.escape(CRITERION) + '</p>'
                    '<p>Annotate start_frame/bottom_frame/end_frame for each full cycle, start_frame/end_frame for each partial. '
                    'Use null for uncertainty. Save completed annotations separately as annotations.json. Existing labels are never replaced.</p>' + ''.join(cards))
        (run / 'review.html').write_text(document)
    print(f'Review ready: {run / "review.html"}; {run / "review_frames/blind_manifest.json"}', flush=True)
