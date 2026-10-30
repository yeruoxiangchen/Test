from __future__ import annotations

import html
import random
from pathlib import Path

from video_count_probe.core import atomic_json, checked_plan, file_sha, read_json, writer_lock
from video_count_probe.wan22_action_probe.review import make_sheet, records
from .protocol import METHODS

CRITERION = ('Record every complete upright -> lowered hips/bent knees -> upright cycle, with its start, bottom and end frame. '
             'Record unfinished attempts separately. Judge whether frames 96..120 show upright rest. '
             'Do not infer hidden events; uncertain values stay null. Normal breathing or small hand motion does not invalidate rest.')


def tri_and(values):
    if any(v is False for v in values):
        return False
    return None if any(v is None for v in values) else True


def validate_annotation(row, record, plan):
    if row['job_id'] != record['job']['id'] or row['video_sha256'] != record['video_sha256']:
        raise ValueError('Annotation video identity/hash mismatch')
    if type(row.get('reviewed')) is not bool:
        raise ValueError('reviewed must be a boolean')
    if not row['reviewed']:
        return
    if not isinstance(row.get('notes'), str) or not row['notes'].strip():
        raise ValueError('Reviewed clips require visual evidence notes')
    if type(row.get('count_judgeable')) is not bool:
        raise ValueError('Reviewed clips require count_judgeable=true/false')
    for key in ('initial_upright', 'final_second_upright_still'):
        if row.get(key) is not None and type(row[key]) is not bool:
            raise ValueError(f'{key} must be boolean or null')
    cycles, partials = row.get('complete_cycles'), row.get('partial_attempts')
    if row['count_judgeable'] and not isinstance(cycles, list):
        raise ValueError('Judgeable counts require a list of complete cycles, including [] for zero')
    if not row['count_judgeable'] and cycles is not None:
        raise ValueError('Uncertain counts must remain null, not an assumed count')
    if partials is not None and not isinstance(partials, list):
        raise ValueError('partial_attempts must be list or null')
    intervals = []
    last = plan['scoring']['last_frame']
    for events, fields in ((cycles or [], ('start_frame', 'bottom_frame', 'end_frame')),
                          (partials or [], ('start_frame', 'end_frame'))):
        previous_end = -1
        for event in events:
            points = [event[k] for k in fields]
            if any(type(v) is not int or not 0 <= v <= last for v in points):
                raise ValueError('Frame indices must be integers within the video')
            if any(a >= b for a, b in zip(points, points[1:])) or points[0] < previous_end:
                raise ValueError('Cycle phases must be ordered, and events must not overlap')
            previous_end = points[-1]
            intervals.append((points[0], points[-1]))
    intervals.sort()
    if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
        raise ValueError('Complete and partial attempts cannot overlap')


def score(row, target, plan):
    empty = {'observed_complete_count': None, 'exact_count': None, 'stopped_after_target': None,
             'final_rest': None, 'target_completed_by_rest': None, 'joint_count_stop': None,
             'extra_partial_after_target': None, 'any_partial_attempt': None, 'initial_upright': None}
    if not row.get('reviewed'):
        return empty
    cycles, partials = row.get('complete_cycles'), row.get('partial_attempts')
    known = row['count_judgeable']
    count = len(cycles) if known else None
    exact = count == target if known else None
    final_rest = row.get('final_second_upright_still')
    stopped = deadline = extra = None
    if known and count < target:
        stopped = deadline = False
    elif known:
        end = cycles[target - 1]['end_frame']
        deadline = end <= plan['scoring']['rest_start_frame']
        extra = any(p['start_frame'] >= end for p in partials) if partials is not None else None
        stopped = False if count > target or extra is True else (True if extra is False else None)
    return {'observed_complete_count': count, 'exact_count': exact, 'stopped_after_target': stopped,
            'final_rest': final_rest, 'target_completed_by_rest': deadline,
            'joint_count_stop': tri_and([exact, stopped, final_rest, deadline]),
            'extra_partial_after_target': extra,
            'any_partial_attempt': bool(partials) if partials is not None else None,
            'initial_upright': row.get('initial_upright')}


def aggregate(rows):
    result = {'expected': len(rows), 'generated': sum(r['generated'] for r in rows),
              'reviewed': sum(r['reviewed'] for r in rows)}
    for metric in ('exact_count', 'stopped_after_target', 'final_rest', 'joint_count_stop'):
        values = [r[metric] for r in rows]
        positive, negative = sum(v is True for v in values), sum(v is False for v in values)
        unknown = len(values) - positive - negative
        result[metric] = {'true': positive, 'false': negative, 'unknown_or_unreviewed': unknown,
                          'success_lower_bound': positive, 'success_upper_bound': positive + unknown}
    return result


def report(root, mode, annotations=None, write=True):
    run = Path(root) / mode
    plan = checked_plan(run / 'plan.json')
    completed = records(run, plan)
    cases = {c['id']: c for c in plan['cases']}
    source = Path(annotations) if annotations else run / 'annotations.json'
    data = read_json(source) if source.exists() else None
    if annotations and data is None:
        raise FileNotFoundError(source)
    labels = {}
    if data:
        if (data['plan_hash'] != plan['plan_hash'] or data.get('rater_type') not in ('ai', 'human')
                or not isinstance(data.get('rater'), str) or not data['rater'].strip()):
            raise ValueError('Annotation plan binding and reviewer provenance are required')
        for row in data['items']:
            jid = row['job_id']
            if jid in labels or jid not in completed:
                raise ValueError('Duplicate, unknown, or ungenerated annotation job')
            validate_annotation(row, completed[jid], plan)
            labels[jid] = row
    rows = []
    for job in plan['jobs']:
        label = labels.get(job['id'], {})
        rows.append({**job, 'target_count': cases[job['case_id']]['target_count'],
                     'generated': job['id'] in completed, 'reviewed': label.get('reviewed', False),
                     **score(label, cases[job['case_id']]['target_count'], plan)})
    pairs = {}
    generated_pairs = 0
    for pair_id in dict.fromkeys(j['pair_id'] for j in plan['jobs']):
        group = {r['method']: r for r in rows if r['pair_id'] == pair_id}
        if set(group) != set(METHODS):
            raise ValueError('Plan contains an incomplete method pair')
        if all(r['generated'] for r in group.values()):
            noises = {completed[r['id']]['initial_latent_sha256'] for r in group.values()}
            if len(noises) != 1:
                raise ValueError(f'Paired initial latent mismatch: {pair_id}')
            generated_pairs += 1
        pairs[pair_id] = group
    paired = {}
    for metric in ('exact_count', 'joint_count_stop'):
        differences = []
        for group in pairs.values():
            a, b = group['phase_prompt'][metric], group['count_prompt'][metric]
            if a is not None and b is not None:
                differences.append(int(a) - int(b))
        paired[metric] = {'decided_pairs': len(differences), 'expected_pairs': len(pairs),
                          'phase_wins': differences.count(1), 'count_wins': differences.count(-1),
                          'ties': differences.count(0),
                          'phase_minus_count': sum(differences) / len(differences) if differences else None}
    reviewed = sum(r['reviewed'] for r in rows)
    decided = sum(r['reviewed'] and all(r[k] is not None for k in
                  ('exact_count', 'stopped_after_target', 'final_rest', 'joint_count_stop')) for r in rows)
    error_records = [read_json(run / 'errors' / (j['id'] + '.json')) for j in plan['jobs']
                     if j['id'] not in completed and (run / 'errors' / (j['id'] + '.json')).exists()]
    errors = sum(r.get('state') != 'interrupted' for r in error_records)
    result = {'plan_hash': plan['plan_hash'], 'generated': len(completed), 'expected': len(rows),
              'reviewed': reviewed, 'decided': decided, 'reviewed_uncertain': reviewed - decided,
              'execution_errors': errors, 'interrupted_jobs': len(error_records) - errors,
              'review_complete': len(completed) == reviewed == len(rows),
              'assessment_complete': len(completed) == decided == len(rows) and errors == 0,
              'scientific_evaluation': plan['scientific_evaluation'],
              'paired_initial_noise_verified': generated_pairs, 'expected_pairs': len(pairs),
              'review_provenance': {'rater': data['rater'], 'rater_type': data['rater_type'],
                                    'path': str(source), 'sha256': file_sha(source)} if data else None,
              'methods': {m: aggregate([r for r in rows if r['method'] == m]) for m in METHODS},
              'by_target_count': {str(n): {m: aggregate([r for r in rows if r['target_count'] == n and r['method'] == m])
                                          for m in METHODS} for n in (1, 2)},
              'paired': paired, 'per_video': rows,
              'peak_allocated_mib': max((r['peak_allocated_mib'] for r in completed.values()), default=None),
              'peak_reserved_mib': max((r['peak_reserved_mib'] for r in completed.values()), default=None),
              'limitations': plan['limits'] + ['Success bounds reflect unresolved annotations/missing results, not statistical confidence intervals.']}
    if write:
        with writer_lock(run / '.writer.lock'):
            atomic_json(run / 'summary.json', result)
    print(f'generated={len(completed)}/{len(rows)} reviewed={reviewed} decided={decided} '
          f'uncertain={reviewed-decided} errors={errors} assessment_complete={result["assessment_complete"]}', flush=True)
    return result


def review(root, mode):
    run = Path(root) / mode
    with writer_lock(run / '.writer.lock'):
        plan = checked_plan(run / 'plan.json')
        completed = records(run, plan)
        jobs = list(plan['jobs'])
        random.Random(52841).shuffle(jobs)
        blind, private, annotations, cards = [], [], [], []
        cases = {c['id']: c for c in plan['cases']}
        for index, job in enumerate(jobs, 1):
            if job['id'] not in completed:
                continue
            record = completed[job['id']]
            sheet = run / 'review_frames' / f'{index:03d}.jpg'
            sidecar = sheet.with_suffix('.json')
            binding = {'video_sha256': record['video_sha256'], 'plan_hash': plan['plan_hash']}
            if not sheet.exists() or not sidecar.exists() or read_json(sidecar) != binding:
                make_sheet(run / record['video'], sheet, plan['config'],
                           f'Clip {index:03d} | squat | 8 fps overview, original 24 fps | target and method hidden')
                atomic_json(sidecar, binding)
            blind.append({'blind_id': index, 'action': 'squat', 'criterion': CRITERION, 'sheet': str(sheet)})
            private.append({'blind_id': index, **job, 'target_count': cases[job['case_id']]['target_count'],
                            'video': record['video'], 'video_sha256': record['video_sha256']})
            annotations.append({'blind_id': index, 'job_id': job['id'], 'video_sha256': record['video_sha256'],
                                'reviewed': False, 'count_judgeable': None, 'complete_cycles': None,
                                'partial_attempts': None, 'initial_upright': None,
                                'final_second_upright_still': None, 'notes': ''})
            cards.append(f'<article><h2>Clip {index:03d}</h2><video controls preload="metadata" width="768" src="{record["video"]}"></video>'
                         f'<p><a href="review_frames/{index:03d}.jpg">Time-labelled overview</a></p></article>')
        atomic_json(run / 'review_frames/blind_manifest.json', {'plan_hash': plan['plan_hash'], 'items': blind,
                    'limits': 'Sheets hide target/method/seed. Reviewer provenance and any prior metadata exposure must still be disclosed.'})
        atomic_json(run / 'review_frames/private_mapping.json', {'plan_hash': plan['plan_hash'], 'items': private})
        atomic_json(run / 'annotations_template.json', {'plan_hash': plan['plan_hash'], 'rater': '', 'rater_type': 'ai', 'items': annotations})
        document = ('<!doctype html><meta charset="utf-8"><title>Squat count and stop review</title>'
                    '<style>body{font:16px/1.6 sans-serif;max-width:950px;margin:24px auto}video{max-width:100%}article{margin:30px 0}</style>'
                    '<h1>Squat count and stop review</h1><p>' + html.escape(CRITERION) + '</p>'
                    '<p>0..120 frames, 24 fps. Complete-cycle annotation: start_frame, bottom_frame, end_frame. '
                    'Partial-attempt annotation: start_frame, end_frame. Use annotations_template.json as a starting point; '
                    'save separately as annotations.json. Existing annotations are never replaced by review.</p>' + ''.join(cards))
        (run / 'review.html').write_text(document)
    print(f'Review ready: {run / "review.html"}; {run / "review_frames/blind_manifest.json"}', flush=True)
