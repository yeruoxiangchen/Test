from __future__ import annotations

import html
from pathlib import Path
import random

from video_count_probe.core import atomic_json, checked_plan, file_sha, read_json, writer_lock
from video_count_probe.wan22_action_probe.review import make_sheet, records
from video_count_probe.wan22_duration_probe.review import CRITERION, aggregate, score
from video_count_probe.wan22_squat_count.review import validate_annotation
from .protocol import CONDITIONS, scoring_plan, verify_reference
from .runtime import verify_noise


def checked_records(run, plan, refs):
    completed = records(run, plan)
    for record in completed.values():
        spec = plan['conditions'][record['job']['condition']]
        if record.get('config') != spec['config'] or record['prompt'] != spec['prompt']:
            raise ValueError('Record configuration/prompt mismatch')
        if any(record[k] != spec['config'][k] for k in ('frames', 'fps', 'height', 'width')):
            raise ValueError('Record geometry mismatch')
        verify_noise(record['job'], record['initial_latent_sha256'], refs)
        if record['initial_latent_shape'] != refs[record['job']['seed']]['initial_latent_shape']:
            raise ValueError('Reference/new latent shapes differ')
    return completed


def paired_response(rows):
    result = []
    for seed in sorted({r['seed'] for r in rows}):
        group = {r['condition']: r for r in rows if r['seed'] == seed}
        counts = {c: group.get(c, {}).get('observed_complete_count') for c in CONDITIONS}
        known = all(n is not None for n in counts.values())
        values = list(counts.values())
        result.append({'seed': seed, 'counts_by_target': counts, 'decided': known,
                       'flat_response': len(set(values)) == 1 if known else None,
                       'strictly_increasing': values[0] < values[1] < values[2] if known else None,
                       'exact_1_2_3': values == [1, 2, 3] if known else None,
                       'observed_n3_minus_n1': values[2] - values[0] if known else None})
    return result


def report(root, mode='pilot', annotations=None, write=True):
    run = Path(root) / mode
    plan = checked_plan(run / 'plan.json')
    _, _, refs, ref_labels, _ = verify_reference(plan)
    completed = checked_records(run, plan, refs)
    source = Path(annotations) if annotations else run / 'annotations.json'
    data = read_json(source) if source.exists() else None
    if annotations and data is None:
        raise FileNotFoundError(source)
    labels = {}
    if data:
        if data['plan_hash'] != plan['plan_hash'] or data.get('rater_type') not in ('ai', 'human') or not data.get('rater', '').strip():
            raise ValueError('Annotation binding/provenance missing')
        for item in data['items']:
            jid = item['job_id']
            if jid in labels or jid not in completed:
                raise ValueError('Duplicate/unknown/ungenerated annotation')
            record = completed[jid]
            validate_annotation(item, record, scoring_plan(plan['conditions'][record['job']['condition']]))
            labels[jid] = item
    rows = []
    if plan['reused_conditions']:
        for seed, record in refs.items():
            rows.append({'id': record['job']['id'], 'condition': 'n2', 'seed': seed, 'generated': True,
                         'reviewed': True, 'provenance': 'reused_reference',
                         **score(ref_labels[seed], plan['conditions']['n2'])})
    for job in plan['jobs']:
        label = labels.get(job['id'], {})
        rows.append({**job, 'generated': job['id'] in completed, 'reviewed': label.get('reviewed', False),
                     'provenance': 'new_generation', **score(label, plan['conditions'][job['condition']])})
    for row in rows:
        row['target_count'] = plan['conditions'][row['condition']]['target_count']
        count = row['observed_complete_count']
        row['observed_minus_requested'] = None if count is None else count - row['target_count']
    new = [r for r in rows if r['provenance'] == 'new_generation']
    errors = [read_json(run / 'errors' / (j['id'] + '.json')) for j in plan['jobs']
              if j['id'] not in completed and (run / 'errors' / (j['id'] + '.json')).exists()]
    decided = sum(r['reviewed'] and all(r[k] is not None for k in ('exact_count', 'count_and_stop', 'last_second_rest')) for r in new)
    result = {'plan_hash': plan['plan_hash'], 'scientific_evaluation': plan['scientific_evaluation'],
              'generated_new': len(completed), 'expected_new': len(plan['jobs']), 'reused_n2': len(rows) - len(new),
              'reviewed_new': sum(r['reviewed'] for r in new), 'decided_new': decided,
              'execution_errors': sum(e.get('state') != 'interrupted' for e in errors),
              'interrupted_jobs': sum(e.get('state') == 'interrupted' for e in errors),
              'assessment_complete': len(completed) == decided == len(plan['jobs']) and
                  all(r['exact_count'] is not None and r['count_and_stop'] is not None for r in rows),
              'reference_provenance': plan['reference'],
              'new_review_provenance': {k: data[k] for k in ('rater', 'rater_type')} | {'path': str(source), 'sha256': file_sha(source)} if data else None,
              'verified_noise_matches_to_n2': len(completed),
              'conditions': {c: aggregate([r for r in rows if r['condition'] == c]) for c in CONDITIONS if any(r['condition'] == c for r in rows)},
              'paired_count_response': paired_response(rows) if mode == 'pilot' else [], 'per_video': rows,
              'resources': {c: {'peak_allocated_mib': max((r['peak_allocated_mib'] for r in completed.values() if r['job']['condition'] == c), default=None),
                                 'peak_reserved_mib': max((r['peak_reserved_mib'] for r in completed.values() if r['job']['condition'] == c), default=None),
                                 'generation_seconds': [r['generation_seconds'] for r in completed.values() if r['job']['condition'] == c]}
                            for c in ('n1', 'n3')}, 'limitations': plan['limits']}
    if write:
        with writer_lock(run / '.writer.lock'):
            atomic_json(run / 'summary.json', result)
    print(f'generated_new={len(completed)}/{len(plan["jobs"])} reused_n2={result["reused_n2"]} '
          f'reviewed_new={result["reviewed_new"]} errors={result["execution_errors"]} '
          f'assessment_complete={result["assessment_complete"]}; {run / "summary.json"}', flush=True)
    return result


def review(root, mode='pilot'):
    run = Path(root) / mode
    with writer_lock(run / '.writer.lock'):
        plan = checked_plan(run / 'plan.json')
        _, _, refs, _, _ = verify_reference(plan)
        completed = checked_records(run, plan, refs)
        jobs = list(plan['jobs'])
        random.Random(72941).shuffle(jobs)
        public, private, labels, cards = [], [], [], []
        for bid, job in enumerate(jobs, 1):
            if job['id'] not in completed:
                continue
            record = completed[job['id']]
            config = plan['conditions'][job['condition']]['config']
            sheet = run / 'review_frames' / f'{bid:03d}.jpg'
            binding = {'video_sha256': record['video_sha256'], 'plan_hash': plan['plan_hash']}
            if not sheet.exists() or not sheet.with_suffix('.json').exists() or read_json(sheet.with_suffix('.json')) != binding:
                make_sheet(run / record['video'], sheet, config, f'Clip {bid:03d} | squat | 8fps overview, original 24fps | target/seed hidden')
                atomic_json(sheet.with_suffix('.json'), binding)
            public.append({'blind_id': bid, 'sheet': str(sheet), 'criterion': CRITERION, 'last_frame': 192, 'last_second_start': 168})
            private.append({'blind_id': bid, **job, 'video': record['video'], 'video_sha256': record['video_sha256']})
            labels.append({'blind_id': bid, 'job_id': job['id'], 'video_sha256': record['video_sha256'], 'reviewed': False,
                           'count_judgeable': None, 'complete_cycles': None, 'partial_attempts': None,
                           'initial_upright': None, 'final_second_upright_still': None, 'notes': ''})
            cards.append(f'<article><h2>Clip {bid:03d}</h2><video controls preload="metadata" src="{record["video"]}"></video>'
                         f'<p><a href="review_frames/{bid:03d}.jpg">Time-labelled frame overview</a></p></article>')
        atomic_json(run / 'review_frames/blind_manifest.json', {'plan_hash': plan['plan_hash'], 'items': public})
        atomic_json(run / 'review_frames/private_mapping.json', {'plan_hash': plan['plan_hash'], 'items': private})
        atomic_json(run / 'annotations_template.json', {'plan_hash': plan['plan_hash'], 'rater': '', 'rater_type': 'ai', 'items': labels})
        (run / 'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Fixed 8s requested-count diagnostic</title>'
            '<style>body{font:16px/1.6 sans-serif;max-width:950px;margin:24px auto}video{width:100%}article{margin:30px 0}</style>'
            '<h1>New videos: count visible actions before revealing target</h1><p>' + html.escape(CRITERION) + '</p>'
            '<p>All clips 193 frames at 24fps. Record shallow cycles in notes. No required completion deadline. '
            'Existing n2 labels are reused separately, not new independent reviews.</p>' + ''.join(cards), encoding='utf-8')
    print(f'Review ready: {run / "review.html"}; {run / "review_frames/blind_manifest.json"}', flush=True)
