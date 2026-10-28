from __future__ import annotations

import html
import random
from pathlib import Path

from video_count_probe.core import atomic_json, checked_plan, file_sha, is_complete, read_json, writer_lock


def records(run, plan):
    identity_path = run / 'identity.json'
    if not identity_path.exists():
        return {}
    identity = read_json(identity_path)
    if identity['plan_hash'] != plan['plan_hash']:
        raise ValueError('Record identity does not match plan')
    return {j['id']: read_json(run / 'records' / (j['id'] + '.json')) for j in plan['jobs'] if is_complete(run, j, identity)}


def annotation_scores(data, plan, completed):
    if data['plan_hash'] != plan['plan_hash'] or data['rater_type'] not in ('ai', 'human') or not data['rater'].strip():
        raise ValueError('Annotation plan/rater provenance required')
    seen, rows = set(), []
    for row in data['items']:
        jid = row['job_id']
        if jid in seen or jid not in completed:
            raise ValueError('Duplicate/unknown/incomplete annotation job')
        seen.add(jid)
        record = completed[jid]
        if row['video_sha256'] != record['video_sha256']:
            raise ValueError('Annotation video hash mismatch')
        decision = row.get('complete_cycle')
        if decision is None:  # Explicitly unreviewed, never scored as failure.
            continue
        if type(decision) is not bool or not isinstance(row.get('notes'), str) or not row['notes'].strip():
            raise ValueError('Use true/false with visual evidence notes, or null for unreviewed')
        rows.append({'job_id': jid, 'action': record['job']['case_id'], 'complete_cycle': decision})
    return rows


def report(root, mode, annotations=None, write=True):
    run = Path(root) / mode
    plan = checked_plan(run / 'plan.json')
    completed = records(run, plan)
    # Re-running status/report must retain existing annotation results.
    source = Path(annotations) if annotations else run / 'annotations.json'
    data = read_json(source) if source.exists() else None
    if annotations and data is None:
        raise FileNotFoundError(source)
    rows = annotation_scores(data, plan, completed) if data else []
    errors = sum((run / 'errors' / (j['id'] + '.json')).exists() and j['id'] not in completed for j in plan['jobs'])
    result = {'plan_hash': plan['plan_hash'], 'generated': len(completed), 'expected': len(plan['jobs']),
              'annotated': len(rows), 'execution_errors': errors,
              'assessment_complete': len(completed) == len(rows) == len(plan['jobs']) and errors == 0,
              'scientific_evaluation': plan['scientific_evaluation'], 'review_provenance': None,
              'complete_action_videos': sum(r['complete_cycle'] for r in rows),
              'per_action': {a: {'reviewed': sum(r['action'] == a for r in rows),
                                 'complete_action_videos': sum(r['action'] == a and r['complete_cycle'] for r in rows)}
                             for a in {c['id'] for c in plan['cases']}},
              'peak_allocated_mib': max((r['peak_allocated_mib'] for r in completed.values()), default=None),
              'peak_reserved_mib': max((r['peak_reserved_mib'] for r in completed.values()), default=None),
              'limitations': plan['limits']}
    if data:
        result['review_provenance'] = {'rater': data['rater'], 'rater_type': data['rater_type'],
                                       'path': str(source), 'sha256': file_sha(source)}
    if write:
        with writer_lock(run / '.writer.lock'):
            atomic_json(run / 'summary.json', result)
    print(f'generated={result["generated"]}/{result["expected"]} annotated={len(rows)} errors={errors} assessment_complete={result["assessment_complete"]}', flush=True)
    return result


def make_sheet(video, destination, config, title):
    import imageio_ffmpeg
    from PIL import Image, ImageDraw
    # Decode at a small spatial size, but keep all temporal frames for verification.
    width = 320
    height = round(config['height'] * width / config['width'])
    generator = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24', bits_per_pixel=24,
                                           output_params=['-vf', f'scale={width}:{height}', '-threads', '1'])
    selected = []
    count = 0
    try:
        metadata = next(generator)
        if tuple(metadata['source_size']) != (config['width'], config['height']) or abs(metadata['fps'] - config['fps']) > 0.01:
            raise ValueError('Encoded video geometry/fps differs from plan')
        if tuple(metadata['size']) != (width, height):
            raise ValueError('Review resize did not produce the requested dimensions')
        for index, raw in enumerate(generator):
            if len(raw) != width * height * 3:
                raise ValueError('Unexpected decoded review frame size')
            if index % 3 == 0:
                selected.append((index, Image.frombytes('RGB', (width, height), raw)))
            count += 1
    finally:
        generator.close()
    if count != config['frames']:
        raise ValueError(f'Wrong review frame count: {count}')
    cols, cell_h = 7, height + 22
    sheet = Image.new('RGB', (cols * width, 32 + ((len(selected) + cols - 1) // cols) * cell_h), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill='black')
    for i, (index, frame) in enumerate(selected):
        x, y = (i % cols) * width, 32 + (i // cols) * cell_h
        draw.text((x + 4, y + 3), f'f{index:03d}  {index / config["fps"]:.3f}s', fill='black')
        sheet.paste(frame, (x, y + 22))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=92)


def review(root, mode):
    run = Path(root) / mode
    with writer_lock(run / '.writer.lock'):
        plan = checked_plan(run / 'plan.json')
        completed = records(run, plan)
        cases = {c['id']: c for c in plan['cases']}
        jobs = list(plan['jobs'])
        random.Random(82415).shuffle(jobs)
        blind, private, annotations, cards = [], [], [], []
        for index, job in enumerate(jobs, 1):
            if job['id'] not in completed:
                continue
            record = completed[job['id']]
            case = cases[job['case_id']]
            sheet = run / 'review_frames' / f'{index:03d}.jpg'
            make_sheet(run / record['video'], sheet, plan['config'],
                       f'Clip {index:03d} | {case["action"]} | sampled 8 fps; original 24 fps')
            blind.append({'blind_id': index, 'action': case['action'], 'criterion': case['criterion'], 'sheet': str(sheet)})
            private.append({'blind_id': index, 'job_id': job['id'], 'video_sha256': record['video_sha256']})
            annotations.append({'blind_id': index, 'job_id': job['id'], 'video_sha256': record['video_sha256'],
                                'complete_cycle': None, 'notes': ''})
            cards.append(f'<article><h2>Clip {index:03d}: {html.escape(case["action"])}</h2><p>{html.escape(case["criterion"])}</p>'
                         f'<video controls loop preload="metadata" width="640" src="{record["video"]}"></video>'
                         f'<p><a href="review_frames/{index:03d}.jpg">Time-labelled frames</a></p></article>')
        atomic_json(run / 'review_frames' / 'blind_manifest.json', {'plan_hash': plan['plan_hash'], 'items': blind,
                    'limits': 'Action capability only. AI frame review is a feasibility assessment, not human ground truth. Seed labels hidden in sheets.'})
        atomic_json(run / 'review_frames' / 'private_mapping.json', {'plan_hash': plan['plan_hash'], 'items': private})
        atomic_json(run / 'annotations_template.json', {'plan_hash': plan['plan_hash'], 'rater': '', 'rater_type': 'ai', 'items': annotations})
        (run / 'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Wan2.2 action capability review</title>'
                                        '<style>body{font:16px sans-serif;max-width:900px;margin:24px auto}article{margin:32px 0}</style>'
                                        '<h1>Action capability review</h1><p>Complete cycles only; uncertain/unreviewed stays null. '
                                        'No count target. Set reviewer provenance in annotations.json.</p>' + ''.join(cards))
    print(f'Review ready: {run / "review.html"}; {run / "review_frames/blind_manifest.json"}', flush=True)
