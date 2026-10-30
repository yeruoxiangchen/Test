"""Frame inspection and explicitly labelled AI review for action calibration.

This file is outside both frozen generation source sets. No model is loaded.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from video_count_probe.core import atomic_json, checked_plan, file_sha, read_json, writer_lock
from video_count_probe.action_calibration.experiment import (
    DEFAULT_ROOT, ability_score, configure_storage, report,
)
from video_count_probe.evaluation import completed_records, validate_annotation
from video_count_probe.review_tools.review_frames import frames, sheet


def context(root):
    run = root / 'pilot'
    plan = checked_plan(run / 'plan.json')
    mapping = read_json(run / 'review_frames/private_mapping.json')
    if mapping['plan_hash'] != plan['plan_hash']:
        raise ValueError('Blind mapping belongs to a different plan')
    return run, plan, {r['blind_id']: r for r in mapping['items']}


def detail(root, ids, start, end, crop):
    run, plan, mapping = context(root)
    if not 0 <= start <= end < plan['config']['frames']:
        raise ValueError('Invalid frame range')
    with writer_lock(run / '.writer.lock'):
        for blind_id in ids:
            item = mapping[blind_id]
            record = read_json(run / 'records' / (item['job_id'] + '.json'))
            video = run / record['video']
            if file_sha(video) != item['video_sha256']:
                raise ValueError('Video changed after blind frame preparation')
            destination = run / 'review_frames/detail' / f'{blind_id:03d}_{start:02d}_{end:02d}.jpg'
            sheet(frames(video), list(range(start, end + 1)), destination,
                  f'Blind {blind_id:03d} | full 16 fps detail', crop=crop, width=360, cols=6)
            print(destination, flush=True)


def annotate(root, items):
    run, plan, mapping = context(root)
    records, errors = completed_records(run, plan)
    if errors or len(records) != len(plan['jobs']):
        raise ValueError('All videos must be verified before this review is finalized')
    if len(items) != len(mapping) or {i['blind_id'] for i in items} != set(mapping):
        raise ValueError('Require one blind label for every clip')
    cases = {c['id']: c for c in plan['cases']}
    annotations = []
    for item in sorted(items, key=lambda i: i['blind_id']):
        row = mapping[item['blind_id']]
        record = records[row['job_id']]
        annotation = {**item, 'job_id': row['job_id'], 'video_sha256': row['video_sha256'],
                      'rater': 'Codex AI frame review', 'complete': True,
                      'annotation_source': 'ai_visual_frame_review'}
        if not isinstance(annotation['complete_cycle_observed'], bool):
            raise ValueError('This capability review requires an explicit cycle-presence decision')
        validate_annotation(annotation, record, cases[record['job']['case_id']], plan)
        ability_score(annotation)
        annotations.append(annotation)
    out = run / 'ai_review_20261027'
    payload = {'plan_hash': plan['plan_hash'], 'annotation_source': 'ai_visual_frame_review',
               'annotations': annotations}
    with writer_lock(run / '.writer.lock'):
        path = out / 'annotations_ai.json'
        if path.exists() and read_json(path) != payload:
            raise ValueError('Existing annotations differ; retain the original and use a separate adjudication file')
        atomic_json(out / 'blind_labels.json', {'plan_hash': plan['plan_hash'], 'labels': items})
        atomic_json(path, payload)
        atomic_json(out / 'protocol.json', {
            'review_type': 'Single AI visual frame review, not independently human-validated',
            'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'initial_sampling_fps': 8, 'original_fps': 16,
            'detail_sheets': [str(p.relative_to(run)) for p in sorted((run / 'review_frames/detail').glob('*.jpg'))],
            'blinding': 'Prompt variants and seeds were not displayed during visual annotation; labels fixed before reading the outcome mapping.',
            'blinding_limit': 'Reviewer knew the experimental design and previous reference findings; appearance may suggest prompt variant.',
            'primary_metric': 'At least one visibly complete action cycle; exact count is not required.',
            'timestamps': 'Evidence timestamps only for ambiguous clips, not an exhaustive count; decisions use complete_cycle_observed.',
            'reference_policy': 'Reuse frozen AI annotations for 18 counted clips, six distinct action/seed units.',
        })
    print(f'AI annotations frozen: {path}', flush=True)


def summarize(root):
    run, plan, mapping = context(root)
    out = run / 'ai_review_20261027'
    old_summary = Path(plan['reference']['root']) / 'summary.json'
    old_sha = file_sha(old_summary)
    result = report(root, 'pilot', [out / 'annotations_ai.json'])
    lookup = {(r['method'], r['action'], r['seed']): r for r in result['per_video']}
    units = sorted({(r['action'], r['seed']) for r in result['per_video']})
    methods = ['short_action', 'uncounted_matched', 'count_n1', 'count_n2', 'count_n3']
    matrix = [{'action': action, 'seed': seed,
               **{method: lookup[method, action, seed]['ability'] for method in methods}}
              for action, seed in units]
    pairs = {}
    for left, right in [('short_action', 'uncounted_matched')] + [('uncounted_matched', m) for m in methods[2:]]:
        differences = [int(r[left] == 'complete_cycle') - int(r[right] == 'complete_cycle') for r in matrix]
        pairs[f'{left}_vs_{right}'] = {'n': len(differences), 'wins': differences.count(1),
                                      'losses': differences.count(-1), 'ties': differences.count(0)}
    records, _ = completed_records(run, plan)
    memory = {key: max(r[key] for r in records.values()) for key in (
        'peak_allocated_mib', 'peak_reserved_mib', 'process_peak_allocated_mib', 'process_peak_reserved_mib')}
    result.update(review_type='AI visual frame review, not human ground truth', paired=pairs,
                  action_seed_matrix=matrix, memory_max_mib=memory,
                  original_human_summary_unchanged=file_sha(old_summary) == old_sha,
                  annotation_sha256=file_sha(out / 'annotations_ai.json'))
    if not result['original_human_summary_unchanged']:
        raise RuntimeError('Original reference summary changed')
    with writer_lock(run / '.writer.lock'):
        atomic_json(out / 'summary_ai.json', result)
        with (out / 'paired_action_seed.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['action', 'seed', *methods])
            writer.writeheader(); writer.writerows(matrix)
        names = {'short_action': '极简动作提示', 'uncounted_matched': '仅移除次数要求',
                 'count_n1': '原次数要求：1 次', 'count_n2': '原次数要求：2 次', 'count_n3': '原次数要求：3 次'}
        table = ''.join(f'<tr><td>{names[m]}</td><td>{g["complete_cycle"]}/{g["n"]}</td>'
                        f'<td>{g["no_complete_cycle"]}</td><td>{g["unusable"]}</td></tr>'
                        for m, g in result['methods'].items())
        by_job = {r['job_id']: r['blind_id'] for r in mapping.values()}
        cards = []
        for row in result['per_video']:
            if row.get('reference_only'):
                continue
            blind_id = by_job[row['id']]
            cards.append(f'<article><h2>盲号 {blind_id:03d} · {html.escape(row["action"])} · {names[row["method"]]} · seed {row["seed"]}</h2>'
                         f'<p>{html.escape(row["ability"])}：{html.escape(row["notes"])}</p>'
                         f'<video controls preload="none" src="../videos/{row["id"]}.mp4"></video>'
                         f'<p><a href="../review_frames/{blind_id:03d}.jpg">时间标注帧图</a></p></article>')
        page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>动作能力对照 · AI 评审</title>'
        page += '<style>body{font:16px/1.7 system-ui;max-width:1000px;margin:32px auto;padding:0 20px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px 16px}video{width:min(100%,650px)}article{border-top:1px solid #ddd;margin-top:28px}h2{font-size:18px}</style>'
        page += '<h1>12 段动作能力对照：AI 评审</h1><p>生成和 AI 评审已完成。指标是至少一个完整动作周期，不是次数恰好正确；未经独立人工复核。</p>'
        page += '<table><tr><th>提示</th><th>完整动作视频</th><th>无完整周期</th><th>不可用</th></tr>' + table + '</table>'
        page += '<p>每种新提示仅 6 个动作/种子单位。三个旧次数条件复用同一组新视频，不构成 18 个独立配对样本。短提示同时改变场景措辞和动作描述，不能单独归因于长度。</p>'
        page += '<p>本轮只检验固定模型与配置下的动作完成度；不能据此声称所有视频模型都不具备该能力，也没有直接验证计数控制方法有效。完整配对结果保存在 summary_ai.json 和 paired_action_seed.csv。</p>'
        page += ''.join(cards) + '</html>'
        (out / 'report_ai.html').write_text(page, encoding='utf-8')
    print(json.dumps({'paired': pairs, 'memory_max_mib': memory}, ensure_ascii=False, indent=2))
    print(f'AI report: {out / "report_ai.html"}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['detail', 'annotate', 'report'])
    p.add_argument('--root', type=Path, default=Path(DEFAULT_ROOT))
    p.add_argument('--ids', type=int, nargs='+')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=80)
    p.add_argument('--crop', type=int, nargs=4)
    p.add_argument('--items-json')
    a = p.parse_args()
    configure_storage(a.root)
    if a.command == 'detail':
        if not a.ids:
            p.error('detail requires --ids')
        detail(a.root, a.ids, a.start, a.end, a.crop)
    elif a.command == 'annotate':
        if not a.items_json:
            p.error('annotate requires --items-json')
        annotate(a.root, json.loads(a.items_json))
    else:
        summarize(a.root)


if __name__ == '__main__':
    main()
