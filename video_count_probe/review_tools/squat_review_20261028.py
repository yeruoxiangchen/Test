"""CPU-only evidence export for the completed paired squat pilot.

Exports existing video frames; it does not classify motion or run generation.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path('/data/zjr/video_count_wan22_squat_count_20261028')
RUN = ROOT / 'pilot'
OUT = RUN / 'ai_review_20261028'


def evidence():
    import imageio_ffmpeg
    from PIL import Image, ImageDraw

    mapping = json.loads((RUN / 'review_frames/private_mapping.json').read_text())
    plan = json.loads((RUN / 'plan.json').read_text())
    if mapping['plan_hash'] != plan['plan_hash']:
        raise ValueError('Plan binding mismatch')
    windows = {2: (84, 108), 4: (0, 18), 5: (96, 120), 7: (84, 108),
               8: (0, 18), 9: (0, 18), 11: (84, 108)}
    width, height = 320, 352
    stride = width * height * 3
    items = []
    for row in mapping['items']:
        bid = row['blind_id']
        if bid not in windows:
            continue
        record = json.loads((RUN / 'records' / (row['id'] + '.json')).read_text())
        video = RUN / record['video']
        if hashlib.sha256(video.read_bytes()).hexdigest() != row['video_sha256']:
            raise ValueError(f'Video hash mismatch: {bid}')
        raw = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-threads', '1',
                              '-i', str(video), '-vf', 'crop=640:704:320:0,scale=320:352',
                              '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-threads', '1', '-'],
                             check=True, capture_output=True).stdout
        if len(raw) != 121 * stride:
            raise ValueError(f'Unexpected frame count: {bid}')
        start, end = windows[bid]
        indices = list(range(start, end + 1))
        cols, cell_h = 5, height + 24
        sheet = Image.new('RGB', (cols * width, 36 + ((len(indices) + cols - 1) // cols) * cell_h), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f'Clip {bid:03d} | original 24 fps | centre crop (x320..959), entire image height | target/method hidden', fill='black')
        for index, frame_no in enumerate(indices):
            x, y = index % cols * width, 36 + index // cols * cell_h
            draw.text((x + 4, y + 4), f'f{frame_no:03d} {frame_no / 24:.4f}s', fill='black')
            frame = Image.frombytes('RGB', (width, height), raw[frame_no * stride:(frame_no + 1) * stride])
            sheet.paste(frame, (x, y + 24))
        path = OUT / 'detail' / f'{bid:03d}_{start:03d}_{end:03d}.jpg'
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path, quality=95)
        items.append({'blind_id': bid, 'frames': [start, end], 'video_sha256': row['video_sha256'],
                      'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        print(path, flush=True)
    (OUT / 'detail_manifest.json').write_text(json.dumps({'plan_hash': plan['plan_hash'], 'items': items,
        'limits': 'Evidence only, not judgments. Overview is 8 fps; these seven selected windows retain 24 fps.'}, indent=2) + '\n')


def publish():
    """Save the explicit judgments from this conversation's visual inspection.

    The fixed list below is manual AI annotation, not a learned motion detector.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from video_count_probe.core import atomic_json, checked_plan, file_sha, read_json, writer_lock
    from video_count_probe.wan22_action_probe.review import records
    from video_count_probe.wan22_action_probe.model import cache_path, read_embedding, text_identity
    from video_count_probe.wan22_squat_count.protocol import MODEL, all_prompts, code_hash
    from video_count_probe.wan22_squat_count.review import report, validate_annotation
    import torch

    plan = checked_plan(RUN / 'plan.json')
    complete = records(RUN, plan)
    if len(complete) != 12:
        raise ValueError('Expected twelve verified videos')
    identity = read_json(RUN / 'identity.json')
    if identity['source_hash'] != code_hash():
        raise ValueError('Frozen generation code changed')
    template = read_json(RUN / 'annotations_template.json')
    if template['plan_hash'] != plan['plan_hash']:
        raise ValueError('Annotation template mismatch')
    detail = read_json(OUT / 'detail_manifest.json')
    if detail['plan_hash'] != plan['plan_hash']:
        raise ValueError('Evidence plan mismatch')
    for item in detail['items']:
        if file_sha(item['path']) != item['sha256']:
            raise ValueError('Evidence sheet changed')
    # Frames are visual phase landmarks, not kinematic ground truth.
    # No method/target labels are used to choose these observations.
    observations = {
        1: ([36, 69, 93], [], True, True,
            'f000-f036直立；约f042开始下降，f063-f075处于低位，f093恢复直立。只见一个完整周期，f096-f120保持站立，没有额外半次。'),
        2: ([9, 45, 100], [], True, False,
            '约f009开始一个下蹲，f039-f075低位停留，之后持续起身，约f100确认站直。24 fps边界图中f096-f099仍有小幅髋膝伸展；按整个f096-f120均站立的严格规则不通过。此临界时刻存在视觉阈值敏感性，另报告宽松复核情景。'),
        3: ([15, 48, 84], [], True, True,
            'f000-f015直立，约f021-f048下降，f045-f054低位，随后起身，f084已经直立。全段一个完整周期，最后一秒站立，无额外动作。'),
        4: ([36, 66, 93], [[0, 36]], False, True,
            '逐帧核对f000-f018：片头已明显屈膝、髋部降低，缺少第一次动作的直立起点；这段到f036的动作单独记为片头残缺周期。f036直立后再次下降，f060-f072低位，f093恢复直立，构成一个完整周期；最后一秒站立。'),
        5: ([9, 66, 110], [], True, False,
            'f009之前直立，约f015开始下蹲，低位保持很久，仅有一次下降和一次起身。24 fps末秒图显示f096仍明显屈膝，约f110才站直；因此虽完成一次，未在4秒前完成，最后一秒并非全程站立。'),
        6: ([24, 66, 93], [], True, True,
            'f000-f024直立，之后逐渐下蹲，f063-f069低位，f093恢复直立。全段一个完整周期，f096-f120保持站立，没有第二次下蹲或额外半次。'),
        7: ([9, 48, 100], [], True, False,
            '约f009开始一个下蹲，f039-f075低位停留，之后起身，约f100确认直立。24 fps边界图显示f096附近仍在完成小幅膝髋伸展；严格末秒规则不通过。其边界也接近4秒，另报告宽松复核情景。'),
        8: ([18, 48, 78], [[0, 12]], False, True,
            '24 fps片头图显示f000已经位于蹲姿，随后到f012起身；这只是片头残缺动作。f018直立后再次下降，约f045-f051低位，f078站直，构成一个完整周期；最后一秒站立。'),
        9: ([21, 45, 72], [[0, 15]], False, True,
            '24 fps片头图显示一开始已经深蹲，f010-f013起身时还有短暂脚部离地，f015才恢复站立；没有可见的第一次完整下降起点，记作残缺周期。f021直立后再下降，f045低位，f072站直，只构成一个完整周期；末秒站立。'),
        10: ([30, 66, 93], [], True, True,
             'f000-f030保持直立，之后下降到f063-f072的低位，再在f093附近站直。一个完整周期，没有第二次下蹲，最后一秒保持站立。'),
        11: ([9, 48, 103], [], True, False,
             'f009附近从站立开始下降，f039-f075处于低位，随后慢慢起身。24 fps边界图中f096仍可见屈膝，约f103才确认站直。一个完整周期，但未在4秒前完成且末秒仍包含起身。'),
        12: ([21, 48, 75], [], True, True,
             'f000-f021直立，之后下降，f045-f051到低位，再起身，f075已经直立。只有一个完整周期，f096-f120站立，没有额外完整或残缺动作。'),
    }
    items = []
    for row in template['items']:
        bid = row['blind_id']
        cycle, partials, initial, rest, notes = observations[bid]
        sheets = [x['path'] for x in detail['items'] if x['blind_id'] == bid]
        label = {**row, 'reviewed': True, 'count_judgeable': True,
                 'complete_cycles': [dict(zip(('start_frame', 'bottom_frame', 'end_frame'), cycle))],
                 'partial_attempts': [dict(zip(('start_frame', 'end_frame'), p)) for p in partials],
                 'initial_upright': initial, 'final_second_upright_still': rest, 'notes': notes,
                 'overview_inspected': str(RUN / 'review_frames' / f'{bid:03d}.jpg'),
                 'detail_sheets_inspected': sheets, 'rest_boundary_sensitive': bid in (2, 7),
                 'temporal_landmarks': 'Approximate visually selected landmarks; not pose-estimator ground truth.'}
        validate_annotation(label, complete[row['job_id']], plan)
        items.append(label)
    annotation = {'plan_hash': plan['plan_hash'], 'rater': 'Codex AI squat frame review', 'rater_type': 'ai',
                  'human_validated': False, 'reviewed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'inspection': 'All 12 time-labelled 8 fps overviews (41 frames each), plus seven 24 fps boundary windows. Not every temporal frame of every video was inspected.',
                  'blinding': 'Sheets hid method, target, and seed. Reviewer had already accessed experiment metadata; this is not an independently blinded evaluation.',
                  'items': items}

    # Audit text wiring and existing embeddings on CPU, without loading T5 or a video model.
    cases = {c['id']: c for c in plan['cases']}
    for rec in complete.values():
        job = rec['job']
        if rec['prompt'] != cases[job['case_id']]['prompts'][job['method']]:
            raise ValueError('Record prompt does not match the frozen plan')
    tid = text_identity(MODEL)
    caches = []
    for prompt in all_prompts(plan):
        path = cache_path(ROOT, prompt, tid)
        tensor = read_embedding(path, prompt, tid)
        caches.append({'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'cache': str(path),
                       'tensor_sha256': hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest(),
                       'shape': list(tensor.shape), 'nonzero_tokens': int((tensor.abs().sum(-1) != 0).sum().item())})
    if len(caches) != 5 or len({x['tensor_sha256'] for x in caches}) != 5:
        raise ValueError('Expected five distinct verified text embeddings')
    times = [r['generation_seconds'] for r in complete.values()]
    audit = {'plan_hash': plan['plan_hash'], 'frozen_source_hash': code_hash(),
             'prompt_binding_checked': 12, 'text_cache_identity_and_hashes_verified': True, 'text_embeddings': caches,
             'unique_video_hashes': len({r['video_sha256'] for r in complete.values()}),
             'unique_initial_noises': len({r['initial_latent_sha256'] for r in complete.values()}),
             'noise_note': 'Three seeds reused across both requested counts and both prompts; 12 clips are not 12 independent seeds.',
             'generation_hours': sum(times) / 3600, 'mean_minutes': statistics.mean(times) / 60,
             'min_minutes': min(times) / 60, 'max_minutes': max(times) / 60,
             'limit': 'Audits stored cache contents and frozen call path; tensors inside the historical forward were not separately logged.'}
    OUT.mkdir(parents=True, exist_ok=True)
    destination = RUN / 'annotations.json'
    with writer_lock(RUN / '.writer.lock'):
        if destination.exists():
            old = read_json(destination)
            if old.get('rater') != annotation['rater'] or old.get('items') != items:
                raise RuntimeError('Existing annotations differ; refusing to overwrite another review')
            annotation = old
        backup = OUT / 'summary_before_ai_review.json'
        if not backup.exists():
            atomic_json(backup, read_json(RUN / 'summary.json'))
        atomic_json(OUT / 'annotations_ai.json', annotation)
        atomic_json(OUT / 'execution_audit.json', audit)
        atomic_json(destination, annotation)
    summary = report(ROOT, 'pilot')
    # Explicit post-hoc annotation sensitivity only. Never replaces primary judgments.
    import copy
    relaxed = copy.deepcopy(annotation)
    relaxed['rater'] += ' / boundary sensitivity, not primary review'
    for item in relaxed['items']:
        if item['blind_id'] in (2, 7):
            item['complete_cycles'][0]['end_frame'] = 96
            item['final_second_upright_still'] = True
    relaxed_path = OUT / 'annotations_boundary_sensitivity.json'
    atomic_json(relaxed_path, relaxed)
    sensitivity = report(ROOT, 'pilot', relaxed_path, write=False)
    atomic_json(OUT / 'summary_boundary_sensitivity.json', sensitivity)
    labels = {x['job_id']: x for x in items}
    initial_state = {}
    for method in plan['methods']:
        group = [labels[r['id']] for r in summary['per_video'] if r['method'] == method]
        initial_state[method] = {'expected': len(group),
                                'initial_upright': sum(x['initial_upright'] is True for x in group),
                                'initial_partial_clips': sum(any(p['start_frame'] == 0 for p in x['partial_attempts']) for x in group)}
    result = {**summary, 'human_validated': False, 'ai_review_complete': True,
              'execution_audit': audit, 'boundary_sensitivity_methods': sensitivity['methods'],
              'initial_state': initial_state,
              'review_method': annotation['inspection'], 'per_video': [{**r, **labels[r['id']]} for r in summary['per_video']]}
    atomic_json(OUT / 'summary_ai.json', result)
    render_report(result, complete)
    print(json.dumps({'methods': summary['methods'], 'by_target_count': summary['by_target_count'],
                      'paired': summary['paired'], 'boundary_sensitivity_methods': sensitivity['methods'],
                      'audit': audit}, ensure_ascii=False, indent=2))
    print(OUT / 'report_ai.html')


def render_report(result, complete):
    methods = {'count_prompt': '直接次数提示', 'phase_prompt': '动作阶段提示'}
    def decision(value):
        return '通过' if value is True else '未通过' if value is False else '不确定'
    rows, cards = [], []
    for r in sorted(result['per_video'], key=lambda x: x['blind_id']):
        bid = r['blind_id']
        rows.append(f'<tr><td>{bid:03d}</td><td>{methods[r["method"]]}</td><td>{r["target_count"]}</td>'
                    f'<td>{r["seed"]}</td><td>{r["observed_complete_count"]}</td><td>{len(r["partial_attempts"])}</td>'
                    f'<td>{decision(r["final_rest"])}</td><td>{decision(r["joint_count_stop"])}</td></tr>')
        details = ' '.join(f'<a href="detail/{Path(p).name}">{Path(p).stem}</a>' for p in r['detail_sheets_inspected'])
        cards.append(f'<article><h2>Clip {bid:03d} · {methods[r["method"]]} · 目标{r["target_count"]}次 · seed={r["seed"]}</h2>'
                     f'<p>{html.escape(r["notes"])}</p><video controls preload="metadata" width="768" '
                     f'src="../{complete[r["id"]]["video"]}"></video>'
                     f'<p><a href="../review_frames/{bid:03d}.jpg">8 fps完整时间轴概览</a> {details}</p></article>')
    metric_rows = []
    for method, group in result['methods'].items():
        values = ''.join(f'<td>{group[k]["true"]}/{group["expected"]}</td>' for k in ('exact_count', 'stopped_after_target', 'final_rest', 'joint_count_stop'))
        metric_rows.append(f'<tr><td>{methods[method]}</td>{values}</tr>')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Wan2.2 深蹲次数提示对照复核</title>
<style>body{font:16px/1.7 system-ui;max-width:1050px;margin:30px auto;padding:0 20px;color:#20242b}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:7px}video{max-width:100%}article{border-top:1px solid #ccc;margin-top:30px}a{color:#1456b8}</style>
<h1>Wan2.2 深蹲次数提示对照：AI 视觉复核</h1>
<p>12/12 视频生成完成，无运行错误，6 对初始噪声核验通过。已查看全部 12 张 8 fps 时间轴概览，并补看 7 段 24 fps 边界窗口。没有查看每段视频的全部原始帧；未经独立人工验证，不宣称独立盲评。</p>
<p><strong>12 段均只有一个完整“直立→下蹲→直立”周期，3 段另有片头残缺动作。</strong>片头已在下蹲或蹲姿，不能把缺少直立起点的动作补算为完整一次。低位停留或局部起伏不重复计数。</p>
<table><tr><th>提示</th><th>完整次数正确</th><th>达到目标后不追加动作</th><th>最后一秒全程站立</th><th>次数与停止联合成功</th></tr>''' + ''.join(metric_rows) + '''</table>
<p>“达到目标后停止”要求先完成目标次数；目标为两次而只做了一次时，此项不通过，并不表示片尾还在持续运动。</p>
<p>002、007 在 4 秒附近完成起身，末秒是否已完全站稳存在视觉阈值敏感性。主结果按严格规则记录，另存将这两段按第96帧完成处理的事后敏感性报告（summary_boundary_sensitivity.json）。完整次数结果不受此项影响。</p>
<p>直接次数提示出现3/6段片头残缺动作，分阶段提示为0/6，起始直立分别为3/6和6/6。这说明分阶段描述改变了片头完整性，但没有增加第二个完整周期。按冻结协议，片头残缺动作单独报告，不事后加入联合成功的扣分条件。</p>
<p>这里只比较两种整段文本提示，没有时间条件路由或模型训练。要求两次而未得到两次，支持继续定位动作周期完整性与次数条件响应的问题；不能据此宣称新控制方法有效，也不能断言模型在所有设置下都做不到两次。三种噪声在两种次数条件下复用，不是12个独立随机样本。</p>
<p>下一步建议：小规模对照视频时长与末秒停止约束，检查是否为单次动作占满时间窗口；随后将方法目标设为完整动作周期的时间调度与次数控制。当前结果不支持仅以更长阶段提示作为有效控制方法，也未启动额外生成或训练。</p>
<p>已检查四种正向提示与生成任务匹配，正负共五份文本嵌入不同且哈希验证通过，12个视频哈希不同，冻结推理源码未变化。历史forward内部输入未另存张量，因此这是缓存及调用路径审计，不是重新运行生成。</p>
<table><tr><th>编号</th><th>提示</th><th>目标</th><th>种子</th><th>完整次数</th><th>残缺动作</th><th>末秒站立</th><th>联合成功</th></tr>''' + ''.join(rows) + '</table>' + ''.join(cards) + '</html>'
    (OUT / 'report_ai.html').write_text(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['evidence', 'publish'])
    args = parser.parse_args()
    for key, suffix in {'TMPDIR': 'tmp', 'TORCH_HOME': 'cache/torch', 'HF_HOME': 'cache/huggingface',
                        'HF_HUB_CACHE': 'cache/huggingface/hub', 'TORCH_EXTENSIONS_DIR': 'cache/torch_extensions',
                        'MPLCONFIGDIR': 'cache/matplotlib', 'XDG_CACHE_HOME': 'cache/xdg'}.items():
        os.environ[key] = str(ROOT / suffix)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
    evidence() if args.command == 'evidence' else publish()


if __name__ == '__main__':
    main()
