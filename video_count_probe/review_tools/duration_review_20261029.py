"""CPU-only evidence extraction and explicit AI annotations for the duration pilot.

Generation code/protocol are unchanged. No model inference or automatic detector.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path('/data/zjr/video_count_wan22_duration_20261029')
RUN = ROOT / 'pilot'
OUT = ROOT / 'analysis_ai_20261029'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def setup():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    for key, suffix in {'TMPDIR': 'tmp', 'TORCH_HOME': 'cache/torch', 'HF_HOME': 'cache/huggingface',
                        'HF_HUB_CACHE': 'cache/huggingface/hub', 'TORCH_EXTENSIONS_DIR': 'cache/torch_extensions',
                        'MPLCONFIGDIR': 'cache/matplotlib'}.items():
        os.environ[key] = str(ROOT / suffix)
    OUT.mkdir(parents=True, exist_ok=True)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def details():
    import imageio_ffmpeg
    from PIL import Image, ImageDraw
    from video_count_probe.core import atomic_json
    plan = read(RUN / 'plan.json')
    mapping = {x['blind_id']: x for x in read(RUN / 'review_frames/private_mapping.json')['items']}
    windows = [(1, 90, 105), (2, 0, 12), (3, 0, 15), (3, 165, 183),
               (6, 0, 30), (6, 30, 54), (6, 150, 192)]
    evidence = []
    for bid in sorted({x[0] for x in windows}):
        row = mapping[bid]
        video = RUN / row['video']
        if sha(video) != row['video_sha256']:
            raise ValueError('Video binding mismatch')
        chosen = [(a, b) for n, a, b in windows if n == bid]
        wanted = {n for a, b in chosen for n in range(a, b + 1)}
        reader = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24',
            output_params=['-vf', 'crop=640:704:320:0,scale=240:264', '-threads', '1'])
        meta = next(reader)
        if meta['size'] != (240, 264) or meta['fps'] != 24:
            raise ValueError(meta)
        frames = {}
        try:
            for n, raw in enumerate(reader):
                if n in wanted:
                    frames[n] = Image.frombytes('RGB', (240, 264), raw)
        finally:
            reader.close()
        for start, end in chosen:
            count = end - start + 1
            sheet = Image.new('RGB', (1440, 32 + ((count + 5) // 6) * 288), 'white')
            draw = ImageDraw.Draw(sheet)
            draw.text((8, 8), f'Clip {bid:03d} | original 24fps | central crop | f{start}..f{end}', fill='black')
            for i, n in enumerate(range(start, end + 1)):
                x, y = (i % 6) * 240, 32 + (i // 6) * 288
                draw.text((x + 4, y + 3), f'f{n:03d} {n / 24:.3f}s', fill='black')
                sheet.paste(frames[n], (x, y + 24))
            path = OUT / f'detail_{bid:03d}_{start:03d}_{end:03d}.jpg'
            sheet.save(path, quality=93)
            evidence.append({'blind_id': bid, 'frames': [start, end], 'path': str(path),
                             'sha256': sha(path), 'video_sha256': row['video_sha256']})
            print(path, flush=True)
    atomic_json(OUT / 'detail_manifest.json', {'plan_hash': plan['plan_hash'], 'items': evidence})


def publish():
    from video_count_probe.core import atomic_json, checked_plan, writer_lock
    from video_count_probe.wan22_duration_probe.review import report

    plan = checked_plan(RUN / 'plan.json')
    expected = '86e400d632302c53f608aa7125e4701f350498d7d4507c07ca69af17f1c191f8'
    if plan['plan_hash'] != expected:
        raise ValueError('This review applies only to the inspected duration pilot')
    annotations = read(RUN / 'annotations_template.json')
    manifest = read(OUT / 'detail_manifest.json')
    mapping = {x['blind_id']: x for x in read(RUN / 'review_frames/private_mapping.json')['items']}
    if annotations['plan_hash'] != expected or manifest['plan_hash'] != expected:
        raise ValueError('Evidence plan mismatch')
    if len(manifest['items']) != 7 or len(annotations['items']) != 6:
        raise ValueError('Expected six overviews and seven inspected detail sheets')
    for item in manifest['items']:
        if sha(item['path']) != item['sha256'] or item['video_sha256'] != mapping[item['blind_id']]['video_sha256']:
            raise ValueError('Detail evidence binding mismatch')

    # Explicit visual judgments entered after inspection, not automated counting.
    judgments = {
        1: ([(9, 45, 102)], [], True, False,
            'One full squat. The person is still rising at frame 96 and stands fully by about frame 102; '
            'therefore the whole final second is not upright rest.'),
        2: ([(12, 27, 45), (48, 66, 84), (90, 108, 129), (129, 144, 159)],
            [(0, 9), (165, 192)], False, False,
            'Starts already crouched and rises: initial partial, not a complete cycle. Four subsequent '
            'upright-lowered-upright cycles, followed by another descent that does not return upright before the clip ends.'),
        3: ([(24, 39, 54), (63, 78, 96), (99, 108, 123), (141, 159, 177)],
            [(0, 12)], False, False,
            'Starts crouched: initial partial. Four subsequent complete cycles. The fourth ascent is ongoing '
            'at the start of the final second (frame 168); upright rest begins only around frames 174-177.'),
        4: ([(18, 48, 78)], [], True, True,
            'One full squat followed by upright rest. No second complete or partial attempt is visible.'),
        5: ([(36, 66, 93)], [], True, True,
            'One full squat with a prolonged low position, then upright rest. Low-position motion is not counted as another cycle.'),
        6: ([(30, 42, 51), (60, 72, 93), (102, 117, 135), (150, 164, 174)],
            [(0, 24), (178, 192)], False, False,
            'Starts crouched with opening-frame ghosting and rises: initial partial. Four visible complete '
            'upright-lowered-upright cycles under the frozen rule. Cycles 1 and 4 are shallow, '
            'with visible hip lowering/knee flexion and full return; no numeric depth threshold was preregistered. '
            'A final descent remains crouched at the end. Excluding both shallow cycles changes the count to two '
            'but still fails stopping because of subsequent extra shallow and incomplete motion.'),
    }
    for item in annotations['items']:
        bid = item['blind_id']
        binding = mapping[bid]
        if item['job_id'] != binding['id'] or item['video_sha256'] != binding['video_sha256']:
            raise ValueError('Overview mapping mismatch')
        overview = RUN / 'review_frames' / f'{bid:03d}.jpg'
        if read(overview.with_suffix('.json')) != {'plan_hash': expected, 'video_sha256': item['video_sha256']}:
            raise ValueError('Overview video binding mismatch')
        cycles, partials, initial, rest, notes = judgments[bid]
        item.update(reviewed=True, count_judgeable=True,
            complete_cycles=[dict(zip(('start_frame', 'bottom_frame', 'end_frame'), x)) for x in cycles],
            partial_attempts=[dict(zip(('start_frame', 'end_frame'), x)) for x in partials],
            initial_upright=initial, final_second_upright_still=rest, notes=notes,
            shallow_cycle_indices_1based=[1, 4] if bid == 6 else [],
            evidence={'overview': str(overview), 'overview_sha256': sha(overview),
                      'detail_sheets': [x for x in manifest['items'] if x['blind_id'] == bid]})
    annotations.update(rater='Codex AI duration frame review', rater_type='ai', human_validated=False,
        review_id='duration_ai_20261029_v1', reviewed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        method='Visual inspection of six 8-fps overviews and seven selected original-24-fps detail windows.',
        blinding='Not independent blinded review: experiment metadata was accessible before inspection; '
                 'overview labels hide conditions/seeds, but duration is visible.',
        frame_landmarks='Approximate visual phase landmarks, not pose-based ground truth or frame-exact event boundaries.',
        primary_rule='Visible upright -> lowered hips/bent knees -> upright; no numerical depth threshold. '
                     'Initial/terminal partial attempts excluded. Temporal completeness is not fitness-quality certification.')
    destination = RUN / 'annotations.json'
    if destination.exists():
        previous = read(destination)
        if previous.get('review_id') != annotations['review_id'] or previous.get('items') != annotations['items']:
            raise FileExistsError('Preserving existing annotations by another reviewer/version')
        annotations['reviewed_at_utc'] = previous['reviewed_at_utc']
    primary_path = OUT / 'annotations_ai.json'
    atomic_json(primary_path, annotations)
    primary = report(ROOT, 'pilot', annotations=primary_path, write=False)
    if not primary['assessment_complete'] or primary['execution_errors']:
        raise ValueError('Review failed validation or generation is incomplete')

    sensitivity = copy.deepcopy(annotations)
    sensitivity['review_id'] += '_posthoc_shallow_exclusion'
    sensitivity['method'] += ' Post-hoc sensitivity only: reclassify clip 006 cycles 1/4 as partial-depth attempts.'
    item = next(x for x in sensitivity['items'] if x['blind_id'] == 6)
    removed = [item['complete_cycles'][i] for i in (0, 3)]
    item['complete_cycles'] = [item['complete_cycles'][i] for i in (1, 2)]
    item['partial_attempts'] = sorted(item['partial_attempts'] + [
        {'start_frame': c['start_frame'], 'end_frame': c['end_frame']} for c in removed], key=lambda x: x['start_frame'])
    item['shallow_cycle_indices_1based'] = []
    item['notes'] += ' POST-HOC SENSITIVITY: both shallow cycles are now partial-depth attempts, not counted full cycles.'
    sensitivity_path = OUT / 'annotations_shallow_sensitivity.json'
    atomic_json(sensitivity_path, sensitivity)
    alternate = report(ROOT, 'pilot', annotations=sensitivity_path, write=False)
    atomic_json(OUT / 'summary_shallow_sensitivity.json', alternate)

    # Publish only after the original validator and scorer accept both interpretations.
    with writer_lock(RUN / '.writer.lock'):
        if destination.exists() and read(destination) != annotations:
            raise FileExistsError('Annotations changed during validation; preserving them')
        if not (OUT / 'summary_before_ai_review.json').exists():
            atomic_json(OUT / 'summary_before_ai_review.json', read(RUN / 'summary.json'))
        atomic_json(destination, annotations)
        primary['new_review_provenance'].update(path=str(destination), sha256=sha(destination))
        atomic_json(RUN / 'summary.json', primary)

    costs = {c: {'mean_generation_minutes': statistics.mean(x['generation_seconds']) / 60,
                 'peak_allocated_gib': x['peak_allocated_mib'] / 1024,
                 'peak_reserved_gib': x['peak_reserved_mib'] / 1024} for c, x in primary['resources'].items()}
    findings = {
        'summary': primary, 'resource_means': costs,
        'generation_time_ratio_c_over_b': costs['c8']['mean_generation_minutes'] / costs['b5']['mean_generation_minutes'],
        'human_validated': False,
        'observation': 'All A/B clips show one cycle. C shows four temporal cycles each, including two shallow cycles '
                       'in seed 2026. All conditions have 0/3 exact-count-and-stop successes.',
        'sensitivity': 'Excluding both shallow cycles gives C seed 2026 count=2, but it still fails stopping. '
                       'C exact_count becomes 1/3, count_and_stop remains 0/3. This is post-hoc, not a new primary metric.',
        'interpretation': 'Repeated action is possible in these 8-second samples; extension did not produce requested-count control. '
                          'A/B does not show improvement after removing clock instructions. B/C is seed-matched, different latent shapes.',
        'next_experiment_proposal_only': 'At fixed 8s, compare requested counts 1/2/3 across the same three seeds, '
                                        'with identical initial noise per seed and harmonized prompt template. Reuse C for n=2; '
                                        'six new clips for n=1/3. No generation or training launched.',
    }
    atomic_json(OUT / 'summary_ai.json', findings)
    render_report(primary, alternate, annotations, mapping, costs, findings)
    print(f'AI review published: {OUT / "report_ai.html"}', flush=True)


def render_report(primary, alternate, annotations, mapping, costs, findings):
    labels = {'a5': 'A：5秒，固定秒数约束（复用）', 'b5': 'B：5秒，取消固定秒数约束', 'c8': 'C：8秒，与B相同提示词'}
    rows = []
    for condition in ('a5', 'b5', 'c8'):
        items = sorted((x for x in primary['per_video'] if x['condition'] == condition), key=lambda x: x['seed'])
        cells = ''.join(f'<td>{x["observed_complete_count"]}{"*" if condition == "c8" and x["seed"] == 2026 else ""}</td>' for x in items)
        stats = primary['conditions'][condition]
        rows.append(f'<tr><td>{labels[condition]}</td>{cells}<td>{stats["count_and_stop"]["true"]}/3</td>'
                    f'<td>{stats["initial_upright"]["true"]}/3</td><td>{stats["last_second_rest"]["true"]}/3</td></tr>')
    cards = []
    for item in annotations['items']:
        job = mapping[item['blind_id']]
        details = ' '.join(f'<a href="{Path(x["path"]).name}">逐帧 {x["frames"][0]}–{x["frames"][1]}</a>' for x in item['evidence']['detail_sheets'])
        cycles = ', '.join(f'{c["start_frame"] / 24:.2f} → {c["bottom_frame"] / 24:.2f} → {c["end_frame"] / 24:.2f}s' for c in item['complete_cycles'])
        cards.append(f'<article><h2>视频 {item["blind_id"]:03d} · {labels[job["condition"]]} · seed {job["seed"]}</h2>'
                     f'<video controls preload="none" src="../pilot/{job["video"]}"></video>'
                     f'<p>{html.escape(item["notes"])}</p><p>完整周期（起始→低点→回正，近似）：{cycles}</p>'
                     f'<p>残缺动作帧区间：{html.escape(str(item["partial_attempts"]))}</p>'
                     f'<p><a href="../pilot/review_frames/{item["blind_id"]:03d}.jpg">8fps总览</a> {details}</p></article>')
    resource_rows = ''.join(f'<tr><td>{labels[c]}</td><td>{v["mean_generation_minutes"]:.2f}分钟</td>'
                            f'<td>{v["peak_allocated_gib"]:.2f} GiB</td><td>{v["peak_reserved_gib"]:.2f} GiB</td></tr>' for c, v in costs.items())
    document = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Wan 时长与次数实验：AI视觉复核</title>'
        '<style>body{font:16px/1.7 sans-serif;max-width:1100px;margin:30px auto;padding:0 20px;color:#17202a}'
        'table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:9px;text-align:left}'
        'video{width:100%;max-width:900px}article{border-top:1px solid #aaa;margin-top:32px}small{color:#555}</style>'
        '<h1>加长后出现多次动作，但“两次后停止”仍未成功</h1>'
        '<p>新增6条全部生成并完成AI帧审阅，另复用3条A组视频及既有标注；无执行错误。'
        '本报告不是独立人工评测，未执行新的生成、训练或自动扩展。</p>'
        '<table><tr><th>条件</th><th>seed 2026</th><th>seed 3407</th><th>seed 7919</th>'
        '<th>恰好两次并停</th><th>站立开场</th><th>末秒站立静止</th></tr>' + ''.join(rows) + '</table>'
        '<p>*按原协议，完整周期指站立→髋部下降且屈膝→重新站立；没有预设蹲深阈值。C组2026含两次浅蹲，'
        '不应称四次健身规范深蹲。事后将两次浅蹲排除，则该视频计数为2，但仍有后续浅蹲及未完成下蹲，'
        '“两次后停止”依然失败。浅蹲排除只作敏感性检查，不替换主评分。</p>'
        '<p>A/B三对初始噪声已核验相同：取消固定秒数约束后，本次仍均为一次。B/C提示词相同，'
        '8秒样本均有多次重复，但三条都从半蹲开场，两条以未完成动作收尾。'
        '这支持继续考察次数与停止控制，不能归结为模型只能做一次，也不能称延长时长解决了计数。</p>'
        '<p>B/C潜变量时间维度不同，同seed不等于同一噪声，8秒片段不是5秒片段的简单延长。'
        '仅三个种子、一种动作、一种模型；不能推断普遍能力、内部机理或方法创新已经成立。</p>'
        '<h2>实测开销</h2><table><tr><th>条件</th><th>平均每条生成耗时</th><th>峰值allocated</th>'
        '<th>峰值reserved</th></tr>' + resource_rows + '</table>'
        f'<p>8秒平均耗时为5秒的 {findings["generation_time_ratio_c_over_b"]:.2f} 倍。以上为当前实验运行数据，'
        '不等于跨实现速度基准或整卡总显存。</p>'
        '<h2>建议下一步</h2><p>若继续，固定8秒、三个种子与采样参数，只改变要求次数为1/2/3；'
        '同seed保持相同初始噪声。复用已有“两次”三条，只新增六条。检查次数能否随目标改变，以及完成后是否停止。'
        '这是待执行的诊断，不是已验证的修复方法；本次未启动。</p>'
        '<p>复核方法：六张8fps总览，加七段原24fps逐帧细节。时间点为目视近似相位标记；'
        '审阅前能够接触实验元信息，不宣称完全盲审。人工复核仍可更正边界与浅蹲判定。</p>'
        '<p><a href="annotations_ai.json">AI逐视频标注</a> · <a href="summary_ai.json">汇总及结论</a> · '
        '<a href="summary_shallow_sensitivity.json">浅蹲排除敏感性</a> · '
        '<a href="../pilot/summary.json">原评分器结果</a></p>' + ''.join(cards) + '</html>')
    (OUT / 'report_ai.html').write_text(document, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['details', 'publish'])
    args = parser.parse_args()
    setup()
    details() if args.stage == 'details' else publish()
