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

ROOT = Path('/data/zjr/video_count_wan22_count_response_20261029')
RUN = ROOT / 'pilot'
OUT = ROOT / 'analysis_ai_20261030'
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
    windows = [(5, 0, 9), (5, 126, 156)]
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
    from video_count_probe.wan22_count_response.review import report
    plan = checked_plan(RUN / 'plan.json')
    if plan['plan_hash'] != 'f420beef1f49ec33cde5b736e6eb280630956070d5c2d9eac51e4f5c2513c617':
        raise ValueError('Wrong inspected experiment')
    data = read(RUN / 'annotations_template.json')
    mapping = {x['blind_id']: x for x in read(RUN / 'review_frames/private_mapping.json')['items']}
    judgments = {
        1: ([(12,27,45),(48,66,84),(90,108,129),(132,147,162)], [(0,9),(168,192)],
            'Four complete cycles, with initial crouched ascent and terminal incomplete fifth descent/rise. End frame is not upright.'),
        2: ([(24,36,54),(63,78,96),(96,108,123),(138,156,177)], [(0,9)],
            'Four complete cycles after an initial crouched ascent. The fourth ascent occurs within the final second; rest begins only afterwards.'),
        3: ([(24,36,51),(63,78,93),(96,108,120),(141,156,174)], [(0,9)],
            'Four complete cycles after an initial partial. Fourth cycle completes during the final second, so the entire final second is not upright rest.'),
        4: ([(12,27,45),(48,66,84),(90,108,129),(129,144,159)], [(0,6),(168,192)],
            'Four complete cycles plus an initial partial and a terminal descent/rise that does not regain upright before frame 192.'),
        5: ([(54,72,93),(102,120,156)], [(0,30),(174,192)],
            'Two clearly complete cycles. Starts with knees already flexed, then descends and rises: initial partial. '
            'Frames 132-136 do not clearly regain fully upright posture before another lowering at 138-145; '
            'this is counted as one continuous cycle through frame 156, not two. '
            'A permissive interpretation splitting at frame 135 gives three instead of two; either exceeds requested one. '
            'Terminal descent remains incomplete. Boundaries reviewed at original 24fps.'),
        6: ([(30,42,57),(60,75,90),(99,117,135),(147,159,171)], [(0,21),(177,192)],
            'Four complete cycles following initial crouched ascent; terminal extra descent remains incomplete. Fourth cycle also ends during the final second.'),
    }
    detail_manifest = read(OUT / 'detail_manifest.json')
    for item in detail_manifest['items']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Inspected detail image changed')
    for item in data['items']:
        bid = item['blind_id']; m = mapping[bid]
        if item['job_id'] != m['id'] or item['video_sha256'] != sha(RUN / m['video']):
            raise ValueError('Video binding changed')
        sheet = RUN / 'review_frames' / f'{bid:03d}.jpg'
        if read(sheet.with_suffix('.json')) != {'plan_hash': plan['plan_hash'], 'video_sha256': item['video_sha256']}:
            raise ValueError('Sheet binding mismatch')
        cycles, partials, notes = judgments[bid]
        item.update(reviewed=True, count_judgeable=True,
            complete_cycles=[dict(zip(('start_frame','bottom_frame','end_frame'), c)) for c in cycles],
            partial_attempts=[dict(zip(('start_frame','end_frame'), c)) for c in partials],
            initial_upright=False, final_second_upright_still=False, notes=notes,
            evidence={'overview':str(sheet), 'overview_sha256':sha(sheet),
                      'details':[x for x in detail_manifest['items'] if x['blind_id']==bid]})
    data.update(rater='Codex AI count-response frame review', rater_type='ai', human_validated=False,
        reviewed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        method='Six 8fps frame overviews and two original-24fps windows for clip 005. Approximate phase landmarks.',
        blinding='Not independent blind assessment: generation metadata accessible; counts first assessed on sheets with target/seed labels hidden.',
        uncertainty='Clip 005 strict count=2, permissive return-to-upright interpretation=3; both fail target 1. Reused n2 seed 2026 has the previously documented shallow-cycle caveat.')
    if (RUN / 'annotations.json').exists():
        raise FileExistsError('Preserve existing annotations; review manually before replacing')
    atomic_json(OUT / 'annotations_ai.json', data)
    result = report(ROOT, annotations=OUT / 'annotations_ai.json', write=False)
    assert result['assessment_complete'] and result['verified_noise_matches_to_n2']==6
    with writer_lock(RUN / '.writer.lock'):
        if (RUN / 'annotations.json').exists():
            raise FileExistsError('Concurrent annotation writer')
        atomic_json(OUT / 'summary_before_review.json', read(RUN / 'summary.json'))
        atomic_json(RUN / 'annotations.json', data)
        result['new_review_provenance'].update(path=str(RUN / 'annotations.json'), sha256=sha(RUN / 'annotations.json'))
        atomic_json(RUN / 'summary.json', result)
    times = [v for r in result['resources'].values() for v in r['generation_seconds']]
    findings = {'result':result, 'human_validated':False, 'mean_new_video_minutes':statistics.mean(times)/60,
        'new_generation_gpu_hours':sum(times)/3600,
        'interpretation':'All nine exact-count-and-stop outcomes fail. Two of three seeds remain at four cycles for targets 1/2/3. Seed 2026 gives 2/4/4 under the frozen criterion, not calibrated count control.',
        'sensitivity':'Clip 005 permissive count=3 instead of 2; reused n2/2026 depth exclusion gives 2 instead of 4. Neither changes any count-and-stop success. These are post-hoc sensitivities, not new primary scores.',
        'next_step':'Stop further descriptive seed expansion for now. If pursuing the topic, test an explicit phase/stop control intervention under fixed duration and noise. This baseline alone proves neither novelty nor method feasibility.'}
    atomic_json(OUT / 'summary_ai.json', findings)
    rows=[]
    for seed in (2026,3407,7919):
        r={x['condition']:x for x in result['per_video'] if x['seed']==seed}
        rows.append('<tr><td>'+str(seed)+'</td>'+''.join('<td>'+str(r[c]['observed_complete_count'])+'</td>' for c in ('n1','n2','n3'))+'</tr>')
    cards=[]
    for item in data['items']:
        m=mapping[item['blind_id']]
        cards.append(f'<h2>Clip {item["blind_id"]:03d}, {m["condition"]}, seed={m["seed"]}</h2>'
            f'<video controls preload="none" src="../pilot/{m["video"]}"></video><p>{html.escape(item["notes"])}</p>'
            f'<a href="../pilot/review_frames/{item["blind_id"]:03d}.jpg">时间标记帧总览</a>')
    (OUT/'report_ai.html').write_text('<!doctype html><meta charset="utf-8"><title>固定8秒次数响应结果</title>'
        '<style>body{font:16px/1.7 sans-serif;max-width:1000px;margin:32px auto}td,th{padding:10px;border:1px solid #ccc}table{border-collapse:collapse}video{width:100%}</style>'
        '<h1>目标次数变化，但没有可靠的次数与停止控制</h1><p>新增6/6完成、审阅6/6，另复用n2三条。AI帧审阅，未经独立人工验证。</p>'
        '<table><tr><th>seed</th><th>要求1次：实际</th><th>要求2次：实际</th><th>要求3次：实际</th></tr>'+''.join(rows)+'</table>'
        '<p>三个目标组的“次数准确”和“次数准确并停止”均为0/3。3407和7919对三种目标均输出4次；2026为2/4/4。'
        '不能称模型完全不受提示影响；可以说本组样本未实现随目标数准确变化。</p>'
        '<p>2026目标1的中途不充分起身按原规则不拆成两次；宽松口径会记为3次。复用的2026目标2含两次浅蹲，排除后记为2次，'
        '但仍有追加动作。上述敏感性均不改变“按目标完成并停止”全部失败。</p>'
        '<p>六条新结果已逐条核对与n2初始噪声SHA一致。仅一个模型、一种动作、三个种子；次数及阶段描述同时变化，非数字词单独消融。'
        '片头半蹲、末尾残缺动作单列，不补算完整周期。帧边界为目视近似。</p>'
        f'<p>六条正式视频总生成耗时{sum(times)/3600:.2f}小时，均值{statistics.mean(times)/60:.1f}分钟/条；不含加载、smoke和先前失败尝试。</p>'
        '<p>下一步建议停止重复扩大基线，转为可验证的相位/停止控制干预；当前结果只是失败现象证据，不是新方法有效性的证据。</p>'
        '<p><a href="annotations_ai.json">逐条标注</a> · <a href="summary_ai.json">完整汇总</a> · '
        '<a href="detail_005_000_009.jpg">005片头逐帧</a> · <a href="detail_005_126_156.jpg">005中段逐帧</a></p>'+''.join(cards),encoding='utf-8')
    print(json.dumps({'paired':result['paired_count_response'],'resources':result['resources'],'report':str(OUT/'report_ai.html')},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['details','publish'])
    args=parser.parse_args()
    setup()
    details() if args.stage=='details' else publish()



