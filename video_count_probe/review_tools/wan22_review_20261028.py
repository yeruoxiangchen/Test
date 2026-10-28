"""Prepare time-labelled evidence for the existing six-video Wan2.2 pilot.

CPU decoding only. Does not change generation code, plans, or videos.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path('/data/zjr/video_count_wan22_action_20261028')
RUN = ROOT / 'pilot'
OUT = RUN / 'ai_review_20261028'
FFMPEG = Path('/home/zjr/Tracker/video_count_probe/.venv/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')


def publish_review():
    """Persist the fixed judgments made by Codex after inspecting the sheets.

    This is a dated manual AI review record, not an automatic video classifier.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from video_count_probe.core import atomic_json, checked_plan, file_sha, read_json, writer_lock
    from video_count_probe.wan22_action_probe.review import annotation_scores, records, report

    plan = checked_plan(RUN / 'plan.json')
    completed = records(RUN, plan)
    if len(completed) != 6:
        raise ValueError('Expected all six original videos to be verified')
    template = read_json(RUN / 'annotations_template.json')
    if template['plan_hash'] != plan['plan_hash']:
        raise ValueError('Annotation plan mismatch')
    judgments = {
        1: (True, 'complete',
            '可确认至少一个完整深蹲：f045/1.875s 直立，f060-f069/2.500-2.875s 髋部明显降低、双膝弯曲，f084-f087/3.500-3.625s 恢复直立。开头已处于下降阶段，约4.5s以后又开始下降，均不额外算完整周期。站立时头部出画，但髋、膝、脚仍可用于判定。',
            [[45, 87]], ['head_cropped', 'initial_partial_cycle', 'terminal_partial_cycle']),
        2: (None, 'uncertain',
            '存在反复跳跃与双脚开合，例如f024-f042/1.000-1.750s、f078-f100/3.250-4.167s；但在张腿的关键阶段，头部与手臂上端被画面裁切，不能确认双手是否举过头顶。约2.25-2.50s还混入明显屈膝下蹲。按原标准保留无法判定，不能把不可见的手臂推断成成功，也不能断言完全没有开合跳。',
            [[24, 42], [54, 60], [78, 100]], ['critical_arms_out_of_frame', 'mixed_squat_motion']),
        3: (False, 'incomplete',
            '检查全部f000-f120原始帧后，可见主要为窄站距原地跳动、交替抬脚；双腿始终没有完成清楚的闭合→张开→闭合。手臂多保持屈肘于胸前，缺少可确认的完整开合跳周期。即使不依赖被裁切的头部，双脚运动也不满足标准。',
            [[0, 120]], ['wrong_motion_narrow_hopping', 'head_cropped']),
        4: (False, 'incomplete',
            '检查全部f000-f120原始帧后，双手主要保持交叉重叠，呈握持手腕/搓动与手指形变；没有辨认出掌面接触后再明确分离的完整拍手。局部运动模糊与手指张合不计为完整动作。此结论是可见动作标准未通过，不推断不可见的物理接触次数。',
            [[0, 120]], ['hands_overlap', 'hand_deformation', 'no_visible_release']),
        5: (False, 'incomplete',
            'f000-f009/0.000-0.375s双手分开并靠近，f011-f018/0.458-0.750s接触/重叠。随后直到f120/5.000s主要保持贴合，只见小幅手指、手腕抖动；逐帧检查没有确认接触后的清楚分离，因此只有接近/接触，未完成规定拍手周期。',
            [[0, 23], [24, 120]], ['approach_without_visible_release', 'hands_remain_overlapped']),
        6: (True, 'complete',
            '可确认完整深蹲：f000-f024/0.000-1.000s直立，f036-f042/1.500-1.750s髋部降低、双膝弯曲，f078-f090/3.250-3.750s恢复直立。下降过程中有局部起伏，不能每次起伏都计一次；约4.0s又开始下蹲，片尾尚未清楚恢复直立，不另算完整周期。头部偶尔出画，主要下肢动作可辨认。',
            [[0, 90]], ['head_cropped', 'partial_bobbing', 'terminal_partial_cycle']),
    }
    items = []
    for row in template['items']:
        index = row['blind_id']
        decision, status, notes, windows, issues = judgments[index]
        detail_paths = sorted((OUT / 'detail').glob(f'{index:03d}_*.jpg'))
        if index in (2, 3, 4, 5) and len(detail_paths) != 5:
            raise ValueError(f'Missing original-frame-rate evidence for clip {index}')
        items.append({**row, 'reviewed': True, 'complete_cycle': decision,
                      'review_status': status, 'notes': notes,
                      'evidence_frame_intervals': windows, 'issues': issues,
                      'overview_sheet': str(RUN / 'review_frames' / f'{index:03d}.jpg'),
                      'detail_sheets_inspected': [str(p) for p in detail_paths],
                      'inspection': '8 fps overview plus all 121 original temporal frames at 24 fps; spatially resized to 480x264' if detail_paths else '8 fps time-labelled overview; 41 sampled frames',
                      'exact_cycle_count': None})
    annotation = {'plan_hash': plan['plan_hash'], 'rater': 'Codex AI visual frame review',
                  'rater_type': 'ai', 'human_validated': False,
                  'reviewed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'blinding': 'Sheets used blind serial IDs, but reviewer also accessed job/seed metadata; this is not an independently blinded review.',
                  'items': items}
    scored = annotation_scores(annotation, plan, completed)
    if len(scored) != 5 or sum(r['complete_cycle'] for r in scored) != 2:
        raise ValueError('Unexpected review aggregation')
    annotation_path = RUN / 'annotations.json'
    with writer_lock(RUN / '.writer.lock'):
        if annotation_path.exists():
            previous = read_json(annotation_path)
            if previous.get('rater') != annotation['rater']:
                raise RuntimeError('Refusing to replace another reviewer')
            if previous['items'] != items:
                raise RuntimeError('Existing judgments differ; save a separate revision')
            annotation = previous
        backup = OUT / 'summary_before_ai_review.json'
        if not backup.exists():
            atomic_json(backup, read_json(RUN / 'summary.json'))
        atomic_json(OUT / 'annotations_ai.json', annotation)
        atomic_json(annotation_path, annotation)
    official = report(ROOT, 'pilot')
    result = {**official, 'ai_reviewed': 6, 'ai_review_complete': True,
              'decided_videos': len(scored), 'uncertain_videos': 1,
              'human_validated': False, 'complete_action_lower_bound_all_six': 2,
              'complete_action_upper_bound_if_uncertain_passes': 3,
              'interpretation': 'Bounds express annotation uncertainty only, not statistical confidence or population performance.',
              'decision': 'Proceed only with a small squat count-and-stop inference comparison; defer clapping and jumping jacks; do not start training from this screen.',
              'per_video': [{**r, 'action': completed[r['job_id']]['job']['case_id'],
                             'seed': completed[r['job_id']]['job']['seed'],
                             'video': completed[r['job_id']]['video']} for r in items],
              'evidence_sha256': {str(p.relative_to(RUN)): file_sha(p)
                                  for p in [*(RUN / 'review_frames').glob('*.jpg'), *(OUT / 'detail').glob('*.jpg')]}}
    atomic_json(OUT / 'summary_ai.json', result)
    names = {'clap': '拍手', 'jumping_jack': '开合跳', 'squat': '深蹲'}
    decisions = {'complete': '完整', 'incomplete': '未见完整周期', 'uncertain': '无法判定'}
    rows, cards = [], []
    for r in result['per_video']:
        label = decisions[r['review_status']]
        rows.append(f'<tr><td>{r["blind_id"]:03d}</td><td>{names[r["action"]]}</td><td>{r["seed"]}</td><td>{label}</td></tr>')
        links = ' '.join(f'<a href="detail/{Path(p).name}">{Path(p).stem}</a>' for p in r['detail_sheets_inspected'])
        cards.append(f'<article><h2>Clip {r["blind_id"]:03d} · {names[r["action"]]} · seed={r["seed"]} · {label}</h2>'
                     f'<p>{html.escape(r["notes"])}</p><video controls preload="metadata" width="768" src="../{r["video"]}"></video>'
                     f'<p><a href="../review_frames/{r["blind_id"]:03d}.jpg">8 fps 概览</a> {links}</p></article>')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Wan2.2 六段动作复核</title>
<style>body{font:16px/1.7 system-ui;max-width:1000px;margin:32px auto;padding:0 20px;color:#20242b}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px}video{max-width:100%}article{border-top:1px solid #ddd;padding-top:16px;margin-top:28px}a{color:#1456b8}</style>
<h1>Wan2.2 六段动作能力复核</h1>
<p><strong>六段均已完成 AI 视觉复核：2 段确认完整、3 段未见完整周期、1 段无法判定。</strong>
这是单一 AI 评审的可行性检查，未经独立人工验证，不能据此估计总体成功率。</p>
<p>深蹲两段都有完整周期；拍手两段都没有清楚的接触后分离；开合跳一段为窄幅跳动，另一段关键手臂被裁切，保留 null。</p>
<p>官方 summary 的 annotated=5、assessment_complete=false 表示有一段无法作二值判定，不代表漏看或生成失败。AI 复核完成状态单独保存在 summary_ai.json。</p>
<table><tr><th>编号</th><th>动作</th><th>种子</th><th>判定</th></tr>''' + ''.join(rows) + '''</table>
<h2>建议下一步</h2><p>将下一轮限制为深蹲，测试“完整做 1 次/2 次，然后保持站立”。固定模型、时长与配对种子，比较直接次数提示与显式动作阶段提示，分别记录完整次数、目标后是否启动额外半个动作、片尾是否保持站立。两段成功只说明存在可用动作基础，尚未证明次数可控或控制方法有效。</p>
<p>开合跳需要先改善全身取景与动作语义；拍手的接触与释放尚不清楚，暂不投入次数控制训练。下一轮生成和训练均未启动。</p>
<h2>复核方法与限制</h2><p>全部六段检查 8 fps 时间标记图；拍手和开合跳另检查全部 121 帧的 24 fps 图。图像只做空间缩小，没有补帧。深蹲只报告至少一个完整周期，不输出未经逐帧标定的精确总次数。此复核接触过种子映射，不宣称独立盲评；不同模型配置比较也不是单变量因果实验。</p>''' + ''.join(cards) + '</html>'
    (OUT / 'report_ai.html').write_text(document)
    print(json.dumps({k: result[k] for k in ['generated', 'ai_reviewed', 'decided_videos', 'uncertain_videos',
                                           'complete_action_videos', 'assessment_complete', 'ai_review_complete']}, ensure_ascii=False))
    print(OUT / 'report_ai.html')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ids', type=int, nargs='+', default=[2, 3, 4, 5])
    parser.add_argument('--publish-review', action='store_true', help='Save the dated judgments from the completed AI visual inspection')
    args = parser.parse_args()
    for key, suffix in {'TMPDIR': 'tmp', 'TORCH_HOME': 'cache/torch',
                        'HF_HOME': 'cache/huggingface', 'HF_HUB_CACHE': 'cache/huggingface/hub',
                        'TORCH_EXTENSIONS_DIR': 'cache/torch_extensions',
                        'MPLCONFIGDIR': 'cache/matplotlib', 'XDG_CACHE_HOME': 'cache/xdg'}.items():
        os.environ[key] = str(ROOT / suffix)
    if args.publish_review:
        publish_review()
        return
    from PIL import Image, ImageDraw

    mapping_data = json.loads((RUN / 'review_frames/private_mapping.json').read_text())
    mapping = {r['blind_id']: r for r in mapping_data['items']}
    plan = json.loads((RUN / 'plan.json').read_text())
    if mapping_data['plan_hash'] != plan['plan_hash']:
        raise ValueError('Plan binding mismatch')
    config = plan['config']
    width, height, fps = 480, 264, config['fps']
    stride = width * height * 3
    output_rows = []
    for blind_id in args.ids:
        row = mapping[blind_id]
        record = json.loads((RUN / 'records' / f"{row['job_id']}.json").read_text())
        video = RUN / record['video']
        if hashlib.sha256(video.read_bytes()).hexdigest() != row['video_sha256']:
            raise ValueError(f'Video hash mismatch: {blind_id}')
        decoded = subprocess.run([
            str(FFMPEG), '-v', 'error', '-threads', '1', '-i', str(video),
            '-vf', f'scale={width}:{height}', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-threads', '1', '-'
        ], check=True, capture_output=True).stdout
        if len(decoded) != config['frames'] * stride:
            raise ValueError(f'Unexpected frame count: {blind_id}')
        for start in range(0, 120, 24):
            end = 120 if start == 96 else start + 23
            indices = list(range(start, end + 1))
            cols, cell_h = 6, height + 22
            sheet = Image.new('RGB', (cols * width, 32 + ((len(indices) + cols - 1) // cols) * cell_h), 'white')
            draw = ImageDraw.Draw(sheet)
            draw.text((8, 8), f'Clip {blind_id:03d} | all original frames {start}-{end} | {fps} fps; spatially resized only', fill='black')
            for offset, index in enumerate(indices):
                x, y = (offset % cols) * width, 32 + (offset // cols) * cell_h
                draw.text((x + 4, y + 3), f'f{index:03d}  {index / fps:.4f}s', fill='black')
                frame = Image.frombytes('RGB', (width, height), decoded[index * stride:(index + 1) * stride])
                sheet.paste(frame, (x, y + 22))
            path = OUT / 'detail' / f'{blind_id:03d}_{start:03d}_{end:03d}.jpg'
            path.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(path, quality=95)
            output_rows.append({'blind_id': blind_id, 'frames': [start, end],
                                'fps': fps, 'video_sha256': row['video_sha256'], 'path': str(path)})
            print(path, flush=True)
    (OUT / 'detail_manifest.json').write_text(json.dumps({
        'plan_hash': plan['plan_hash'], 'items': output_rows,
        'limits': 'Creating evidence sheets is not an annotation. Reviewer must explicitly inspect them.'
    }, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
