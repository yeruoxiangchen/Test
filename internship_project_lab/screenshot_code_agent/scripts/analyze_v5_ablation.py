#!/usr/bin/env python3
"""Archive and analyze the completed ablation while the extended batch continues."""
import argparse
import collections
import hashlib
import json
import re
import shutil
import statistics
from pathlib import Path

from Levenshtein import distance
from PIL import Image, ImageChops

from analyze_v5_supplementary import phase_summary
from screenshot_agent.control import atomic_json
from screenshot_agent.report import read_events

ROOT = Path(__file__).resolve().parents[1]
PHASES = ['clean_available', 'clean_initial_only', 'degraded_available', 'degraded_initial_only']
LABELS = ['原图，允许裁剪/重读', '原图，只读一次', '降质图，允许裁剪/重读', '降质图，只读一次']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def layout_normalize(text):
    # Auxiliary post-hoc diagnostic only. Do not change the primary CER.
    return '\n'.join(line.rstrip() for line in text.splitlines()
        if line.strip() and not re.fullmatch(r'[-=]{3,}', line.strip()))


def analyze(run, output):
    state = read(run/'status.json')
    if state['status'] != 'complete' or not state['model_released']:
        raise ValueError('Wait for the ablation and its model cleanup to finish')
    source = ROOT/'outputs/v5_eval_source'
    frozen = read(source/'source_state.json')
    if any(sha(source/name) != digest for name,digest in frozen['sha256'].items()):
        raise ValueError('Frozen runtime changed')
    plan = {p['name']:p for p in state['phases']}
    summaries, readings, failures, configs, input_hashes = {}, {}, [], [], {}
    pixel_checks = 0
    for phase in PHASES:
        folder = run/phase
        status = read(folder/'evaluation/status.json')
        if status['status'] != 'complete' or status['completed'] != 20:
            raise ValueError('Expected all twenty attempts in each phase')
        if sha(Path(plan[phase]['manifest'])) != plan[phase]['manifest_sha256']:
            raise ValueError('Manifest changed during evaluation')
        config = read(folder/'evaluation/batch_job.json')['config']
        available = phase.endswith('_available')
        if config['screenshots'] != {'allow_crop':available, 'allow_reread':available}:
            raise ValueError('Screenshot tool policy differs from the plan')
        configs.append({k:v for k,v in config.items() if k != 'screenshots'})
        summary = phase_summary(folder, 'vlm')
        readings[phase] = {}
        diagnostic = collections.defaultdict(list)
        normalized_exact = 0
        for row in summary['rows']:
            directory = Path(row['run_dir'])
            actual = read(directory/'result.json')
            if actual['resolved'] != row['resolved'] or not actual['model_inference']:
                raise ValueError('Report does not match raw model verdicts')
            if row['resolved'] and not (row['validation']['patch_replay']['applied']
                    and row['validation']['final']['returncode'] == 0
                    and not row['validation']['final']['timed_out']):
                raise ValueError('Successful verdict lacks independent verification')
            task = read(directory/'task.json')
            input_image = Path(task['images'][0]['path'])
            with Image.open(input_image) as original, Image.open(directory/'screenshots/shot.png') as used:
                a,b = original.convert('RGB'),used.convert('RGB')
                if a.size != b.size or ImageChops.difference(a,b).getbbox() is not None:
                    raise ValueError('Model screenshot differs from the declared input')
            pixel_checks += 1
            events = read_events(directory/'events.jsonl')
            reading = next(e['reading'] for e in events if e['event'] == 'screenshot_read')
            readings[phase][row['task_id']] = reading['raw_text']
            reference = Path(task['images'][0]['reference']).read_text()
            a,b = layout_normalize(reference),layout_normalize(reading['raw_text'])
            diagnostic[reading['kind']].append(distance(a,b)/max(1,len(a)))
            normalized_exact += a == b
            for name in ['result.json','events.jsonl','patch.diff','verification_test.log','config.json']:
                if (directory/name).exists():
                    input_hashes[str(directory/name)] = sha(directory/name)
            if row['resolved']:
                continue
            edits = [e['arguments'] for e in events if e['event'] == 'tool_start' and e.get('name') == 'edit_file']
            edit_failures = []
            for event in events:
                if event['event'] == 'tool_result' and event.get('name') == 'edit_file':
                    reply = json.loads(event['output'])
                    if reply.get('returncode'):
                        edit_failures.append(reply['output'])
            failures.append({'phase':phase, 'task_id':row['task_id'], 'exit_reason':row.get('exit_reason'),
                'changed_files':row.get('changed_files',[]), 'source_edit_attempts':len(edits),
                'failed_source_edits':len(edit_failures), 'first_edit_arguments':edits[0] if edits else None,
                'last_edit_arguments':edits[-1] if edits else None,
                'error_examples':list(dict.fromkeys(edit_failures))[:2], 'run_dir':str(directory)})
        summary.pop('rows')
        summary['auxiliary_layout_normalized_cer'] = {k:statistics.mean(v) for k,v in diagnostic.items()}
        summary['auxiliary_normalized_exact'] = normalized_exact
        summaries[phase] = summary
    if any(c != configs[0] for c in configs):
        raise ValueError('Models or numerical budgets differ')
    pairs = []
    for variant in ['clean','degraded']:
        on = read(run/(variant+'_available')/'evaluation/report.json')['records']
        off = {r['task_id']:r for r in read(run/(variant+'_initial_only')/'evaluation/report.json')['records']}
        if {r['task_id'] for r in on} != set(off):
            raise ValueError('Task pairs differ')
        pairs.append({'variant':variant, 'tasks':20,
            'identical_first_transcriptions':sum(readings[variant+'_available'][k] == readings[variant+'_initial_only'][k] for k in off),
            'available_only_pass':[r['task_id'] for r in on if r['resolved'] and not off[r['task_id']]['resolved']],
            'initial_only_pass':[r['task_id'] for r in on if not r['resolved'] and off[r['task_id']]['resolved']],
            'both_fail':[r['task_id'] for r in on if not r['resolved'] and not off[r['task_id']]['resolved']]})
    total = sum(s['runs'] for s in summaries.values())
    passed = sum(s['resolved'] for s in summaries.values())
    crops = sum(s['crop_tool_calls'] for s in summaries.values())
    rereads = sum(s['screenshot_reads']-s['runs'] for s in summaries.values())
    stats = read(run/'final_model_stats.json')
    result = {'complete':True, 'scope':'Completed controlled-capture ablation only; extended repository batch is separate.',
        'runs':total, 'passed':passed, 'unique_tasks':20, 'unique_failed_tasks':len({r['task_id'] for r in failures}),
        'organic_user_screenshots':0, 'crop_calls':crops, 'additional_reads':rereads,
        'same_models_and_numerical_budgets':True, 'input_pixel_checks':pixel_checks,
        'summaries':summaries, 'pairs':pairs, 'failures':failures,
        'elapsed_minutes':(state['finished_at']-state['started_at'])/60,
        'peak_cuda_allocated_gib':stats['peak_cuda_memory_bytes']/2**30,
        'primary_scores_changed':False, 'new_model_calls':0,
        'auxiliary_metric_scope':'Post-hoc diagnostic removes blank lines, trailing spaces and pure ---/=== divider lines. Primary CER remains untouched.'}
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'analysis.json', result)
    for name in ['status.json','plan.json','pilot_passed.json','final_model_stats.json']:
        shutil.copy2(run/name, output/name)
    shutil.copy2(source/'source_state.json',output/'source_state.json')
    for phase in PHASES:
        destination = output/phase
        destination.mkdir()
        for name in ['report.json','report.md','batch_job.json','status.json']:
            shutil.copy2(run/phase/'evaluation'/name,destination/name)
        for row in read(run/phase/'evaluation/report.json')['records']:
            original = Path(row['run_dir'])
            target = destination/'trials'/original.name
            target.mkdir(parents=True)
            names = ['result.json','config.json','task.json','patch.diff','baseline_test.log','verification_test.log']
            if not row['resolved']:
                names += ['events.jsonl','conversation.json']
            for name in names:
                if (original/name).exists():
                    shutil.copy2(original/name,target/name)
    lines = ['# v5 截图功能消融分析','',
        f'本组已完成 {total}/{total} 次运行，独立验证通过 {passed} 次（{passed/total:.1%}）。这是同一批 20 个受控微型任务的四组条件，不是 80 个独立任务。另有 2 次端到端预检，不计入主表。',
        '50 题真实仓库来源的三组扩展评测独立运行，本报告不将其未完成成绩并入。','',
        '| 条件 | 通过 / 尝试 | 成功率 | 平均耗时 | 平均工具调用 | 首轮原始 CER |',
        '|---|---:|---:|---:|---:|---:|']
    for phase,label in zip(PHASES,LABELS):
        s = summaries[phase]
        lines.append(f"| {label} | {s['resolved']}/{s['runs']} | {s['resolved_fraction']:.0%} | {s['elapsed_seconds_mean']:.1f} 秒 | {s['tool_calls_mean']:.2f} | {s['reading_cer_mean']:.2%} |")
    lines += ['', '## 能得出的结论','',
        '截图读取 → 检索源码 → 修改 → 测试 → 补丁重放的工程闭环已在这些任务中工作。76 次通过均有实际补丁应用和独立测试记录；其余 4 次保留为失败。',
        f'整轮裁剪调用 {crops} 次，额外重读 {rereads} 次。同一图像的两种工具配置，20/20 题首轮转写完全一致；原图、降质图均如此。',
        '因此本轮没有测到“实际裁剪/重读操作带来的收益”。原图组 100% 对 90% 的差异发生在不同的工具清单和后续执行轨迹下，不能写成“裁剪提升 10 个百分点”。降质组两边均为 95%。',
        '这些微型题主要只有一个短源码文件，测试可见，图片证据未必是完成修复所必需。不能将 95% 用作真实复杂仓库、自然用户截图或 SWE-bench 的成功率。','',
        '## 四次失败的实际轨迹','',
        '| 条件 / 任务 | 证据与失败位置 |','|---|---|',
        '| 原图、只读一次 / count_words | 改为 `re.split` 后空字符串仍返回 1，固定测试要求 0。模型已经收到失败反馈，但随后重复文字解释，输出达到 2048 token，没有继续编辑；错误补丁被独立验证拒绝。 |',
        '| 原图、只读一次 / csv_fields | 最终补丁未处理空字符串，返回 `[\'\']` 而非 `[]`。尝试补救时 `old_string` 多了一个不存在的 `}`，多次精确匹配失败；18 次代码调用耗尽。 |',
        '| 降质图、允许裁剪重读 / merge | 提出的方向是复制字典后更新，但编辑参数把换行写成字面量反斜杠转义，无法匹配真实多行源码；重复失败到步数上限，产生空补丁。 |',
        '| 降质图、只读一次 / merge | 同样发生换行转义与精确匹配失败，耗尽步数，产生空补丁。 |',
        '', '四次失败涉及 3 个不同任务；另外 17/20 题在四组都通过。没有运行时请求错误或视觉 JSON 格式错误。可观察到的直接失败点是代码边界条件、编辑参数和失败后恢复，未出现靠裁剪解决歧义的实际轨迹。',
        '文件读取反馈存在嵌套 JSON 转义；它可能增加代码模型构造多行编辑参数的难度。这是待验证的接口改进方向，当前记录不足以单独证明其因果作用。','',
        '## 如何理解读取误差','',
        '主表 CER 保留空白和终端分隔线长度差异。辅助诊断只去掉空行、行尾空格及纯横线/等号分隔行，不修改主成绩：10 个代码截图在原图和降质图中均与参考文本一致；终端截图的平均 CER 分别约为 0.31% 和 0.27%。',
        '每种图像条件有 18/20 题在上述辅助规范化后完全一致。因此原始 6.3% CER 不应解读成约 6.3% 的代码逻辑识别错误。降质后的平均 CER 略低，也不能据此宣称模糊图片更好。','',
        '## 后续优先级','',
        '1. 保持扩展集冻结运行，等待 50 题三组结果，判断真实仓库中的主要限制。',
        '2. 在单独的新版本中改进文件读取/编辑错误反馈，减少多层转义；为文字重复、无效编辑及长度截断提供准确结束状态，再对这些失败任务做独立回归。',
        '3. 为裁剪功能补充确实需要局部读取的截图，检查实际调用和重读后证据变化；当前工具可用性实验不能替代该验证。',
        '4. 现有证据优先支持工程修正和代码模型执行能力排查，尚不足以决定做 SFT/RL。',
        '', '## 运行与证据','',
        f"服务墙钟约 {result['elapsed_minutes']:.2f} 分钟，包含模型加载和额外预检；PyTorch 峰值已分配显存 {result['peak_cuda_allocated_gib']:.2f} GiB。GPU 7 模型已释放。",
        'API 标价为零，硬件与时间成本另计。未新增模型调用；80 份实际输入已逐像素核对，冻结运行源码和原始结果指纹已验证。',
        '分析、逐题补丁、测试结果和四条失败轨迹见 `docs/artifacts/v5_ablation/`；完整原始轨迹仍位于 `outputs/v5_supplementary/ablation/`。']
    document = ROOT/'docs/experiment_report_v5_ablation.md'
    document.write_text('\n'.join(lines)+'\n')
    shutil.copy2(document,output/'report.md')
    if any(sha(Path(path)) != digest for path,digest in input_hashes.items()):
        raise ValueError('Analysis unexpectedly changed a raw result')
    atomic_json(output/'verification.json',{'raw_results_unchanged':True, 'runtime_unchanged':True,
        'source_files_verified':len(frozen['sha256']), 'image_pixel_checks':pixel_checks,
        'raw_sha256':input_hashes,
        'artifact_sha256':{str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p.name != 'verification.json'}})
    return {k:result[k] for k in ['runs','passed','unique_tasks','crop_calls','additional_reads','elapsed_minutes','peak_cuda_allocated_gib']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=Path('outputs/v5_supplementary/ablation'))
    parser.add_argument('--output',type=Path,default=Path('docs/artifacts/v5_ablation'))
    args = parser.parse_args()
    print(json.dumps(analyze(args.run.resolve(),args.output.resolve()),indent=2))
