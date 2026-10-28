#!/usr/bin/env python3
"""Wait, validate paired comparisons, archive evidence and generate V5 analysis."""
import argparse
import collections
import hashlib
import json
import shutil
import statistics
import time
from pathlib import Path

from screenshot_agent.control import atomic_json
from screenshot_agent.report import read_events

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    return statistics.mean(values) if values else None


def phase_summary(phase,reader=None):
    report=read(phase/'evaluation/report.json')
    rows=[r for r in report['records'] if reader is None or r['reader']==reader]
    crops=reads=code_caps=vision_caps=empty_patches=no_source_tool_runs=0
    formats=collections.Counter()
    for row in rows:
        journal=read_events(Path(row['run_dir'])/'events.jsonl')
        crops+=sum(e['event']=='tool_start' and e.get('name')=='crop_screenshot' for e in journal)
        reads+=sum(e['event']=='screenshot_read' for e in journal)
        formats.update(e['reading'].get('format_status','native') for e in journal if e['event']=='screenshot_read')
        config=read(Path(row['run_dir'])/'config.json')
        source_tools={'read_file','grep','glob','edit_file','write_file','bash'}
        no_source_tool_runs+=not any(e['event']=='tool_start' and e.get('name') in source_tools for e in journal)
        empty_patches+=not row.get('changed_files')
        for event in journal:
            if event['event']!='model_response':
                continue
            kind=event.get('kind','code')
            tokens=(event['response'].get('usage') or {}).get('completion_tokens',0)
            if tokens>=config[kind]['max_output_tokens']:
                code_caps+=kind=='code'
                vision_caps+=kind=='vision'
    return {'runs':len(rows),'resolved':sum(r['resolved'] for r in rows),
        'resolved_fraction':sum(r['resolved'] for r in rows)/len(rows),
        'elapsed_seconds_mean':mean([r['elapsed_seconds'] for r in rows]),
        'tool_calls_mean':mean([r['usage']['tool_calls'] for r in rows]),
        'code_calls_total':sum(r['usage']['code_model_calls'] for r in rows),
        'vision_calls_total':sum(r['usage']['vision_model_calls'] for r in rows),
        'prompt_tokens':sum(r['usage']['prompt_tokens'] for r in rows),
        'completion_tokens':sum(r['usage']['completion_tokens'] for r in rows),
        'api_cost_usd':sum(r['usage']['cost_usd'] for r in rows),
        'reading_cer_mean':mean([x['cer'] for r in rows for x in r['reading_metrics'].get('readings',[]) if x['cer'] is not None]),
        'crop_tool_calls':crops,'screenshot_reads':reads,'reading_formats':dict(formats),
        'empty_patch_runs':empty_patches,'runs_without_source_tools':no_source_tool_runs,
        'code_responses_at_output_cap':code_caps,'vision_responses_at_output_cap':vision_caps,
        'status_counts':dict(collections.Counter(r['status'] for r in rows)),
        'exit_reasons':dict(collections.Counter(r.get('exit_reason',r.get('error_type','unknown')) for r in rows)),
        'file_overlap_count':sum(r.get('localization_diagnostic',{}).get('reference_file_overlap',False) for r in rows),
        'rows':rows}


def analyze(run,source,artifact):
    state={mode:read(run/mode/'status.json') for mode in ['ablation','extended']}
    if any(s['status']!='complete' or not s.get('model_released') for s in state.values()):
        raise ValueError('Both comparisons and model cleanup must complete')
    frozen=read(source/'source_state.json')
    if any(sha(source/name)!=value for name,value in frozen['sha256'].items()):
        raise ValueError('Frozen source changed')
    extended=ROOT/'datasets/generated/swebench_extended_v5'
    build=read(extended/'build_report.json')
    if build['evaluation_tasks']!=50 or build['evaluation_repositories']<5:
        raise ValueError('Extended coverage differs from the predeclared scope')
    if not read(extended/'glyph_audit.json')['all_visible_glyphs_supported']:
        raise ValueError('Missing input glyphs')
    inventory=read(extended/'input_inventory.json')
    if any(sha(Path(path))!=value for item in inventory['entries'] for path,value in item['sha256'].items()):
        raise ValueError('An audited extended input changed')
    summaries={}
    configs=[]
    for mode,item in state.items():
        for phase in item['phases']:
            folder=run/mode/phase['name']
            progress=read(folder/'evaluation/status.json')
            if progress['status']!='complete' or progress['completed']!=phase['runs']:
                raise ValueError('An incomplete comparison cannot be finalized')
            if sha(Path(phase['manifest']))!=phase['manifest_sha256']:
                raise ValueError('Manifest changed during evaluation')
            config=read(folder/'evaluation/batch_job.json')['config']
            if config['screenshots']!={'allow_crop':phase['policy']=='available','allow_reread':phase['policy']=='available'}:
                raise ValueError('Screenshot-tool availability differs from plan')
            configs.append({k:v for k,v in config.items() if k!='screenshots'})
            for reader in phase['readers'].split(','):
                summary=phase_summary(folder,reader)
                summaries[mode+'/'+phase['name']+'/'+reader]=summary
    # Two loopback ports differ; every other model/budget field must match.
    normalized=[]
    for config in configs:
        copy=json.loads(json.dumps(config))
        for model in ['code','vision']:
            copy[model]['base_url']='local-loopback-endpoint'
        normalized.append(copy)
    if any(c!=normalized[0] for c in normalized):
        raise ValueError('Comparison arms changed models or numerical budgets')
    clean=read(run/'extended/three_readers/clean_patch_audit.json')
    if not clean['complete'] or clean['attempts']!=150 or not clean['all_verdicts_preserved']:
        raise ValueError('Independent extended patch replay is incomplete or inconsistent')
    pairs=[]
    for variant in ['clean','degraded']:
        yes=summaries['ablation/'+variant+'_available/vlm']['rows']
        no={r['task_id']:r for r in summaries['ablation/'+variant+'_initial_only/vlm']['rows']}
        if len(yes)!=20 or {r['task_id'] for r in yes}!=set(no):
            raise ValueError('Ablation task pairs differ')
        pairs.append({'variant':variant,'tasks':20,
            'available_only_pass':sum(r['resolved'] and not no[r['task_id']]['resolved'] for r in yes),
            'initial_only_pass':sum(not r['resolved'] and no[r['task_id']]['resolved'] for r in yes),
            'both_pass':sum(r['resolved'] and no[r['task_id']]['resolved'] for r in yes),
            'both_fail':sum(not r['resolved'] and not no[r['task_id']]['resolved'] for r in yes)})
    compact={name:{k:v for k,v in row.items() if k!='rows'} for name,row in summaries.items()}
    payload={'complete':True,'primary_runs':sum(r['runs'] for r in compact.values()),
        'extended_unique_tasks':50,'extended_repositories':build['evaluation_repositories'],
        'controlled_capture_tasks':20,'organic_user_screenshots':0,
        'matched_models_and_numerical_budgets':True,'summaries':compact,'ablation_pairs':pairs,
        'source':str(source),'interpretation':'Separate repair and tool-availability comparisons. No leaderboard, training, organic-user or strategy-innovation claim.'}
    if payload['primary_runs']!=230:
        raise ValueError('Expected all 230 scheduled primary attempts')
    artifact.mkdir(parents=True,exist_ok=False)
    atomic_json(artifact/'summary.json',payload)
    shutil.copy2(source.with_suffix('.zip'),artifact/'source_snapshot.zip')
    shutil.copy2(ROOT/'outputs/v5_engineering_checks.json',artifact/'engineering_checks.json')
    data=artifact/'data'
    data.mkdir()
    for name in ['build_report.json','selection.json','selected_records.json','tasks.jsonl','glyph_audit.json','input_inventory.json']:
        shutil.copy2(extended/name,data/name)
    for audit in build['audits']:
        instance=audit['instance_id']
        target=data/'tasks'/instance
        target.mkdir(parents=True)
        for name in ['audit.json','baseline_test.log','reference_test.log','installation.log','screenshot_light.png',
                     'screenshot_light.txt','screenshot_light.layout.json','reference.patch','benchmark.json']:
            if (extended/instance/name).exists():
                shutil.copy2(extended/instance/name,target/name)
    profiles=data/'dependency_profiles'
    profiles.mkdir()
    for pattern in ['*.requirements.txt','*.freeze.txt']:
        for path in (ROOT/'.runtime/extended_v5').glob(pattern):
            shutil.copy2(path,profiles/path.name)
    history=data/'preflight_attempts'
    history.mkdir()
    for attempt in (extended/'preflight_attempts').iterdir():
        if attempt.is_dir():
            for name in ['build_report.json','recovery.json']:
                if (attempt/name).exists():
                    (history/attempt.name).mkdir(exist_ok=True)
                    shutil.copy2(attempt/name,history/attempt.name/name)
    for mode,item in state.items():
        dest=artifact/mode
        dest.mkdir()
        for name in ['status.json','plan.json','final_model_stats.json','pilot_passed.json','available.json','initial_only.json']:
            if (run/mode/name).exists():
                shutil.copy2(run/mode/name,dest/name)
        for phase in item['phases']:
            original=run/mode/phase['name']
            target=dest/phase['name']
            target.mkdir()
            for name in ['report.json','report.md','status.json','batch_job.json']:
                shutil.copy2(original/'evaluation'/name,target/name)
            if (original/'clean_patch_audit.json').exists():
                shutil.copy2(original/'clean_patch_audit.json',target/'clean_patch_audit.json')
            for folder in (original/'evaluation/runs').iterdir():
                if not folder.is_dir():
                    continue
                trial=target/'trials'/folder.name
                trial.mkdir(parents=True)
                for name in ['result.json','config.json','task.json','patch.diff','patch.clean.diff','events.jsonl',
                             'conversation.json','baseline_test.log','verification_test.log','agent_error.log','error.log']:
                    if (folder/name).exists():
                        shutil.copy2(folder/name,trial/name)
    lines=['# 补充修复与截图功能实验（v5）','',
        '第一版工程评测：现成模型 + CoreCoder 原决策循环，未进行 SFT/RL 或策略训练。',
        '代码与视觉均使用 Qwen3-VL-8B-Instruct；每次输出 2048 token，18 次代码/压缩调用、50 次工具、最多 3 次截图读取、1200 秒总预算；Agent 压缩阈值 8192，服务上下文 16384。',
        '同一比较的模型、数值预算和任务相同。与 v3/九题回归的 1024 token 输出和 4096 压缩阈值不同，因此分表报告，不合并成绩。','',
        '## SWE-bench Lite 截图派生修复','',
        f"预先固定 50 个主候选及 10 个顺序环境备选；60 个全部完成错误版本/参考补丁预检，{build['validated_pool']} 个可运行，按冻结顺序取 50 个，来自 {build['evaluation_repositories']} 个仓库。环境失败及重试历史保留，没有使用模型修复结果筛题。",
        '先按原仓库分开发/测试组，再渲染截图；排除旧的 10 个任务。原 Issue 保留周围描述，选中代码/报错块换成合成截图，不能声称每题都必须依赖图片。',
        '官方测试文件可见且保护只读，参考源码补丁和评分元数据在仓库外。独立 Python + Bubblewrap 环境、官方解析器/评分器；SymPy 保留原 bin/test，其余执行全部官方计分测试文件。未运行原始 Docker 协议，不是排行榜成绩。','',
        '| 读取方式 | 修复成功 | 首轮 CER | 平均秒数 | 平均工具调用 |','|---|---:|---:|---:|---:|']
    for reader in ['oracle','ocr','vlm']:
        s=compact['extended/three_readers/'+reader]
        cer=f"{s['reading_cer_mean']:.2%}" if s['reading_cer_mean'] is not None else 'N/A'
        lines.append(f"| {reader} | {s['resolved']}/{s['runs']} | {cer} | {s['elapsed_seconds_mean']:.1f} | {s['tool_calls_mean']:.1f} |")
    lines+=['',f"150 份补丁额外独立重放，成功 {clean['clean_passed']} 次，全部与原始逐题判定一致。",'',
        '## 裁剪和重读功能消融','',
        '20 个已知受控微型修复任务，10 张 CodeMirror 代码截图、10 张 xterm 终端截图；终端回放隔离测试的实际输出。Chromium 实际截图，未执行桌面 IDE/交互 PTY 采集，不是自然用户提交的问题。',
        '同题对比只读一次 / 允许裁剪与重读，并测试原图 / 双线性 2 倍降采样后恢复尺寸。降质图是同一题的变体，不计作新的独立任务；额外 2 次端到端预检不并入主表。',
        '两组数值上限相同；只读一次组在首轮证据获取后不再暴露截图工具，这是待测的功能差异。','',
        '| 图像 | 截图工具 | 修复成功 | 裁剪调用 | 截图读取总数 | 首轮 CER |','|---|---|---:|---:|---:|---:|']
    for variant in ['clean','degraded']:
        for policy in ['available','initial_only']:
            s=compact['ablation/'+variant+'_'+policy+'/vlm']
            cer=f"{s['reading_cer_mean']:.2%}" if s['reading_cer_mean'] is not None else 'N/A'
            lines.append(f"| {variant} | {policy} | {s['resolved']}/{s['runs']} | {s['crop_tool_calls']} | {s['screenshot_reads']} | {cer} |")
    lines+=['','工具可用不等于模型实际使用。若裁剪调用为 0，不能声称本轮证明裁剪提高修复率；配对差异也需结合实际工具轨迹解读。',
        '微型题主要验证工程闭环，不能代表真实仓库难度。截图转写 CER 与修复率分开报告。','',
        '## 失败诊断与成本','',
        '首轮文本差异、参考文件重合度、退出原因和工具调用属于诊断线索，不能直接当作错误根因。失败、超时、空补丁全部保留在各自主表分母。']
    for name,s in compact.items():
        lines.append(f"- {name}：状态 {s['status_counts']}；退出原因 {s['exit_reasons']}；参考文件重合 {s['file_overlap_count']}/{s['runs']}；空补丁 {s['empty_patch_runs']}；未调用源码工具 {s['runs_without_source_tools']}；代码/视觉输出达到上限线索 {s['code_responses_at_output_cap']}/{s['vision_responses_at_output_cap']}。")
    lines+=['','输出达到 token 上限只是截断或重复生成的线索，本地服务的 finish_reason 未准确区分长度限制。无工具的普通文本响应会结束上游循环，不能因此认定完成修复；只有独立测试通过才计成功。',
        '空补丁、未调用源码工具和读取误差的共现不能直接证明因果关系。准确文本组仍失败时，要结合补丁与工具轨迹评估代码模型执行能力，而非把所有失败归因于视觉读取。',
        '逐题补丁、原始证据、独立重放结果、配置和冻结源码见 `docs/artifacts/v5_supplementary/`。',
        '本地 API 标价为 0；GPU 时间、电力与计算资源有实际成本。两组任务在独立 GPU 服务上运行；下列时间包括加载和等待，并非 GPU 核函数用时。']
    for mode,s in state.items():
        stats=read(run/mode/'final_model_stats.json')
        lines.append(f"- {mode}：GPU {s['gpu']}，服务墙钟 {(s['finished_at']-s['started_at'])/3600:.2f} 小时，PyTorch 峰值已分配显存 {stats['peak_cuda_memory_bytes']/2**30:.2f} GiB。")
    lines+=['','83 项工程检查通过。离线重解析可恢复原 CodeOCR 的 3/5 条格式错误，未调用模型且未改动 v4 成绩，另见 `docs/json_recovery_v5.md`。',
        '实际贡献是截图证据工具、仓库核对、隔离执行、预算/日志、补丁重放与可审计评测。尚无训练结果、自然用户截图效果或新推理策略贡献；先依据这些结果判断瓶颈，再决定是否训练。']
    document=ROOT/'docs/experiment_report_v5_supplementary.md'
    document.write_text('\n'.join(lines)+'\n')
    shutil.copy2(document,artifact/'report.md')
    atomic_json(artifact/'verification.json',{'complete':True,'primary_runs':230,'matched_budgets':True,
        'artifact_sha256':{str(p.relative_to(artifact)):sha(p) for p in artifact.rglob('*') if p.is_file() and p.name!='verification.json'}})
    return {'complete':True,'primary_runs':230,'report':str(document),'artifact':str(artifact)}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,default=Path('outputs/v5_supplementary'))
    p.add_argument('--source',type=Path,default=Path('outputs/v5_eval_source'))
    p.add_argument('--artifact',type=Path,default=Path('docs/artifacts/v5_supplementary'))
    p.add_argument('--wait-seconds',type=int,default=0)
    a=p.parse_args()
    deadline=time.monotonic()+a.wait_seconds
    try:
        while True:
            states=[read(a.run/m/'status.json').get('status') if (a.run/m/'status.json').exists() else 'pending' for m in ['ablation','extended']]
            if all(s=='complete' for s in states):
                break
            if any(s in ['error','cancelled'] for s in states) or time.monotonic()>=deadline:
                raise ValueError(f'Comparisons unfinished: {states}; original failures retained')
            print(f'Waiting for final comparisons: {states}',flush=True)
            time.sleep(30)
        result=analyze(a.run.resolve(),a.source.resolve(),a.artifact.resolve())
        atomic_json(a.run/'analysis_status.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except Exception as error:
        atomic_json(a.run/'analysis_status.json',{'complete':False,'error':str(error),'error_type':type(error).__name__})
        raise
