#!/usr/bin/env python3
"""Archive the completed same-budget nine-task regression and frozen runtime."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from screenshot_agent.control import atomic_json

ROOT=Path(__file__).resolve().parents[1]


def archive(run,source,output):
    run,source,output=run.resolve(),source.resolve(),output.resolve()
    frozen=json.loads((source/'source_state.json').read_text())
    changed=[p for p,digest in frozen['sha256'].items() if hashlib.sha256((source/p).read_bytes()).hexdigest()!=digest]
    if changed:
        raise ValueError(f'Frozen runtime changed: {changed}')
    report=json.loads((run/'evaluation/report.json').read_text())
    status=json.loads((run/'evaluation/status.json').read_text())
    clean=json.loads((run/'clean_patch_audit.json').read_text())
    if status['status']!='complete' or status['completed']!=27 or len(report['records'])!=27 or not clean['all_verdicts_preserved']:
        raise ValueError('Regression and independent replay must finish first')
    output.mkdir(parents=True,exist_ok=False)
    for src,name in [(run/'evaluation/report.json','report.json'),(run/'evaluation/report.md','report.md'),
                     (run/'launch.json','launch.json'),(run/'final_model_stats.json','model_stats.json'),
                     (run/'clean_patch_audit.json','clean_patch_audit.json'),
                     (source/'source_state.json','source_state.json'),(source.with_suffix('.zip'),'source_snapshot.zip')]:
        shutil.copy2(src,output/name)
    configs=set()
    for folder in sorted((run/'evaluation/runs').iterdir()):
        if not folder.is_dir():
            continue
        dest=output/'trials'/folder.name
        dest.mkdir(parents=True)
        configs.add(hashlib.sha256((folder/'config.json').read_bytes()).hexdigest())
        for name in ['result.json','task.json','config.json','events.jsonl','conversation.json','patch.diff',
                     'patch.clean.diff','baseline_test.log','verification_test.log','agent_error.log']:
            if (folder/name).exists():
                shutil.copy2(folder/name,dest/name)
        check=run/'clean_patch_checks'/folder.name
        for name in ['check.json','verification_test.log']:
            if (check/name).exists():
                shutil.copy2(check/name,dest/('clean_'+name))
    if len(configs)!=1:
        raise ValueError('Paired arms used different configurations')
    old=json.loads((ROOT/'docs/artifacts/v3/real_model_report.json').read_text())
    comparisons=[]
    for reader in ['oracle','ocr','vlm']:
        before=[r for r in old['records'] if r['reader']==reader]
        after=[r for r in report['records'] if r['reader']==reader]
        comparisons.append({'reader':reader,'old_resolved':sum(r['resolved'] for r in before),
            'current_resolved':sum(r['resolved'] for r in after),'attempted':len(after),
            'passed_tasks':[r['task_id'] for r in after if r['resolved']]})
    atomic_json(output/'comparison.json',comparisons)
    hashes={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file()}
    atomic_json(output/'verification.json',{'runtime_unchanged':True,'runtime_source_root':str(source),
        'config_count':len(configs),'runs':27,'clean_passed':clean['clean_passed'],'artifact_sha256':hashes})
    lines=['# 九题同预算回归（v5）','',
        '对既有 9 个 SWE-bench Lite 截图派生任务重新运行 oracle / CPU OCR / VLM，共 27 次；每题三组使用相同模型、配置和测试。',
        'Qwen3-VL-8B-Instruct，代码输出 1024 token，视觉 2048 token，18 次代码调用、50 次工具、3 次截图读取、1200 秒预算；Agent 压缩阈值 4096，服务上下文 16384。',
        '冻结版本包含文件路径检索、缓存补丁导出和 Unicode JSONL 修复。后续 JSON 宽松解析、截图工具开关不在此冻结版本中。','',
        '| 读取方式 | 修复前 v3 | 本轮 | 本轮通过任务 |','|---|---:|---:|---|']
    for row in comparisons:
        lines.append(f"| {row['reader']} | {row['old_resolved']}/9 | {row['current_resolved']}/9 | {', '.join(row['passed_tasks'])} |")
    lines+=['',f"27 份补丁全部独立重放，成功 {clean['clean_passed']} 次，判定与逐题原始成绩一致。",'',
        'OCR 从 0/9 到 1/9，与修复后的检索能力一致；小样本回归不是统计显著性证明，也不能把差异唯一归因到某个单独修复。',
        '三组均仅完成同一个 pytest 任务，其余 8 题仍失败。视觉读取更准确尚未转化为本轮修复率优势；需要扩大修复任务并分析代码模型与上下文预算。',
        '所有测试为 Agent 可见且保护只读；使用独立 Python + Bubblewrap 适配环境及官方解析器，未运行官方 Docker 协议，不是排行榜分数。截图为合成数据，真实用户截图仍为 0。',
        '证据与冻结源码：`docs/artifacts/v5_regression/`。与接下来的代码 2048 token、压缩阈值 8192 新实验分表报告，不合并成绩。']
    (ROOT/'docs/experiment_report_v5_regression.md').write_text('\n'.join(lines)+'\n')
    return comparisons


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,default=Path('outputs/v5_regression_gpu6'))
    p.add_argument('--source',type=Path,default=Path('outputs/v5_regression_source'))
    p.add_argument('--output',type=Path,default=Path('docs/artifacts/v5_regression'))
    a=p.parse_args()
    print(json.dumps(archive(a.run,a.source,a.output),indent=2))
