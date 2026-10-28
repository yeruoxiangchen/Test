#!/usr/bin/env python3
"""Postprocess the finished full-data run without changing readings or scores."""

import argparse
import collections
import hashlib
import json
import re
import shutil
import time
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def analyze(run):
    root = Path(__file__).resolve().parents[1]
    status = load(run/'status.json')
    if status['status']!='complete' or status.get('completed')!=400:
        raise ValueError('A complete 400-reading run is required')
    report = load(run/'combined/report.json')
    records = report['records']
    if len(records)!=400:
        raise ValueError('Coverage mismatch')
    dataset = Path(load(run/'source_state.json')['dataset'])
    groups = collections.defaultdict(list)
    for row in records:
        text = (dataset/row['reference']).read_text()
        source = '含中文' if re.search(r'[\u3400-\u9fff]',text) else '不含中文'
        length = '50–60 行' if len(text.splitlines())<=60 else '61–80 行' if len(text.splitlines())<=80 else '81–115 行'
        for kind,label in [('overall','全部'),('language',source),('length',length)]:
            groups[(kind,label,row['reader'])].append(row)
    summaries = []
    for (kind,label,reader),rows in sorted(groups.items()):
        summaries.append({'kind':kind,'label':label,'reader':reader,'readings':len(rows),
            'sources':len({r['source_id'] for r in rows}),
            'cer_including_errors_mean':sum(r.get('metrics',{}).get('cer',1) for r in rows)/len(rows),
            'request_errors':sum(r['status']=='error' for r in rows),
            'schema_invalid':sum(not r.get('reading_schema_valid',True) for r in rows if r['status']=='complete'),
            'output_cap_cues':sum(r.get('output_cap_reached',False) for r in rows),
            'empty_predictions':sum(not r.get('reading',{}).get('raw_text','').strip() for r in rows),
            'syntax_valid':sum(r.get('hypothesis_parses',False) for r in rows)})
    worst = {}
    for reader in ['ocr','vlm']:
        rows = sorted((r for r in records if r['reader']==reader),key=lambda r:r.get('metrics',{}).get('cer',1),reverse=True)[:10]
        worst[reader] = [{k:r.get(k) for k in ['sample_id','source_id','reader','ratio','status','metrics',
                                             'reading_schema_valid','output_cap_reached','error']} for r in rows]
    paired = {}
    for ratio in [1,2]:
        by_source = collections.defaultdict(dict)
        for r in records:
            if r['ratio']==ratio:
                by_source[r['source_id']][r['reader']] = r.get('metrics',{}).get('cer',1)
        paired[ratio] = {'sources':len(by_source),'vlm_lower_cer':sum(r['vlm']<r['ocr'] for r in by_source.values()),
                         'ocr_lower_cer':sum(r['ocr']<r['vlm'] for r in by_source.values()),
                         'ties':sum(r['ocr']==r['vlm'] for r in by_source.values())}
    output = root/'docs/artifacts/codeocr_full_v4_analysis'
    output.mkdir(parents=True,exist_ok=True)
    payload = {'records':summaries,'worst_cases':worst,'paired':paired,'source_model_results_changed':False,
               'vision_requests':sum(r.get('usage',{}).get('vision_model_calls',0) for r in records),
               'prompt_tokens':sum(r.get('usage',{}).get('prompt_tokens',0) for r in records),
               'completion_tokens':sum(r.get('usage',{}).get('completion_tokens',0) for r in records),
               'local_api_price_usd':sum(r.get('usage',{}).get('cost_usd',0) for r in records)}
    (output/'diagnostics.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    lines = ['# CodeOCR 全量读取诊断','', '分析基于已完成的 400 条原始读取，不重试失败样本，不修改预测或主报告。',
             '表内两档输入合并平均；同一代码的两张变体属于同一个源任务。请求异常按 CER=1 计入。','',
             '| 分组 | 读取方式 | 源代码数 | 读取数 | CER | 请求异常 | JSON 异常 | 达到输出上限线索 | 语法可解析 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['label']} | {s['reader']} | {s['sources']} | {s['readings']} | {s['cer_including_errors_mean']:.2%} | "
                     f"{s['request_errors']} | {s['schema_invalid']} | {s['output_cap_cues']} | {s['syntax_valid']} |")
    lines += ['', '达到 2048 输出 token 仅表示可能截断或重复生成，不直接认定根因。当前本地服务的 finish_reason 不能准确区分长度截断。',
              'JSON 异常输出保留原文并计入字符误差；语法可解析不代表转写正确。按语言/长度分组是描述性分析，不能单独证明这些特征造成错误。','',
              '| 档位 | 配对源任务 | VLM CER 更低 | OCR CER 更低 | 相同 |','|---|---:|---:|---:|---:|']
    for ratio,s in paired.items():
        lines.append(f"| {ratio} | {s['sources']} | {s['vlm_lower_cer']} | {s['ocr_lower_cer']} | {s['ties']} |")
    for reader,rows in worst.items():
        lines += ['',f'## {reader} 字符误差最高的十次读取','']
        for row in rows:
            cer=(row.get('metrics') or {}).get('cer',1)
            lines.append(f"- `{row['source_id']}` / {row['ratio']}×：CER={cer:.2%}，JSON 合规={row['reading_schema_valid']}，输出上限线索={row['output_cap_reached']}。")
    lines += ['', '逐条预测和模型响应见主实验归档；这些诊断不构成修复成功率或真实用户截图效果。',
              f"本轮记录视觉请求 {payload['vision_requests']} 次，输入 token {payload['prompt_tokens']}，输出 token {payload['completion_tokens']}。",
              '本地 API 标价为零，硬件与运行时间成本另计。']
    document = root/'docs/codeocr_full_v4_analysis.md'
    document.write_text('\n'.join(lines)+'\n')
    shutil.copy2(document,output/'analysis.md')
    (output/'verification.json').write_text(json.dumps({'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.iterdir() if p.is_file() and p.name!='verification.json'}},indent=2)+'\n')
    return {'complete':True,'report':str(document)}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--wait-seconds',type=int,default=0)
    args=parser.parse_args()
    started=time.monotonic()
    while True:
        state=load(args.run/'status.json') if (args.run/'status.json').exists() else {}
        if state.get('status')=='complete':
            break
        if state.get('status') in ['failed','cancelled'] or time.monotonic()-started>=args.wait_seconds:
            raise SystemExit('Full-data experiment unfinished; no final analysis generated')
        time.sleep(30)
    result=analyze(args.run.resolve())
    (args.run/'analysis_status.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
