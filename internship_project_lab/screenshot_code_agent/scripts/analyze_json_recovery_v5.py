#!/usr/bin/env python3
"""Reparse saved invalid VLM payloads, with no inference or changes to V4 scores."""
import ast
import hashlib
import json
from pathlib import Path

from screenshot_agent.codeocr_benchmark import upstream
from screenshot_agent.control import atomic_json
from screenshot_agent.images import parse_vision_response

ROOT=Path(__file__).resolve().parents[1]


def analyze():
    original=ROOT/'outputs/codeocr_full_v4/combined/report.json'
    before=hashlib.sha256(original.read_bytes()).hexdigest()
    records=json.loads(original.read_text())['records']
    _,_,metrics=upstream(ROOT/'vendor/CodeOCR')
    dataset=ROOT/'datasets/generated/codeocr_full_v4'
    rows=[]
    for old in records:
        if old['reader']!='vlm' or old.get('reading_schema_valid',True):
            continue
        recovered=parse_vision_response(old['reading']['raw_text'])
        reference=(dataset/old['reference']).read_text()
        try:
            ast.parse(recovered.raw_text)
            parses=True
        except (SyntaxError,ValueError):
            parses=False
        rows.append({'sample_id':old['sample_id'],'source_id':old['source_id'],'format_status':recovered.format_status,
            'before_cer':old['metrics']['cer'],'after_cer':metrics(reference,recovered.raw_text)['cer'],
            'syntax_valid':parses,'recovered_reading':recovered.model_dump()})
    if hashlib.sha256(original.read_bytes()).hexdigest()!=before:
        raise ValueError('Original report changed')
    output=ROOT/'docs/artifacts/v5_json_recovery'
    output.mkdir(parents=True,exist_ok=False)
    payload={'saved_invalid_readings':len(rows),'recovered':sum(r['format_status']=='relaxed_json' for r in rows),
        'model_requests':0,'original_scores_modified':False,'original_report_sha256':before,'records':rows,
        'scope':'Post-hoc parser diagnostic on the same five saved malformed payloads. Not a new model run or replacement score.'}
    atomic_json(output/'analysis.json',payload)
    lines=['# 视觉响应 JSON 恢复检查（v5）','',
        '离线重新解析 CodeOCR 全量实验中已保存的 5 条不合规视觉输出；没有调用模型、补写代码或替换 v4 成绩。',
        '仅允许 JSON 字符串中的原始控制字符；保留代码转义，不猜测缺失引号或截断内容。无法恢复时保留原始响应并显式标记。','',
        '| 样本 | 解析状态 | 原 CER | 重解析 CER | Python 语法通过 |','|---|---|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['sample_id']} | {r['format_status']} | {r['before_cer']:.2%} | {r['after_cer']:.2%} | {r['syntax_valid']} |")
    lines+=['',f"可恢复 {payload['recovered']}/{len(rows)} 条。这里只证明解析器能挽回部分格式损失，不能证明原本未识别清楚的图片能被读对。",
        '新增集成测试还覆盖歧义/截断输出保留、转义不被重复解码、坏图片失败落盘、请求超时记账，以及首次读取后隐藏截图工具。',
        '这些属于系统健壮性改进，不属于模型训练或 Agent 推理策略创新。']
    document=ROOT/'docs/json_recovery_v5.md'
    document.write_text('\n'.join(lines)+'\n')
    atomic_json(output/'verification.json',{'sha256':{'analysis.json':hashlib.sha256((output/'analysis.json').read_bytes()).hexdigest()},'original_report_unchanged':True})
    return {k:v for k,v in payload.items() if k!='records'}


if __name__=='__main__':
    print(json.dumps(analyze(),indent=2))
