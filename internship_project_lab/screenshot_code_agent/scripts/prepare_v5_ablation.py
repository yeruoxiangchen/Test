#!/usr/bin/env python3
"""Freeze image degradation and paired screenshot-tool ablation tasks."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from screenshot_agent.config import Task
from screenshot_agent.control import atomic_json


def build(source,output):
    source,output=source.resolve(),output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(s) for s in (source/'tasks.jsonl').read_text().splitlines() if s.strip()]
    clean,degraded,audits=[],[],[]
    for row in rows:
        task=Task.from_file(source/row['task_file'])
        folder=output/task.task_id
        folder.mkdir()
        image=Path(task.images[0].path)
        with Image.open(image) as src:
            src.convert('RGB').resize((src.width//2,src.height//2),Image.Resampling.BILINEAR).resize(
                src.size,Image.Resampling.BILINEAR).save(folder/'degraded.png')
        task.images[0].path=str(folder/'degraded.png')
        atomic_json(folder/'task.json',task.model_dump())
        clean.append({**row,'task_file':str(source/row['task_file'])})
        degraded.append({**row,'task_file':str(folder/'task.json')})
        audits.append({'task_id':task.task_id,'group_id':task.group_id,'split':task.split,
            'clean_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
            'degraded_sha256':hashlib.sha256((folder/'degraded.png').read_bytes()).hexdigest()})
    for name,manifest in [('clean.jsonl',clean),('degraded.jsonl',degraded)]:
        (output/name).write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    report={'tasks':len(rows),'independent_new_tasks':0,'degradation':'Downsample each dimension by 2, bilinear; restore original dimensions with bilinear.',
        'policy_arms':['initial_only','crop_reread_available'],'runs':len(rows)*2*2,
        'pairing':'Same repository, task ID, split, code/vision model and numerical budgets; only image variant/tool availability differ.',
        'predeclared_before_model_calls':True,'organic_user_screenshots':0,'audits':audits}
    atomic_json(output/'build_report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=Path('datasets/generated/tool_captures_v5'))
    p.add_argument('--output',type=Path,default=Path('datasets/generated/tool_ablation_v5'))
    a=p.parse_args()
    print(json.dumps(build(a.source,a.output),indent=2))
