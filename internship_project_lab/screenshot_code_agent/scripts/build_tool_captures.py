#!/usr/bin/env python3
"""Capture genuine widgets around controlled bugs; retain exact provenance."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from screenshot_agent.config import Task
from screenshot_agent.control import Events, atomic_json
from screenshot_agent.repository import git
from screenshot_agent.sandbox import BubblewrapSandbox

ROOT=Path(__file__).resolve().parents[1]


class CaptureSandbox(BubblewrapSandbox):
    def command(self,argv,mounts=None):
        command=super().command(argv,mounts)
        # Chromium reserves a large virtual address cage. Physical RAM is
        # separately limited by the capture service's cgroup (MemoryMax=4G).
        return [p.replace('--nofile=128:128','--nofile=1024:1024') for p in command
                if not p.startswith('--as=')]


def build(output,limit=20):
    if output.exists():
        raise ValueError('Use a new capture dataset directory')
    output.mkdir(parents=True)
    original=ROOT/'datasets/generated/v1'
    definitions=[json.loads(s) for s in (original/'tasks.jsonl').read_text().splitlines() if s.strip()][:limit]
    plans,manifest,audits=[],[],[]
    for index,row in enumerate(definitions):
        task=Task.from_file(original/row['task_file'])
        instance='capture_'+task.task_id
        folder=output/instance
        folder.mkdir()
        shutil.copytree(task.repo,folder/'repo',ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache'))
        if task.task_id=='synthetic_csv_fields':
            testfile=next((folder/'repo/tests').glob('test*.py'))
            body=testfile.read_text()
            body += "\nclass TestExtraCSV(unittest.TestCase):\n    def test_preserve_empty_fields(self):\n        self.assertEqual(m.csv_fields(',a,,'), ['', 'a', '', ''])\n"
            testfile.write_text(body)
        events=Events(folder/'preflight_events.jsonl')
        sb=BubblewrapSandbox(folder/'repo',events,['tests'],60)
        before=sb.run(task.test_command,log_path=folder/'baseline_test.log')
        sb.cancel()
        gold=folder/'gold_check'
        shutil.copytree(folder/'repo',gold)
        git(gold,'init','-q')
        git(gold,'apply','--binary','-',stdin=Path(task.reference_patch).read_bytes())
        verifier=BubblewrapSandbox(gold,events,['tests'],60)
        after=verifier.run(task.test_command,log_path=folder/'reference_test.log')
        verifier.cancel()
        shutil.rmtree(gold)
        if before['returncode']==0 or after['returncode'] or before['timed_out'] or after['timed_out']:
            raise RuntimeError(f'Capture task failed buggy/gold validation: {instance}')
        kind='code' if index%2==0 else 'error'
        text=(folder/'repo/miniapp.py').read_text() if kind=='code' else '$ '+task.test_command+'\n'+before['output']
        plans.append({'id':instance,'kind':kind,'filename':'miniapp.py' if kind=='code' else 'terminal',
                      'text':text,'font_size':18})
        data=task.model_dump()
        data.update(task_id=instance,repo='repo',provenance='controlled_codemirror_xterm_capture',
                    reference_patch='reference.patch',replay=None,benchmark=None)
        data['images']=[{'image_id':'shot','path':'screenshot.png','reference':'screenshot.txt','layout':'screenshot.layout.json'}]
        shutil.copy2(task.reference_patch,folder/'reference.patch')
        atomic_json(folder/'task.json',data)
        manifest.append({'task_id':instance,'task_file':instance+'/task.json','split':task.split,
                         'group_id':task.group_id,'provenance':data['provenance']})
        audits.append({'task_id':instance,'base_task':task.task_id,'kind':kind,'buggy_exit':before['returncode'],
                       'reference_exit':after['returncode'],'source_repo_sha256':hashlib.sha256((folder/'repo/miniapp.py').read_bytes()).hexdigest()})
    atomic_json(output/'capture_plan.json',plans)
    sandbox=CaptureSandbox(output,Events(output/'capture_events.jsonl'),[],300)
    execution=sandbox.run(['node','/grader/capture.cjs'],mounts={
        '/grader/capture.cjs':ROOT/'scripts/capture_widgets.cjs','/grader/node_modules':ROOT/'.capture-runtime/node_modules',
        '/grader/browsers':ROOT/'.cache/capture-browsers','/grader/font.otf':ROOT/'.cache/NotoSansMonoCJKsc-Regular.otf',
        '/grader/plan.json':output/'capture_plan.json'},log_path=output/'capture.log')
    sandbox.cancel()
    if execution['returncode'] or execution['timed_out']:
        raise RuntimeError('Browser capture failed: '+execution['output'][-5000:])
    (output/'tasks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    report={'captures':len(plans),'tasks':len(manifest),'organic_user_screenshots':0,
        'code_captures':sum(p['kind']=='code' for p in plans),'terminal_captures':sum(p['kind']=='error' for p in plans),
        'source':'Actual CodeMirror/xterm widgets in Chromium; terminal replays actual isolated test output.',
        'scope':'Controlled bugs and controlled tool captures. Not naturally submitted user issues or unseen repositories.',
        'audits':audits,'browser':json.loads((output/'browser.json').read_text())}
    atomic_json(output/'build_report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('datasets/generated/tool_captures_v5'))
    p.add_argument('--limit',type=int,default=20)
    args=p.parse_args()
    print(json.dumps(build(args.output.resolve(),args.limit),ensure_ascii=False,indent=2))
