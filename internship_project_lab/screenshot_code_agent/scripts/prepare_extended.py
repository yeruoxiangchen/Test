#!/usr/bin/env python3
"""Freeze extended tasks, install isolated profiles, and audit buggy/gold versions."""

import argparse
import json
import shutil
import time
from pathlib import Path

from screenshot_agent.extended_data import build_extended, select_extended, setup_profiles

parser = argparse.ArgumentParser()
parser.add_argument('command',choices=['select','profiles','build'])
parser.add_argument('--output',type=Path,default=Path('datasets/generated/swebench_extended_v5'))
parser.add_argument('--profiles',type=Path,default=Path('.runtime/extended_v5'))
parser.add_argument('--timeout',type=int,default=180)
parser.add_argument('--retry-failed',action='store_true',help='Archive completed infrastructure failures and their private prefixes before retrying')
args = parser.parse_args()
if args.command=='select':
    excluded = set(json.loads(Path('datasets/generated/swebench_lite_v3/selection.json').read_text())['selected_ids'])
    result = select_extended(Path('datasets/swebench_lite_test.parquet'),args.output,excluded)
    print(json.dumps({'selected':len(result),'ids':[r['instance_id'] for r in result]},indent=2))
elif args.command=='profiles':
    records = json.loads((args.output/'selected_records.json').read_text())
    setup_profiles(records,Path('.runtime/python39/python').resolve(),args.profiles.resolve())
else:
    if args.retry_failed:
        attempt=args.output/'preflight_attempts'/str(time.time_ns())
        report=args.output/'build_report.json'
        rows=json.loads(report.read_text())['audits'] if report.exists() else []
        failed=[r for r in rows if r['status']!='validated']
        if failed:
            attempt.mkdir(parents=True)
            shutil.copy2(report,attempt/'build_report.json')
            for row in failed:
                name=row['instance_id']
                shutil.move(str(args.output/name),str(attempt/name))
                prefix=args.profiles/'tasks'/name
                if prefix.exists():
                    (attempt/'prefixes').mkdir(exist_ok=True)
                    shutil.move(str(prefix),str(attempt/'prefixes'/name))
            (attempt/'recovery.json').write_text(json.dumps({'reason':'Infrastructure-only retry, before any model outcomes',
                'instances':[r['instance_id'] for r in failed]},indent=2))
            print(f'Archived {len(failed)} failed preflights: {attempt}',flush=True)
    result = build_extended(args.output,Path('vendor/swe-bench-tasks'),Path('vendor/SWE-bench'),args.profiles,
                            Path('.cache/source_archives'),args.timeout)
    print(json.dumps({k:v for k,v in result.items() if k!='audits'},indent=2))
