#!/usr/bin/env python3
"""Offline check of known development contracts; never a model success metric.

Run visible assertions against the original and the reference source patch to
catch incorrectly authored assertions. Reference source is confined to separate
CPU sandboxes and is never included in model inputs or repair workspaces.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import run_harness_compare as harness

from screenshot_agent.config import Task
from screenshot_agent.control import atomic_json
from screenshot_agent.workflow import RepairWorkflow
from screenshot_agent.workflow_cases import CASES
from screenshot_agent.workflow_config import load_workflow_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', action='append', help='Optional known case(s) to recheck')
    args = parser.parse_args()
    output = harness.storage(args.output)
    cfg = load_workflow_config(ROOT / 'configs/workflow-v4.json')
    manifest = json.loads((args.source / 'prepared.json').read_text())['tasks'][:10]
    rows = []
    for item in manifest:
        if item['task_id'] not in CASES or (args.task and item['task_id'] not in args.task):
            continue
        task = Task.from_file(Path(item['task_file']))
        folder = output / task.task_id
        folder.mkdir(exist_ok=False)
        started = time.monotonic()
        flow = RepairWorkflow(task, cfg, folder, counter=lambda m, t: 100, model_inference=False)
        row = {'task_id': task.task_id}
        try:
            flow.setup()
            row['original'] = flow.feedback.known_issue()
            source_patch = json.loads(Path(task.benchmark).read_text())['record']['patch']
            patch = folder / 'reference_source.patch'
            patch.write_text(source_patch)
            row['reference'] = flow.feedback.known_issue(patch)
            row['ready'] = (row['original']['status'] == 'baseline_reproduced' and
                            row['reference']['status'] == 'passed')
        except Exception as error:
            row.update(ready=False, error=repr(error))
        finally:
            if flow.sandbox:
                flow.sandbox.cancel()
            if flow.public_display:
                flow.public_display.close()
        row['seconds'] = round(time.monotonic() - started, 2)
        rows.append(row)
        atomic_json(output / 'report.json', {
            'scope': 'Known development assertion validation, NOT agent evaluation',
            'reference_source_used_offline': True, 'model_calls': 0, 'hidden_tests_executed': False,
            'tasks': rows, 'ready': bool(rows) and all(r['ready'] for r in rows)})
        print(json.dumps(row), flush=True)
    if not rows or not all(r['ready'] for r in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
