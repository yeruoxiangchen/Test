#!/usr/bin/env python3
"""Run real public baselines for the fixed smoke inputs, without a model or GPU."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import run_harness_compare as harness

from screenshot_agent.config import Task
from screenshot_agent.control import atomic_json
from screenshot_agent.scale_data import sha256
from screenshot_agent.workflow import RepairWorkflow, implementation
from screenshot_agent.workflow_config import load_workflow_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/workflow-v4.json')
    args = parser.parse_args()
    args.output = harness.storage(args.output)
    selected = json.loads((args.source / 'prepared.json').read_text())['tasks'][:10]
    if len(selected) != 10:
        raise ValueError('Expected ten prepared inputs')
    cfg = load_workflow_config(args.config)
    checked_source = implementation()
    rows = []
    for index, item in enumerate(selected, 1):
        task = Task.from_file(Path(item['task_file']))
        folder = args.output / task.task_id
        folder.mkdir(exist_ok=False)
        start = time.monotonic()
        flow = RepairWorkflow(task, cfg, folder, counter=lambda m, t: 100, model_inference=False)
        row = {'task_id': task.task_id, 'task_sha256': sha256(Path(item['task_file']))}
        try:
            flow.setup()
            baseline = flow.feedback.baseline()
            cases = baseline.get('cases') or {}
            row.update(command=baseline['command'], returncode=baseline['returncode'],
                       timed_out=baseline['timed_out'], cases=len(cases),
                       passing=sum(v == 'passed' for v in cases.values()),
                       log=baseline.get('log'), selected=flow.state.get('public_test_selection'))
            row['ready'] = bool(row['passing'] and not row['timed_out'] and row['returncode'] in (0, 1))
            acceptance = flow.feedback.known_issue()
            row['development_acceptance'] = acceptance
            row['ready'] = row['ready'] and acceptance['status'] in {'not_configured', 'baseline_reproduced'}
        except Exception as error:
            row.update(ready=False, error=repr(error))
        finally:
            if flow.sandbox:
                flow.sandbox.cancel()
            if flow.public_display:
                flow.public_display.close()
        row['seconds'] = round(time.monotonic() - start, 2)
        rows.append(row)
        if implementation() != checked_source:
            raise RuntimeError('Implementation changed during public preflight; rerun in a new directory')
        atomic_json(args.output / 'report.json', {'source': str(args.source), 'implementation': checked_source,
                    'config': cfg.model_dump(),
                    'completed': len(rows), 'total': 10, 'ready': len(rows) == 10 and all(r['ready'] for r in rows),
                    'model_calls': 0, 'hidden_tests_executed': False, 'tasks': rows})
        print(json.dumps({'completed': index, **row}), flush=True)
    if not all(r['ready'] for r in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
