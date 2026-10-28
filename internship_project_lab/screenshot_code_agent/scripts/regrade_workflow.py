#!/usr/bin/env python3
"""Replay saved workflow patches with the corrected grader; never call a model."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from screenshot_agent.config import Task
from screenshot_agent.control import Events, atomic_json
from screenshot_agent.harness import PROTOCOL
from screenshot_agent.hidden_eval import grade_hidden, validate_prepared
from screenshot_agent.repository import replay_patch
from screenshot_agent.scale_data import immutable_json, sha256, tree_hash

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path('/data/zjr/screenshot_code_agent/repair_workflow_20261027')
DEFAULT_OUTPUT = Path('/data/zjr/screenshot_code_agent/workflow_regrade_20261028')
TERMINAL = {'verified', 'unresolved', 'error'}
METHOD = 'saved_patch_hidden_regrade_v1'


def read(path):
    return json.loads(path.read_text())


def source_finished(source, task_id):
    folder = source / 'runs' / task_id
    try:
        with (folder / 'workflow.lock').open('r') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            return (read(folder / 'result.json')['status'] in TERMINAL
                    and read(folder / 'workflow.json')['status'] in TERMINAL)
    except (FileNotFoundError, BlockingIOError):
        return False


def completed_selection(source, output, tasks):
    """Freeze the available cohort once; later completions cannot change its denominator."""
    path = output / 'plan.json'
    if path.exists():
        plan = read(path)
        if plan.get('selection_mode') != 'completed_only':
            raise ValueError('Output belongs to another selection mode')
        by_id = {row['task_id']: row for row in tasks}
        return [by_id[task_id] for task_id in plan['task_ids']]
    selected = [row for row in tasks if source_finished(source, row['task_id'])]
    if not selected:
        raise ValueError('No completed, unlocked source attempts are ready')
    return selected


def regrade_one(source, output, row):
    """Original patches/results/baselines stay unchanged, including on resume."""
    task_id = row['task_id']
    if not task_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in task_id):
        raise ValueError('Invalid task id')
    if task_id in {'.', '..'}:
        raise ValueError('Invalid task id')
    original = source / 'runs' / task_id
    # An existing workflow holds an exclusive lock until result export finishes.
    with (original / 'workflow.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        prior = read(original / 'result.json')
        state = read(original / 'workflow.json')
        if (prior['task_id'] != task_id or prior['protocol'] != PROTOCOL
                or prior['status'] not in TERMINAL or state['status'] not in TERMINAL):
            raise ValueError('Source attempt has not finished under the hidden-test protocol')
        paths = {'result': original / 'result.json', 'patch': original / 'patch.diff',
                 'workflow': original / 'workflow.json', 'config': original / 'config.json',
                 'task': Path(row['task_file'])}
        identity = {key: sha256(path) for key, path in paths.items()}
        destination = output / 'tasks' / task_id
        previous = destination / 'result.json'
        if previous.exists():
            result = read(previous)
            if result['source_sha256'] != identity:
                raise ValueError('Source attempt changed since regrading; use a new output directory')
            return result
        validate_prepared(paths['task'].parent, verify_repo=False)
        if tree_hash(original / 'baseline') != state['baseline_hash']:
            raise ValueError('Saved baseline changed; refuse to regrade a different input')
        task = Task.from_file(paths['task'])
        attempt = destination / ('attempt_' + str(time.time_ns()))
        attempt.mkdir(parents=True)
        patch = attempt / 'patch.diff'
        shutil.copyfile(paths['patch'], patch)
        if sha256(patch) != identity['patch']:
            raise ValueError('Patch copy failed verification')
        result = {'method': METHOD, 'protocol': PROTOCOL, 'task_id': task_id,
                  'source_sha256': identity, 'original_result': str(paths['result']),
                  'original_resolved': prior['resolved'], 'original_status': prior['status'],
                  'resolved': False, 'runtime_failure': False, 'status': 'unresolved',
                  'model_requests': 0, 'source_modified': False, 'attempt': str(attempt)}
        started = time.time()
        tree = {'max_files': task.runtime.max_repo_files, 'max_bytes': task.runtime.max_repo_bytes,
                'allowed_symlinks': task.runtime.allowed_symlinks}
        result['patch_replay'] = replay_patch(original / 'baseline', attempt / 'verification', patch, **tree)
        if result['patch_replay']['applied']:
            timeout = read(paths['config'])['limits']['command_timeout_seconds']
            result['hidden'] = grade_hidden(task, attempt / 'verification', Events(attempt / 'events.jsonl'),
                                          attempt / 'test.log', timeout=timeout)
            result['runtime_failure'] = bool(result['hidden'].get('runtime_failure'))
            result['resolved'] = bool(result['hidden']['resolved'])
            result['status'] = ('error' if result['runtime_failure'] else
                                'verified' if result['resolved'] else 'unresolved')
        if any(sha256(path) != identity[key] for key, path in paths.items()):
            raise ValueError('Source artifacts changed during regrading')
        result.update(finished_at=time.time(), elapsed_seconds=time.time() - started)
        atomic_json(previous, result)
        return result


def summarize(output, tasks):
    rows = [read(output / 'tasks' / row['task_id'] / 'result.json') for row in tasks
            if (output / 'tasks' / row['task_id'] / 'result.json').exists()]
    value = {'method': METHOD, 'total': len(tasks), 'completed': len(rows),
             'complete': len(rows) == len(tasks), 'resolved': sum(r['resolved'] for r in rows),
             'runtime_failures': sum(r['runtime_failure'] for r in rows), 'model_requests': 0,
             'source_modified': False, 'tasks': rows}
    value['valid_report'] = value['complete'] and not value['runtime_failures']
    plan = read(output / 'plan.json') if (output / 'plan.json').exists() else {}
    value['selection_mode'] = plan.get('selection_mode', 'all')
    value['source_total'] = plan.get('source_total', len(tasks))
    atomic_json(output / 'report.json', value)
    text = (f"# 已有补丁独立重评\n\n已重评：{value['completed']}/{value['total']}；"
            f"隐藏测试通过：{value['resolved']}；评分环境异常：{value['runtime_failures']}。\n\n"
            '仅重放原始补丁，不调用模型、不修改旧结果，不将参考测试反馈给 Agent。'
            '不完整或含环境异常的报告不能视为最终成功率。\n')
    if value['selection_mode'] == 'completed_only':
        text += f"\n这是启动时已完成的{len(tasks)}题快照，不是原始{value['source_total']}题的最终成绩。\n"
    (output / 'report.md').write_text(text)
    return value


def run(args):
    source, output = args.source.resolve(), args.output.resolve()
    if not output.is_relative_to('/data/zjr'):
        raise ValueError('Regrading output must live under /data/zjr')
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Regrading needs a separate output directory outside the source run')
    manifest = read(source / 'prepared.json')
    if manifest['protocol'] != PROTOCOL:
        raise ValueError('Regrading requires hidden-reference-test inputs')
    tasks = manifest['tasks']
    source_total = len(tasks)
    mode = 'completed_only' if args.completed_only else 'task' if args.task_id else 'all'
    if args.task_id:
        tasks = [row for row in tasks if row['task_id'] == args.task_id]
        if not tasks:
            raise ValueError('Task id is not in the original manifest')
    output.mkdir(parents=True, exist_ok=True)
    code = [Path(__file__), ROOT / 'scripts/official_grade.py',
            *[ROOT / 'src/screenshot_agent' / (name + '.py') for name in
              ('hidden_eval', 'runtime', 'sandbox', 'benchmark', 'repository')]]
    with (output / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.completed_only:
            tasks = completed_selection(source, output, tasks)
        plan = {'method': METHOD, 'source': str(source), 'prepared_sha256': sha256(source / 'prepared.json'),
                'selection_mode': mode, 'source_total': source_total,
                'task_ids': [row['task_id'] for row in tasks],
                'implementation': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in code}}
        immutable_json(output / 'plan.json', plan)

        def update(stage, **fields):
            value = {'stage': stage, 'pid': os.getpid(), 'updated_at': time.time(),
                     'total': len(tasks), 'source': str(source), 'model_requests': 0, **fields}
            atomic_json(output / 'status.json', value)
            print(json.dumps(value), flush=True)

        try:
            while True:
                completed = 0
                for row in tasks:
                    if source_finished(source, row['task_id']):
                        completed += 1
                if completed == len(tasks):
                    break
                if not args.wait_for_source:
                    raise ValueError('Source attempts are unfinished; use --wait-for-source or --task-id')
                update('waiting_for_source', source_completed=completed)
                time.sleep(args.poll_seconds)
            for index, row in enumerate(tasks):
                update('regrading', completed=index, task_id=row['task_id'])
                result = regrade_one(source, output, row)
                report = summarize(output, tasks)
                if result['runtime_failure']:
                    raise RuntimeError('Grader infrastructure failed; stopping before further tasks. See ' + result['attempt'])
            update('completed', completed=len(tasks), resolved=report['resolved'], report=str(output / 'report.json'))
        except BaseException as error:
            update('failed', error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'status'])
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--task-id', help='Regrade one completed task, e.g. a CPU-only smoke check')
    selection.add_argument('--completed-only', action='store_true', help='Freeze and grade just the already finished tasks')
    parser.add_argument('--wait-for-source', action='store_true')
    parser.add_argument('--poll-seconds', type=int, choices=range(5, 301), default=60)
    args = parser.parse_args()
    if args.completed_only and args.wait_for_source:
        parser.error('--completed-only cannot be combined with --wait-for-source')
    if args.action == 'status':
        path = args.output / 'status.json'
        print(json.dumps(read(path) if path.exists() else {'stage': 'not_started'}, ensure_ascii=False, indent=2))
    else:
        run(args)


if __name__ == '__main__':
    main()
