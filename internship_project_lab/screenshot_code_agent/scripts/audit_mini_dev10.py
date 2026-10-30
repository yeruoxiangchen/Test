#!/usr/bin/env python3
"""Read completed mini/v4 runs; do not infer, execute candidate code, or rescore."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def audit_task(folder, previous):
    result = read(folder / 'result.json')
    hidden = result['validation'].get('hidden', {})
    official = hidden.get('official', {})
    messages = read(folder / 'conversation.json')
    calls, observations = {}, []
    for index, message in enumerate(messages):
        for call in message.get('tool_calls') or []:
            calls[call['id']] = {'message_index': index,
                'thought': (message.get('content') or '')[-1800:],
                'command': json.loads(call['function']['arguments']).get('command', '')}
        if message.get('role') != 'tool':
            continue
        text = message.get('content', '')
        match = re.search(r'<returncode>(.*?)</returncode>', text)
        item = {**calls.get(message.get('tool_call_id'), {}), 'observation_index': index,
                'returncode': int(match[1]) if match else None, 'output_tail': text[-1800:]}
        item['tmp_missing'] = bool(re.search(r'/tmp/[^\s\'\"]+', text) and
            ('No such file or directory' in text or "can't open file" in text))
        item['git_restore_blocked'] = 'index.lock' in text and 'Read-only file system' in text
        item['development_script_command'] = bool(re.search(
            r'python(?:3)?\s+\.sca_scratch/development_acceptance\.py', item.get('command', '')))
        observations.append(item)
    event_counts, finishes = Counter(), Counter()
    pending, model_seconds = None, 0.0
    last_prompt_has_developer_case = False
    for line in (folder / 'events.jsonl').open():
        event = json.loads(line)
        kind = event.get('event')
        event_counts[kind] += 1
        if kind == 'model_request':
            pending = event['time']
            last_prompt_has_developer_case = any('Frozen read-only script:' in str(m.get('content', ''))
                                                for m in event.get('messages', []))
        if kind in {'model_response', 'model_error'} and pending is not None:
            model_seconds += event['time'] - pending
            pending = None
        if kind == 'model_response':
            for choice in event['response'].get('choices', []):
                finishes[choice.get('finish_reason')] += 1
    return {
        'task_id': result['task_id'], 'resolved': result['resolved'], 'v4_resolved': previous['resolved'],
        'exit_reason': result['exit_reason'], 'framework_exit': result.get('framework_exit'),
        'runtime_failure': bool(result.get('runtime_failure')), 'usage': result['usage'],
        'seconds': result['finished_at'] - result['started_at'],
        'changed_files': result['changed_files'], 'patch_bytes': result['patch_bytes'],
        'backup_or_temporary_files': [name for name in result['changed_files'] if
            re.search(r'\.(?:backup|bak|temp|fixed|fixed_backup)$', name) or
            Path(name).name in {'test_fix.py', 'temp_function.txt', 'temp_physical_func.txt'}],
        'grader_started': hidden.get('grader_started'), 'parsed': official.get('parsed'),
        'final_test_timed_out': hidden.get('test', {}).get('timed_out'),
        'final_test_returncode': hidden.get('test', {}).get('returncode'),
        'test_counts': {k:{s:len(v) for s,v in group.items()} for k,group in official.get('tests_status', {}).items()},
        'failed_tests': {k:group.get('failure', []) for k,group in official.get('tests_status', {}).items()},
        'final_output_tail': hidden.get('test', {}).get('output', '')[-1800:],
        'tool_nonzero_returns': sum(row['returncode'] != 0 for row in observations),
        'tmp_missing_observations': [row for row in observations if row['tmp_missing']],
        'blocked_restore_observations': [row for row in observations if row['git_restore_blocked']],
        'development_script_observations': [row for row in observations if row['development_script_command']],
        'last_tools': observations[-6:], 'last_assistant': next((m for m in reversed(messages)
            if m.get('role') == 'assistant'), {}),
        'last_prompt_retains_developer_case': last_prompt_has_developer_case,
        'events': dict(event_counts), 'finish_reasons': dict(finishes),
        'model_wait_seconds': model_seconds, 'result_sha256': digest(folder / 'result.json'),
        'patch_sha256': digest(folder / 'patch.diff'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/data/zjr/screenshot_code_agent/mini_dev10_20261029'))
    parser.add_argument('--baseline', type=Path, default=Path('/data/zjr/screenshot_code_agent/v4_dev10_20261029'))
    parser.add_argument('--output', type=Path, default=Path('/data/zjr/screenshot_code_agent/mini_dev10_analysis_20261029'))
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to('/data/zjr') or any(output.is_relative_to(p.resolve())
            for p in (args.source, args.baseline)):
        raise ValueError('Analysis must use a separate /data/zjr directory')
    source_tasks, prior_tasks = [read(p / 'prepared.json')['tasks'] for p in (args.source, args.baseline)]
    assert source_tasks == prior_tasks and len(source_tasks) == 10
    original_report = read(args.source / 'report.json')['arms']['mini']
    prior_report = read(args.baseline / 'report.json')
    plan, old_plan = read(args.source / 'comparison_plan.json'), read(args.baseline / 'workflow_plan.json')
    assert plan['model'] == old_plan['model']
    for key in ('limits', 'service_context_tokens', 'max_prompt_tokens', 'total_prompt_tokens', 'total_completion_tokens'):
        assert plan['config'][key] == old_plan['config'][key], key
    shared = ['src/screenshot_agent/workflow_cases.py', 'src/screenshot_agent/hidden_eval.py',
              'src/screenshot_agent/benchmark.py', 'src/screenshot_agent/runtime.py']
    parity = {name:digest(args.source/'runtime'/name) == digest(args.baseline/'runtime'/name) for name in shared}
    assert all(parity.values())
    rows = []
    for task in source_tasks:
        prior = read(args.baseline / 'runs' / task['task_id'] / 'result.json')
        row = audit_task(args.source / 'mini/runs' / task['task_id'], prior)
        rows.append(row)
        write(output / 'tasks' / (task['task_id'] + '.json'), row)
    assert sum(row['resolved'] for row in rows) == original_report['resolved']
    assert sum(row['usage']['code_model_calls'] for row in rows) == original_report['model_calls']
    clocks = {}
    for key, path in [('mini', args.source), ('v4', args.baseline)]:
        status = read(path / 'status.json')
        assert status['stage'] == 'completed' and status['pending'] == status['running'] == 0
        clocks[key] = {k:datetime.fromtimestamp(status[k], timezone.utc).isoformat()
                       for k in ('started_at', 'finished_at')}
        clocks[key]['wall_seconds'] = status['finished_at'] - status['started_at']
    paired = {key:[] for key in ('both_success','mini_only','v4_only','both_failed')}
    for row in rows:
        key = ('both_success' if row['resolved'] and row['v4_resolved'] else
               'mini_only' if row['resolved'] else 'v4_only' if row['v4_resolved'] else 'both_failed')
        paired[key].append(row['task_id'])
    totals = {
        'model_wait_seconds': sum(row['model_wait_seconds'] for row in rows),
        'model_errors': sum(row['events'].get('model_error', 0) for row in rows),
        'parsed_grades': sum(bool(row['parsed']) for row in rows),
        'started_graders': sum(bool(row['grader_started']) for row in rows),
        'tmp_missing_tasks': [row['task_id'] for row in rows if row['tmp_missing_observations']],
        'tmp_missing_observations': sum(len(row['tmp_missing_observations']) for row in rows),
        'blocked_restore_tasks': [row['task_id'] for row in rows if row['blocked_restore_observations']],
        'tasks_with_backup_or_temporary_files': sum(bool(row['backup_or_temporary_files']) for row in rows),
        'budget_exceeded_tasks': [row['task_id'] for row in rows if row['exit_reason'] == 'budget_exceeded'],
    }
    summary = {'audited_at':datetime.now(timezone.utc).isoformat(), 'source':str(args.source),
        'baseline':str(args.baseline), 'mini':original_report, 'v4':{k:v for k,v in prior_report.items() if k!='tasks'},
        'clocks':clocks, 'paired':paired, 'totals':totals, 'shared_source_parity':parity,
        'same_tasks_model_budgets':True, 'new_inference':False, 'candidate_tests_executed':False,
        'note':'Known development tasks; adapted mini environment. Small sample, no causal attribution of all failures to adapter.'}
    write(output/'analysis.json', summary)
    print(json.dumps({'output':str(output), 'mini_resolved':original_report['resolved'], 'paired':paired,
                      'clocks':clocks, 'totals':totals}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
