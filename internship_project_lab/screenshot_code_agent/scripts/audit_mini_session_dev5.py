#!/usr/bin/env python3
"""Audit the frozen five-task run without inference, candidate execution or regrading."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from audit_mini_dev10 import audit_task, read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(
        '/data/zjr/screenshot_code_agent/mini_session_dev5_20261029'))
    parser.add_argument('--old-mini', type=Path, default=Path(
        '/data/zjr/screenshot_code_agent/mini_dev10_20261029'))
    parser.add_argument('--v4', type=Path, default=Path(
        '/data/zjr/screenshot_code_agent/v4_dev10_20261029'))
    parser.add_argument('--output', type=Path, default=Path(
        '/data/zjr/screenshot_code_agent/mini_session_dev5_analysis_20261029'))
    args = parser.parse_args()
    output = args.output.resolve()
    inputs = (args.source, args.old_mini, args.v4)
    if not output.is_relative_to('/data/zjr') or any(
            output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output)
            for p in inputs):
        raise ValueError('Use a separate /data/zjr analysis directory')
    source_report = read(args.source / 'report.json')
    status = read(args.source / 'status.json')
    assert source_report['complete'] and status['stage'] == 'completed'
    assert status['pending'] == status['running'] == 0
    tasks = read(args.source / 'prepared.json')['tasks']
    assert len(tasks) == len({t['task_id'] for t in tasks}) == 5
    comparisons = {key: {'resolved': 0, 'model_calls': 0, 'task_seconds': 0.0}
                   for key in ('mini_session', 'old_mini', 'v4')}
    rows = []
    for task in tasks:
        tid = task['task_id']
        folder = args.source / 'mini/runs' / tid
        current = read(folder / 'result.json')
        old = read(args.old_mini / 'mini/runs' / tid / 'result.json')
        v4 = read(args.v4 / 'runs' / tid / 'result.json')
        row = audit_task(folder, v4)
        row['old_mini_resolved'] = old['resolved']
        row['agent_error_type'] = current.get('agent_error_type')
        row['agent_error'] = current.get('agent_error')
        row['patch_replay'] = current['validation'].get('patch_replay')
        row['delivery'] = (read(folder / 'delivery_check.json')
                           if (folder / 'delivery_check.json').exists() else None)
        for key, result in [('mini_session', current), ('old_mini', old), ('v4', v4)]:
            comparisons[key]['resolved'] += bool(result['resolved'])
            comparisons[key]['model_calls'] += result['usage']['code_model_calls']
            comparisons[key]['task_seconds'] += result['finished_at'] - result['started_at']
        event_counts = Counter()
        null_commands, restores = [], []
        for line in (folder / 'events.jsonl').open():
            event = json.loads(line)
            event_counts[event['event']] += 1
            if event['event'] == 'sandbox_command':
                command = event.get('command', '')
                if isinstance(command, str) and any(
                        s in command for s in ('git restore ', 'git checkout ')):
                    restores.append({'command': command, 'returncode': event['returncode']})
            if event['event'] != 'model_response':
                continue
            for choice in event['response']['choices']:
                for call in choice['message'].get('tool_calls') or []:
                    try:
                        command = json.loads(call['function']['arguments']).get('command', '')
                    except (ValueError, AttributeError):
                        continue
                    if isinstance(command, str) and '\x00' in command:
                        index = command.index('\x00')
                        null_commands.append({'tool_call_id': call['id'], 'position': index,
                                              'excerpt': command[max(0, index - 100):index + 100]})
        row['events'] = dict(event_counts)
        row['null_commands'] = null_commands
        row['git_restore_commands'] = restores
        row['temporary_suffix_files'] = [f for f in current['changed_files']
                                         if f.endswith(('.tmp', '.backup', '.bak', '.temp'))]
        write(output / 'tasks' / (tid + '.json'), row)
        rows.append(row)
    official = source_report['arms']['mini']
    assert comparisons['mini_session']['resolved'] == official['resolved']
    assert comparisons['mini_session']['model_calls'] == official['model_calls']
    summary = {
        'audited_at': datetime.now(timezone.utc).isoformat(), 'source': str(args.source),
        'previous_mini': str(args.old_mini), 'previous_v4': str(args.v4),
        'same_five_tasks': comparisons, 'original_report': source_report,
        'wall_seconds': status['finished_at'] - status['started_at'],
        'started_at_utc': datetime.fromtimestamp(status['started_at'], timezone.utc).isoformat(),
        'finished_at_utc': datetime.fromtimestamp(status['finished_at'], timezone.utc).isoformat(),
        'totals': {
            'patch_replayed': sum(r['patch_replay']['applied'] for r in rows),
            'grader_started': sum(bool(r['grader_started']) for r in rows),
            'parsed_grades': sum(bool(r['parsed']) for r in rows),
            'delivery_checked': sum(r['delivery'] is not None for r in rows),
            'delivery_accepted': sum(bool(r['delivery'] and r['delivery']['passed']) for r in rows),
            'tmp_missing_feedbacks': sum(len(r['tmp_missing_observations']) for r in rows),
            'blocked_git_restores': sum(len(r['blocked_restore_observations']) for r in rows),
            'successful_git_restores': sum(c['returncode'] == 0 for r in rows
                                           for c in r['git_restore_commands']),
            'null_command_tasks': [r['task_id'] for r in rows if r['null_commands']],
            'agent_exception_tasks': [r['task_id'] for r in rows if r['agent_error_type']],
            'temporary_file_tasks': [r['task_id'] for r in rows if r['temporary_suffix_files']],
        },
        'new_inference': False, 'candidate_tests_executed': False,
        'original_scores_modified': False,
        'interpretation': 'Known development subset, no screenshots. runtime_failures=0 does not '
                          'exclude the recorded tool argument exception. No causal claim from one run.'}
    write(output / 'analysis.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
