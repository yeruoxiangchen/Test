"""Offline timing and outcome audit; never execute candidates or change grades."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path('/data/zjr/screenshot_code_agent')
SOURCE = ROOT/'mini_verified_dev5_20261029'
OUTPUT = ROOT/'mini_verified_dev5_analysis_20261030'


def read(path):
    return json.loads(path.read_text())


def main():
    report, status = read(SOURCE/'report.json'), read(SOURCE/'status.json')
    assert report['complete'] and status['stage'] == 'completed'
    rows = []
    for task in read(SOURCE/'prepared.json')['tasks']:
        p = SOURCE/'mini/runs'/task['task_id']
        result = read(p/'result.json')
        official = result['validation']['hidden']['official']
        counts, seconds, tools = Counter(), Counter(), Counter()
        pending, timeouts, actions = None, [], []
        for line in (p/'events.jsonl').open():
            e = json.loads(line)
            kind = e['event']
            counts[kind] += 1
            if kind == 'model_request':
                pending = e['time']
            if kind in ('model_response', 'model_error') and pending is not None:
                seconds['model_requests'] += e['time'] - pending
                pending = None
            if kind == 'tool_start':
                tools[e['name']] += 1
                if e['name'] in ('bash', 'edit_file', 'check_patch'):
                    cmd = e['arguments'].get('command', '')
                    repeated = e['name'] == 'check_patch' or (
                        e['name'] == 'bash' and 'echo ' in cmd and
                        not any(t in cmd for t in (' >', 'python', 'sed ', 'cat ')))
                    actions.append((e['time'], repeated))
            if kind == 'sandbox_command':
                cmd = str(e.get('command', ''))
                group = ('visible_behavior' if 'development_acceptance.py' in cmd else
                         'pytest' if 'pytest' in cmd else 'other_commands')
                seconds[group] += e.get('elapsed_seconds', 0)
                if e.get('timed_out'):
                    timeouts.append({'group': group, 'seconds': e['elapsed_seconds'], 'command': cmd})
        suffix = 0
        for _, repeated in reversed(actions):
            if not repeated:
                break
            suffix += 1
        row = {'task_id': task['task_id'], 'resolved': result['resolved'],
               'exit_reason': result['exit_reason'], 'usage': result['usage'],
               'task_seconds': result['finished_at'] - result['started_at'],
               'events': dict(counts), 'tool_counts': dict(tools), 'seconds': dict(seconds),
               'timeouts': timeouts, 'terminal_check_or_echo_actions': suffix,
               'terminal_check_or_echo_span': actions[-1][0]-actions[-suffix][0] if suffix else 0,
               'parsed': official.get('parsed'),
               'test_counts': {k: {s: len(v) for s, v in g.items()}
                               for k, g in official.get('tests_status', {}).items()},
               'failed_tests': {k: g.get('failure', []) for k, g in official.get('tests_status', {}).items()},
               'candidate_selection': result.get('candidate_selection'),
               'delivery': read(p/'delivery_check.json') if (p/'delivery_check.json').exists() else None}
        rows.append(row)
    assert len(rows) == 5 and sum(r['resolved'] for r in rows) == report['arms']['mini']['resolved']
    assert sum(r['usage']['code_model_calls'] for r in rows) == report['arms']['mini']['model_calls']
    value = {'source': str(SOURCE), 'report': report, 'tasks': rows,
             'wall_seconds': status['finished_at']-status['started_at'],
             'note': 'Event durations are approximate disjoint model/sandbox categories; controller overhead '
                     'and final grading may be outside them. Known development set, no images.',
             'candidate_execution': False, 'new_inference': False, 'original_scores_modified': False}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/'analysis.json').write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'output': str(OUTPUT), 'resolved': sum(r['resolved'] for r in rows),
                      'wall_seconds': value['wall_seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
