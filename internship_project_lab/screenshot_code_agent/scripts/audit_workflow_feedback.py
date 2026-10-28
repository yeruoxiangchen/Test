#!/usr/bin/env python3
"""Read a fixed regrade cohort and check feedback/context contracts without inference.

No tests, sandboxes, model services or output files are created. Token counting
loads only the local tokenizer, never model weights. Original artifacts remain
unchanged; the saved conversation is copied in memory before compaction.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from corecoder.context import estimate_tokens

from screenshot_agent.context import RepairContext
from screenshot_agent.harness import QwenTokenCounter


def read(path):
    return json.loads(path.read_text())


def audit(source, regrade, context_task):
    plan = read(regrade / 'plan.json')
    if Path(plan['source']).resolve() != source.resolve():
        raise ValueError('Regrade cohort does not belong to the requested source')
    report = read(regrade / 'report.json')
    if not (report['complete'] and report['valid_report']):
        raise ValueError('The selected regrade cohort is not complete and valid')

    pairs, stages, retries = Counter(), {}, Counter()
    for task_id in plan['task_ids']:
        flow = read(source / 'runs' / task_id / 'workflow.json')
        result = read(regrade / 'tasks' / task_id / 'result.json')
        public = flow.get('public_test_passed')
        public_label = 'unknown' if public is None else str(public).lower()
        hidden_label = str(result['resolved']).lower()
        pairs[f'public={public_label},hidden={hidden_label}'] += 1
        retries[str(flow['retry_count'])] += 1
        for stage in flow['plan']:
            stages.setdefault(stage['phase'], Counter())[str(stage.get('exit_reason'))] += 1

    folder = source / 'runs' / context_task
    config = read(folder / 'config.json')
    messages = read(folder / 'conversation.json')
    result = read(folder / 'result.json')
    estimated_before = estimate_tokens(messages)
    counter = QwenTokenCounter(config['tokenizer_path'])
    exact_before = counter(messages, [])
    events = []
    context = RepairContext(config['max_context_tokens'], messages[0]['content'],
                            SimpleNamespace(emit=lambda *a, **kw: events.append((a, kw))))
    changed = context.maybe_compress(messages)

    return {
        'source': str(source),
        'regrade': str(regrade),
        'cohort_size': len(plan['task_ids']),
        'resolved': report['resolved'],
        'public_vs_hidden': dict(pairs),
        'phase_exit_counts': {stage: dict(counts) for stage, counts in stages.items()},
        'retry_counts': dict(retries),
        'context_replay': {
            'task_id': context_task,
            'original_exit_reason': result.get('exit_reason'),
            'original_code_calls': result['usage']['code_model_calls'],
            'max_prompt_tokens': config['max_prompt_tokens'],
            'compression_trigger_estimate': int(config['max_context_tokens'] * .70),
            'estimated_history_tokens': estimated_before,
            'exact_history_tokens_without_system_or_tools': exact_before,
            'compaction_changed_history': changed,
            'exact_history_tokens_after_compaction': counter(messages, []),
            'uncompressed_history_already_over_limit': (
                not changed and exact_before > config['max_prompt_tokens']),
            'limitation': 'History-only count; this does not reconstruct the full API request.',
        },
        'model_requests': 0,
        'scope': 'Offline observations, not a new repair evaluation or a causal ablation.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(
        '/data/zjr/screenshot_code_agent/repair_workflow_20261027'))
    parser.add_argument('--regrade', type=Path, default=Path(
        '/data/zjr/screenshot_code_agent/workflow_regrade_partial_20261028'))
    parser.add_argument('--context-task', default='pvlib__pvlib-python-1026')
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.regrade, args.context_task),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
