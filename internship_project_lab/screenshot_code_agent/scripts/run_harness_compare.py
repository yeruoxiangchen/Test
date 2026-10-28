#!/usr/bin/env python3
"""Pinned single-agent framework comparison. Long preparation/evaluation is explicit."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import run_scale_4gpu as scale

from screenshot_agent.config import Task
from screenshot_agent.control import atomic_json
from screenshot_agent.harness import PROTOCOL, HarnessConfig, QwenTokenCounter, verify_mini
from screenshot_agent.harness_runner import run_harness_task
from screenshot_agent.hidden_eval import prepare_hidden_task, validate_prepared
from screenshot_agent.scale_data import immutable_json, read, sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path('/data/zjr/screenshot_code_agent/harness_compare_20261027')
DEFAULT_SOURCE = ROOT / 'outputs/scale_toolsfix_dev_20261025/dev/plan.json'


def storage(output, create=True):
    output = Path(output).resolve()
    if not output.is_relative_to('/data/zjr'):
        raise ValueError('New data, outputs, caches and temporary files must live under /data/zjr')
    for name, relative in {'TMPDIR':'tmp', 'XDG_CACHE_HOME':'cache', 'TORCH_HOME':'cache/torch',
        'HF_HOME':'cache/huggingface', 'HF_HUB_CACHE':'cache/huggingface/hub',
        'TORCH_EXTENSIONS_DIR':'cache/torch_extensions', 'MPLCONFIGDIR':'cache/matplotlib',
        'VLLM_CACHE_ROOT':'cache/vllm', 'TRITON_CACHE_DIR':'cache/triton',
        'CUDA_CACHE_PATH':'cache/cuda', 'NUMBA_CACHE_DIR':'cache/numba'}.items():
        path = output / relative
        if create:
            path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    return output


def stage_runtime(c, runtime, cache):
    values = runtime.model_dump()
    for field in ('rootfs', 'python_overlay', 'library_overlay', 'prefix'):
        if not values.get(field):
            continue
        source = Path(values[field]).resolve()
        if source.is_relative_to('/data'):
            continue
        identity = hashlib.sha256(str(source).encode()).hexdigest()
        target = cache / field / identity
        receipt = target.parent / (identity + '.json')
        if receipt.exists():
            if read(receipt)['source'] != str(source) or not target.is_dir():
                raise ValueError('Runtime cache receipt is inconsistent')
        else:
            target.mkdir(parents=True, exist_ok=True)
            c.update(stage='copying_runtime_to_data', runtime_source=str(source), runtime_target=str(target))
            command = ['rsync', '-aH', '--no-owner', '--no-group', '--partial', '--info=progress2',
                       str(source) + '/', str(target) + '/']
            # Deduplicate unchanged files across read-only OCI roots on /data.
            prior = next((p for p in sorted(target.parent.glob('*.json')) if p != receipt), None)
            if prior:
                command.insert(1, '--link-dest=' + str(prior.with_suffix('')))
            if shutil.which('ionice'):
                command = ['ionice', '-c', '2', '-n', '7', *command]
            entry = c.launch(command, 'copy_' + field + '_' + identity[:12])
            c.wait(entry)
            atomic_json(receipt, {'source':str(source), 'target':str(target), 'completed_at':time.time(),
                                  'method':'rsync archive/partial, source retained; transfer checksums verified by rsync'})
        values[field] = str(target)
    return type(runtime).model_validate(values)


def prepare(c, args):
    source = read(args.source_plan)
    rows = source['tasks']
    if len(rows) != 100 or len({r['task_id'] for r in rows}) != 100:
        raise ValueError('Formal comparison requires the same 100 distinct development tasks')
    identity = {'source_plan':str(args.source_plan.resolve()), 'sha256':sha256(args.source_plan),
                'protocol':PROTOCOL, 'task_ids':[r['task_id'] for r in rows]}
    immutable_json(args.output / 'selection.json', identity)
    receipts = []
    for i, row in enumerate(rows):
        folder = args.output / 'data/tasks' / row['task_id']
        c.update(stage='preparing_hidden_inputs', completed=i, total=len(rows), task_id=row['task_id'])
        if folder.exists() and not (folder / 'prepared.json').exists():
            archived = args.output / 'prepare_attempts' / f"{row['task_id']}_{time.time_ns()}"
            archived.parent.mkdir(parents=True, exist_ok=True)
            folder.rename(archived)
        if folder.exists():
            receipt = validate_prepared(folder)
        else:
            task = Task.from_file(Path(row['task_file']))
            runtime = stage_runtime(c, task.runtime, args.output / 'cache/task_runtimes')
            receipt = prepare_hidden_task(row['task_file'], folder, runtime=runtime)
        receipts.append(receipt)
    manifest = {'protocol':PROTOCOL, 'tasks':receipts, 'selection_sha256':sha256(args.output / 'selection.json')}
    immutable_json(args.output / 'prepared.json', manifest)
    c.update(stage='prepared', completed=len(receipts), total=len(rows))


def freeze(args):
    runtime = scale.frozen_runtime(args.output)
    pin = runtime / 'MINI_UPSTREAM.json'
    if not pin.exists():
        shutil.copy2(ROOT / 'MINI_UPSTREAM.json', pin)
    if pin.read_bytes() != (ROOT / 'MINI_UPSTREAM.json').read_bytes():
        raise ValueError('mini upstream pin differs from frozen run')
    return runtime


def framework_environment():
    import corecoder
    package = Path(corecoder.__file__).parent
    return {'python_packages': {name:importlib.metadata.version(name) for name in (
        'corecoder','openai','pydantic','jinja2','platformdirs','tokenizers','PyYAML','rich','python-dotenv')},
        'corecoder_sources':{str(p.relative_to(package)):sha256(p) for p in sorted(package.rglob('*.py'))}}


def capacity_check(c, args, config, key, *, chat=None, counter=None):
    """Check long prompt admission before spending any real repair attempts."""
    counter = counter or QwenTokenCounter(config.tokenizer_path)
    filler = ' x' * max(1, config.max_prompt_tokens - 512)
    messages = [{'role':'user', 'content':f'Read this padding and reply OK only.\n{filler}'}]
    tokens = counter(messages, [])
    if tokens > config.max_prompt_tokens:
        raise ValueError('Capacity smoke exceeds its declared input allowance')
    payload = {'model':'sca-coder','messages':messages,'max_tokens':16,'temperature':0}
    def query():
        started = time.monotonic()
        if chat is None:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            request = urllib.request.Request(f'http://127.0.0.1:{args.port}/v1/chat/completions',
                data=json.dumps(payload).encode(), headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
            with opener.open(request, timeout=config.limits.request_timeout_seconds) as response:
                value = json.load(response)
        else:
            value = chat(payload)
        if not value.get('choices') or not value['choices'][0]['message'].get('content'):
            raise ValueError('Long-context capacity smoke returned no answer')
        return {'seconds':round(time.monotonic()-started,3), 'usage':value.get('usage'),
                'finish_reason':value['choices'][0].get('finish_reason')}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(query) for _ in range(args.workers)}
        rows = []
        while pending:
            done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            rows.extend(future.result() for future in done)
            c.update(stage='long_context_capacity_check', probe_prompt_tokens=tokens,
                     probe_completed=len(rows), probe_total=args.workers)
    value = {'prompt_tokens':tokens, 'concurrency':args.workers, 'responses':rows,
             'note':'Long-prefill admission with short outputs; not a throughput benchmark or worst-case decode guarantee'}
    atomic_json(args.output / 'capacity_check.json', value)
    return value


def results(output, tasks):
    arms = {}
    for arm in ('corecoder', 'mini'):
        records = []
        for task in tasks:
            p = output / arm / 'runs' / task['task_id'] / 'result.json'
            if p.exists():
                value = read(p)
                if value.get('protocol') != PROTOCOL or value.get('framework') != arm or value['task_id'] != task['task_id']:
                    raise ValueError('Result identity/protocol does not match the comparison manifest')
                records.append(value)
        arms[arm] = records
    return arms


def report(output):
    prepared = read(output / 'prepared.json')
    arms = results(output, prepared['tasks'])
    totals = {}
    for arm, rows in arms.items():
        totals[arm] = {'completed':len(rows), 'total':len(prepared['tasks']),
            'resolved':sum(r['resolved'] for r in rows),
            'runtime_failures':sum(bool(r.get('runtime_failure')) for r in rows),
            'zero_patch':sum(not r.get('changed_files') for r in rows),
            'model_calls':sum(r.get('usage', {}).get('code_model_calls', 0) for r in rows),
            'tool_calls':sum(r.get('usage', {}).get('tool_calls', 0) for r in rows),
            'task_seconds':sum(max(0, r.get('finished_at', 0)-r.get('started_at', 0)) for r in rows),
            'prompt_tokens':sum(r.get('usage', {}).get('accounted_prompt_tokens', 0) for r in rows),
            'completion_tokens':sum(r.get('usage', {}).get('accounted_completion_tokens', 0) for r in rows),
            'exit_reasons':dict(Counter(r.get('exit_reason',r['status']) for r in rows))}
    complete = all(t['completed'] == t['total'] for t in totals.values())
    paired = Counter()
    paired_tasks = {name:[] for name in ('both_success','mini_only','corecoder_only','both_failed')}
    maps = {a:{r['task_id']:r for r in rows} for a, rows in arms.items()}
    for task_id in sorted(maps['corecoder'].keys() & maps['mini'].keys()):
        a, b = maps['corecoder'][task_id]['resolved'], maps['mini'][task_id]['resolved']
        group = 'both_success' if a and b else 'mini_only' if b else 'corecoder_only' if a else 'both_failed'
        paired[group] += 1
        paired_tasks[group].append(task_id)
    value = {'protocol':PROTOCOL, 'complete':complete, 'arms':totals, 'paired':dict(paired), 'paired_tasks':paired_tasks,
             'interpretation':'Same Qwen model and 100 development tasks; reference tests hidden during repair. '
                              'Do not compare percentage deltas directly against the old visible-test 19/100.'}
    atomic_json(output / 'report.json', value)
    lines = ['# CoreCoder / mini-SWE-agent 对照', '', value['interpretation'], '',
             '| 框架 | 已结束 | 修复成功 | 运行异常 |', '|---|---:|---:|---:|']
    for arm, item in totals.items():
        lines.append(f"| {arm} | {item['completed']}/{item['total']} | {item['resolved']}/{item['total']} | {item['runtime_failures']} |")
    lines += ['', '全部运行异常保留在固定100题分母；完整配对：' + json.dumps(dict(paired), ensure_ascii=False)]
    lines += ['', '| 框架 | 模型调用 | 工具调用 | 输入token（含失败估计） | 输出token（含失败估计） | 单题时间合计(h) | 零补丁 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for arm, item in totals.items():
        lines.append(f"| {arm} | {item['model_calls']} | {item['tool_calls']} | {item['prompt_tokens']} | "
                     f"{item['completion_tokens']} | {item['task_seconds']/3600:.2f} | {item['zero_patch']} |")
    lines += ['', '时间是各题时间之和，包含任务快照及最终验证；并行运行时不等于整体墙钟时长。'
              '逐题差异和退出原因见 report.json。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    return value


def run(c, args):
    verify_mini()
    prepared = read(args.output / 'prepared.json')
    if prepared['protocol'] != PROTOCOL or len(prepared['tasks']) != 100:
        raise ValueError('All 100 hidden-test inputs must be prepared before model startup')
    runtime = freeze(args)
    config = HarnessConfig.model_validate(read(runtime / 'configs/harness-compare.json'))
    args.service_context_tokens = config.service_context_tokens
    args.max_num_seqs = args.workers
    args.split = 'dev'
    plan = {'protocol':PROTOCOL, 'frameworks':['corecoder','mini'], 'config':config.model_dump(),
            'prepared_sha256':sha256(args.output / 'prepared.json'),
            'source_sha256':sha256(runtime / 'source_state.json'), 'mini_pin_sha256':sha256(runtime / 'MINI_UPSTREAM.json'),
            'model':scale.verify_resource_receipts(), 'framework_environment':framework_environment(), 'workers':args.workers,
            'gpu_policy':scale.gpu_policy(args)}
    immutable_json(args.output / 'comparison_plan.json', plan)
    pending = []
    for i, receipt in enumerate(prepared['tasks']):
        c.update(stage='validating_inputs', completed=i, total=100)
        validate_prepared(Path(receipt['task_file']).parent)
        for arm in ('corecoder','mini'):
            directory = args.output / arm / 'runs' / receipt['task_id']
            if (directory / 'result.json').exists():
                old = read(directory / 'result.json')
                if old['protocol'] != PROTOCOL or old['framework'] != arm or old['task_id'] != receipt['task_id']:
                    raise ValueError('Mismatched completed attempt')
                expected = config.model_copy(update={'framework':arm, 'code':config.code.model_copy(
                    update={'base_url':f'http://127.0.0.1:{args.port}/v1'})}).model_dump()
                if read(directory / 'config.json') != expected:
                    raise ValueError('Completed attempt used different model or budget settings')
                # Completed API timeouts stay in the denominator. Only interrupted
                # attempts can be explicitly archived and resumed.
                if old['status'] != 'cancelled':
                    continue
            if directory.exists():
                if not args.resume_interrupted:
                    raise ValueError(f'Incomplete attempt retained: {directory}; use --resume-interrupted')
                archive = args.output / 'interrupted_attempts' / arm / f"{receipt['task_id']}_{time.time_ns()}"
                archive.parent.mkdir(parents=True, exist_ok=True)
                directory.rename(archive)
            pending.append((arm, receipt))
    if not pending:
        return report(args.output)
    service, env = scale.start_service(c, args, runtime, 'coder', read(runtime / 'configs/scale_4gpu_resources.json'))
    try:
        capacity_check(c, args, config, env['SCA_CODE_API_KEY'])
        active = []
        for arm in ('corecoder','mini'):
            cfg = config.model_copy(update={'framework':arm, 'code':config.code.model_copy(
                update={'base_url':f'http://127.0.0.1:{args.port}/v1'})})
            immutable_json(args.output / arm / 'config.json', cfg.model_dump())
        while pending or active:
            if service['proc'].poll() is not None:
                raise RuntimeError('Shared model server exited; task attempts retained for explicit resume')
            while pending and len(active) < args.workers:
                arm, task = pending.pop(0)
                directory = args.output / arm / 'runs' / task['task_id']
                command = [sys.executable, '-u', str(runtime / 'scripts/run_harness_compare.py'), 'task',
                           '--output', str(args.output), '--task', task['task_file'],
                           '--config', str(args.output / arm / 'config.json'), '--run-dir', str(directory)]
                entry = c.launch(command, arm + '_' + task['task_id'], env=env, cwd=runtime)
                active.append((entry, directory, time.monotonic()))
            for entry, directory, started in list(active):
                if time.monotonic() - started > config.limits.max_wall_seconds + 1200:
                    raise RuntimeError('Task outer timeout; partial attempt retained')
                if entry['proc'].poll() is not None:
                    if entry['proc'].returncode or not (directory / 'result.json').exists():
                        raise RuntimeError('Task process failed: ' + entry['log'])
                    active.remove((entry,directory,started))
            c.update(stage='evaluating', pending=len(pending), running=len(active), total=200)
            if active:
                time.sleep(1)
    finally:
        scale.terminate(service['proc'])
    return report(args.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare','run','all','status','report','task','check'])
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--source-plan', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--gpus', default='2,5,6,7')
    p.add_argument('--workers', type=int, choices=[1,2,4,8], default=2)
    p.add_argument('--port', type=int, default=18846)
    p.add_argument('--gpu-memory-utilization', type=float, default=.8)
    p.add_argument('--gpu-headroom-mib', type=int, default=1536)
    p.add_argument('--allow-shared-gpus', action='store_true')
    p.add_argument('--wait-for-gpus', action='store_true')
    p.add_argument('--trace-server-signals', action='store_true')
    p.add_argument('--resume-interrupted', action='store_true')
    p.add_argument('--task', type=Path)
    p.add_argument('--config', type=Path)
    p.add_argument('--run-dir', type=Path)
    args = p.parse_args()
    args.output = storage(args.output, create=args.action != 'status')
    if args.action == 'status':
        state = read(args.output / 'status.json') if (args.output / 'status.json').exists() else {'stage':'not_started'}
        state['controller_alive'] = scale.alive(state.get('pid'), state.get('start_ticks'))
        for child in state.get('children', []):
            child['alive'] = scale.alive(child['pid'], child.get('start_ticks'))
        if state.get('stage') == 'copying_runtime_to_data' and state.get('children'):
            log = Path(state['children'][-1]['log'])
            if log.is_file():
                with log.open('rb') as stream:
                    stream.seek(max(0, log.stat().st_size - 4096))
                    lines = stream.read().decode(errors='replace').replace('\r','\n').splitlines()
                state['copy_progress'] = next((line.strip() for line in reversed(lines) if line.strip()), '')
        if (args.output / 'prepared.json').exists():
            state['arms'] = {arm:{'completed':len(rows), 'resolved':sum(r['resolved'] for r in rows),
                'runtime_failures':sum(bool(r.get('runtime_failure')) for r in rows)}
                for arm, rows in results(args.output, read(args.output / 'prepared.json')['tasks']).items()}
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return
    if args.action == 'check':
        from screenshot_agent.harness import QwenTokenCounter, load_mini
        config = HarnessConfig.model_validate(read(ROOT / 'configs/harness-compare.json'))
        load_mini(args.output / 'checks')
        count = QwenTokenCounter(config.tokenizer_path)([{'role':'user','content':'hello'}], [])
        row = read(args.source_plan)['tasks'][0]
        folder = args.output / 'checks/real_input'
        if folder.exists() and not (folder / 'prepared.json').exists():
            folder.rename(folder.with_name('real_input_incomplete_' + str(time.time_ns())))
        receipt = prepare_hidden_task(row['task_file'], folder)
        validate_prepared(folder)
        checked = {'upstream':verify_mini()['commit'], 'tokenizer_smoke_tokens':count, 'gpu_started':False,
                   'real_input':receipt['task_id'], 'reference_test_roundtrip_verified':True,
                   'note':'CPU-only single-input check; no task/model evaluation or runtime copy started'}
        atomic_json(args.output / 'checks/check.json', checked)
        print(json.dumps(checked))
        return
    if args.action == 'task':
        run_harness_task(Task.from_file(args.task), HarnessConfig.model_validate(read(args.config)), args.run_dir)
        return
    if args.action == 'report':
        print(json.dumps(report(args.output), ensure_ascii=False, indent=2))
        return
    def cancel(signum, frame):
        raise KeyboardInterrupt('Cancellation requested')
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, cancel)
    c = scale.Controller(args.output, args.action)
    try:
        if args.action in ('prepare','all'):
            prepare(c, args)
        if args.action in ('run','all'):
            run(c,args)
        c.update(stage='prepared' if args.action == 'prepare' else 'completed', finished_at=time.time())
    except KeyboardInterrupt:
        c.update(stage='cancelled', finished_at=time.time())
        raise SystemExit(130) from None
    except Exception as error:
        c.update(stage='failed', error=str(error), finished_at=time.time())
        raise
    finally:
        c.close()


if __name__ == '__main__':
    main()
