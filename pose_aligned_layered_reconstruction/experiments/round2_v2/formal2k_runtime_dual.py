"""Explicit two-GPU training revision; unchanged model/data and global batch 2."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from . import formal2k_run as base
from . import formal2k_runtime as previous_runtime
from .formal2k_run import (DEFAULT_RUN, PYTHON, WORKSPACE, utc, read, atomic, sha,
    bind, canonical, journal, verify_receipts, seal_stage,
    stage_progress)

AMENDMENT_NAME = 'runtime_amendments/002_gpu12_dual.json'
GPUS = [1, 2]


def derived_plan(plan, amendment_record):
    stages = []
    for stage in plan['stages']:
        command = list(stage['command'])
        if stage['kind'] == 'train':
            pos = command.index('--grad_accum') + 1
            if command[pos] != '2':
                raise ValueError('Expected the original global-batch-2 training command')
            command[pos] = '1'
        stages.append({**stage, 'command': command})
    return {**plan, 'stages': stages, 'world_size': 2, 'gpus': GPUS, 'gpu': '1,2',
            'runtime_amendment': amendment_record, 'effective_batch': 2,
            'concurrent_gpu_children': 2,
            'allocator_cap_mib': None, 'minimum_free_mib': None}


def freeze_amendment(args):
    root = Path(args.run_dir).resolve()
    plan, identity = previous_runtime.verify_protocol(root)
    path = root / AMENDMENT_NAME
    if path.exists():
        raise FileExistsError('Runtime amendment is immutable and already exists')
    lock = (root / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if path.exists():
        raise FileExistsError('Runtime amendment was frozen before lock acquisition')
    if (root / 'runtime_amendments/002_gpu7_memory.json').exists():
        raise ValueError('A competing 002 memory revision exists; explicit reconciliation is required')
    status = read(root / 'status.json')
    train_names = {stage['name'] for stage in plan['stages'] if stage['kind'] == 'train'}
    if (status['state'] not in ('failed', 'stopped') or status.get('child_pid')
            or status.get('identity_hash') != identity
            or status.get('runtime_amendment', {}).get('identity_hash') != plan['runtime_amendment']['identity_hash']
            or any(item['name'] in train_names for item in status['completed_stages'])
            or any((root / name).exists() for name in train_names)):
        raise ValueError('Dual-GPU revision requires a stopped/failed queue before any training starts')
    if list((root / 'stage_receipts').glob('*.json')):
        raise ValueError('Existing sealed stages require explicit identity migration before a dual revision')
    if 'OutOfMemoryError' not in read(root / 'data/failure.json').get('exception', ''):
        raise ValueError('Expected preserved data-preparation OOM evidence')
    receipt_paths = sorted((root / 'data/receipts').glob('*.json'))
    if len(receipt_paths) != 6:
        raise ValueError('Expected the six successful data receipts from the original failed attempt')
    history_sources = [root / 'status.json', root / 'data/failure.json',
                       root / 'data/runtime_binding.json', root / 'data/progress.json',
                       root / 'data/status.json', Path(status['current_log'])]
    history_sources += [p for p in (root / 'memory_probe/cap16_gpu7/report.json',
                                   root / 'memory_probe/cap16_gpu7.log') if p.exists()]
    history_sources = [p.resolve(strict=True) for p in history_sources]
    history = []
    for source in history_sources:
        destination = root / 'runtime_amendments/002_gpu12_dual_history' / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        history.append({**bind(destination), 'original_path': str(source)})
    receipt_bindings = [bind(p) for p in receipt_paths]
    old_probe_script = root / 'memory_probe/probe_largest.py'
    probe_script_binding = bind(old_probe_script) if old_probe_script.exists() else None
    sources = []
    candidates = [Path(__file__), Path(__file__).with_name('run_formal2k_dual.sh')]
    for source in (p for p in candidates if p.exists()):
        record = bind(source)
        destination = root / 'runtime_amendments/002_gpu12_dual_sources' / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        record['snapshot'] = str(destination)
        sources.append(record)
    amendment = {'format': 'palr.formal2k.runtime_amendment.v1', 'created_at_utc': utc(),
        'parent_protocol': bind(root / 'protocol.json'), 'parent_protocol_identity': identity,
        'parent_runtime_amendment': bind(root / previous_runtime.AMENDMENT_NAME),
        'parent_runtime_identity': plan['runtime_amendment']['identity_hash'],
        'authorization': 'User requested two GPUs; the user owns manual launch and long-running monitoring.',
        'old_gpu': plan['gpu'], 'gpus': GPUS, 'data_gpus': [1], 'world_size': 2,
        'grad_accum': 1, 'effective_batch': 2, 'concurrent_training_stages': 1, 'concurrent_gpu_children': 2,
        'admission': 'Required GPUs must each have used <=64 MiB and utilization <=5% on two checks ten seconds apart; no device reservation is claimed.',
        'allocator_cap_mib': None, 'capacity_probe_required': False,
        'on_failure': 'Preserve artifacts and stop the queue; no automatic retry, object filtering or model changes.',
        'unchanged': 'Dataset, eight-view inputs, model sources, initialization seed, optimizer, learning rate, global batch, update/exposure budgets and evaluation design.',
        'distribution_change': 'Training only: world_size 1->2 and grad_accum 2->1. Same global sample pairs; rank-dependent diffusion streams differ from the single-GPU trajectory. Each arm uses the same two-rank policy.',
        'command_transform': 'Change only --grad_accum 2 to 1 in frozen training commands; launch them with standalone torch.distributed.run, two ranks. Data remains one process on GPU 1.',
        'failed_attempt_history': history, 'preserved_data_receipts': receipt_bindings,
        'historical_probe_script': probe_script_binding, 'source_code': sources}
    atomic(path, {'identity_hash': canonical(amendment), 'amendment': amendment})
    journal(root, 'runtime_amendment_frozen', identity_hash=canonical(amendment),
            parent_protocol_identity=identity, gpus=GPUS, world_size=2, effective_batch=2,
            automatic_launch=False, capacity_probe_required=False)
    atomic(root / 'status.json', {
        'state': 'ready_manual_launch', 'identity_hash': canonical(amendment),
        'parent_protocol_identity': identity, 'gpu': '1,2', 'gpus': GPUS,
        'active_gpus': [], 'child_pid': None, 'current_stage': None,
        'completed_stages': [], 'queue': [stage['name'] for stage in plan['stages']],
        'runtime_amendment': {**bind(path), 'identity_hash': canonical(amendment)},
        'updated_at_utc': utc(), 'capacity_probe_required': False,
    })


def verify_protocol(root):
    plan, identity = previous_runtime.verify_protocol(root)
    path = root / AMENDMENT_NAME
    envelope = read(path)
    amendment = envelope['amendment']
    if (canonical(amendment) != envelope['identity_hash'] or amendment['parent_protocol_identity'] != identity
            or amendment['parent_runtime_identity'] != plan['runtime_amendment']['identity_hash']):
        raise ValueError('Runtime amendment identity mismatch')
    records = [amendment['parent_protocol'], amendment['parent_runtime_amendment'],
               *amendment['failed_attempt_history'], *amendment['preserved_data_receipts'], *amendment['source_code']]
    if amendment['historical_probe_script'] is not None:
        records.append(amendment['historical_probe_script'])
    for item in records:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Runtime amendment source changed: ' + item['path'])
        if 'snapshot' in item and sha(item['snapshot']) != item['sha256']:
            raise ValueError('Runtime source snapshot changed: ' + item['snapshot'])
    if (amendment['gpus'] != GPUS or amendment['data_gpus'] != [1]
            or amendment['world_size'] != 2 or amendment['grad_accum'] != 1
            or amendment['effective_batch'] != 2 or amendment['allocator_cap_mib'] is not None
            or amendment['capacity_probe_required'] is not False):
        raise ValueError('Fixed dual-GPU policy changed')
    record = {'path': str(path), 'sha256': sha(path), 'identity_hash': envelope['identity_hash']}
    return derived_plan(plan, record), envelope['identity_hash']


def gpu_usage(index):
    result = subprocess.run(['nvidia-smi', f'--id={index}',
        '--query-gpu=memory.used,memory.free,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    used, free, total, utilization = (int(s.strip()) for s in result.stdout.strip().split(','))
    return {'memory_mib': used, 'free_mib': free, 'total_mib': total, 'utilization_percent': utilization}


def admission_ready(usages):
    return bool(usages) and all(value['memory_mib'] <= 64 and value['utilization_percent'] <= 5
                                for value in usages.values())


def worker_command(command, stage):
    if command[:3] != [PYTHON, '-u', '-m']:
        raise ValueError('Expected the frozen Python module command')
    if stage['kind'] == 'data':
        return list(command)
    return [PYTHON, '-u', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
            '--nproc_per_node=2', '--module', command[3], *command[4:]]


def validate_rng_sidecars(checkpoint):
    """Reject a newly saved main checkpoint with missing/stale rank RNG state."""
    import torch
    checkpoint = Path(checkpoint)
    main_time = checkpoint.stat().st_mtime_ns
    result = {}
    for rank in range(2):
        path = Path(str(checkpoint) + f'.rank{rank}.rng.pt')
        if not path.is_file() or path.stat().st_mtime_ns < main_time:
            raise ValueError(f'Rank {rank} RNG is missing/stale for {checkpoint}; manually recover a complete numbered checkpoint, never mix generations')
        value = torch.load(path, map_location='cpu', weights_only=False)
        if (set(value) != {'python', 'numpy', 'torch', 'cuda'}
                or not isinstance(value['python'], tuple) or len(value['python']) != 3
                or not isinstance(value['numpy'], tuple) or len(value['numpy']) != 5):
            raise ValueError('Invalid rank RNG schema: ' + str(path))
        for key in ('torch', 'cuda'):
            state = value[key]
            if not torch.is_tensor(state) or state.dtype != torch.uint8 or state.ndim != 1 or not state.numel():
                raise ValueError('Invalid RNG tensor: ' + str(path) + ' ' + key)
        result[f'rank{rank}_rng'] = bind(path)
    return result


def verify_stage_complete(root, stage, plan):
    if stage['kind'] == 'data':
        item = read(root / 'data/status.json')
        if (item['state'] != 'complete' or item['identity_hash'] != plan['data_identity']
            or item['sample_count'] != plan['sample_count'] or item['object_count'] != 2000
            or sha(item['manifest']['path']) != item['manifest']['sha256']):
            raise ValueError('Incomplete or mismatched data preparation')
        return {'manifest': item['manifest'], 'sample_count': item['sample_count']}
    folder = root / stage['name']
    item, config = read(folder / 'complete.json'), read(folder / 'run_config.json')
    # Compare the saved configuration with the actual frozen CLI, including
    # scalar types; --init is intentionally replaced by --resume on recovery.
    command = stage['command']
    for index, token in enumerate(command):
        if not token.startswith('--'):
            continue
        key = token[2:]
        if key == 'init' and config.get('resume'):
            continue
        expected = command[index + 1] if index + 1 < len(command) and not command[index + 1].startswith('--') else True
        actual = config.get(key)
        if isinstance(actual, (float, int)) and not isinstance(actual, bool) and expected is not True:
            matches = actual == float(expected)
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(f'Completed training option differs: {stage["name"]} {key}')
    if (item['step'] != stage['expected'] or config['max_steps'] != stage['expected']
        or item['data']['sha256'] != sha(plan['training_manifest'])
        or item['data'] != config['data_identity'] or item['data']['sample_count'] != plan['sample_count']
        or len(item['data']['object_uids']) != 2000 or config['effective_batch'] != 2
        or item['data']['world_size'] != 2 or item['data']['grad_accum'] != 1 or config['grad_accum'] != 1 or config['max_samples'] != 0
        or config['cache_frozen_geometry'] or config['finetune_lora'] or config['no_gradient_checkpointing']):
        raise ValueError('Completed training identity/budget differs: ' + stage['name'])
    for field in ('initial_rank_parameter_hashes', 'final_rank_parameter_hashes'):
        hashes = item.get(field, [])
        if len(hashes) != 2 or len(set(hashes)) != 1:
            raise ValueError('Two-rank parameter consistency check failed: ' + field)
    if stage['kind'] == 'train' and 'condition_mode' in stage:
        if config['condition_mode'] != stage['condition_mode'] or not config['freeze_geometry']:
            raise ValueError('Completed SS control mode differs')
        parent = config['bindings']['parent']
        expected_parent = read(plan['data_plan']['path'])['plan']['parent_audit']['ss30k']['checkpoint']
        if (parent['path'] != expected_parent['path'] or parent['sha256'] != expected_parent['sha256']
            or parent['weights'] != 'ema' or not parent['freeze_parent']):
            raise ValueError('SS parent differs from frozen data audit')
        smoke = read(plan['smoke_report']['path'])
        accepted = next(r for r in smoke['results'] if r['condition_mode'] == stage['condition_mode'])
        if config['trainable_parameter_count'] != accepted['trainable_parameter_count']:
            raise ValueError('SS trainable capacity differs from real-GPU acceptance')
        if set(config['init_adapter_hashes'] or {}) != {'layer_encoder', 'view_gate', 'layer_block_condition'}:
            raise ValueError('SS shared-adapter initial hashes are missing')
        if stage['name'] != 'ss_full':
            reference = read(root / 'ss_full/run_config.json')
            for key in ('data_identity', 'init_adapter_hashes'):
                if config[key] != reference[key]:
                    raise ValueError('SS arms differ in ' + key)
            if config['bindings']['stock'] != reference['bindings']['stock']:
                raise ValueError('SS arms differ in frozen stock model')
    checkpoint = folder / 'checkpoints' / f"step_{stage['expected']:06d}.pt"
    rng = validate_rng_sidecars(checkpoint)
    return {'complete': bind(folder / 'complete.json'), 'run_config': bind(folder / 'run_config.json'),
            'checkpoint': bind(checkpoint), **rng}


def run(args):
    root = Path(args.run_dir).resolve()
    plan, identity = verify_protocol(root)
    lock = (root / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = os.environ.copy()
    env.update(CUDA_HOME='/usr/local/cuda-12.6',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONFAULTHANDLER='1',
               TORCH_CUDA_ARCH_LIST='8.6', SPCONV_ALGO='native',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               MPLCONFIGDIR=str(root / 'matplotlib_cache'))
    env['PATH'] = '/home/zjr/anaconda3/envs/reconviagen/bin:' + env.get('PATH', '')
    for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
                'GROUP_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE', 'V2_NPROC'):
        env.pop(key, None)
    state = {'state': 'starting', 'identity_hash': identity, 'gpu': '1,2', 'gpus': GPUS,
             'supervisor_pid': os.getpid(), 'started_at_utc': utc(), 'completed_stages': [],
             'queue': [s['name'] for s in plan['stages']], 'current_stage': None, 'child_pid': None,
             'runtime_amendment': plan['runtime_amendment'],
             'admission_policy': {'maximum_used_mib': 64, 'maximum_utilization_percent': 5,
                                  'torch_allocator_cap_mib': None, 'capacity_probe_required': False}}
    stop = [False]
    def request_stop(signum, frame):
        stop[0] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    def publish():
        state['updated_at_utc'] = utc()
        atomic(root / 'status.json', state)
    process = None
    journal(root, 'supervisor_started', pid=os.getpid(), gpu=plan['gpu'], resume=args.resume,
            runtime_amendment=plan['runtime_amendment'])
    try:
        for stage in plan['stages']:
            verify_protocol(root)
            receipts = verify_receipts(root, plan, identity)
            if stage['kind'] == 'train' and 'prepare_occlusion' not in receipts:
                raise ValueError('Training requires a sealed, complete data preparation receipt')
            if 'condition_mode' in stage and 'lari' not in receipts:
                raise ValueError('SS requires a sealed shared LaRI checkpoint')
            state.update(current_stage=stage['name'], state='checking_stage', child_pid=None)
            active_gpus = [1] if stage['kind'] == 'data' else GPUS
            state['active_gpus'] = active_gpus
            state['queue'] = state['queue'][1:]
            publish()
            completion = root / ('data/status.json' if stage['kind'] == 'data' else stage['name'] + '/complete.json')
            already_complete = completion.exists() and (stage['kind'] != 'data' or read(completion).get('state') == 'complete')
            if already_complete:
                if not args.resume:
                    raise ValueError('Completed stage already exists; explicit --resume required')
                if stage['name'] not in receipts:
                    raise ValueError('Completed stage has no immutable receipt; manual recovery audit required: ' + stage['name'])
                evidence = verify_stage_complete(root, stage, plan)
                state['completed_stages'].append({'name': stage['name'], 'reused_completed_stage': True, **evidence})
                publish()
                continue
            command = list(stage['command'])
            folder = root / stage['name']
            resume_checkpoint = None
            if stage['kind'] == 'train' and folder.exists():
                checkpoint = folder / 'checkpoints/last.pt'
                if not args.resume or not checkpoint.exists():
                    raise ValueError('Existing incomplete training needs explicit --resume and a checkpoint: ' + str(folder))
                if '--init' in command:
                    pos = command.index('--init')
                    del command[pos:pos+2]
                command += ['--resume', str(checkpoint)]
                validate_rng_sidecars(checkpoint)
                resume_checkpoint = checkpoint
            idle = 0
            while idle < 2:
                if stop[0] or (root / 'STOP').exists():
                    raise InterruptedError('Stop requested before GPU stage')
                usage = {str(gpu): gpu_usage(gpu) for gpu in active_gpus}
                idle = idle + 1 if admission_ready(usage) else 0
                state.update(state='waiting_for_fixed_gpu', last_gpu_usage=usage)
                publish()
                if idle < 2:
                    time.sleep(10)
            verify_protocol(root)
            verify_receipts(root, plan, identity)
            log = root / 'logs' / f"{stage['name']}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            if resume_checkpoint is not None:
                resume_source = {'checkpoint': bind(resume_checkpoint),
                                 'previous_run_config': bind(folder / 'run_config.json'),
                                 **validate_rng_sidecars(resume_checkpoint)}
                launch_path = root / 'resume_sources' / f"{stage['name']}_{time.time_ns()}.json"
                atomic(launch_path, {**resume_source, 'previous_config_content': read(folder / 'run_config.json'),
                                    'protocol_identity': identity, 'recorded_at_utc': utc()})
                journal(root, 'resume_source_bound', name=stage['name'], record=bind(launch_path), **resume_source)
            command = worker_command(command, stage)
            child_env = {**env, 'CUDA_VISIBLE_DEVICES': ','.join(map(str, active_gpus))}
            with log.open('a') as stream:
                process = subprocess.Popen(command, cwd=WORKSPACE, env=child_env, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                state.update(state='running', child_pid=process.pid, current_command=command,
                             current_log=str(log), stage_started_at_utc=utc())
                journal(root, 'stage_started', name=stage['name'], pid=process.pid, command=command, log=str(log))
                while process.poll() is None:
                    if stop[0] or (root / 'STOP').exists():
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=45)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        raise InterruptedError('Stop requested; last saved checkpoint retained')
                    state['progress'] = stage_progress(root, stage)
                    state['last_gpu_usage'] = {str(gpu): gpu_usage(gpu) for gpu in active_gpus}
                    publish()
                    time.sleep(10)
                code = process.returncode
            journal(root, 'stage_exited', name=stage['name'], exit_status=code)
            state.update(child_pid=None, last_exit_status=code)
            if code:
                raise RuntimeError(f"Stage {stage['name']} failed with exit {code}; following stages were not launched")
            verify_protocol(root)
            verify_receipts(root, plan, identity)
            evidence = verify_stage_complete(root, stage, plan)
            evidence['runtime_amendment'] = bind(root / AMENDMENT_NAME)
            receipt = seal_stage(root, stage, evidence, identity)
            state['completed_stages'].append({'name': stage['name'], 'completed_at_utc': utc(),
                                             'receipt': receipt, **evidence})
            publish()
            process = None
        state.update(state='training_complete_pending_evaluation', current_stage=None, child_pid=None,
                     finished_at_utc=utc(), queue=[])
        atomic(root / 'training_matrix.json', {'identity_hash': identity, 'stages': state['completed_stages'],
               'state': state['state'], 'evaluation_complete': False, 'finished_at_utc': utc()})
        journal(root, 'all_training_stages_complete', evaluation_complete=False)
        publish()
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        state.update(state='stopped' if isinstance(error, InterruptedError) else 'failed',
                     error=str(error), traceback=traceback.format_exc(), child_pid=None)
        publish()
        journal(root, state['state'], error=str(error))
        raise


def monitor(args):
    root = Path(args.run_dir).resolve()
    while True:
        state = read(root / 'status.json') if (root / 'status.json').exists() else {'state': 'not_started'}
        if args.json:
            print(json.dumps(state, ensure_ascii=False), flush=True)
        else:
            lines = [f'[{utc()}] Formal2K 双卡训练 GPU 1,2',
                     f"状态：{state['state']}  阶段：{state.get('current_stage') or '-'}  实际使用：{state.get('active_gpus', [])}"]
            heartbeat = state.get('updated_at_utc')
            if heartbeat:
                age = max(0., (datetime.now(timezone.utc) - datetime.fromisoformat(heartbeat)).total_seconds())
                lines.append(f'最近状态更新：{age:.0f} 秒前')
                if age > 120 and state['state'] in ('running', 'waiting_for_fixed_gpu', 'starting', 'checking_stage'):
                    lines.append('警告：状态心跳已超过 120 秒未更新；请检查后台服务和日志。')
            original = read(root / 'protocol.json')['protocol']
            for stage in original['stages']:
                progress = stage_progress(root, stage)
                line = f"  {stage['name']}: {progress['progress']}/{stage['expected']}"
                if 'flow_loss' in progress:
                    line += f"  flow_loss={progress['flow_loss']:.6f}"
                elif 'loss' in progress:
                    line += f"  loss={progress['loss']:.6f}"
                lines.append(line)
            if state.get('last_gpu_usage'):
                lines.append('显存/利用率：' + json.dumps(state['last_gpu_usage'], ensure_ascii=False))
            if state.get('current_log'):
                lines.append('当前日志：' + state['current_log'])
            if state.get('error'):
                lines.append('错误：' + state['error'])
            if args.watch and sys.stdout.isatty():
                print('\033[2J\033[H', end='')
            print('\n'.join(lines), flush=True)
        if not args.watch:
            return
        time.sleep(args.watch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default=str(DEFAULT_RUN))
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('freeze-amendment')
    runner = sub.add_parser('run')
    runner.add_argument('--resume', action='store_true')
    watcher = sub.add_parser('monitor')
    watcher.add_argument('--watch', type=float, default=0)
    watcher.add_argument('--json', action='store_true')
    args = parser.parse_args()
    if args.action == 'monitor' and (args.watch < 0 or 0 < args.watch < 1):
        parser.error('--watch must be zero or at least one second')
    {'freeze-amendment': freeze_amendment, 'run': run, 'monitor': monitor}[args.action](args)


if __name__ == '__main__':
    main()
