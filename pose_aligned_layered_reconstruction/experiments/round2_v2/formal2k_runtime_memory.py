"""Explicit memory-policy revision chained to the immutable GPU-7 amendment.

The original experiment protocol stays immutable. This supervisor differs from
its frozen predecessor only in GPU admission, allocator wrapper and explicit
runtime-amendment evidence. No model, dataset or optimizer setting changes.
There are no automatic retries or further allocator-limit adjustments.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback
from . import formal2k_run as base
from . import formal2k_runtime as previous_runtime
from .formal2k_run import (DEFAULT_RUN, PYTHON, WORKSPACE, utc, read, atomic, sha,
    bind, canonical, journal, verify_receipts, verify_stage_complete, seal_stage,
    stage_progress)

AMENDMENT_NAME = 'runtime_amendments/002_gpu7_memory.json'
WORKER_MODULE = 'pose_aligned_layered_reconstruction.experiments.round2_v2.formal2k_gpu_worker'
ALLOCATOR_CAP_MIB = 22528
MINIMUM_FREE_MIB = 23552


def verify_bound_items(value):
    """Check explicitly content-bound probe artifacts, recursively."""
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            if sha(value['path']) != value['sha256']:
                raise ValueError('Memory probe artifact changed: ' + value['path'])
        elif 'sample' in value and 'sha256' in value:
            if sha(value['sample']) != value['sha256']:
                raise ValueError('Memory probe sample changed: ' + value['sample'])
        for item in value.values():
            verify_bound_items(item)
    elif isinstance(value, list):
        for item in value:
            verify_bound_items(item)


def verify_memory_probe(path, expected_data_identity):
    probe = read(path)
    if (probe.get('passed') is not True or probe.get('gpu') != 7
            or probe.get('cuda_visible_devices') != '7'
            or probe.get('allocator_cap_mib') != ALLOCATOR_CAP_MIB
            or probe.get('minimum_free_mib') != MINIMUM_FREE_MIB
            or probe.get('data_plan_identity') != expected_data_identity
            or probe.get('failure_case_completed') is not True
            or probe.get('largest_support_case_completed') is not True
            or probe.get('indices') != [35, 1918]
            or [case.get('index') for case in probe.get('cases', [])] != [35, 1918]
            or not all(case.get('passed') is True for case in probe.get('cases', []))):
        raise ValueError('A successful GPU-7 probe at the fixed 22 GiB allocator cap is required')
    verify_bound_items(probe)
    return probe


def freeze_amendment(args):
    root = Path(args.run_dir).resolve()
    plan, identity = previous_runtime.verify_protocol(root)
    path = root / AMENDMENT_NAME
    if path.exists():
        raise FileExistsError('Runtime amendment is immutable and already exists')
    lock = (root / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if path.exists():
        raise FileExistsError('Runtime amendment was frozen before the lock was acquired')
    status = read(root / 'status.json')
    train_stages = {stage['name'] for stage in plan['stages'] if stage['kind'] == 'train'}
    receipts = verify_receipts(root, plan, identity)
    if (status['state'] != 'failed' or status.get('child_pid') or status.get('gpu') != 7
            or plan['gpu'] != 7 or status.get('identity_hash') != identity
            or status.get('runtime_amendment', {}).get('identity_hash') != plan['runtime_amendment']['identity_hash']):
        raise ValueError('This revision requires the failed GPU-7 queue under amendment 001')
    if (any(item['name'] in train_stages for item in status['completed_stages'])
            or train_stages & receipts.keys()
            or any((root / name).exists() for name in train_stages)):
        raise ValueError('Training already began; this pre-training memory revision no longer applies')
    failure = read(root / 'data/failure.json')
    if 'OutOfMemoryError' not in failure.get('exception', ''):
        raise ValueError('Expected the preserved data-preparation CUDA OOM evidence')
    probe_path = Path(args.memory_probe_report).resolve(strict=True)
    verify_memory_probe(probe_path, plan['data_identity'])
    receipt_paths = sorted((root / 'data/receipts').glob('*.json'))
    if len(receipt_paths) != 6:
        raise ValueError('Expected the six new immutable data receipts from the failed attempt')
    history_sources = [root / 'status.json', root / 'data/failure.json',
                       root / 'data/runtime_binding.json', root / 'data/progress.json',
                       root / 'data/status.json', Path(status['current_log'])]
    # Resolve every required artifact before creating the immutable revision.
    history_sources = [source.resolve(strict=True) for source in history_sources]
    original_receipts = [bind(source) for source in receipt_paths]
    history = []
    for source in [*history_sources, *receipt_paths]:
        destination = root / 'runtime_amendments/002_gpu7_memory_history' / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        history.append({**bind(destination), 'original_path': str(source)})
    sources = []
    for source in (Path(__file__),):
        record = bind(source)
        destination = root / 'runtime_amendments/002_gpu7_memory_sources' / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        record['snapshot'] = str(destination)
        sources.append(record)
    amendment = {'format': 'palr.formal2k.runtime_amendment.v1', 'created_at_utc': utc(),
        'parent_protocol': bind(root / 'protocol.json'), 'parent_protocol_identity': identity,
        'parent_runtime_amendment': bind(root / previous_runtime.AMENDMENT_NAME),
        'parent_runtime_identity': plan['runtime_amendment']['identity_hash'],
        'authorization': 'Continuation of the user-authorized single-GPU training and GPU-switch request.',
        'reason': ('The 12 GiB allocator-capped data worker failed while requesting 1.01 GiB, '
                   'with 11.17 GiB allocated by PyTorch and 9.07 GiB physically free on the GPU. '
                   'A later 16 GiB probe completed that decode with 15150.448 MiB peak PyTorch '
                   'allocation, then stopped because ninja was missing from PATH; it did not '
                   'establish completion of the largest-support case. The largest support is '
                   'about 1.926 times the failing case. This fixed 22 GiB revision therefore '
                   'requires initially idle GPU 7 and a new successful 22 GiB probe with the '
                   'correct environment; the original failure and six receipts are preserved.'),
        'old_gpu': 7, 'gpu': 7, 'world_size': 1, 'concurrent_gpu_children': 1,
        'old_allocator_cap_mib': plan['allocator_cap_mib'],
        'minimum_free_mib': MINIMUM_FREE_MIB, 'allocator_cap_mib': ALLOCATOR_CAP_MIB,
        'maximum_admission_used_mib': 64, 'maximum_admission_utilization_percent': 5,
        'admission': 'Initially idle GPU 7 only: used <=64 MiB, utilization <=5%, and free >=23552 MiB on two checks spaced ten seconds apart.',
        'memory_scope': '22 GiB PyTorch allocator cap for data and every training arm; third-party native CUDA allocations are not capped. GPU admission is not an exclusive device reservation.',
        'on_failure': 'Preserve artifacts and stop queue; never silently increase cap, lower geometry resolution or drop objects.',
        'unchanged': 'All dataset identities, sample order, model sources, initialization, optimizer/update budgets and evaluation design.',
        'memory_probe': bind(probe_path), 'failed_attempt_history': history,
        'preserved_data_receipts': original_receipts, 'source_code': sources,
        'unchanged_worker': bind(Path(__file__).with_name('formal2k_gpu_worker.py'))}
    atomic(path, {'identity_hash': canonical(amendment), 'amendment': amendment})
    journal(root, 'runtime_amendment_frozen', identity_hash=canonical(amendment),
            parent_protocol_identity=identity, gpu=7, minimum_free_mib=MINIMUM_FREE_MIB,
            allocator_cap_mib=ALLOCATOR_CAP_MIB, parent_runtime_identity=amendment['parent_runtime_identity'])


def verify_protocol(root):
    plan, identity = previous_runtime.verify_protocol(root)
    path = root / AMENDMENT_NAME
    envelope = read(path)
    amendment = envelope['amendment']
    if (canonical(amendment) != envelope['identity_hash'] or amendment['parent_protocol_identity'] != identity
            or amendment['parent_runtime_identity'] != plan['runtime_amendment']['identity_hash']):
        raise ValueError('Runtime amendment identity mismatch')
    for item in [amendment['parent_protocol'], amendment['parent_runtime_amendment'],
                 amendment['memory_probe'], amendment['unchanged_worker'],
                 *amendment['failed_attempt_history'], *amendment['preserved_data_receipts'],
                 *amendment['source_code']]:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Runtime amendment source changed: ' + item['path'])
        if 'snapshot' in item and sha(item['snapshot']) != item['sha256']:
            raise ValueError('Runtime source snapshot changed: ' + item['snapshot'])
    if (amendment['gpu'] != 7 or amendment['minimum_free_mib'] != MINIMUM_FREE_MIB
            or amendment['allocator_cap_mib'] != ALLOCATOR_CAP_MIB
            or amendment['maximum_admission_used_mib'] != 64
            or amendment['maximum_admission_utilization_percent'] != 5
            or amendment['world_size'] != 1 or amendment['concurrent_gpu_children'] != 1):
        raise ValueError('The fixed idle-GPU-7 / 22 GiB / 23 GiB-free policy changed')
    verify_memory_probe(amendment['memory_probe']['path'], plan['data_identity'])
    return {**plan, 'gpu': amendment['gpu'],
        'minimum_free_mib': amendment['minimum_free_mib'], 'allocator_cap_mib': amendment['allocator_cap_mib'],
        'runtime_amendment': {'path': str(path), 'sha256': sha(path), 'identity_hash': envelope['identity_hash']}}, identity


def gpu_usage(index):
    result = subprocess.run(['nvidia-smi', f'--id={index}',
        '--query-gpu=memory.used,memory.free,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    used, free, total, utilization = (int(s.strip()) for s in result.stdout.strip().split(','))
    return {'memory_mib': used, 'free_mib': free, 'total_mib': total, 'utilization_percent': utilization}


def admission_ready(usage, plan):
    return (usage['memory_mib'] <= 64 and usage['utilization_percent'] <= 5
            and usage['free_mib'] >= plan['minimum_free_mib'])


def worker_command(command, plan):
    if command[:3] != [PYTHON, '-u', '-m']:
        raise ValueError('Expected the frozen Python module command')
    return [PYTHON, '-u', '-m', WORKER_MODULE,
            '--max-memory-mib', str(plan['allocator_cap_mib']), '--module', command[3], '--', *command[4:]]


def run(args):
    root = Path(args.run_dir).resolve()
    plan, identity = verify_protocol(root)
    lock = (root / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(plan['gpu']), V2_NPROC='1', WORLD_SIZE='1',
               CUDA_HOME='/usr/local/cuda-12.6', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONFAULTHANDLER='1',
               TORCH_CUDA_ARCH_LIST='8.6', SPCONV_ALGO='native',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               MPLCONFIGDIR=str(root / 'matplotlib_cache'))
    env['PATH'] = '/home/zjr/anaconda3/envs/reconviagen/bin:' + env.get('PATH', '')
    for key in ('RANK', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(key, None)
    state = {'state': 'starting', 'identity_hash': identity, 'gpu': plan['gpu'],
             'supervisor_pid': os.getpid(), 'started_at_utc': utc(), 'completed_stages': [],
             'queue': [s['name'] for s in plan['stages']], 'current_stage': None, 'child_pid': None,
             'runtime_amendment': plan['runtime_amendment'],
             'admission_policy': {'minimum_free_mib': plan['minimum_free_mib'],
                                  'torch_allocator_cap_mib': plan['allocator_cap_mib'], 'shared_gpu': False,
                                  'maximum_used_mib': 64, 'maximum_utilization_percent': 5,
                                  'exclusive_reservation': False}}
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
                resume_checkpoint = checkpoint
            idle = 0
            while idle < 2:
                if stop[0] or (root / 'STOP').exists():
                    raise InterruptedError('Stop requested before GPU stage')
                usage = gpu_usage(plan['gpu'])
                idle = idle + 1 if admission_ready(usage, plan) else 0
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
                                 'previous_run_config': bind(folder / 'run_config.json')}
                launch_path = root / 'resume_sources' / f"{stage['name']}_{time.time_ns()}.json"
                atomic(launch_path, {**resume_source, 'previous_config_content': read(folder / 'run_config.json'),
                                    'protocol_identity': identity, 'recorded_at_utc': utc()})
                journal(root, 'resume_source_bound', name=stage['name'], record=bind(launch_path), **resume_source)
            command = worker_command(command, plan)
            with log.open('a') as stream:
                process = subprocess.Popen(command, cwd=WORKSPACE, env=env, stdout=stream,
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
                    state['last_gpu_usage'] = gpu_usage(plan['gpu'])
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

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default=str(DEFAULT_RUN))
    sub = parser.add_subparsers(dest='action', required=True)
    freezer = sub.add_parser('freeze-amendment')
    freezer.add_argument('--memory-probe-report', required=True)
    runner = sub.add_parser('run')
    runner.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    {'freeze-amendment': freeze_amendment, 'run': run}[args.action](args)


if __name__ == '__main__':
    main()
