"""User-authorized GPU switch without modifying frozen training sources.

The original experiment protocol stays immutable. This supervisor differs from
its frozen predecessor only in GPU admission, allocator wrapper and explicit
runtime-amendment evidence. No model, dataset or optimizer setting changes.
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
from .formal2k_run import (DEFAULT_RUN, PYTHON, WORKSPACE, utc, read, atomic, sha,
    bind, canonical, journal, verify_receipts, verify_stage_complete, seal_stage,
    stage_progress)

AMENDMENT_NAME = 'runtime_amendments/001_gpu7.json'
WORKER_MODULE = 'pose_aligned_layered_reconstruction.experiments.round2_v2.formal2k_gpu_worker'


def freeze_amendment(args):
    root = Path(args.run_dir).resolve()
    plan, identity = base.verify_protocol(root)
    path = root / AMENDMENT_NAME
    if path.exists():
        raise FileExistsError('Runtime amendment is immutable and already exists')
    status = read(root / 'status.json')
    if status['state'] != 'stopped' or status.get('child_pid') or status['completed_stages']:
        raise ValueError('This amendment requires a stopped queue before its first GPU stage')
    if (root / 'data/runtime_binding.json').exists() or (root / 'lari').exists():
        raise ValueError('GPU work already began; this pre-training switch no longer applies')
    sources = []
    for source in (Path(__file__), Path(__file__).with_name('formal2k_gpu_worker.py')):
        record = bind(source)
        destination = root / 'runtime_amendments/001_gpu7_sources' / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        record['snapshot'] = str(destination)
        sources.append(record)
    old_status = root / 'runtime_amendments/gpu4_stopped_status.json'
    atomic(old_status, status)
    amendment = {'format': 'palr.formal2k.runtime_amendment.v1', 'created_at_utc': utc(),
        'parent_protocol': bind(root / 'protocol.json'), 'parent_protocol_identity': identity,
        'authorization': 'User requested: 卡4被占了可以换个卡吗',
        'old_gpu': plan['gpu'], 'gpu': 7, 'world_size': 1, 'concurrent_gpu_children': 1,
        'minimum_free_mib': 18432, 'allocator_cap_mib': 12288,
        'admission': 'Shared GPU; at least 18 GiB free on two checks spaced ten seconds apart.',
        'memory_scope': '12 GiB PyTorch allocator cap; third-party native CUDA allocations are not capped.',
        'on_failure': 'Preserve artifacts and stop queue; never silently increase cap, lower geometry resolution or drop objects.',
        'unchanged': 'All dataset identities, sample order, model sources, initialization, optimizer/update budgets and evaluation design.',
        'parent_runtime_status': bind(old_status), 'source_code': sources}
    atomic(path, {'identity_hash': canonical(amendment), 'amendment': amendment})
    journal(root, 'runtime_amendment_frozen', identity_hash=canonical(amendment),
            parent_protocol_identity=identity, gpu=7, minimum_free_mib=18432, allocator_cap_mib=12288)


def verify_protocol(root):
    plan, identity = base.verify_protocol(root)
    path = root / AMENDMENT_NAME
    envelope = read(path)
    amendment = envelope['amendment']
    if canonical(amendment) != envelope['identity_hash'] or amendment['parent_protocol_identity'] != identity:
        raise ValueError('Runtime amendment identity mismatch')
    for item in [amendment['parent_protocol'], amendment['parent_runtime_status'], *amendment['source_code']]:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Runtime amendment source changed: ' + item['path'])
    if amendment['minimum_free_mib'] < amendment['allocator_cap_mib'] + 4096:
        raise ValueError('Shared GPU admission lost its memory headroom')
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
    return usage['free_mib'] >= plan['minimum_free_mib']


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
                                  'torch_allocator_cap_mib': plan['allocator_cap_mib'], 'shared_gpu': True}}
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
    sub.add_parser('freeze-amendment')
    runner = sub.add_parser('run')
    runner.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    {'freeze-amendment': freeze_amendment, 'run': run}[args.action](args)


if __name__ == '__main__':
    main()
