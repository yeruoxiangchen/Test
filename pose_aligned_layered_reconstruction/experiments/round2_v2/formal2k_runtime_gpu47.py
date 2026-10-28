"""Migrate the frozen Formal2K queue to data GPU 4 and training GPUs 4,7."""
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
from . import formal2k_runtime_memoryfix as previous
from .formal2k_run import (DEFAULT_RUN, PYTHON, WORKSPACE, utc, read, atomic,
    sha, bind, canonical, journal, verify_receipts, seal_stage, stage_progress)
from .formal2k_runtime_memoryfix import (gpu_usage, admission_ready, worker_command,
    validate_rng_sidecars, verify_stage_complete)

AMENDMENT_NAME = 'runtime_amendments/004_gpu47.json'
GPUS = [4, 7]


def derived_plan(plan, amendment_record):
    return {**plan, 'gpu': '4,7', 'gpus': GPUS, 'runtime_amendment': amendment_record}


def freeze_amendment(args):
    root = Path(args.run_dir).resolve()
    path = root / AMENDMENT_NAME
    lock = (root / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if path.exists():
        raise FileExistsError('Runtime amendment is immutable and already exists')
    plan, identity = previous.verify_protocol(root)
    status = read(root / 'status.json')
    train_names = {stage['name'] for stage in plan['stages'] if stage['kind'] == 'train'}
    if (status['state'] != 'stopped' or status.get('current_stage') != 'prepare_occlusion'
            or status.get('child_pid') or status.get('identity_hash') != identity
            or status.get('runtime_amendment', {}).get('identity_hash') != identity
            or status.get('completed_stages') or any((root / name).exists() for name in train_names)
            or list((root / 'stage_receipts').glob('*.json'))):
        raise ValueError('GPU migration requires the stopped data queue before any training starts')
    parent = read(root / previous.AMENDMENT_NAME)['amendment']
    receipts = [bind(p) for p in sorted((root / 'data/receipts').glob('*.json'))]
    if len(receipts) != 66 or receipts != parent['preserved_data_receipts']:
        raise ValueError('GPU migration must preserve all 66 completed data receipts')
    history_sources = [root / 'status.json', root / 'data/runtime_binding.json',
                       root / 'data/progress.json', root / 'data/status.json', root / 'data/failure.json']
    if args.previous_waiting_status:
        waiting_path = Path(args.previous_waiting_status).resolve(strict=True)
        waiting = read(waiting_path)
        if (waiting.get('state') != 'waiting_for_fixed_gpu' or waiting.get('identity_hash') != identity
                or waiting.get('child_pid') or waiting.get('completed_stages')
                or waiting.get('runtime_amendment', {}).get('identity_hash') != identity):
            raise ValueError('Prior waiting evidence differs from the stopped parent queue')
        history_sources.insert(0, waiting_path)
    log = root / 'logs/prepare_occlusion.log'
    if log.exists():
        history_sources.append(log)
    history = []
    for source in history_sources:
        source = source.resolve(strict=True)
        destination = root / 'runtime_amendments/004_gpu47_history' / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        history.append({**bind(destination), 'original_path': str(source)})
    sources = []
    for filename in ('formal2k_runtime_gpu47.py', 'run_formal2k_gpu47.sh'):
        source = Path(__file__).with_name(filename)
        record = bind(source)
        destination = root / 'runtime_amendments/004_gpu47_sources' / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        sources.append({**record, 'snapshot': str(destination)})
    amendment = {'format': 'palr.formal2k.runtime_amendment.v1', 'created_at_utc': utc(),
        'parent_runtime_amendment': bind(root / previous.AMENDMENT_NAME), 'parent_runtime_identity': identity,
        'authorization': 'User explicitly requested migration and restart of the waiting queue on GPUs 4 and 7.',
        'gpus': GPUS, 'data_gpus': [4], 'world_size': 2, 'grad_accum': 1, 'effective_batch': 2,
        'allocator_cap_mib': None, 'capacity_probe_required': False,
        'unchanged': 'All worker commands, 003 data-memory repair, dataset, models, precision, optimizer, update budgets, admission and resume/completion guards.',
        'queue_history': history, 'preserved_data_receipts': receipts, 'source_code': sources}
    identity = canonical(amendment)
    atomic(path, {'identity_hash': identity, 'amendment': amendment})
    journal(root, 'runtime_amendment_frozen', identity_hash=identity, gpus=GPUS, automatic_launch=False)
    atomic(root / 'status.json', {'state': 'ready_manual_launch', 'identity_hash': identity,
        'gpu': '4,7', 'gpus': GPUS, 'active_gpus': [], 'required_gpus': [],
        'child_pid': None, 'current_stage': None, 'completed_stages': [],
        'queue': [stage['name'] for stage in plan['stages']],
        'runtime_amendment': {**bind(path), 'identity_hash': identity}, 'updated_at_utc': utc(),
        'capacity_probe_required': False})


def verify_protocol(root):
    plan, identity = previous.verify_protocol(root)
    path = root / AMENDMENT_NAME
    envelope = read(path)
    amendment = envelope['amendment']
    if canonical(amendment) != envelope['identity_hash'] or amendment['parent_runtime_identity'] != identity:
        raise ValueError('GPU migration amendment identity mismatch')
    for record in [amendment['parent_runtime_amendment'], *amendment['queue_history'],
                   *amendment['preserved_data_receipts'], *amendment['source_code']]:
        if sha(record['path']) != record['sha256']:
            raise ValueError('GPU migration source changed: ' + record['path'])
        if 'snapshot' in record and sha(record['snapshot']) != record['sha256']:
            raise ValueError('GPU migration snapshot changed: ' + record['snapshot'])
    if (amendment['gpus'] != GPUS or amendment['data_gpus'] != [4]
            or amendment['world_size'] != 2 or amendment['grad_accum'] != 1
            or amendment['effective_batch'] != 2 or amendment['allocator_cap_mib'] is not None
            or amendment['capacity_probe_required'] is not False):
        raise ValueError('Fixed GPU migration policy changed')
    return derived_plan(plan, {**bind(path), 'identity_hash': envelope['identity_hash']}), envelope['identity_hash']


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
    state = {'state': 'starting', 'identity_hash': identity, 'gpu': '4,7', 'gpus': GPUS, 'active_gpus': [], 'required_gpus': [],
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
            required_gpus = [4] if stage['kind'] == 'data' else GPUS
            state['required_gpus'] = required_gpus
            state['active_gpus'] = []
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
                usage = {str(gpu): gpu_usage(gpu) for gpu in required_gpus}
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
            child_env = {**env, 'CUDA_VISIBLE_DEVICES': ','.join(map(str, required_gpus))}
            with log.open('a') as stream:
                process = subprocess.Popen(command, cwd=WORKSPACE, env=child_env, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                state.update(state='running', child_pid=process.pid, active_gpus=list(required_gpus), current_command=command,
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
                    state['last_gpu_usage'] = {str(gpu): gpu_usage(gpu) for gpu in required_gpus}
                    publish()
                    time.sleep(10)
                code = process.returncode
            journal(root, 'stage_exited', name=stage['name'], exit_status=code)
            state.update(child_pid=None, active_gpus=[], last_exit_status=code)
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
                     finished_at_utc=utc(), queue=[], active_gpus=[], required_gpus=[])
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
                     error=str(error), traceback=traceback.format_exc(), child_pid=None, active_gpus=[])
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
            gpus = state.get('gpus', GPUS)
            actual = state.get('active_gpus', []) if state.get('child_pid') else []
            required = state.get('required_gpus', [])
            lines = [f"[{utc()}] Formal2K 双卡训练 GPU {','.join(map(str, gpus))}",
                     f"状态：{state['state']}  阶段：{state.get('current_stage') or '-'}  实际使用：{actual}"]
            if state['state'] == 'waiting_for_fixed_gpu':
                lines.append('等待设备：' + str(required))
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
    freezer = sub.add_parser('freeze-amendment')
    freezer.add_argument('--previous-waiting-status', help='Saved status before stopping the waiting queue')
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
