"""Run the already frozen mesh cases only on currently idle candidate GPUs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'pose_aligned_layered_reconstruction/outputs/round2_v2_heldout_mesh_20261018'
RANGES = ((0, 6), (6, 11), (11, 16), (16, 32), (32, 48), (48, 64))
MODULE = 'pose_aligned_layered_reconstruction.experiments.round2_v2.heldout_mesh_eval'


class AttachedWorker:
    """Track this experiment's still-running worker after scheduler handoff."""
    def __init__(self, item):
        self.item = item

    def poll(self):
        status = OUT / 'workers' / (self.item['name'] + '_status.json')
        state = json.loads(status.read_text()) if status.exists() else {}
        if state.get('state') == 'complete':
            return 0
        if state.get('state') == 'failed':
            return 1
        command_path = Path(f'/proc/{self.item["pid"]}/cmdline')
        try:
            command = command_path.read_bytes().split(b'\0')
        except FileNotFoundError:
            return 1
        if MODULE.encode() not in command or self.item['name'].encode() not in command:
            return 1
        return None


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def gpu_usage():
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True)
    return {int(parts[0]): (int(parts[1]), int(parts[2]))
            for line in result.splitlines() if (parts := line.split(','))}


def repair_range(args):
    """Resume a failed range without touching the running coordinator/workers."""
    start, end = args.repair_range
    gpu = args.repair_gpu
    if (start, end) not in RANGES or gpu not in range(6):
        raise ValueError('repair must use one frozen range and GPU 0–5')
    payload = json.loads((OUT / 'plan.json').read_text())
    status = json.loads((OUT / 'scheduler_status.json').read_text())
    if status['identity_hash'] != payload['identity_hash']:
        raise ValueError('repair plan mismatch')
    for item in status['active']:
        if max(start, item['range'][0]) < min(end, item['range'][1]):
            raise ValueError('cannot repair a range still owned by an active worker')
    for observation in range(2):
        memory, utilization = gpu_usage()[gpu]
        if memory > 32 or utilization > 5:
            raise ValueError(f'repair GPU {gpu} is occupied')
        if observation == 0:
            time.sleep(20)
    name = f'gpu{gpu}_{start:02d}_{end:02d}_repair_{int(time.time())}'
    repair_dir = OUT / 'repairs'
    repair_dir.mkdir(exist_ok=True)
    log_path = OUT / 'logs' / (name+'.log')
    command = [sys.executable, '-u', '-m', MODULE, 'worker', '--start', str(start),
               '--end', str(end), '--name', name]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    with log_path.open('w') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        record = {'state': 'running', 'identity_hash': payload['identity_hash'], 'range': [start, end],
                  'gpu': gpu, 'pid': process.pid, 'name': name, 'command': command, 'log': str(log_path),
                  'started_at_utc': utc(), 'unchanged_frozen_generation_protocol': True}
        path = repair_dir / (name+'_status.json')
        save(path, record)
        print(json.dumps({'event': 'independent_range_repair_started', **record}), flush=True)
        while process.poll() is None:
            time.sleep(10)
        record.update(exit_status=process.returncode, finished_at_utc=utc())
        if process.returncode == 0:
            for row in payload['plan']['rows']:
                if not start <= row['object_index'] < end:
                    continue
                for case in row['cases']:
                    case_path = OUT / 'records' / f'{row["index"]:03d}_{case["seed"]}_{case["mode"]}.json'
                    result = json.loads(case_path.read_text())
                    if not result['success'] or result['identity_hash'] != payload['identity_hash']:
                        raise ValueError('repair did not complete every fixed case')
            record['state'] = 'complete'
        else:
            record['state'] = 'failed'
        save(path, record)
        print(json.dumps({'event': 'independent_range_repair_finished', **record}), flush=True)
        if process.returncode:
            raise SystemExit(process.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', default='4,5,0,1,2,3')
    parser.add_argument('--max-workers', type=int, default=4)
    parser.add_argument('--resume-from-pid', type=int, help='Replace only this experiment scheduler, retaining live workers.')
    parser.add_argument('--max-retries', type=int, default=1)
    parser.add_argument('--repair-range', type=int, nargs=2)
    parser.add_argument('--repair-gpu', type=int)
    parser.add_argument('--repeatability-gpu', type=int)
    args = parser.parse_args()
    if args.repeatability_gpu is not None:
        if args.repair_range is not None or args.resume_from_pid is not None:
            parser.error('repeatability audit is separate from repair/handoff')
        import runpy
        audit = runpy.run_path(str(Path(__file__).with_name('repeatability_audit.py')))
        audit['run_gpu_audit'](args.repeatability_gpu)
        return
    if args.repair_range is not None:
        if args.repair_gpu is None or args.resume_from_pid is not None:
            parser.error('repair requires --repair-gpu and cannot hand off the coordinator')
        repair_range(args)
        return
    devices = [int(x) for x in args.gpus.split(',')]
    if len(set(devices)) != len(devices) or not set(devices) <= set(range(6)) or not 1 <= args.max_workers <= 6:
        parser.error('only GPUs 0–5 are candidates; do not use occupied GPUs 6/7')
    payload = json.loads((OUT / 'plan.json').read_text())
    plan = payload['plan']
    previous = None
    if args.resume_from_pid:
        if args.resume_from_pid == os.getpid():
            raise ValueError('cannot hand off from self')
        proc = Path(f'/proc/{args.resume_from_pid}')
        command = proc.joinpath('cmdline').read_bytes().split(b'\0')
        if (proc.stat().st_uid != os.getuid() or not any(
            Path(os.fsdecode(arg)).name == Path(__file__).name for arg in command if arg)):
            raise ValueError('handoff target is not the owned experiment scheduler')
        previous = json.loads((OUT / 'scheduler_status.json').read_text())
        if previous['identity_hash'] != payload['identity_hash']:
            raise ValueError('handoff experiment identity mismatch')
        # SIGTERM only the coordinator PID, never the worker process group.
        os.kill(args.resume_from_pid, signal.SIGTERM)
        for _ in range(50):
            if not proc.exists() or proc.joinpath('stat').read_text().split()[2] == 'Z':
                break
            time.sleep(.1)
        else:
            raise RuntimeError('previous scheduler did not stop')
        previous = json.loads((OUT / 'scheduler_status.json').read_text())
        archive = OUT / 'scheduler_history'
        archive.mkdir(exist_ok=True)
        save(archive / f'status_before_handoff_{args.resume_from_pid}.json', previous)
    def case_names(start, end):
        return [f'{row["index"]:03d}_{case["seed"]}_{case["mode"]}' for row in plan['rows']
                if start <= row['object_index'] < end for case in row['cases']]
    def finished(start, end):
        for name in case_names(start, end):
            path = OUT / 'records' / (name + '.json')
            if not path.exists():
                return False
            record = json.loads(path.read_text())
            if not record['success'] or record['identity_hash'] != payload['identity_hash']:
                raise ValueError('invalid existing record')
        return True
    active, completed, failures, history, attempts = {}, [], [], [], {}
    if previous:
        completed = previous['completed']
        history = previous.get('failed_attempts', []) + previous['failures']
        for item in [*completed, *history, *previous['active']]:
            key = tuple(item['range'])
            attempts[key] = attempts.get(key, 0) + 1
        for item in previous['active']:
            if finished(*item['range']):
                completed.append(dict(item, exit_status=0, recovered_complete_at_utc=utc()))
            else:
                active[item['gpu']] = dict(item, process=AttachedWorker(item), log_stream=None)
    owned = {tuple(item['range']) for item in active.values()}
    queue = [item for item in RANGES if item not in owned and not finished(*item)]
    idle_observations = {gpu: 0 for gpu in devices}
    logs = OUT / 'logs'
    logs.mkdir(exist_ok=True)
    started = previous['started_at_utc'] if previous else utc()
    print(json.dumps({'event': 'scheduler_start', 'at_utc': started, 'candidate_gpus': devices,
                      'remaining_ranges': queue, 'identity_hash': payload['identity_hash']}), flush=True)
    while queue or active:
        for gpu in list(active):
            item = active[gpu]
            code = item['process'].poll()
            if code is None:
                continue
            if item['log_stream'] is not None:
                item['log_stream'].close()
            result = {key: value for key, value in item.items() if key not in ('process', 'log_stream')}
            result.update(exit_status=code, finished_at_utc=utc())
            if code != 0 or not finished(*item['range']):
                if attempts[tuple(item['range'])] <= args.max_retries:
                    history.append(result)
                    queue.insert(0, tuple(item['range']))
                else:
                    failures.append(result)
            else:
                completed.append(result)
            del active[gpu]
            print(json.dumps({'event': 'worker_finished', **result}), flush=True)
        usage = gpu_usage()
        for gpu in devices:
            memory, utilization = usage[gpu]
            idle_observations[gpu] = idle_observations[gpu]+1 if memory <= 32 and utilization <= 5 else 0
        for gpu in devices:
            if not queue or len(active) >= args.max_workers:
                break
            if gpu in active or idle_observations[gpu] < 2:
                continue
            # A second fresh query narrows the check/start race with other jobs.
            memory, utilization = gpu_usage()[gpu]
            if memory > 32 or utilization > 5:
                idle_observations[gpu] = 0
                continue
            start, end = queue.pop(0)
            attempts[start, end] = attempts.get((start, end), 0) + 1
            name = f'gpu{gpu}_{start:02d}_{end:02d}_attempt{attempts[start,end]}'
            path = logs / (name+'.log')
            stream = path.open('a')
            command = [sys.executable, '-u', '-m', MODULE, 'worker', '--start', str(start),
                       '--end', str(end), '--name', name]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            active[gpu] = {'gpu': gpu, 'range': [start, end], 'name': name, 'pid': process.pid,
                           'started_at_utc': utc(), 'log': str(path), 'command': command,
                           'idle_memory_mib_at_start': memory, 'idle_utilization_at_start': utilization,
                           'process': process, 'log_stream': stream}
            print(json.dumps({'event': 'worker_started', **{k:v for k,v in active[gpu].items()
                                                          if k not in ('process', 'log_stream')}}), flush=True)
        state = {'state': 'running' if active or queue else 'complete', 'identity_hash': payload['identity_hash'],
                 'started_at_utc': started, 'updated_at_utc': utc(), 'candidate_gpus': devices,
                 'queue': queue, 'completed': completed, 'failures': failures,
                 'failed_attempts': history, 'scheduler_pid': os.getpid(), 'max_workers': args.max_workers,
                 'max_retries': args.max_retries,
                 'active': [{k:v for k,v in item.items() if k not in ('process','log_stream')} for item in active.values()],
                 'last_gpu_usage': usage}
        if failures:
            state['state'] = 'worker_failed'
        save(OUT / 'scheduler_status.json', state)
        if failures and not active and not queue:
            raise SystemExit('worker failed; see scheduler_status.json and logs; no case excluded')
        if queue or active:
            time.sleep(20)
    if not all(finished(*item) for item in RANGES):
        raise RuntimeError('not all fixed cases completed')
    print(json.dumps({'event': 'all_workers_complete', 'at_utc': utc(), 'records': 320}), flush=True)


if __name__ == '__main__':
    main()
