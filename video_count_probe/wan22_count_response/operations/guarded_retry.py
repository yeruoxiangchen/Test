"""Retry the frozen protocol; monitor competing GPU processes without changing inference.

Only this launcher's own subprocess group can receive termination signals.
This monitor is not a reservation system: unrelated jobs may still enter between polls.
"""
from __future__ import annotations

import csv
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))
from video_count_probe.core import atomic_json, checked_plan, read_json, writer_lock
from video_count_probe.wan22_action_probe.cli import configure_storage
from video_count_probe.wan22_count_response.protocol import ROOT, code_hash

MIN_CAPACITY_MIB = 22500
PRESSURE_CONFIRMATIONS = 3


def event(kind, **data):
    print(json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'event': kind, **data}), flush=True)


def snapshot(device):
    row = next(csv.reader(subprocess.check_output([
        'nvidia-smi', '--id=' + device,
        '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=10).splitlines()))
    index, uuid, used, free, util = [x.strip() for x in row]
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                      '--format=csv,noheader,nounits'], text=True, timeout=10)
    processes = []
    for record in csv.reader(output.splitlines()):
        if len(record) == 3 and record[0].strip() == uuid:
            processes.append({'pid': int(record[1]), 'memory_mib': record[2].strip()})
    return {'physical_gpu': int(index), 'uuid': uuid, 'used_mib': int(used), 'free_mib': int(free),
            'utilization': int(util), 'processes': processes}


def foreign_processes(info, child_pid):
    # CUDA inference runs in the direct child. Permit its own process-group descendants.
    other = []
    for row in info['processes']:
        try:
            owned = row['pid'] == child_pid or os.getpgid(row['pid']) == child_pid
        except ProcessLookupError:
            continue
        if not owned:
            other.append(row)
    return other


def startup_ready(info):
    # Match the frozen runtime's gpu_preflight; a tiny CUDA context is allowed.
    return info['free_mib'] >= MIN_CAPACITY_MIB and info['used_mib'] <= 1000 and info['utilization'] <= 10


def capacity_for_our_process(info, other):
    """Estimate total capacity available to this job, INCLUDING its current pool.

    Free memory alone falls during normal inference and is not a stop criterion.
    NVML process and device samples are not atomic: confirm sustained pressure.
    """
    try:
        external_mib = sum(int(p['memory_mib']) for p in other)
    except (ValueError, TypeError):
        return None
    return info['free_mib'] + info['used_mib'] - external_mib


def stop_owned(child):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def wait_idle(device):
    started, stable, polls = time.monotonic(), 0, 0
    while time.monotonic() - started < 1800:
        info = snapshot(device)
        idle = startup_ready(info)
        stable = stable + 1 if idle else 0
        if polls % 6 == 0:
            event('waiting_for_start_capacity', stable_checks=stable, required_free_mib=MIN_CAPACITY_MIB, **info)
        if stable >= 3:
            return info
        polls += 1
        time.sleep(5)
    raise RuntimeError('GPU did not meet startup capacity requirements; no competing job was terminated')


def run_stage(root, stage, device):
    if stage != 'encode':
        event('gpu_start_capacity_confirmed', **wait_idle(device))
    command = [sys.executable, '-B', '-u', '-m', 'video_count_probe.wan22_count_response.cli',
               stage, '--root', str(root)]
    if stage == 'smoke':
        command += ['--retry-failures']
    child = subprocess.Popen(command, cwd=WORKSPACE, start_new_session=True)
    event('stage_start', stage=stage, child_pid=child.pid)
    previous = None
    pressure_checks = 0
    try:
        while child.poll() is None:
            if stage != 'encode':
                info = snapshot(device)
                other = foreign_processes(info, child.pid)
                capacity = capacity_for_our_process(info, other)
                pressure_checks = pressure_checks + 1 if capacity is not None and capacity < MIN_CAPACITY_MIB else 0
                if pressure_checks:
                    event('gpu_headroom_pressure', stage=stage, other_processes=other,
                          available_job_capacity_mib=capacity, required_capacity_mib=MIN_CAPACITY_MIB,
                          consecutive_checks=pressure_checks, **info)
                    if pressure_checks >= PRESSURE_CONFIRMATIONS:
                        event('gpu_headroom_insufficient', stage=stage,
                              action='terminate_own_child_only_after_sustained_pressure', **info)
                        stop_owned(child)
                        return 75
                current = tuple(row['pid'] for row in info['processes'])
                if current != previous:
                    event('gpu_process_change', stage=stage, available_job_capacity_mib=capacity,
                          other_processes=other, action='continue_unless_sustained_capacity_shortage', **info)
                    previous = current
            time.sleep(2)
        code = child.wait()
        event('stage_complete', stage=stage, exit_code=code)
        return code
    finally:
        stop_owned(child)


def main():
    root = configure_storage(Path(os.environ.get('WAN22_COUNT_RESPONSE_ROOT', ROOT)))
    for variable, directory in {'CUDA_CACHE_PATH': 'cuda', 'TRITON_CACHE_DIR': 'triton'}.items():
        location = root / 'cache' / directory
        location.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(location)
    device = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not device.isdigit():
        raise ValueError('Set exactly one physical GPU index in CUDA_VISIBLE_DEVICES')
    for key, value in {'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'TOKENIZERS_PARALLELISM': 'false',
                       'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}.items():
        os.environ[key] = value
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, _frame: sys.exit(128 + signum))
    with writer_lock(root / '.sequence.lock'):
        # Preserve original OOM and subsequent interrupted attempts before retry.
        archive = root / 'operations/oom_first_attempt'
        archive.mkdir(parents=True, exist_ok=True)
        for path in (root / 'smoke/errors').glob('*.json'):
            if not (archive / path.name).exists():
                shutil.copyfile(path, archive / path.name)
            history = root / 'operations/attempt_history' / (datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + path.name)
            history.parent.mkdir(parents=True, exist_ok=True)
            if not history.exists():
                shutil.copyfile(path, history)
        identity = read_json(root / 'smoke/identity.json')
        if identity['source_hash'] != code_hash():
            raise ValueError('Frozen generation code changed; refusing to mix results')
        atomic_json(root / 'operations/guarded_retry.json', {
            'plan_hash': checked_plan(root / 'pilot/plan.json')['plan_hash'],
            'source_hash': identity['source_hash'], 'physical_gpu': device,
            'logical_cuda_gpu': 0, 'guard_interval_seconds': 2,
            'inference_changes': False, 'policy': 'memory_capacity_v2',
            'min_capacity_mib': MIN_CAPACITY_MIB, 'pressure_confirmations': PRESSURE_CONFIRMATIONS,
            'note': 'Tiny competing processes do not trigger termination. Stop own child only for sustained insufficient capacity.'})
        event('guarded_sequence_started', pid=os.getpid(), physical_gpu=device, logical_cuda_gpu=0)
        for stage in ('encode', 'smoke', 'pilot'):
            code = run_stage(root, stage, device)
            if code:
                return code
    return 0


if __name__ == '__main__':
    try:
        result = main()
    except Exception as exc:
        event('launcher_error', exception=repr(exc))
        raise
    event('sequence_exit', exit_code=result)
    raise SystemExit(result)
