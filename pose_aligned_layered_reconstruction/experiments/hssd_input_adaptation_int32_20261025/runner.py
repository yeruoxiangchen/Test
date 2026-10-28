"""Serial, resumable repair queue. Status/watch use only the standard library."""
import argparse
import fcntl
import os
import signal
import subprocess
import time
from collections import deque
from datetime import datetime, timezone

if __package__:
    from .common import *
else:
    from common import *


def check_gpu():
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not gpu.isdigit():
        raise ValueError('Select one physical GPU with HSSD_INPUT_GPU')
    used = int(subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.used',
                                       '--format=csv,noheader,nounits'], text=True).strip())
    if used > 1024:
        raise RuntimeError(f'GPU {gpu} occupied ({used} MiB); choose another GPU explicitly')


def state(value):
    row = {'time': now(), 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'pid': os.getpid(), **value}
    write(REPAIR/'status.json', row)
    write(OUT/'status.json', row)


def stage(name, module, *args, timeout=21600):
    log = REPAIR/'logs'/f'{name}_{time.time_ns()}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    event(name, phase='starting')
    with log.open('w') as stream:
        child = subprocess.Popen([PYTHON, '-m', MODULE+'.'+module, *args],
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                state({'status': 'running', 'stage': name, 'child_pid': child.pid,
                       'elapsed_seconds': time.monotonic()-start, 'log': str(log)})
                code = child.poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f'{name} exited {code}; see {log}')
                    return
                if time.monotonic()-start > timeout:
                    raise TimeoutError(f'{name} exceeded {timeout} seconds')
                time.sleep(3)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()


def ensure_gate(name, module, *args):
    path = REPAIR/(name+'.json')
    if path.exists():
        check_gate(name)  # A failed or changed gate must never silently pass.
        return
    stage(name, module, *args)
    check_gate(name)


def run(action, lock_fd=None):
    OUT.mkdir(parents=True, exist_ok=True)
    lock = os.fdopen(lock_fd, 'a') if lock_fd is not None else (OUT/'queue.lock').open('a')
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if action != 'summary':
                check_gpu()
            freeze_amendment()
            if action in ['preflight', 'all']:
                ensure_gate('kernel_check', 'check_kernel')
                ensure_gate('decoder_small', 'validate_decoder', 'small')
            if action == 'all':
                ensure_gate('decoder_large', 'validate_decoder', 'large')
                stage('repair_mesh', 'worker', 'mesh')
            if action in ['all', 'summary']:
                stage('repair_summary', 'worker', 'summary', timeout=14400)
            state({'status': 'preflight_passed' if action == 'preflight' else 'all_complete_review_required',
                   'stage': None})
        except BaseException as error:
            previous = read(REPAIR/'status.json') if (REPAIR/'status.json').exists() else {}
            state({**previous, 'time': now(),
                   'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                   'error': str(error) or type(error).__name__})
            raise


def start():
    OUT.mkdir(parents=True, exist_ok=True)
    REPAIR.mkdir(parents=True, exist_ok=True)
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_gpu()
        freeze_amendment()
        log = REPAIR/f'launcher_{time.time_ns()}.log'
        with log.open('w') as stream:
            child = subprocess.Popen([PYTHON, str(HERE/'runner.py'), 'all', '--lock-fd', str(lock.fileno())],
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(lock.fileno(),))
        print(f'Started InputAdapt4 int32 repair on GPU {os.environ["CUDA_VISIBLE_DEVICES"]}, PID {child.pid}; log {log}', flush=True)


def audit():
    """CPU-only handoff after the two interactive numerical checks."""
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ident = freeze_amendment()
        checks = {name: check_gate(name) for name in ['kernel_check', 'decoder_small']}
        snapshot = read(REPAIR/'original_results.json')
        row = {'time': now(), 'runtime_identity': ident, 'checks': checks,
               'original_meshes': len(snapshot['meshes']), 'ss': len(snapshot['supports']),
               'missing': snapshot['missing'], 'long_queue_started': False}
        write(REPAIR/'preflight.json', row)
        state({'status': 'preflight_passed_pending_long_checks', 'stage': None})
        print('Original sources/results verified; kernel and real-decoder checks passed. Long queue not started.', flush=True)


def status():
    s = read(REPAIR/'status.json') if (REPAIR/'status.json').exists() else {'status': 'not_started'}
    print(f'[{now()}] HSSD InputAdapt4 int32 repair GPU {s.get("gpu", "-")}', flush=True)
    age = (datetime.now(timezone.utc)-datetime.fromisoformat(s['time'])).total_seconds() if s.get('time') else None
    print(f'状态：{s["status"]}  阶段：{s.get("stage") or "无活动阶段"}  心跳距今：{round(age) if age is not None else "-"} 秒', flush=True)
    if s['status'] == 'running' and age is not None and age > 30:
        print('心跳已过期，需检查进程；旧running状态不能证明仍在运行。', flush=True)
    for name in ['kernel_check', 'decoder_small', 'decoder_large']:
        path = REPAIR/(name+'.json')
        value = read(path) if path.exists() else {}
        print(f'  {name}: {"通过" if value.get("passed") else "未通过/尚未完成"}', flush=True)
    jobs = base.new_jobs()
    ss = sum((base.case_dir(v, a, o['uid'], seed)/'ss.json').exists() for v, a, o, seed in jobs)
    meshes = sum((base.case_dir(v, a, o['uid'], seed)/'mesh.json').exists() for v, a, o, seed in jobs)
    print(f'新增 SS={ss}/54 mesh={meshes}/54；逻辑结果={meshes+42}/96', flush=True)
    if s.get('error'):
        print('错误：'+s['error'], flush=True)
    if s.get('log'):
        print('日志：'+s['log'], flush=True)
        with Path(s['log']).open(errors='replace') as f:
            print(''.join(deque(f, maxlen=2)).rstrip(), flush=True)
    if s['status'] == 'all_complete_review_required':
        print('计算与汇总已完成，请审阅原输出目录的REPORT.md、figures/和world/。', flush=True)


def stop():
    s = read(REPAIR/'status.json')
    if s.get('status') != 'running':
        print('No running repair queue recorded.'); return
    pid = s['pid']
    cmd = Path(f'/proc/{pid}/cmdline')
    if not cmd.exists():
        state({**s, 'time': now(), 'status': 'interrupted', 'stage': None, 'error': 'Queue process no longer exists'})
        return
    parts = cmd.read_bytes().split(b'\0')
    if str(HERE/'runner.py').encode() not in parts:
        raise ValueError('Refusing to signal an unrelated process')
    os.kill(pid, signal.SIGTERM)
    print(f'Sent SIGTERM to repair queue {pid}; completed artifacts and sampled latents are retained.')


def interrupt(signum, frame):
    raise KeyboardInterrupt(f'Signal {signum}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['audit', 'preflight', 'all', 'start', 'summary', 'status', 'watch', 'stop'])
    p.add_argument('--lock-fd', type=int)
    args = p.parse_args()
    signal.signal(signal.SIGTERM, interrupt)
    if args.action == 'start':
        start()
    elif args.action == 'audit':
        audit()
    elif args.action in ['status', 'watch']:
        try:
            while True:
                status()
                if args.action == 'status': break
                time.sleep(10)
        except KeyboardInterrupt:
            pass
    elif args.action == 'stop':
        stop()
    else:
        run(args.action, args.lock_fd)
