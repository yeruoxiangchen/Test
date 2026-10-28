"""Serial bounded replay queue; read-only status and explicit resume."""
import argparse
import fcntl
import os
import subprocess
import time
from pathlib import Path

from .common import OUT, MODULE, PYTHON, now, read, write


def child(stage, args, timeout):
    log = OUT / 'logs' / f'{stage}_{time.time_ns()}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open('w') as stream:
        process = subprocess.Popen([PYTHON, '-u', '-m', MODULE + '.replay', *args],
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                write(OUT / 'status.json', {'time': now(), 'status': 'running', 'stage': stage,
                    'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'pid': os.getpid(), 'child_pid': process.pid,
                    'elapsed_seconds': time.monotonic()-started, 'log': str(log)})
                if time.monotonic()-started > timeout:
                    raise TimeoutError(f'{stage} exceeded {timeout}s; partial traces retained')
                time.sleep(3)
            if process.returncode:
                raise RuntimeError(f'{stage} exited {process.returncode}; see {log}')
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            child('prepare', ['prepare'], 180)
            child('same_process', ['worker', '--group', 'same'], 600)
            child('fresh_process', ['worker', '--group', 'fresh'], 600)
            child('summary', ['summary'], 300)
            write(OUT / 'status.json', {'time': now(), 'status': 'replay_complete_review_required', 'stage': None,
                'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')})
        except BaseException as error:
            old = read(OUT / 'status.json') if (OUT / 'status.json').exists() else {}
            write(OUT / 'status.json', {**old, 'time': now(), 'status': 'failed', 'error': str(error)})
            raise


def start():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    log = OUT / f'launcher_{time.time_ns()}.log'
    with log.open('w') as stream:
        p = subprocess.Popen(['bash', str(Path(__file__).with_name('run.sh')), 'run'],
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    print(f'Started bounded replay PID {p.pid}; log: {log}', flush=True)


def status():
    s = read(OUT / 'status.json') if (OUT / 'status.json').exists() else {'status': 'not_started'}
    print(f'[{now()}] HSSD MeshReplay1 GPU {s.get("gpu", "-")}')
    print(f'状态：{s["status"]}  阶段：{s.get("stage") or "无活动阶段"}  最近心跳：{s.get("time", "-")}')
    if s.get('error'):
        print('错误：' + s['error'])
    if s.get('log'):
        print('日志：' + s['log'])
        lines = Path(s['log']).read_text(errors='replace').splitlines()
        for line in lines[-2:]:
            print(line)
    for group in ['same', 'fresh']:
        path = OUT / f'{group}_complete.json'
        if path.exists():
            print(group + ': ' + ', '.join(read(path)['results']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'start', 'status', 'watch'])
    action = parser.parse_args().action
    if action == 'run':
        run()
    elif action == 'start':
        start()
    else:
        while True:
            status()
            if action == 'status':
                break
            time.sleep(10)
