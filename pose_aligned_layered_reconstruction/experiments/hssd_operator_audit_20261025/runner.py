import argparse
import fcntl
import os
import subprocess
import time
from pathlib import Path
from .common import OUT, MODULE, PYTHON, read, write, now


def child(stage, module, args, timeout=600):
    log = OUT / 'logs' / f'{stage}_{time.time_ns()}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with log.open('w') as stream:
        process = subprocess.Popen([PYTHON, '-u', '-m', MODULE + '.' + module, *args],
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                write(OUT / 'status.json', {'time': now(), 'status': 'running', 'stage': stage,
                    'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'pid': os.getpid(), 'child_pid': process.pid,
                    'elapsed_seconds': time.monotonic()-start, 'log': str(log)})
                if time.monotonic()-start > timeout:
                    raise TimeoutError(stage + ' reached its bounded runtime')
                time.sleep(3)
            if process.returncode:
                raise RuntimeError(f'{stage} exited {process.returncode}; see {log}')
        finally:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired: process.kill(); process.wait()


def run(action):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if action == 'audit':
                child('prepare', 'probe', ['prepare'])
                for i in range(3): child(f'block_{i}', 'probe', ['block', '--id', str(i)])
                for i in range(2): child(f'decoder_{i}', 'probe', ['decoder', '--id', str(i)])
                child('summary', 'probe', ['summary'])
            elif action == 'pincheck':
                child('pincheck', 'kernel_check', ['run'], timeout=1200)
            elif action == 'fair-smoke':
                child('fair_smoke', 'fair', ['smoke'], timeout=1200)
            elif action == 'fair':
                child('fair', 'fair', ['run'], timeout=43200)
            write(OUT / 'status.json', {'time': now(), 'status': action + '_complete_review_required',
                'stage': None, 'gpu': os.environ['CUDA_VISIBLE_DEVICES']})
        except BaseException as e:
            old = read(OUT / 'status.json') if (OUT / 'status.json').exists() else {}
            write(OUT / 'status.json', {**old, 'time': now(), 'status': 'failed', 'error': str(e)})
            raise


def start(action):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (OUT / f'launcher_{time.time_ns()}.log').open('w') as stream:
        p = subprocess.Popen(['bash', str(Path(__file__).with_name('run.sh')), action],
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    print('Started', action, 'PID', p.pid, flush=True)


def status():
    s = read(OUT / 'status.json') if (OUT / 'status.json').exists() else {'status': 'not_started'}
    print(f'[{now()}] HSSD OperatorAudit GPU {s.get("gpu", "-")}')
    print('状态：' + s['status'] + '  阶段：' + (s.get('stage') or '无活动阶段'))
    if s.get('error'): print('错误：' + s['error'])
    if s.get('log'):
        print('日志：' + s['log'])
        print('\n'.join(Path(s['log']).read_text(errors='replace').splitlines()[-3:]))
    print('单块进程：' + str(sum((OUT / f'block_{i}.json').exists() for i in range(3))) + '/3')
    print('固定latent解码进程：' + str(sum((OUT / f'decoder_{i}.json').exists() for i in range(2))) + '/2')
    if (OUT / 'fair/status.json').exists():
        print('公平mesh复核：', read(OUT / 'fair/status.json'))


if __name__ == '__main__':
    choices = ['audit', 'pincheck', 'fair-smoke', 'fair']
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=choices+['start-'+a for a in choices]+['status', 'watch'])
    action = p.parse_args().action
    if action.startswith('start-'): start(action[6:])
    elif action in ['status', 'watch']:
        while True:
            status()
            if action == 'status': break
            time.sleep(10)
    else: run(action)
