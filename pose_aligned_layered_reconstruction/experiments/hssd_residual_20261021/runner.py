"""Finite single-card queues; training and expensive mesh evaluation are separate."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from .common import ARMS, OUT, PACKAGE, MODULE, PYTHON, STEPS, policy, sources, verify, write, now


def require_smoke():
    receipt = json.loads((OUT / 'smoke.json').read_text())
    if not receipt['passed'] or receipt['sources'] != sources():
        raise ValueError('Missing/changed preflight: run preflight before training')
    for key in ['base', 'manifest']:
        verify(receipt[key])


def validate_training():
    require_smoke()
    for domain in ['clean', 'occluded']:
        runs = []
        for arm in ARMS:
            root = OUT / 'train' / f'{domain}_{arm}'
            run = json.loads((root / 'run.json').read_text())
            done = json.loads((root / 'complete.json').read_text())
            if run['contract'] != policy(domain, arm) or done['contract'] != run['contract'] or done['step'] != STEPS:
                raise ValueError('Missing or changed completed training')
            rows = [json.loads(line) for line in (root / 'history.jsonl').read_text().splitlines()]
            if [r['step'] for r in rows] != list(range(1, STEPS + 1)):
                raise ValueError('Non-contiguous training history')
            runs.append((run, [(r['uid'], r['diffusion_time'], r['learning_rate']) for r in rows]))
        if len({r[0]['initial_residual_sha'] for r in runs}) != 1 or len({r[0]['trainable_parameter_count'] for r in runs}) != 1:
            raise ValueError('Residual initialization / capacity mismatch')
        if any(r[1] != runs[0][1] for r in runs):
            raise ValueError('Unpaired training order/noise/schedule')


def run_child(stage, command):
    logs = OUT / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / (stage + '.log')
    with log.open('a') as stream:
        stream.write(json.dumps({'time': now(), 'command': command}) + '\n')
        stream.flush()
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        try:
            while child.poll() is None:
                write(OUT / 'status.json', {'time': now(), 'status': 'running', 'stage': stage,
                    'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'pid': os.getpid(), 'child_pid': child.pid,
                    'log': str(log)})
                time.sleep(3)
            if child.returncode:
                raise RuntimeError(f'{stage} exited {child.returncode}; see {log}')
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait()


def final_comparison():
    from ...mesh_metrics import aggregate_mesh_records
    result = {'scope': 'HSSD16 fitting diagnostic; oracle privileged; no automatic expansion', 'domains': {}}
    for domain in ['clean', 'occluded']:
        summaries = {arm: json.loads((OUT / 'evaluations' / f'{domain}_{arm}' / 'summary.json').read_text()) for arm in ARMS}
        records = []
        for arm, label in [('appearance_control', 'parent'), ('predicted', 'candidate')]:
            folder = OUT / 'evaluations' / f'{domain}_{arm}'
            records += [{**json.loads(p.read_text()), 'mode': label} for p in sorted(folder.glob('[0-9][0-9][0-9]/*_candidate.json'))]
        if len(records) != 64:
            raise ValueError('Incomplete predicted vs appearance control comparison')
        uids = sorted({r['uid'] for r in records})
        gt_by_uid = {}
        for row in records:
            verify(row['gt'])
            gt_by_uid.setdefault(row['uid'], set()).add(row['gt']['sha256'])
        if any(len(v) != 1 for v in gt_by_uid.values()) or len(uids) != 16:
            raise ValueError('Different GT across residual arms')
        paired = aggregate_mesh_records(records, uids) if all(r['success'] for r in records) else {'all_meshes_successful': False}
        result['domains'][domain] = {'arms_vs_frozen_base': summaries, 'predicted_vs_matched_appearance_control': paired}
    write(OUT / 'comparison.json', result)


def queue(action):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if action == 'preflight':
                run_child('preflight', [PYTHON, '-u', '-m', MODULE + '.smoke'])
                state = 'preflight_complete'
            else:
                require_smoke()
                if action == 'evaluate':
                    validate_training()
                for domain in ['clean', 'occluded']:
                    for arm in ARMS:
                        module = '.train' if action == 'train' else '.evaluate'
                        command = [PYTHON, '-u', '-m', MODULE + module, '--domain', domain, '--arm', arm]
                        run_child(f'{action}_{domain}_{arm}', command)
                        if action == 'evaluate':
                            run_child(f'summary_{domain}_{arm}', command + ['--summarize'])
                if action == 'evaluate':
                    final_comparison()
                else:
                    validate_training()
                state = 'evaluation_complete_review_required' if action == 'evaluate' else 'training_complete_pending_evaluation'
            write(OUT / 'status.json', {'time': now(), 'status': state, 'stage': None, 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')})
        except BaseException as exc:
            old = json.loads((OUT / 'status.json').read_text()) if (OUT / 'status.json').exists() else {}
            write(OUT / 'status.json', {**old, 'time': now(), 'status': 'failed', 'error': str(exc)})
            raise


def start(action):
    OUT.mkdir(parents=True, exist_ok=True)
    # Check the same advisory lock used by the worker; never launch a second queue.
    with (OUT / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_smoke()
        if action == 'evaluate':
            validate_training()
    # Release the launch check before the child acquires its lifetime lock.
    # Simultaneous launches are still serialized by the worker's nonblocking lock.
    logs = OUT / 'logs'
    logs.mkdir(exist_ok=True)
    log = logs / f'launcher_{action}_{time.time_ns()}.log'
    with log.open('w') as stream:
        process = subprocess.Popen(['bash', str(Path(__file__).with_name('run.sh')), action],
            cwd=PACKAGE.parent, start_new_session=True, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, close_fds=True)
    write(OUT / 'launcher.json', {'time': now(), 'pid': process.pid, 'action': action, 'log': str(log)})
    print(f'Started PID {process.pid}; log: {log}', flush=True)


def status():
    path = OUT / 'status.json'
    state = json.loads(path.read_text()) if path.exists() else {'status': 'not_started'}
    print(f'[{now()}] HSSD Residual16 GPU {state.get("gpu", "-")}')
    print(f'状态：{state["status"]}  阶段：{state.get("stage")}  最近心跳：{state.get("time")}')
    if state.get('error'):
        print('错误：' + state['error'])
    for domain in ['clean', 'occluded']:
        for arm in ARMS:
            history = OUT / 'train' / f'{domain}_{arm}' / 'history.jsonl'
            last = json.loads(history.read_text().splitlines()[-1]) if history.exists() and history.stat().st_size else {}
            evals = list((OUT / 'evaluations' / f'{domain}_{arm}').glob('[0-9][0-9][0-9]/*_candidate.json'))
            print(f'  {domain}/{arm}: {last.get("step", 0)}/{STEPS} loss={last.get("flow_loss", "-")} mesh={len(evals)}/32')
    if state.get('log'):
        print('日志：' + state['log'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['preflight', 'train', 'evaluate', 'start-train', 'start-eval', 'status', 'watch'])
    a = p.parse_args()
    if a.action.startswith('start-'):
        start('train' if a.action == 'start-train' else 'evaluate')
    elif a.action in ['status', 'watch']:
        while True:
            status()
            if a.action != 'watch':
                break
            time.sleep(10)
    else:
        queue(a.action)


if __name__ == '__main__':
    main()
