"""Single-card bounded train/SS queues; mesh decoding is a separate user action."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from .common import ARMS, EVAL_SEEDS, OUT, PACKAGE, MODULE, PYTHON, STEPS, policy, sources, verify, write, now


def require_smoke():
    receipt = json.loads((OUT / 'smoke.json').read_text())
    if not receipt['passed'] or receipt['sources'] != sources():
        raise ValueError('Missing/changed preflight; run preflight')
    for key in ['base', 'manifest']:
        verify(receipt[key])


def validate_training():
    for domain in ['clean', 'occluded']:
        runs = []
        for arm in ARMS:
            root = OUT / 'train' / f'{domain}_{arm}'
            run, done = (json.loads((root / p).read_text()) for p in ['run.json', 'complete.json'])
            if run['contract'] != policy(domain, arm) or done['contract'] != run['contract'] or done['step'] != STEPS:
                raise ValueError('Missing or changed completed training')
            rows = [json.loads(line) for line in (root / 'history.jsonl').read_text().splitlines()]
            if [r['step'] for r in rows] != list(range(1, STEPS + 1)):
                raise ValueError('Non-contiguous training history')
            runs.append((run, [(r['uid'], r['diffusion_time'], r['learning_rate']) for r in rows]))
        if len({r[0]['initial_residual_sha'] for r in runs}) != 1 or len({r[0]['trainable_parameter_count'] for r in runs}) != 1:
            raise ValueError('Residual initialization/capacity mismatch')
        if any(r[1] != runs[0][1] for r in runs):
            raise ValueError('Unpaired training order/time/schedule')


def run_child(stage_name, command):
    logs = OUT / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / (stage_name + '.log')
    with log.open('a') as stream:
        stream.write(json.dumps({'time': now(), 'command': command}) + '\n')
        stream.flush()
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        try:
            while child.poll() is None:
                write(OUT / 'status.json', {'time': now(), 'status': 'running', 'stage': stage_name,
                    'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'pid': os.getpid(), 'child_pid': child.pid, 'log': str(log)})
                time.sleep(3)
            if child.returncode:
                raise RuntimeError(f'{stage_name} exited {child.returncode}; see {log}')
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait()


def ss_summary():
    from ...mesh_metrics import aggregate_mesh_records
    from .evaluate import identity, verify_record, root_for
    result = {'scope': 'SS64 surface-cell diagnostic; no final mesh or generalization claim', 'domains': {}}
    for domain in ['clean', 'occluded']:
        manifest = json.loads(Path(policy(domain, 'appearance_control')['manifest']['path']).read_text())
        rows = {}
        for arm in ('base', *ARMS):
            root = root_for(domain, arm)
            if json.loads((root / 'ss_complete.json').read_text()) != identity(domain, arm):
                raise ValueError('Incomplete SS sampling')
            rr = []
            for index, source in enumerate(manifest['samples']):
                for seed in EVAL_SEEDS:
                    row = json.loads((root / f'{index:03d}' / f'{seed}_ss.json').read_text())
                    verify_record(row, source, seed)
                    rr.append(row)
            rows[arm] = rr
        pairs = [(a, 'base') for a in ARMS] + [(a, 'appearance_control') for a in ARMS if a != 'appearance_control']
        pairs += [('oracle_cell', 'oracle_point'), ('predicted_cell', 'predicted_point')]
        contrasts = {}
        for arm, control in pairs:
            records = []
            for a, b in zip(rows[arm], rows[control]):
                if (a['uid'], a['seed'], a['gt']['sha256']) != (b['uid'], b['seed'], b['gt']['sha256']):
                    raise ValueError('SS comparison GT differs')
            for name, mode in [(arm, 'candidate'), (control, 'parent')]:
                records += [{**r, 'mode': mode, 'surface': {k: r['occupancy'][k] for k in ['iou', 'precision', 'recall', 'f1']} if r['success'] else {}} for r in rows[name]]
            contrasts[arm + '_vs_' + control] = (aggregate_mesh_records(records, sorted({r['uid'] for r in records}), EVAL_SEEDS)
                if all(r['success'] for r in records) else {'all_successful': False, 'failed': [r for r in records if not r['success']]})
            comparison = contrasts[arm + '_vs_' + control]
            if 'all_meshes_successful' in comparison:
                comparison['all_ss_successful'] = comparison.pop('all_meshes_successful')
        result['domains'][domain] = {'comparisons': contrasts, 'records': rows}
    write(OUT / 'ss_summary.json', result)


def queue(action):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if action == 'preflight':
                run_child('preflight', [PYTHON, '-u', '-m', MODULE + '.smoke'])
                state = 'preflight_complete'
            elif action == 'diagnose':
                run_child('geometry_diagnostic', [PYTHON, '-u', '-m', MODULE + '.diagnose'])
                state = 'geometry_diagnostic_complete_review_required'
            else:
                require_smoke()
                if action in ['train', 'run']:
                    for domain in ['clean', 'occluded']:
                        for arm in ARMS:
                            run_child(f'train_{domain}_{arm}', [PYTHON, '-u', '-m', MODULE + '.train', '--domain', domain, '--arm', arm])
                    validate_training()
                    state = 'training_complete_pending_ss'
                if action in ['ss', 'run', 'mesh']:
                    validate_training()
                    phase = 'mesh' if action == 'mesh' else 'ss'
                    for domain in ['clean', 'occluded']:
                        for arm in ('base', *ARMS):
                            run_child(f'{phase}_{domain}_{arm}', [PYTHON, '-u', '-m', MODULE + '.evaluate', phase, '--domain', domain, '--arm', arm])
                        if phase == 'mesh':
                            run_child(f'mesh_summary_{domain}', [PYTHON, '-u', '-m', MODULE + '.evaluate', 'summary', '--domain', domain])
                    if phase == 'ss':
                        ss_summary()
                    state = 'mesh_complete_review_required' if phase == 'mesh' else 'ss_complete_review_required'
            write(OUT / 'status.json', {'time': now(), 'status': state, 'stage': None, 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')})
        except BaseException as exc:
            path = OUT / 'status.json'
            old = json.loads(path.read_text()) if path.exists() else {}
            write(path, {**old, 'time': now(), 'status': 'failed', 'error': str(exc)})
            raise


def start(action):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if action not in ['preflight', 'diagnose']:
            require_smoke()
        if action in ['ss', 'mesh']:
            validate_training()
    logs = OUT / 'logs'
    logs.mkdir(exist_ok=True)
    log = logs / f'launcher_{action}_{time.time_ns()}.log'
    with log.open('w') as stream:
        process = subprocess.Popen(['bash', str(Path(__file__).with_name('run.sh')), action], cwd=PACKAGE.parent,
            start_new_session=True, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, close_fds=True)
    write(OUT / 'launcher.json', {'time': now(), 'pid': process.pid, 'action': action, 'log': str(log)})
    print(f'Started PID {process.pid}; log: {log}', flush=True)


def status():
    path = OUT / 'status.json'
    state = json.loads(path.read_text()) if path.exists() else {'status': 'not_started'}
    print(f'[{now()}] HSSD CellQuery16 GPU {state.get("gpu", "-")}')
    print(f'状态：{state["status"]}  阶段：{state.get("stage") or "无活动阶段"}  最近心跳：{state.get("time")}')
    if state.get('error'):
        print('错误：' + state['error'])
    for domain in ['clean', 'occluded']:
        for arm in ('base', *ARMS):
            history = OUT / 'train' / f'{domain}_{arm}' / 'history.jsonl'
            last = json.loads(history.read_text().splitlines()[-1]) if history.exists() and history.stat().st_size else {}
            root = OUT / 'evaluations' / f'{domain}_{arm}'
            print(f'  {domain}/{arm}: train={last.get("step",0)}/{0 if arm == "base" else STEPS} SS={len(list(root.glob("[0-9][0-9][0-9]/*_ss.json")))}/48 mesh={len(list(root.glob("[0-9][0-9][0-9]/*_candidate.json")))}/48')
    if state.get('log'):
        print('日志：' + state['log'])


def main():
    actions = ['preflight', 'diagnose', 'train', 'ss', 'mesh', 'run']
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=actions + ['start-' + x for x in actions] + ['status', 'watch'])
    a = p.parse_args()
    if a.action.startswith('start-'):
        start(a.action[6:])
    elif a.action in ['status', 'watch']:
        while True:
            status()
            if a.action == 'status': break
            time.sleep(10)
    else:
        queue(a.action)


if __name__ == '__main__': main()
