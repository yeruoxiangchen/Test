"""Single-GPU, process-isolated, resumable data/training/evaluation queue."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from . import common as c


def process_start(pid):
    try: return Path(f'/proc/{int(pid)}/stat').read_text().split(') ', 1)[1].split()[19]
    except (OSError, ValueError): return None


def policy():
    path = c.OUT / 'gpu_policy.json'
    return c.read(path) if path.exists() else {'gpus': '6', 'minimum_free_mib': 18432, 'allocator_cap_mib': 16384}


def choose_gpu():
    cfg = policy(); allowed = {int(x) for x in cfg['gpus'].split(',')}
    assert cfg['minimum_free_mib'] >= cfg['allocator_cap_mib'] + 2048
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
    for line in text.splitlines():
        index, free = [int(x.strip()) for x in line.split(',')]
        if index in allowed and free >= cfg['minimum_free_mib']: return index
    return None


def job(name, module, args, outputs, gpu=True):
    return name, [c.PYTHON, '-B', '-m', c.MODULE + '.' + module, *map(str, args)], outputs, gpu


def preflight_jobs():
    for arm in c.CONFIG['arms']:
        for stage in ('ss', 'shape'):
            folder = c.OUT / 'smoke/training' / arm / stage
            yield job(f'smoke_{arm}_{stage}', 'train_group', ['--arm', arm, '--stage', stage,
                      '--manifest', c.DATA / 'smoke.json', '--output', folder, '--max-steps', 2, '--smoke'],
                      [folder / 'final.pt', folder / 'complete.json'])
    for arm in c.CONFIG['arms']:
        output = c.OUT / 'smoke/evaluations' / arm / 'native_train_interface/seed_42/result.json'
        yield job(f'smoke_sample_{arm}', 'evaluate_group', ['--arm', arm, '--manifest', c.DATA / 'smoke.json',
                  '--group', 'native_train_interface', '--seed', 42, '--smoke'], [output])


def selection_jobs():
    for scene in sorted({r['scene'] for r in c.plan()['anchors']}):
        yield job('select_' + scene, 'data_worker', ['select', '--scene', scene],
                  [c.DATA / 'scene_groups' / f'{scene}.json'], False)
    yield job('assemble', 'data_worker', ['assemble'], [c.DATA / 'groups.json', c.DATA / 'objects.json'], False)


def data_jobs(skip_selection=False, skip_probe=False):
    if not skip_selection: yield from selection_jobs()
    from .support_probe import GROUP, UIDS
    pilot = set(c.plan()['pilot_endpoints'])
    rows = sorted(c.read(c.DATA / 'objects.json')['objects'], key=lambda r: (r['uid'] not in pilot, r['split'], r['uid']))
    for row in rows:
        uid = row['uid']; out = c.codec_folder(uid)
        if skip_probe and uid in UIDS: continue
        if uid in c.plan()['native_reuse']:
            yield job('reuse_' + uid, 'data_worker', ['reuse', '--uid', uid], [c.OUT / 'reuse' / f'{uid}.json'], False)
            continue
        yield from codec_jobs(uid)
    groups = c.read(c.DATA / 'groups.json')
    for group in groups['pilot_groups'] + groups['groups']:
        if skip_probe and group['id'] == GROUP: continue
        outputs = [c.DATA / 'conditions' / group['id'] / uid / name for uid in group['uids'] for name in ('pose.pt', 'condition.json')]
        yield job('conditions_' + group['id'], 'data_worker', ['conditions', '--group', group['id']], outputs)
    yield job('finalize_data', 'data_worker', ['finalize'], [c.OUT / 'data_summary.json'] +
              [c.DATA / f'{s}.json' for s in ('train', 'val', 'test', 'pilot_train', 'pilot_val')], False)


def codec_jobs(uid):
    out = c.codec_folder(uid)
    for phase, resolution, outputs in [
        ('dual', 512, ['dual_512.vxz', 'dual_512.json', 'transform.json', 'target_t2.ply']),
        ('encode', 512, ['shape_512.npz', 'encode_512.json']),
        ('dual', 1024, ['dual_1024.vxz', 'dual_1024.json']),
        ('encode', 1024, ['shape_1024.npz', 'encode_1024.json']),
        ('ss', 512, ['ss.npz', 'ss.json']),
        ('decode', 512, ['decoded_t2.ply', 'decoded_world.ply', 'codec.json', 'codec.png'])]:
        yield job(f'{uid}_{phase}_{resolution}', 'data_worker', ['codec', '--uid', uid, '--phase', phase,
                  '--resolution', resolution], [out / f for f in outputs])


def support_jobs():
    from .support_probe import GROUP, UIDS
    for uid in UIDS: yield from codec_jobs(uid)
    outputs = [c.DATA / 'conditions' / GROUP / uid / name for uid in UIDS for name in ('pose.pt', 'condition.json')]
    yield job('conditions_' + GROUP, 'data_worker', ['conditions', '--group', GROUP], outputs)
    yield job('support_probe_manifest', 'support_probe', ['manifest'], [c.DATA / 'support_probe.json'], False)
    for arm in c.CONFIG['arms']:
        folder = c.OUT / 'support_probe/training' / arm / 'shape'
        yield job('support_probe_' + arm, 'train_group', ['--arm', arm, '--stage', 'shape',
                  '--manifest', c.DATA / 'support_probe.json', '--output', folder, '--max-steps', 2, '--smoke'],
                  [folder / 'final.pt', folder / 'complete.json'])
    yield job('support_probe_verify', 'support_probe', ['verify'], [c.OUT / 'support_probe/verification.json'], False)


def training_jobs():
    if not (c.OUT / 'data_summary.json').exists(): raise RuntimeError('Complete start-data first')
    for arm in c.CONFIG['arms']:
        for stage in ('ss', 'shape'):
            folder = c.OUT / 'training' / arm / stage
            yield job('train_' + arm + '_' + stage, 'train_group', ['--arm', arm, '--stage', stage,
                      '--manifest', c.DATA / 'train.json', '--output', folder], [folder / 'final.pt', folder / 'complete.json'])


def evaluation_jobs(test=False):
    from .summary import evaluation_groups
    for group in evaluation_groups(test):
        for seed in c.CONFIG['evaluation_seeds']:
            for arm in c.CONFIG['arms']:
                manifest = c.DATA / f'{group["split"]}.json'
                out = c.OUT / 'evaluations' / arm / group['id'] / f'seed_{seed}' / 'result.json'
                yield job(f'eval_{group["id"]}_{seed}_{arm}', 'evaluate_group', ['--arm', arm, '--manifest', manifest,
                          '--group', group['id'], '--seed', seed], [out])
    yield job('test_summary' if test else 'summary', 'summary', ['--test'] if test else [],
              [c.OUT / ('test_summary.json' if test else 'summary.json'), c.OUT / ('TEST_REPORT.md' if test else 'REPORT.md')])


def jobs(mode):
    if mode in ('preflight', 'train', 'all'): yield from preflight_jobs()
    if mode in ('support-check', 'data', 'all'): yield from selection_jobs()
    if mode in ('support-check', 'data', 'train', 'all'): yield from support_jobs()
    if mode in ('data', 'all'): yield from data_jobs(skip_selection=True, skip_probe=True)
    if mode in ('train', 'all'): yield from training_jobs()
    if mode in ('eval', 'all'): yield from evaluation_jobs()
    if mode == 'test': yield from evaluation_jobs(True)


def run(mode):
    env = c.environment(); os.environ.update(env); c.OUT.mkdir(parents=True, exist_ok=True)
    with (c.OUT / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = c.identity(); c.immutable(c.OUT / 'runtime.json', identity)
        active, child, gpu, log = None, None, None, None
        base = {'pid': os.getpid(), 'process_start': process_start(os.getpid()), 'mode': mode,
                'runtime_identity': identity['identity'], 'gpu_policy': policy()}
        def status(state, **extra):
            c.write(c.OUT / 'status.json', {**base, 'time': c.now(), 'status': state, 'stage': active,
                                          'gpu': gpu, 'log': str(log) if log else None, **extra})
        def stop(signum, frame): raise KeyboardInterrupt('Stop requested')
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        done = 0
        try:
            for name, args, outputs, needs_gpu in jobs(mode):
                active = name; gpu = None
                receipt = c.OUT / 'stages' / f'{name}.json'
                if receipt.exists():
                    old = c.read(receipt)
                    c.check_receipt_identity(name, old, identity['identity'])
                    assert old['command_identity'] == c.digest(args), 'Stage command changed'
                    for row in old['artifacts']:
                        path = Path(row['path']); assert path.is_file() and path.stat().st_size == row['bytes'], path
                    done += 1; continue
                while needs_gpu and gpu is None:
                    gpu = choose_gpu()
                    if gpu is None: status('waiting_for_gpu', completed_stages=done); time.sleep(10)
                original = args
                memory = None
                if needs_gpu:
                    memory = c.OUT / 'memory' / f'{name}_{time.time_ns()}.json'
                    args = [c.PYTHON, '-B', '-m', c.MODULE + '.launch_stage', '--report', str(memory), '--', *args[3:]]
                log = c.OUT / 'logs' / f'{name}_{time.time_ns()}.log'; log.parent.mkdir(parents=True, exist_ok=True)
                with log.open('w') as stream:
                    stream.write('COMMAND ' + json.dumps(args) + '\n'); stream.flush()
                    child = subprocess.Popen(args, cwd=c.REPO, env={**env, 'CUDA_VISIBLE_DEVICES': str(gpu) if needs_gpu else '',
                         'T2_REL_ALLOCATOR_MIB': str(policy()['allocator_cap_mib'])}, stdout=stream,
                         stderr=subprocess.STDOUT, start_new_session=True, stdin=subprocess.DEVNULL)
                    while child.poll() is None:
                        status('running', completed_stages=done, child_pid=child.pid, child_start=process_start(child.pid)); time.sleep(5)
                    if child.returncode: raise RuntimeError(f'{name} exited {child.returncode}; see {log}')
                child = None
                if name.startswith(('eval_', 'smoke_sample_')):
                    record = c.read(outputs[0])
                    for obj in record['objects']:
                        folder = outputs[0].parent / obj['uid']; outputs.append(folder / 'ss.npz')
                        if obj['success']: outputs += [folder / n for n in ('shape.npz', 'mesh.ply', 'mesh_world.ply')]
                artifacts = []
                for path in outputs:
                    assert path.is_file() and path.stat().st_size > 0, path
                    artifacts.append({'path': str(path), 'bytes': path.stat().st_size})
                c.write(receipt, {'time': c.now(), 'runtime_identity': identity['identity'], 'command_identity': c.digest(original),
                                 'artifacts': artifacts, 'log': str(log), 'memory_report': str(memory) if memory else None})
                done += 1
            active, gpu = None, None
            status({'all': 'all_complete_review_required', 'data': 'data_complete_review_required',
                    'preflight': 'preflight_complete', 'support-check': 'support_check_complete', 'train': 'training_complete_pending_evaluation',
                    'eval': 'evaluation_complete_review_required', 'test': 'test_complete_review_required'}[mode], completed_stages=done)
        except BaseException as exc:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
            status('stopped' if isinstance(exc, KeyboardInterrupt) else 'failed', completed_stages=done, error=str(exc))
            if not isinstance(exc, KeyboardInterrupt): raise


def launch(mode):
    env = c.environment(); c.OUT.mkdir(parents=True, exist_ok=True)
    with (c.OUT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (c.OUT / 'queue.lock').open('a') as running:
            try: fcntl.flock(running, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: raise SystemExit('已有自有队列运行，使用watch。')
        last = c.OUT / 'launcher.json'
        if last.exists():
            old = c.read(last)
            if process_start(old['pid']) == old['process_start']: raise SystemExit('队列已启动。')
        if not (c.OUT / 'plan.json').exists():
            from .prepare import main
            main()
        if (c.OUT / 'runtime.json').exists() and c.read(c.OUT / 'runtime.json') != c.identity():
            raise SystemExit('源码身份已改变；拒绝混用旧回执，请先审查修订。')
        cfg = policy()
        if os.environ.get('T2_REL_GPUS'): cfg['gpus'] = os.environ['T2_REL_GPUS']
        assert all(x.strip().isdigit() for x in cfg['gpus'].split(','))
        c.write(c.OUT / 'gpu_policy.json', cfg)
        with (c.OUT / 'launcher.log').open('a') as stream:
            child = subprocess.Popen([c.PYTHON, '-B', '-m', c.MODULE + '.runner', 'run', '--mode', mode],
                    cwd=c.REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        c.write(last, {'pid': child.pid, 'process_start': process_start(child.pid), 'time': c.now(), 'mode': mode})
        print(f'已启动 {mode} PID={child.pid}；单卡串行，候选GPU={cfg["gpus"]}，PyTorch分配上限16GiB。')


def show():
    file = c.OUT / 'status.json'
    if not file.exists(): print('尚未启动长队列。'); return
    state = c.read(file); alive = process_start(state['pid']) == state['process_start']
    print(f'[{c.now()}] HSSD TRELLIS.2 Relations192\n状态：{state["status"]}  阶段：{state.get("stage") or "无活动阶段"}  GPU：{state.get("gpu")}')
    print(f'模式：{state["mode"]}；已完成阶段：{state.get("completed_stages", 0)}；PID：{state["pid"]}；进程身份：{"匹配" if alive else "已结束或当前进程空间不可见"}')
    print(f'更新：{state["time"]}；候选卡：{state["gpu_policy"]["gpus"]}；启动需空余18GiB，分配上限16GiB。')
    groups = c.DATA / 'objects.json'
    if groups.exists():
        objects = c.read(groups)['objects']
        print(f'原始目标192；含邻居需标签{len(objects)}；编码512={sum((c.codec_folder(r["uid"])/"shape_512.npz").exists() for r in objects)}；codec={sum((c.codec_folder(r["uid"])/"codec.json").exists() for r in objects)}')
    else: print('原始目标192；邻居分组尚在准备。')
    for arm in c.CONFIG['arms']:
        for stage in ('ss', 'shape'):
            progress = c.OUT / 'training' / arm / stage / 'progress.json'
            if progress.exists():
                r = c.read(progress); print(f'  {arm}/{stage}: {r["step"]}/{r["total_steps"]} loss={r["loss"]:.6f} 对象暴露={r["object_exposures"]}')
    if state.get('log'):
        log = Path(state['log']); print('日志：' + str(log))
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 2400)); lines = stream.read().decode(errors='replace').splitlines()
            print('\n'.join(lines[-2:]))
    if state.get('error'): print('错误：' + state['error'])
    if 'complete' in state['status']: print('当前模式已结束，无需继续等待；下一阶段命令见README。')


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['run', 'start', 'status', 'watch', 'stop'])
    p.add_argument('--mode', choices=['preflight', 'support-check', 'data', 'train', 'eval', 'all', 'test'], default='all'); a = p.parse_args()
    if a.action == 'run': run(a.mode)
    elif a.action == 'start': launch(a.mode)
    elif a.action == 'stop':
        state = c.read(c.OUT / 'status.json')
        if process_start(state['pid']) == state['process_start']:
            os.kill(state['pid'], signal.SIGTERM); print('已请求停止自有队列及其计算子进程，保留结果。')
        else: print('队列已结束。')
    elif a.action == 'status': show()
    else:
        while True: show(); time.sleep(10)


if __name__ == '__main__': main()
