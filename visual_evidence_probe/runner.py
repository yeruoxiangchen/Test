"""Bounded smoke, eight-object inference queue, and progress reporting."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .common import ROOT, SOURCE, OUT, ARMS, SEEDS, now, read, write, binding, folder, emit, verify


def inspect():
    import torch
    source = json.loads(SOURCE.read_text())
    row = source['samples'][0]
    sample = torch.load(row['sample'], map_location='cpu', weights_only=False)
    print(json.dumps({'sample': row, 'schema': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)}
        if torch.is_tensor(v) else str(v)[:400] for k, v in sample.items()}}, indent=2), flush=True)
    from pose_aligned_reconstruction.current_30k import SS30K_CHECKPOINT
    cp = torch.load(SS30K_CHECKPOINT.path, map_location='cpu', mmap=True, weights_only=False)
    print(json.dumps({'ss_checkpoint': str(SS30K_CHECKPOINT.path), 'step': cp['step'],
        'args': cp['args'], 'summary': {k: cp['model_summary'][k] for k in ('stage', 'sampler_defaults', 'projection')},
        'ema_tensors': len(cp['ema_trainable_state'])},
        default=str, indent=2), flush=True)
    print(json.dumps({'cuda_available': torch.cuda.is_available(), 'gpu_count': torch.cuda.device_count(),
        'torch': torch.__version__}), flush=True)


def smoke():
    import gc
    import numpy as np
    import torch
    from .runtime import setup, load_ss, sample_ss, load_slat, mesh_from_support, parity
    from .data import OBS_KEYS, select_camera_order
    from pose_aligned_layered_reconstruction.dino_context import build_masked_dino_contexts
    from pose_aligned_layered_reconstruction.experiments.hssd_operator_audit_20261025.kernel_policy import FixedNative
    root = ROOT / 'outputs/runtime_smoke_20261025'
    root.mkdir(parents=True, exist_ok=True)
    device = setup()
    source = read(SOURCE)['samples'][0]
    full = torch.load(source['sample'], map_location='cpu', weights_only=False)
    indices = select_camera_order(full)[:2]
    sample = {k: full[k] for k in OBS_KEYS}
    sample.update(intrinsics=full['intrinsics'][indices], extrinsics=full['extrinsics'][indices])
    sample.update(build_masked_dino_contexts(full['masked_visual_patch_features'][indices].float()))
    sample['visual_patch_features'] = sample['masked_visual_patch_features']
    with torch.no_grad(), FixedNative() as policy:
        print('Loading frozen SS30K', flush=True)
        ss = load_ss(device)
        result = {'time': now(), 'source': source, 'parent_equation_parity': parity(ss, sample, 0)}
        coords = sample_ss(ss, sample, 0, 42, root / 'ss.npz')
        del ss
        gc.collect(); torch.cuda.empty_cache()
        print('Loading frozen SLat30K', flush=True)
        slat = load_slat(device)
        a, _ = mesh_from_support(slat, sample, coords, 0, 42, root / 'first.obj')
        b, _ = mesh_from_support(slat, sample, coords, 0, 42, root / 'replay.obj')
        exact = np.array_equal(a.faces, b.faces) and np.array_equal(a.vertices, b.vertices)
        result.update(replay_exact=exact, vertices=len(a.vertices), faces=len(a.faces), kernel_policy=policy.manifest())
        write(root / 'result.json', result)
        if not exact:
            raise ValueError('Identical-input mesh replay failed')
    print(json.dumps({'smoke_passed': True, 'result': str(root / 'result.json')}), flush=True)


def status():
    state = read(OUT / 'status.json') if (OUT / 'status.json').exists() else {'status': 'not_started'}
    count = len(list((OUT / 'objects').glob('*/results/*/*.json')))
    workers = [read(p) for p in sorted((OUT / 'workers').glob('*.json'))]
    print(json.dumps({'time': now(), 'status': state, 'mesh_records': count,
                      'expected': 8 * len(ARMS) * len(SEEDS), 'workers': workers}, ensure_ascii=False, indent=2), flush=True)


def run_queue(gpus):
    import fcntl
    from .data import prepare
    smoke_result = ROOT / 'outputs/runtime_smoke_20261025/result.json'
    if not smoke_result.exists() or not read(smoke_result).get('replay_exact'):
        raise ValueError('Run and pass smoke first')
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = prepare()
        write(OUT / 'runtime_smoke_binding.json', binding(smoke_result))
        waiting, running = list(range(8)), []
        failed = []
        while waiting or running:
            used = {g for _, g, _, _ in running}
            for gpu in gpus:
                if gpu in used or not waiting:
                    continue
                index = waiting.pop(0)
                log = OUT / 'logs' / f'{index:03d}_{time.time_ns()}.log'
                log.parent.mkdir(parents=True, exist_ok=True)
                stream = log.open('w')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                p = subprocess.Popen([sys.executable, '-u', '-m', 'visual_evidence_probe.runner', 'worker',
                    '--index', str(index)], env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                running.append((index, gpu, p, stream))
            remaining = []
            for index, gpu, process, stream in running:
                if process.poll() is None:
                    remaining.append((index, gpu, process, stream))
                else:
                    stream.close()
                    if process.returncode:
                        failed.append({'index': index, 'gpu': gpu, 'exit_code': process.returncode})
            running = remaining
            write(OUT / 'status.json', {'time': now(), 'status': 'running', 'waiting': waiting,
                'active': [{'index': i, 'gpu': g, 'pid': p.pid} for i, g, p, _ in running], 'failed': failed})
            time.sleep(2)
        if failed:
            write(OUT / 'status.json', {'time': now(), 'status': 'failed', 'failed': failed})
            raise RuntimeError(f'{len(failed)} workers failed; retained all logs/artifacts')
        from .report import summarize
        summarize()
        write(OUT / 'status.json', {'time': now(), 'status': 'complete_review_required'})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['inspect', 'smoke', 'prepare', 'run', 'worker', 'status', 'report'])
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--gpus', default='0,1,2,3')
    a = p.parse_args()
    if a.action == 'inspect':
        inspect()
    elif a.action == 'smoke':
        smoke()
    elif a.action == 'prepare':
        from .data import prepare
        prepare()
    elif a.action == 'worker':
        from .worker import run
        run(a.index)
    elif a.action == 'run':
        run_queue([int(g) for g in a.gpus.split(',')])
    elif a.action == 'status':
        status()
    elif a.action == 'report':
        from .report import summarize
        summarize()


if __name__ == '__main__':
    main()
