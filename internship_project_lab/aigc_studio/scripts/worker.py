"""Start one dedicated ComfyUI worker on a verified idle GPU (or CPU for diagnostics)."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def available_gpus():
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True)
    used = set(processes.strip().splitlines())
    result = []
    for line in gpu.strip().splitlines():
        index, uid, free, total, utilization = (p.strip() for p in line.split(','))
        if uid not in used and int(free) >= int(total) - 1024 and int(utilization) <= 5:
            result.append(index)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='auto', help='physical GPU index or auto')
    parser.add_argument('--port', type=int, default=18791)
    parser.add_argument('--wait', action='store_true', help='wait for an idle GPU instead of exiting')
    parser.add_argument('--cpu', action='store_true', help='schema/integration diagnostics, not normal generation')
    parser.add_argument('--disable-cache', action='store_true', help='execute every workflow node for benchmarking')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('port must be between 1024 and 65535')
    root = Path(os.environ['AIGC_DATA_ROOT'])
    vendor = root / 'vendor' / 'ComfyUI'
    python = root / 'comfy-venv' / 'bin' / 'python'
    if not python.exists() or not (vendor / 'main.py').exists():
        parser.error('Run bash scripts/setup_comfy.sh first')
    runtime = root / 'run' / f'comfy-{args.port}'
    runtime.mkdir(parents=True, exist_ok=True)
    # Keep locks inherited by the inference process, including across exec.
    locks = []
    port_lock = (runtime / 'worker.lock').open('w')
    fcntl.flock(port_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    locks.append(port_lock)
    if not args.cpu:
        stable_gpu, samples = None, 0
        while True:
            available = available_gpus()
            choices = available if args.gpu == 'auto' else [g for g in available if g == args.gpu]
            chosen = choices[0] if choices else None
            samples = samples + 1 if chosen is not None and chosen == stable_gpu else (1 if chosen is not None else 0)
            stable_gpu = chosen
            if samples >= 3:
                gpu_lock = (root / 'run' / f'gpu-{chosen}.lock').open('w')
                try:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    gpu_lock.close()
                    samples = 0
                else:
                    locks.append(gpu_lock)
                    os.environ['CUDA_VISIBLE_DEVICES'] = chosen
                    print(f'Starting on idle GPU {chosen}', flush=True)
                    break
            if not chosen and not args.wait:
                parser.exit(2, 'No idle GPU: use --wait to wait, or retry after existing jobs finish.\n')
            print('Waiting for 3 consecutive idle GPU checks…', flush=True)
            time.sleep(10)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for lock in locks:
        os.set_inheritable(lock.fileno(), True)
    for folder in ['input', 'output', 'temp', 'user']:
        (runtime / folder).mkdir(exist_ok=True)
    model_paths = runtime / 'extra_model_paths.yaml'
    model_paths.write_text('studio:\n  base_path: ' + json.dumps(str(root / 'models')) + '\n  is_default: true\n  checkpoints: checkpoints\n  diffusion_models: diffusion_models\n  text_encoders: text_encoders\n  vae: vae\n')
    command = [str(python), '-B', str(vendor / 'main.py'), '--listen', '127.0.0.1', '--port', str(args.port),
               '--extra-model-paths-config', str(model_paths), '--disable-auto-launch', '--disable-all-custom-nodes', '--offline',
               '--input-directory', str(runtime / 'input'), '--output-directory', str(runtime / 'output'),
               '--temp-directory', str(runtime / 'temp'), '--user-directory', str(runtime / 'user')]
    command += ['--cpu'] if args.cpu else ['--lowvram', '--preview-method', 'none']
    if args.disable_cache:
        command.append('--cache-none')
    (runtime / 'worker.json').write_text(json.dumps({'pid': os.getpid(), 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'port': args.port, 'cache_disabled': args.disable_cache, 'started': time.time()}))
    os.chdir(vendor)
    os.execv(str(python), command)


if __name__ == '__main__':
    main()
