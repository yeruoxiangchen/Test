"""Run independent mesh shards in a user service; never train any model."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .evaluate_mesh import DEFAULT_OUTPUT, PACKAGE, load_plan, aggregate, read_completed
from .prepare_official_experiment import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--gpus', default='1,2,3,4,5,6')
    parser.add_argument('--attempt', default='formal_v1')
    args = parser.parse_args()
    gpus = [int(v) for v in args.gpus.split(',')]
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0 or max(gpus) > 7:
        raise ValueError('invalid GPU list')
    if not args.attempt.replace('_', '').isalnum():
        raise ValueError('invalid attempt name')
    output = Path(args.output_dir).resolve()
    plan, identity_hash = load_plan(output)
    # Smoke the first object and the largest predicted-support object before
    # starting the formal fan-out; never select these by a mesh accuracy score.
    for index in (0, 15):
        for case in plan['rows'][index]['cases']:
            read_completed(output/'records'/(case['name']+'.json'), identity_hash)
    jobs = output / 'jobs' / args.attempt
    jobs.mkdir(parents=True, exist_ok=False)
    started = time.time()
    status = dict(state='starting', attempt=args.attempt, identity_hash=identity_hash,
                  expected_records=256, started_epoch=started, jobs=[])
    processes, logs = [], []
    def stopped(signum, frame):
        raise InterruptedError('background mesh supervisor received signal ' + str(signum))
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    try:
        for rank, gpu in enumerate(gpus):
            indices = list(range(rank, len(plan['rows']), len(gpus)))
            name = f'{args.attempt}_gpu{gpu}'
            command = [sys.executable, '-u', '-m', 'pose_aligned_layered_reconstruction.evaluate_mesh', 'worker',
                       '--output_dir', str(output), '--indices', ','.join(map(str,indices)), '--worker_name', name]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       ATTN_BACKEND='flash_attn', SPCONV_ALGO='native', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MAX_JOBS='4',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            env['PATH'] = '/home/zjr/anaconda3/envs/reconviagen/bin:/usr/local/cuda-12/bin:' + env.get('PATH','')
            stream = (jobs / f'gpu{gpu}.log').open('x', buffering=1)
            logs.append(stream)
            process = subprocess.Popen(command, cwd=PACKAGE.parent, env=env, stdout=stream, stderr=subprocess.STDOUT)
            processes.append(process)
            status['jobs'].append(dict(gpu=gpu, pid=process.pid, indices=indices, log=str(jobs/f'gpu{gpu}.log'), command=command))
        status['state'] = 'running'
        last_heartbeat = 0.
        while True:
            codes = [p.poll() for p in processes]
            status.update(return_codes=codes, completed_records=len(list((output/'records').glob('*.json'))),
                          elapsed_seconds=time.time()-started, updated_epoch=time.time())
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError('mesh shard failed: ' + str(codes))
            if time.time()-last_heartbeat >= 30:
                atomic_json(output/'background_status.json', status)
                print(f"mesh progress {status['completed_records']}/256; worker return codes {codes}", flush=True)
                last_heartbeat = time.time()
            if all(code == 0 for code in codes):
                break
            time.sleep(2)
        status['state'] = 'validating_and_aggregating'
        atomic_json(output/'background_status.json', status)
        report = aggregate(output)
        status.update(state='complete', completed_records=report['record_count'], elapsed_seconds=time.time()-started,
                      report=str(output/'aggregate.json'), updated_epoch=time.time())
        atomic_json(output/'background_status.json', status)
        print('All 256 paired meshes validated and aggregated.', flush=True)
    except BaseException as error:
        status.update(state='failed', error=repr(error), elapsed_seconds=time.time()-started, updated_epoch=time.time())
        atomic_json(output/'background_status.json', status)
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for stream in logs:
            stream.close()


if __name__ == '__main__':
    main()
