from __future__ import annotations

import os
from pathlib import Path
import time
import traceback

from video_count_probe.core import atomic_json, checked_plan, digest, file_sha, is_complete, read_json, writer_lock
from video_count_probe.runtime import versions
from video_count_probe.wan22_action_probe.model import cache_path, model_receipt, read_embedding, text_identity
from video_count_probe.wan22_action_probe.review import records
from video_count_probe.wan22_action_probe.runtime import ensure_shared_headroom, gpu_preflight, load_pipe
from video_count_probe.wan22_squat_count.runtime import initial_latents, infer
from .protocol import all_prompts, code_hash, verify_reference


def current_identity(plan, model_dir):
    identity = {'plan_hash': plan['plan_hash'], 'source_hash': code_hash(),
                'model_receipt_hash': digest(model_receipt(model_dir)), 'packages': versions()}
    for key in ('model_receipt_hash', 'packages'):
        if identity[key] != plan['reference_identity'][key]:
            raise ValueError(f'{key} differs from reused reference; do not mix model/environment')
    return identity


def require_smoke(root, plan, identity, conditions, budget):
    path = Path(root) / 'smoke/plan.json'
    if not path.exists():
        raise RuntimeError('Run prepare and smoke before pilot')
    smoke = checked_plan(path)
    if smoke['reference'] != plan['reference']:
        raise RuntimeError('Smoke reference differs from pilot')
    completed = records(path.parent, smoke)
    for condition in conditions:
        found = [r for r in completed.values() if r['job']['condition'] == condition]
        if len(found) != 1:
            raise RuntimeError(f'Run smoke --condition {condition} before pilot; its full geometry is unverified')
        record = found[0]
        for key in ('source_hash', 'model_receipt_hash', 'packages'):
            if record['identity'][key] != identity[key]:
                raise RuntimeError('Smoke code/model/environment differs; use a new root')
        expected = {**plan['conditions'][condition]['config'], 'steps': 2}
        if smoke['conditions'][condition]['config'] != expected or record.get('config') != expected:
            raise RuntimeError('Smoke geometry/sampling configuration mismatch')
        if budget is not None and budget < record['peak_reserved_mib']:
            raise RuntimeError(f'{condition} smoke reserved {record["peak_reserved_mib"]:.0f} MiB; '
                               f'requested budget {budget} MiB is smaller. Select sufficient capacity.')


def pending_jobs(run, plan, identity, condition, limit, retry):
    completed = {j['id']: read_json(run / 'records' / (j['id'] + '.json'))
                 for j in plan['jobs'] if is_complete(run, j, identity)}
    pending = [j for j in plan['jobs'] if j['id'] not in completed and
               (condition == 'all' or j['condition'] == condition)]
    if limit is not None:
        pending = pending[:limit]
    for job in pending:
        path = run / 'errors' / (job['id'] + '.json')
        if path.exists() and read_json(path).get('state') != 'interrupted' and not retry:
            raise RuntimeError(f'Prior failure for {job["id"]}; inspect errors/ then use --retry-failures')
    return completed, pending


def verify_noise(job, sha, reference_records):
    if job['condition'] == 'b5' and sha != reference_records[job['seed']]['initial_latent_sha256']:
        raise ValueError('A/B initial latent mismatch; the same seed must reproduce the same 121-frame noise')


def generate(root, mode, model_dir, condition='all', limit=None, retry=False, budget=None):
    run = Path(root) / mode
    plan = checked_plan(run / 'plan.json')
    _, _, reference_records, _, _ = verify_reference(plan)
    identity = current_identity(plan, model_dir)
    if identity['packages']['diffusers'] != '0.35.1':
        raise RuntimeError('Expected diffusers==0.35.1')
    with writer_lock(run / '.writer.lock'):
        identity_file = run / 'identity.json'
        if identity_file.exists() and read_json(identity_file) != identity:
            raise RuntimeError('Generation code/model/environment changed; use a new output root')
        completed, pending = pending_jobs(run, plan, identity, condition, limit, retry)
        if not pending:
            print(f'No pending videos in {condition}; verified {len(completed)}/{len(plan["jobs"])}. No GPU model loaded.', flush=True)
            return
        if mode == 'pilot':
            require_smoke(root, plan, identity, {j['condition'] for j in pending}, budget)
        import numpy as np
        import torch
        from diffusers.utils import export_to_video
        tid = text_identity(model_dir)
        embeddings = {}
        for prompt in all_prompts(plan):
            path = cache_path(root, prompt, tid)
            if not path.exists():
                raise RuntimeError(f'Missing CPU text cache; run encode first: {path}')
            embeddings[prompt] = read_embedding(path, prompt, tid)
        gpu = gpu_preflight(budget)
        with writer_lock(Path(root) / 'gpu_locks' / (gpu['uuid'] + '.lock')):
            ensure_shared_headroom(gpu)
            atomic_json(identity_file, identity)
            print(f'Loading Wan2.2 without T5 for {len(pending)} pending videos. 8s memory is unverified until smoke completes.', flush=True)
            pipe = load_pipe(model_dir, plan['conditions']['b5']['config'])
            (run / 'videos').mkdir(parents=True, exist_ok=True)
            for index, job in enumerate(pending, 1):
                spec = plan['conditions'][job['condition']]
                config, prompt = spec['config'], spec['prompt']
                path = run / 'videos' / (job['id'] + '.mp4')
                temporary = path.with_suffix('.partial.mp4')
                output = latents = generator = None
                started = time.perf_counter()
                print(f'[{index}/{len(pending)}] {job["condition"]} seed={job["seed"]} '
                      f'frames={config["frames"]} steps={config["steps"]}', flush=True)
                try:
                    ensure_shared_headroom(gpu)
                    generator, latents, noise_sha = initial_latents(pipe, config, job['seed'])
                    verify_noise(job, noise_sha, reference_records)
                    latent_shape = list(latents.shape)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    begin = time.perf_counter()
                    output = infer(pipe, config, embeddings[prompt], embeddings[config['negative_prompt']],
                                   latents, generator, 'cuda:0')
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - begin
                    expected = (config['frames'], config['height'], config['width'], 3)
                    if tuple(output.shape) != expected or not np.isfinite(output).all():
                        raise ValueError(f'Unexpected video shape/values: {output.shape}, expected {expected}')
                    peak, reserved = torch.cuda.max_memory_allocated() / 2**20, torch.cuda.max_memory_reserved() / 2**20
                    export_to_video(output, str(temporary), fps=config['fps'], macro_block_size=1)
                    os.replace(temporary, path)
                    record = {'status': 'ok', 'job': job, 'identity': identity, 'config': config,
                              'prompt': prompt, 'video': str(path.relative_to(run)), 'video_sha256': file_sha(path),
                              'initial_latent_sha256': noise_sha, 'initial_latent_shape': latent_shape,
                              'noise_relation_to_a': 'identical_verified' if job['condition'] == 'b5' else 'different_shape_seed_matched_only',
                              'generation_seconds': elapsed, 'total_seconds': time.perf_counter() - started,
                              'peak_allocated_mib': peak, 'peak_reserved_mib': reserved, 'gpu': gpu,
                              **{k: config[k] for k in ('frames', 'fps', 'height', 'width')}}
                    atomic_json(run / 'records' / (job['id'] + '.json'), record)
                    completed[job['id']] = record
                    (run / 'errors' / (job['id'] + '.json')).unlink(missing_ok=True)
                    print(f'ok {job["id"]} generation={elapsed:.1f}s allocated={peak:.0f}MiB reserved={reserved:.0f}MiB '
                          f'verified_new={len(completed)}/{len(plan["jobs"])}', flush=True)
                except BaseException as exc:
                    temporary.unlink(missing_ok=True)
                    atomic_json(run / 'errors' / (job['id'] + '.json'), {
                        'job': job, 'identity': identity, 'gpu': gpu,
                        'state': 'interrupted' if isinstance(exc, (SystemExit, KeyboardInterrupt)) else 'failed',
                        'exception': repr(exc), 'traceback': traceback.format_exc()})
                    raise
                finally:
                    del output, latents, generator
