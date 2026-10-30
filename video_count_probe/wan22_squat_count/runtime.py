from __future__ import annotations

import hashlib
import os
import time
import traceback
from pathlib import Path

from video_count_probe.core import atomic_json, checked_plan, digest, file_sha, is_complete, read_json, writer_lock
from video_count_probe.runtime import versions
from video_count_probe.wan22_action_probe.model import cache_path, model_receipt, read_embedding, text_identity
from video_count_probe.wan22_action_probe.runtime import ensure_shared_headroom, gpu_preflight, load_pipe
from .protocol import code_hash, verify_reference


def current_identity(plan, model_dir):
    return {'plan_hash': plan['plan_hash'], 'source_hash': code_hash(),
            'model_receipt_hash': digest(model_receipt(model_dir)), 'packages': versions()}


def initial_latents(pipe, config, seed):
    import torch
    generator = torch.Generator(device='cpu').manual_seed(seed)
    latents = pipe.prepare_latents(1, pipe.transformer.config.in_channels,
                                  config['height'], config['width'], config['frames'],
                                  torch.float32, torch.device('cpu'), generator)
    return generator, latents, hashlib.sha256(latents.numpy().tobytes()).hexdigest()


def infer(pipe, config, positive, negative, latents, generator, device):
    import torch
    with torch.no_grad():
        return pipe(prompt_embeds=positive.to(device), negative_prompt_embeds=negative.to(device),
                    height=config['height'], width=config['width'], num_frames=config['frames'],
                    num_inference_steps=config['steps'], guidance_scale=config['guidance_scale'],
                    latents=latents, generator=generator, output_type='np').frames[0]


def verify_paired_noise(job, noise_sha, committed):
    for record in committed.values():
        if record['job']['pair_id'] == job['pair_id'] and record['initial_latent_sha256'] != noise_sha:
            raise ValueError(f'Paired initial latent mismatch: {job["pair_id"]}')


def generate(root, mode, model_dir, limit=None, retry_failures=False, shared_gpu_memory_mib=None):
    run = Path(root) / mode
    plan = checked_plan(run / 'plan.json')
    verify_reference(plan)
    identity = current_identity(plan, model_dir)
    config = plan['config']
    if identity['packages']['diffusers'] != '0.35.1':
        raise RuntimeError('Expected diffusers==0.35.1')
    with writer_lock(run / '.writer.lock'):
        identity_file = run / 'identity.json'
        if identity_file.exists() and read_json(identity_file) != identity:
            raise RuntimeError('Generation code/model/environment changed; use a new --root')
        committed = {j['id']: read_json(run / 'records' / (j['id'] + '.json'))
                     for j in plan['jobs'] if is_complete(run, j, identity)}
        pending = [j for j in plan['jobs'] if j['id'] not in committed]
        if limit is not None:
            if limit < 1:
                raise ValueError('--limit must be positive')
            pending = pending[:limit]
        if not pending:
            print('All videos already verified complete; no GPU or text model will load.', flush=True)
            return
        for job in pending:
            error = run / 'errors' / (job['id'] + '.json')
            if error.exists() and read_json(error).get('state') != 'interrupted' and not retry_failures:
                raise RuntimeError(f'Prior failure for {job["id"]}; inspect errors/ then use --retry-failures')
        import numpy as np
        import torch
        from diffusers.utils import export_to_video
        cases = {c['id']: c for c in plan['cases']}
        text_id = text_identity(model_dir)
        prompts = [config['negative_prompt']] + [cases[j['case_id']]['prompts'][j['method']] for j in pending]
        embeddings = {}
        for prompt in dict.fromkeys(prompts):
            path = cache_path(root, prompt, text_id)
            if not path.exists():
                raise RuntimeError(f'Missing text cache {path}; run encode first')
            embeddings[prompt] = read_embedding(path, prompt, text_id)
        gpu = gpu_preflight(shared_gpu_memory_mib)
        with writer_lock(Path(root) / 'gpu_locks' / (gpu['uuid'] + '.lock')):
            ensure_shared_headroom(gpu)
            atomic_json(identity_file, identity)
            print(f'Loading Wan2.2-5B without T5; {len(pending)} pending videos. Prior capability-run allocated peak ~14.56 GiB; this prompt pair still requires smoke verification.', flush=True)
            pipe = load_pipe(model_dir, config)
            (run / 'videos').mkdir(parents=True, exist_ok=True)
            for index, job in enumerate(pending, 1):
                output = None
                started = time.perf_counter()
                path = run / 'videos' / (job['id'] + '.mp4')
                temporary = path.with_name(job['id'] + '.partial.mp4')
                print(f'[{index}/{len(pending)}] {job["case_id"]} seed={job["seed"]} {job["method"]} steps={config["steps"]}', flush=True)
                try:
                    ensure_shared_headroom(gpu)
                    prompt = cases[job['case_id']]['prompts'][job['method']]
                    generator, latents, noise_sha = initial_latents(pipe, config, job['seed'])
                    verify_paired_noise(job, noise_sha, committed)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    begin = time.perf_counter()
                    output = infer(pipe, config, embeddings[prompt], embeddings[config['negative_prompt']],
                                   latents, generator, 'cuda:0')
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - begin
                    expected = (config['frames'], config['height'], config['width'], 3)
                    if tuple(output.shape) != expected or not np.isfinite(output).all():
                        raise ValueError(f'Invalid output geometry/values: {output.shape}; expected {expected}')
                    peak = torch.cuda.max_memory_allocated() / 2**20
                    reserved = torch.cuda.max_memory_reserved() / 2**20
                    export_to_video(output, str(temporary), fps=config['fps'], macro_block_size=1)
                    os.replace(temporary, path)
                    record = {'status': 'ok', 'job': job, 'identity': identity, 'prompt': prompt,
                              'video': str(path.relative_to(run)), 'video_sha256': file_sha(path),
                              'initial_latent_sha256': noise_sha, 'generation_seconds': elapsed,
                              'total_seconds': time.perf_counter() - started, 'peak_allocated_mib': peak,
                              'peak_reserved_mib': reserved, 'gpu': gpu, 'frames': config['frames'],
                              'fps': config['fps'], 'height': config['height'], 'width': config['width']}
                    atomic_json(run / 'records' / (job['id'] + '.json'), record)
                    committed[job['id']] = record
                    (run / 'errors' / (job['id'] + '.json')).unlink(missing_ok=True)
                    print(f'ok {job["id"]} generation={elapsed:.1f}s allocated={peak:.0f}MiB reserved={reserved:.0f}MiB; verified={len(committed)}/{len(plan["jobs"])}', flush=True)
                except BaseException as exc:
                    temporary.unlink(missing_ok=True)
                    atomic_json(run / 'errors' / (job['id'] + '.json'),
                                {'job': job, 'identity': identity, 'gpu': gpu,
                                 'state': 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
                                 'exception': repr(exc), 'traceback': traceback.format_exc()})
                    raise
                finally:
                    del output
