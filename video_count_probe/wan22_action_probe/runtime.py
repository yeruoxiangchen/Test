from __future__ import annotations

import faulthandler
import hashlib
import inspect
import os
import sys
import time
import traceback
from pathlib import Path

from video_count_probe.core import atomic_json, checked_plan, digest, file_sha, is_complete, read_json, writer_lock
from video_count_probe.runtime import check_gpu, versions
from .model import cache_path, model_receipt, read_embedding, text_identity, validate_configs
from .protocol import code_hash, verify_reference


def environment(model_dir, without_model=False):
    print('Importing torch / Diffusers; slow /home I/O can delay this ...', flush=True)
    faulthandler.dump_traceback_later(60, repeat=False)
    try:
        import torch
        import imageio_ffmpeg
        from diffusers import AutoencoderKLWan, WanPipeline
        packages = versions()
        if packages['diffusers'] != '0.35.1':
            raise RuntimeError('This runner was validated with diffusers==0.35.1')
        if 'expand_timesteps' not in inspect.signature(WanPipeline.__init__).parameters:
            raise RuntimeError('WanPipeline lacks Wan2.2 timestep expansion')
        if 'patch_size' not in inspect.signature(AutoencoderKLWan.__init__).parameters:
            raise RuntimeError('AutoencoderKLWan lacks Wan2.2 VAE patch support')
        result = {'python': sys.executable, 'packages': packages, 'source_hash': code_hash(),
                  'ffmpeg': imageio_ffmpeg.get_ffmpeg_exe(), 'cuda_available': torch.cuda.is_available(),
                  'real_checkpoint_inference_tested': False}
        if (Path(model_dir) / 'wan22_metadata_verified.json').exists():
            from transformers import T5TokenizerFast
            from diffusers.pipelines.wan.pipeline_wan import prompt_clean
            from .protocol import NEGATIVE, PROMPTS
            metadata = model_receipt(model_dir, metadata_only=True)
            validate_configs(model_dir)
            tokenizer = T5TokenizerFast.from_pretrained(model_dir, subfolder='tokenizer', local_files_only=True)
            lengths = {name: len(tokenizer(prompt_clean(prompt), truncation=False).input_ids)
                       for name, prompt in {'negative': NEGATIVE, **PROMPTS}.items()}
            if max(lengths.values()) > 512:
                raise ValueError('Prompt would be truncated')
            result.update(metadata_revision=metadata['revision'], prompt_token_lengths=lengths)
        if not without_model:
            receipt = model_receipt(model_dir)
            validate_configs(model_dir)
            if not result['cuda_available']:
                raise RuntimeError('CUDA unavailable in this process')
            result['revision'] = receipt['revision']
        return result
    finally:
        faulthandler.cancel_dump_traceback_later()


def current_identity(plan, model_dir):
    return {'plan_hash': plan['plan_hash'], 'source_hash': code_hash(),
            'model_receipt_hash': digest(model_receipt(model_dir)), 'packages': versions()}


def gpu_preflight(shared_gpu_memory_mib=None):
    if shared_gpu_memory_mib is None:
        return check_gpu(min_free_mib=22500)
    if type(shared_gpu_memory_mib) is not int or shared_gpu_memory_mib <= 0:
        raise ValueError('--shared-gpu-memory-mib must be a positive integer')
    # Explicit opt-in permits sharing, but does not remove the free-memory check.
    headroom_mib = 2048
    gpu = check_gpu(allow_busy=True, min_free_mib=shared_gpu_memory_mib + headroom_mib)
    import torch
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    if (shared_gpu_memory_mib + headroom_mib) * 2**20 > total_bytes:
        raise ValueError('Memory budget plus headroom exceeds device capacity')
    torch.cuda.set_per_process_memory_fraction(shared_gpu_memory_mib * 2**20 / total_bytes, device=0)
    gpu.update(allocator_budget_mib=shared_gpu_memory_mib, required_headroom_mib=headroom_mib,
               cap_scope='PyTorch caching allocator only; CUDA context/library allocations are additional')
    print(f'Shared GPU: PyTorch allocator limit={shared_gpu_memory_mib} MiB; '
          f'require {headroom_mib} MiB extra headroom. Other jobs can still change usage.', flush=True)
    return gpu


def ensure_shared_headroom(gpu):
    if 'allocator_budget_mib' not in gpu:
        return
    import torch
    free_bytes, _ = torch.cuda.mem_get_info(0)
    # This process's reserved pool is already subtracted from global free memory.
    usable_mib = (free_bytes + torch.cuda.memory_reserved(0)) / 2**20
    required_mib = gpu['allocator_budget_mib'] + gpu['required_headroom_mib']
    if usable_mib < required_mib:
        raise RuntimeError(f'Shared GPU headroom shrank: free + our reserved pool = {usable_mib:.0f} MiB; '
                           f'require {required_mib} MiB. Stop here; select a less occupied GPU.')


def load_pipe(model_dir, config):
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline
    from .vae_compat import install
    validate_configs(model_dir)
    vae = AutoencoderKLWan.from_pretrained(model_dir, subfolder='vae', torch_dtype=torch.float32,
                                          local_files_only=True)
    install(vae)
    # Large T5 is deliberately absent. Four CPU embeddings are prepared by encode.
    pipe = WanPipeline.from_pretrained(model_dir, vae=vae, text_encoder=None, tokenizer=None,
                                       torch_dtype=torch.bfloat16, local_files_only=True)
    if not pipe.config.expand_timesteps or pipe.vae_scale_factor_spatial != 16:
        raise RuntimeError('Loaded pipeline does not use Wan2.2 geometry')
    pipe.transformer.eval()
    pipe.vae.eval()
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload(gpu_id=0)
    return pipe


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
            raise RuntimeError('Code/model/environment changed; use a new --root')
        pending = [j for j in plan['jobs'] if not is_complete(run, j, identity)]
        if limit is not None:
            if limit < 1:
                raise ValueError('--limit must be positive')
            pending = pending[:limit]
        if not pending:
            print('All videos already verified complete; no GPU model will load.', flush=True)
            return
        for job in pending:
            if (run / 'errors' / (job['id'] + '.json')).exists() and not retry_failures:
                raise RuntimeError(f'Previous failure for {job["id"]}; inspect errors/, then use --retry-failures')
        import torch
        import numpy as np
        from diffusers.utils import export_to_video
        cases = {c['id']: c for c in plan['cases']}
        text_id = text_identity(model_dir)
        prompts = [config['negative_prompt']] + [cases[j['case_id']]['prompt'] for j in pending]
        embeddings = {}
        for prompt in dict.fromkeys(prompts):
            path = cache_path(root, prompt, text_id)
            if not path.exists():
                raise RuntimeError(f'Missing CPU text cache: {path}; run encode first')
            embeddings[prompt] = read_embedding(path, prompt, text_id)
        # This is a preflight guard, not a claim that the real model peak is known.
        gpu = gpu_preflight(shared_gpu_memory_mib)
        with writer_lock(Path(root) / 'gpu_locks' / (gpu['uuid'] + '.lock')):
            ensure_shared_headroom(gpu)
            atomic_json(identity_file, identity)
            print(f'Loading Wan2.2-5B without T5; {len(pending)} pending videos. Real peak is unverified until smoke passes.', flush=True)
            pipe = load_pipe(model_dir, config)
            (run / 'videos').mkdir(parents=True, exist_ok=True)
            for number, job in enumerate(pending, 1):
                output = None
                started = time.perf_counter()
                path = run / 'videos' / (job['id'] + '.mp4')
                temp = path.with_name(job['id'] + '.partial.mp4')
                print(f'[{number}/{len(pending)}] {job["case_id"]} seed={job["seed"]} steps={config["steps"]}', flush=True)
                try:
                    ensure_shared_headroom(gpu)
                    prompt = cases[job['case_id']]['prompt']
                    generator = torch.Generator(device='cpu').manual_seed(job['seed'])
                    latents = pipe.prepare_latents(1, pipe.transformer.config.in_channels,
                                                  config['height'], config['width'], config['frames'],
                                                  torch.float32, torch.device('cpu'), generator)
                    noise_sha = hashlib.sha256(latents.numpy().tobytes()).hexdigest()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    begin = time.perf_counter()
                    output = pipe(prompt_embeds=embeddings[prompt].to('cuda:0'),
                                  negative_prompt_embeds=embeddings[config['negative_prompt']].to('cuda:0'),
                                  height=config['height'], width=config['width'], num_frames=config['frames'],
                                  num_inference_steps=config['steps'], guidance_scale=config['guidance_scale'],
                                  latents=latents, generator=generator, output_type='np').frames[0]
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - begin
                    expected_shape = (config['frames'], config['height'], config['width'], 3)
                    if tuple(output.shape) != expected_shape or not np.isfinite(output).all():
                        raise ValueError(f'Bad decoded output: {output.shape}; expected {expected_shape}')
                    peak = torch.cuda.max_memory_allocated() / 2**20
                    reserved = torch.cuda.max_memory_reserved() / 2**20
                    export_to_video(output, str(temp), fps=config['fps'], macro_block_size=1)
                    os.replace(temp, path)
                    record = {'status': 'ok', 'job': job, 'identity': identity, 'prompt': prompt,
                              'video': str(path.relative_to(run)), 'video_sha256': file_sha(path),
                              'initial_latent_sha256': noise_sha, 'generation_seconds': elapsed,
                              'total_seconds': time.perf_counter() - started, 'peak_allocated_mib': peak,
                              'peak_reserved_mib': reserved, 'gpu': gpu, 'frames': config['frames'], 'fps': config['fps'],
                              'height': config['height'], 'width': config['width']}
                    atomic_json(run / 'records' / (job['id'] + '.json'), record)
                    (run / 'errors' / (job['id'] + '.json')).unlink(missing_ok=True)
                    print(f'ok {job["id"]} generation={elapsed:.1f}s allocated={peak:.0f}MiB reserved={reserved:.0f}MiB', flush=True)
                except BaseException as exc:
                    temp.unlink(missing_ok=True)
                    atomic_json(run / 'errors' / (job['id'] + '.json'),
                                {'job': job, 'identity': identity, 'gpu': gpu,
                                 'exception': repr(exc), 'traceback': traceback.format_exc()})
                    raise
                finally:
                    del output
