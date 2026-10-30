from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path
import signal
import sys
import unittest

from video_count_probe.cli import stage_log
from video_count_probe.core import atomic_json, checked_plan
from video_count_probe.wan22_action_probe.cli import configure_storage
from .protocol import MODEL, REFERENCE, ROOT, all_prompts, code_hash, prepare, under_data, verify_reference


def check(root, model_dir):
    plan = checked_plan(root / 'pilot/plan.json')
    _, _, reference_records, _, _ = verify_reference(plan)
    print('Checking frozen reference, local configs and prompt lengths on CPU; no GPU model will load.', flush=True)
    faulthandler.dump_traceback_later(60, repeat=False)
    try:
        from diffusers import WanPipeline
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean
        from transformers import T5TokenizerFast
        from video_count_probe.wan22_action_probe.model import validate_configs
        from .runtime import current_identity, initial_latents, verify_noise
        from types import MethodType, SimpleNamespace
        identity = current_identity(plan, model_dir)
        configs = validate_configs(model_dir)
        tokenizer = T5TokenizerFast.from_pretrained(model_dir, subfolder='tokenizer', local_files_only=True)
        lengths = [len(tokenizer(prompt_clean(p), truncation=False).input_ids) for p in all_prompts(plan)]
        if max(lengths) > 512 or identity['packages']['diffusers'] != '0.35.1':
            raise ValueError('Prompt truncation or wrong Diffusers version')
        temporal = configs['vae']['scale_factor_temporal']
        spatial = configs['vae']['scale_factor_spatial']
        shapes = {}
        for condition in ('b5', 'c8'):
            cfg = plan['conditions'][condition]['config']
            if (cfg['frames'] - 1) % temporal or cfg['height'] % (spatial * 2) or cfg['width'] % (spatial * 2):
                raise ValueError('Frame/spatial geometry would be rounded by WanPipeline')
            shapes[condition] = [1, configs['transformer']['in_channels'], (cfg['frames'] - 1) // temporal + 1,
                                 cfg['height'] // spatial, cfg['width'] // spatial]
        # Exercise the installed noise sampler at real geometry; no model weights are needed.
        holder = SimpleNamespace(vae_scale_factor_temporal=temporal, vae_scale_factor_spatial=spatial,
                                 transformer=SimpleNamespace(config=SimpleNamespace(in_channels=shapes['b5'][1])))
        holder.prepare_latents = MethodType(WanPipeline.prepare_latents, holder)
        noise_checks = []
        for seed in plan['seeds']:
            _, tensor, sha = initial_latents(holder, plan['conditions']['b5']['config'], seed)
            verify_noise({'condition': 'b5', 'seed': seed}, sha, reference_records)
            if list(tensor.shape) != shapes['b5']:
                raise ValueError('Unexpected real 5s noise shape')
            noise_checks.append({'seed': seed, 'sha256': sha, 'matches_reference': True})
            del tensor
        _, tensor, _ = initial_latents(holder, plan['conditions']['c8']['config'], plan['seeds'][0])
        if list(tensor.shape) != shapes['c8']:
            raise ValueError('Unexpected real 8s noise shape')
        del tensor
        result = {'python': sys.executable, 'packages': identity['packages'], 'source_hash': code_hash(),
                  'reference_clips_verified': 3, 'prompt_token_lengths': lengths,
                  'expected_initial_latent_shapes': shapes, 'b_c_prompt_identical': True,
                  'actual_cpu_a_b_noise_checks': noise_checks,
                  'gpu_smoke_checked_by_this_command': False, 'note': 'Run smoke to measure both lengths on the selected GPU.'}
        atomic_json(root / 'environment.json', result)
        print(json.dumps(result, indent=2), flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()


def main():
    parser = argparse.ArgumentParser(description='Six-video duration diagnostic: reused A + new B/C; long runs explicit.')
    parser.add_argument('command', choices=['prepare', 'check', 'test', 'encode', 'smoke', 'pilot', 'status', 'review', 'report'])
    parser.add_argument('--root', type=Path, default=os.environ.get('WAN22_DURATION_ROOT', ROOT))
    parser.add_argument('--model-dir', type=Path, default=MODEL)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--mode', choices=['smoke', 'pilot'], default='pilot')
    parser.add_argument('--condition', choices=['all', 'b5', 'c8'], default='all')
    parser.add_argument('--shared-gpu-memory-mib', type=int)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--retry-failures', action='store_true')
    parser.add_argument('--annotations', type=Path)
    args = parser.parse_args()
    generation = args.command in ('smoke', 'pilot')
    if args.shared_gpu_memory_mib is not None and (not generation or args.shared_gpu_memory_mib <= 0):
        parser.error('--shared-gpu-memory-mib requires smoke/pilot and a positive integer')
    if args.limit is not None and (not generation or args.limit < 1):
        parser.error('--limit requires smoke/pilot and a positive integer')
    if args.condition != 'all' and not generation:
        parser.error('--condition applies only to smoke/pilot')
    if args.retry_failures and not generation:
        parser.error('--retry-failures applies only to smoke/pilot')
    root, model_dir = configure_storage(args.root), under_data(args.model_dir)
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    if not generation:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    faulthandler.enable()

    def stop(signum, _frame):
        print(json.dumps({'event': 'termination_signal', 'signal': signal.Signals(signum).name, 'pid': os.getpid()}), flush=True)
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    with stage_log(root, args.command):
        print(f'root={root}\nmodel_dir={model_dir}', flush=True)
        if args.command == 'prepare':
            prepare(root, args.reference)
        elif args.command == 'test':
            suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / 'tests'))
            if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
                raise SystemExit(1)
        elif args.command == 'check':
            check(root, model_dir)
        elif args.command == 'encode':
            from video_count_probe.wan22_action_probe.model import encode
            plan = checked_plan(root / 'pilot/plan.json')
            verify_reference(plan)
            from .runtime import current_identity
            current_identity(plan, model_dir)
            encode(root, model_dir, all_prompts(plan))
        else:
            from .review import report, review
            if generation:
                from .runtime import generate
                generate(root, args.command, model_dir, args.condition, args.limit,
                         args.retry_failures, args.shared_gpu_memory_mib)
                review(root, args.command)
                report(root, args.command)
                print('Invocation finished. Check generated_new/expected_new; explicit visual annotation is required. No training started.', flush=True)
            elif args.command == 'review':
                review(root, args.mode)
            else:
                report(root, args.mode, args.annotations, write=args.command == 'report')


if __name__ == '__main__':
    main()
