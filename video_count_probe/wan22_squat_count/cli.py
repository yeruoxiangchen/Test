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


def main():
    parser = argparse.ArgumentParser(description='12-video paired squat count-and-stop pilot; inference only.')
    parser.add_argument('command', choices=['prepare', 'test', 'check', 'encode', 'smoke', 'pilot', 'status', 'review', 'report'])
    parser.add_argument('--root', type=Path, default=os.environ.get('WAN22_SQUAT_ROOT', ROOT))
    parser.add_argument('--model-dir', type=Path, default=MODEL)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--mode', choices=['smoke', 'pilot'], default='pilot')
    parser.add_argument('--shared-gpu-memory-mib', type=int)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--retry-failures', action='store_true')
    parser.add_argument('--annotations', type=Path)
    args = parser.parse_args()
    if args.shared_gpu_memory_mib is not None and (args.command not in ('smoke', 'pilot') or args.shared_gpu_memory_mib <= 0):
        parser.error('--shared-gpu-memory-mib requires smoke/pilot and a positive integer')
    if args.limit is not None and (args.command not in ('smoke', 'pilot') or args.limit < 1):
        parser.error('--limit requires smoke/pilot and a positive integer')
    root = configure_storage(args.root)
    model_dir = under_data(args.model_dir)
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
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
        elif args.command in ('check', 'encode'):
            plan = checked_plan(root / 'pilot/plan.json')
            verify_reference(plan)
            if args.command == 'encode':
                from video_count_probe.wan22_action_probe.model import encode
                encode(root, model_dir, all_prompts(plan))
            else:
                print('Checking imports, local model receipt, and prompt lengths on CPU; no GPU model will load.', flush=True)
                faulthandler.dump_traceback_later(60, repeat=False)
                try:
                    import torch
                    from diffusers import AutoencoderKLWan, WanPipeline
                    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
                    from transformers import T5TokenizerFast
                    from video_count_probe.runtime import versions
                    from video_count_probe.wan22_action_probe.model import model_receipt, validate_configs
                    receipt = model_receipt(model_dir)
                    validate_configs(model_dir)
                    tokenizer = T5TokenizerFast.from_pretrained(model_dir, subfolder='tokenizer', local_files_only=True)
                    lengths = [len(tokenizer(prompt_clean(p), truncation=False).input_ids) for p in all_prompts(plan)]
                    if max(lengths) > 512 or versions()['diffusers'] != '0.35.1':
                        raise ValueError('Prompt truncation or wrong Diffusers version')
                    result = {'python': sys.executable, 'packages': versions(), 'revision': receipt['revision'],
                              'prompt_count': len(lengths), 'prompt_token_lengths': lengths,
                              'source_hash': code_hash(), 'cuda_available': torch.cuda.is_available(),
                              'real_new_prompt_gpu_smoke_tested': False}
                    atomic_json(root / 'environment.json', result)
                    print(json.dumps(result, indent=2), flush=True)
                finally:
                    faulthandler.cancel_dump_traceback_later()
        else:
            from .review import report, review
            if args.command in ('smoke', 'pilot'):
                from .runtime import generate
                generate(root, args.command, model_dir, args.limit, args.retry_failures, args.shared_gpu_memory_mib)
                review(root, args.command)
                report(root, args.command)
                print('This invocation finished. Check generated/expected; visual review is still required. No training started.', flush=True)
            elif args.command == 'review':
                review(root, args.mode)
            else:
                result = report(root, args.mode, args.annotations, write=args.command == 'report')
                print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
