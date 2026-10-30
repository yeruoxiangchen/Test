from __future__ import annotations

import argparse
import faulthandler
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from video_count_probe.cli import stage_log
from video_count_probe.core import atomic_json
from .protocol import MODEL, NEGATIVE, PROMPTS, REFERENCE, ROOT, prepare, under_data


def configure_storage(root):
    root = under_data(root)
    paths = {'TMPDIR': root / 'tmp', 'TORCH_HOME': root / 'cache/torch', 'HF_HOME': root / 'cache/huggingface',
             'HF_HUB_CACHE': root / 'cache/huggingface/hub', 'TORCH_EXTENSIONS_DIR': root / 'cache/torch_extensions',
             'MPLCONFIGDIR': root / 'cache/matplotlib', 'XDG_CACHE_HOME': root / 'cache/xdg'}
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    # tempfile may have been imported before the CLI configured its environment.
    import tempfile
    tempfile.tempdir = str(paths['TMPDIR'])
    return root


def main():
    parser = argparse.ArgumentParser(description='Independent six-video Wan2.2 action capability probe. Long tasks are explicit.')
    parser.add_argument('command', choices=['prepare', 'metadata', 'download', 'encode', 'check', 'test',
                                            'smoke', 'pilot', 'status', 'review', 'report'])
    parser.add_argument('--root', type=Path, default=os.environ.get('WAN22_ACTION_ROOT', ROOT))
    parser.add_argument('--model-dir', type=Path, default=os.environ.get('WAN22_ACTION_MODEL_DIR', MODEL))
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--mode', choices=['smoke', 'pilot'], default='pilot')
    parser.add_argument('--without-model', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--retry-failures', action='store_true')
    parser.add_argument('--shared-gpu-memory-mib', type=int,
                        help='Explicit GPU sharing: cap this process\'s PyTorch allocator; require budget + 2048 MiB free. '
                             'Example: 17408 (17 GiB). CUDA/library allocations are additional.')
    parser.add_argument('--download-attempts', type=int, default=5)
    parser.add_argument('--annotations', type=Path)
    args = parser.parse_args()
    if args.shared_gpu_memory_mib is not None:
        if args.command not in ('smoke', 'pilot') or args.shared_gpu_memory_mib <= 0:
            parser.error('--shared-gpu-memory-mib requires smoke/pilot and a positive integer')
    root = configure_storage(args.root)
    model_dir = under_data(args.model_dir)
    online = args.command in ('metadata', 'download')
    os.environ['HF_HUB_OFFLINE'] = '0' if online else '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '0' if online else '1'
    faulthandler.enable()

    def stop(signum, _frame):
        print(json.dumps({'event': 'termination_signal', 'signal': signum, 'pid': os.getpid()}), flush=True)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    with stage_log(root, args.command):
        print(f'root={root}\nmodel_dir={model_dir}', flush=True)
        if args.command == 'prepare':
            prepare(root, args.reference)
        elif args.command in ('metadata', 'download'):
            from .model import download
            download(model_dir, metadata_only=args.command == 'metadata', attempts=args.download_attempts)
        elif args.command == 'encode':
            from .model import encode
            encode(root, model_dir, [NEGATIVE, *PROMPTS.values()])
        elif args.command == 'check':
            from .runtime import environment
            result = environment(model_dir, args.without_model)
            atomic_json(root / ('environment_without_model.json' if args.without_model else 'environment.json'), result)
            print(json.dumps(result, indent=2), flush=True)
        elif args.command == 'test':
            subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s',
                            str(Path(__file__).parent / 'tests'), '-v'], check=True)
        else:
            from .review import report, review
            if args.command in ('smoke', 'pilot'):
                from .runtime import generate
                # Explicit prepare is required; never replace a frozen plan.
                generate(root, args.command, model_dir, args.limit, args.retry_failures,
                         shared_gpu_memory_mib=args.shared_gpu_memory_mib)
                review(root, args.command)
                report(root, args.command)
                print('Generation finished. Visual assessment is still required; no automatic training or expansion.', flush=True)
            elif args.command == 'review':
                review(root, args.mode)
            else:
                result = report(root, args.mode, args.annotations, write=args.command == 'report')
                print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
