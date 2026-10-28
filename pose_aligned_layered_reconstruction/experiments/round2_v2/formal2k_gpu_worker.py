"""Run one unchanged module under a PyTorch CUDA allocator memory ceiling.

Wrapper options precede --module. Everything after the module name is passed
through unchanged, with one optional leading ``--`` separator removed.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-memory-mib', type=int, default=12288,
                        help='PyTorch allocator ceiling in MiB; must precede --module')
    parser.add_argument('--module', required=True, help='Python module to execute')
    end = len(argv)
    for index, value in enumerate(argv):
        if value == '--module':
            end = min(index + 2, len(argv))
            break
        if value.startswith('--module='):
            end = index + 1
            break
    args = parser.parse_args(argv[:end])
    if args.max_memory_mib <= 0:
        parser.error('--max-memory-mib must be positive')
    if not args.module:
        parser.error('--module must not be empty')
    remainder = argv[end:]
    if remainder[:1] == ['--']:
        remainder = remainder[1:]
    args.module_args = remainder
    return args


def main(argv=None):
    args = parse_args(argv)
    # Keep help/parser checks usable without initializing CUDA. Training modules
    # set their own seeds after this wrapper; no seed is changed here.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable to the memory-limited worker')
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    requested_bytes = args.max_memory_mib * 1024 ** 2
    if requested_bytes > properties.total_memory:
        raise ValueError('requested allocator ceiling exceeds the visible GPU memory')
    fraction = requested_bytes / properties.total_memory
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    print(json.dumps({
        'phase': 'gpu_allocator_limit_installed',
        'pid': os.getpid(),
        'device': 'cuda:0',
        'gpu_name': properties.name,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu_total_memory_mib': properties.total_memory / 1024 ** 2,
        'max_memory_mib': args.max_memory_mib,
        'allocator_fraction': fraction,
        'scope': ('PyTorch CUDA caching allocator only; CUDA context and allocations '
                  'by third-party native libraries are not included. This is not '
                  'an OS-enforced total-process GPU-memory cap.'),
        'module': args.module,
    }, ensure_ascii=False), flush=True)
    sys.argv = [args.module, *args.module_args]
    runpy.run_module(args.module, run_name='__main__')


if __name__ == '__main__':
    main()
