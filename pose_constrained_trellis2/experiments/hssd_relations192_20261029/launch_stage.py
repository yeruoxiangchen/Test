"""Useful GPU stage launcher with a bounded allocator and measured peaks."""
import argparse
import os
import runpy
import sys
from pathlib import Path
from . import common as c


def main():
    p = argparse.ArgumentParser(); p.add_argument('--report', required=True); p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args(); target = a.command[1:] if a.command[0] == '--' else a.command
    allowed = {c.MODULE + '.' + name for name in ('data_worker', 'train_group', 'evaluate_group', 'summary')}
    if target[0] not in allowed: raise ValueError('Unexpected module')
    report = Path(a.report).resolve()
    assert report.is_relative_to(c.OUT / 'memory')
    os.environ.update(c.environment())
    import torch
    torch.cuda.set_device(0); total = torch.cuda.get_device_properties(0).total_memory
    cap = int(os.environ.get('T2_REL_ALLOCATOR_MIB', '16384'))
    if cap <= 0 or cap * 1024**2 >= total: raise ValueError('Invalid allocator cap')
    torch.cuda.set_per_process_memory_fraction(cap * 1024**2 / total, 0)
    torch.cuda.reset_peak_memory_stats(); state = 'complete'; error = None
    try:
        sys.argv = target
        runpy.run_module(target[0], run_name='__main__')
    except BaseException as exc:
        if not isinstance(exc, SystemExit) or exc.code not in (None, 0): state = 'failed'; error = repr(exc)
        raise
    finally:
        c.write(report, {'time': c.now(), 'status': state, 'error': error, 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
                         'allocator_cap_mib': cap, 'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
                         'peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2,
                         'scope': 'PyTorch allocator; excludes other processes and some CUDA-library allocations'})


if __name__ == '__main__': main()
