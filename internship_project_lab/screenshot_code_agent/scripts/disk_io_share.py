#!/usr/bin/env python3
"""Read-only per-disk block READ attribution over one bounded BCC sampling window.

Uses the installed biotop probe with lengths saved at request start and atomic
counters. Does not modify the system tool, process priorities, or running jobs.
"""

import argparse
import ast
import json
import os
import pwd
import re
import time
from collections import defaultdict
from pathlib import Path


def probe_source(path=Path('/usr/sbin/biotop-bpfcc')):
    tree = ast.parse(path.read_text())
    values = [node.value.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == 'bpf_text' for target in node.targets)
              and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)]
    if len(values) != 1:
        raise ValueError('Unrecognized installed biotop source; no probes attached')
    source = values[0]
    changes = {
        'BPF_HASH(start, struct request *);':
            'BPF_HASH(start, struct request *);\nBPF_HASH(saved_bytes, struct request *, u64);',
        'start.update(&req, &ts);':
            'u64 length = req->__data_len;\n    saved_bytes.update(&req, &length);\n    start.update(&req, &ts);',
        'valp->us += delta_us;': '__sync_fetch_and_add(&valp->us, delta_us);',
        'valp->bytes += req->__data_len;':
            'u64 *length = saved_bytes.lookup(&req);\n        if (length) __sync_fetch_and_add(&valp->bytes, *length);',
        'valp->io++;': '__sync_fetch_and_add(&valp->io, 1);',
        'start.delete(&req);': 'start.delete(&req);\n    saved_bytes.delete(&req);',
        'u64 ts;': 'u64 ts;\n    if ((req->cmd_flags & REQ_OP_MASK) != REQ_OP_READ && '
                   '(req->cmd_flags & REQ_OP_MASK) != REQ_OP_WRITE) return 0;',
        'return 0;    // missed tracing issue':
            'whobyreq.delete(&req);\n        saved_bytes.delete(&req);\n        return 0; // no tracked start',
    }
    for old, new in changes.items():
        if source.count(old) != 1:
            raise ValueError('Unsupported installed probe layout; use a compatible BCC version')
        source = source.replace(old, new)
    return source


def disk_identity(name, sys_root=Path('/sys/class/block')):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('Use a disk name such as sdb')
    path = sys_root / name
    if (path / 'partition').exists():
        raise ValueError('Use the whole device (sdb), not a partition (sdb1)')
    major, minor = map(int, (path / 'dev').read_text().strip().split(':'))
    return path, major, minor


def read_disk_stats(path):
    values = list(map(int, (path / 'stat').read_text().split()))
    return {'read_bytes': values[2] * 512, 'write_bytes': values[6] * 512}


def summarize(records, major, minor, elapsed, disk_delta):
    rows = {}
    for item in records:
        if (item['major'], item['minor']) != (major, minor):
            continue
        row = rows.setdefault(item['pid'], {'pid': item['pid'], 'names': set(),
                             'read_bytes': 0, 'read_ios': 0, 'read_us': 0, 'write_bytes': 0})
        row['names'].add(item['comm'])
        if item['write']:
            row['write_bytes'] += item['bytes']
        else:
            row['read_bytes'] += item['bytes']
            row['read_ios'] += item['ios']
            row['read_us'] += item['us']
    total_bytes = sum(row['read_bytes'] for row in rows.values())
    total_ios = sum(row['read_ios'] for row in rows.values())
    if total_ios and total_bytes == 0:
        raise RuntimeError('Read requests observed but byte counter is zero: percentages are INVALID')
    for row in rows.values():
        row['names'] = sorted(row['names'])
        row['read_share_pct'] = 100 * row['read_bytes'] / total_bytes if total_bytes else 0
        row['read_io_share_pct'] = 100 * row['read_ios'] / total_ios if total_ios else 0
        row['read_mib_s'] = row['read_bytes'] / 2**20 / elapsed
        row['read_issue_to_done_ms'] = row['read_us'] / 1000 / row['read_ios'] if row['read_ios'] else None
    return {'elapsed_seconds': elapsed, 'read_bytes': total_bytes, 'read_ios': total_ios,
            'diskstats_read_bytes': disk_delta['read_bytes'],
            'trace_to_diskstats_read_pct': 100 * total_bytes / disk_delta['read_bytes']
                if disk_delta['read_bytes'] > 0 else None,
            'rows': sorted(rows.values(), key=lambda row: row['read_bytes'], reverse=True),
            'scope': 'Share of attributed READ bytes/requests on this disk; includes PID 0. '
                     'Not disk utilization or causal blame. Diskstats comparison is approximate due to '
                     'in-flight requests, merging/requeues and missing events.'}


def process_user(pid):
    if pid == 0:
        return 'unknown'
    try:
        status = (Path('/proc') / str(pid) / 'status').read_text()
        uid = int(next(line for line in status.splitlines() if line.startswith('Uid:')).split()[1])
        try:
            return pwd.getpwuid(uid).pw_name
        except KeyError:
            return str(uid)
    except (OSError, StopIteration):
        return 'exited/unknown'


def collect(seconds, major, minor, disk, source):
    from bcc import BPF

    bpf = BPF(text=source)
    attached = []
    hooks = [('blk_account_io_start', 'trace_pid_start'), ('blk_mq_start_request', 'trace_req_start'),
             ('blk_account_io_done', 'trace_req_completion')]
    if BPF.get_kprobe_functions(b'blk_start_request'):
        hooks.insert(1, ('blk_start_request', 'trace_req_start'))
    try:
        for event, function in hooks:
            bpf.attach_kprobe(event=event, fn_name=function)
            attached.append(event)
        before = read_disk_stats(disk)
        started = time.monotonic()
        print(f'Sampling {disk.name} for {seconds:g}s; no workload is started...', flush=True)
        try:
            time.sleep(seconds)
        except KeyboardInterrupt:
            pass
        # Freeze all probe updates before computing totals and percentages.
        for event in attached:
            bpf.detach_kprobe(event=event)
        attached.clear()
        elapsed = max(time.monotonic() - started, .001)
        after = read_disk_stats(disk)
        records = [{'pid': int(key.pid), 'comm': bytes(key.name).split(b'\0')[0].decode(errors='replace'),
                    'major': int(key.major), 'minor': int(key.minor), 'write': bool(key.rwflag),
                    'bytes': int(value.bytes), 'ios': int(value.io), 'us': int(value.us)}
                   for key, value in bpf['counts'].items()]
        return summarize(records, major, minor, elapsed,
                         {name: max(0, after[name] - before[name]) for name in before})
    finally:
        bpf.cleanup()


def display(report, top):
    print(f"\nREAD total: {report['read_bytes']/2**20:.2f} MiB, {report['read_ios']} requests; "
          f"{report['elapsed_seconds']:.1f}s")
    print('PID      USER             COMM                     READ_MiB  MiB/s  READ%   R_IO%   AVGms')
    users = defaultdict(int)
    rows = [row for row in report['rows'] if row['read_ios']]
    for row in rows:
        row['user'] = process_user(row['pid'])
        users[row['user']] += row['read_bytes']
    for row in rows[:top]:
        names = '/'.join(row['names']) or 'unattributed'
        print(f"{row['pid']:<8} {row['user']:<16.16} {names:<24.24} {row['read_bytes']/2**20:>8.2f} "
              f"{row['read_mib_s']:>6.2f} {row['read_share_pct']:>6.1f} "
              f"{row['read_io_share_pct']:>6.1f} {row['read_issue_to_done_ms']:>7.2f}")
    if len(rows) > top:
        print(f'{len(rows)-top} additional PIDs are included in the denominator but not displayed.')
    print('\nREAD byte shares by user (unknown/exited remains in denominator):')
    for name, count in sorted(users.items(), key=lambda item: item[1], reverse=True):
        print(f"  {name}: {count/2**20:.2f} MiB, {100*count/report['read_bytes'] if report['read_bytes'] else 0:.1f}%")
    coverage = report['trace_to_diskstats_read_pct']
    print(f"\nDiskstats READ: {report['diskstats_read_bytes']/2**20:.2f} MiB; "
          f"trace/diskstats: {coverage:.1f}%" if coverage is not None else '\nNo diskstats reads in this window.')
    print(report['scope'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--disk', default='sdb')
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--top', type=int, default=20)
    parser.add_argument('--source-only', action='store_true', help='Print patched probe without attaching it')
    parser.add_argument('--json-output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 300 or not 1 <= args.top <= 200:
        parser.error('seconds must be 1..300 and top must be 1..200')
    source = probe_source()
    if args.source_only:
        print(source)
        return
    if os.geteuid() != 0:
        parser.error('Use sudo /usr/bin/python3 for read-only BPF process attribution')
    disk, major, minor = disk_identity(args.disk)
    result = collect(args.seconds, major, minor, disk, source)
    display(result, args.top)
    if args.json_output:
        # Do not overwrite an earlier diagnostic report.
        with args.json_output.open('x') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
