"""Select four idle GPUs, or explicitly allow light sharing with a memory margin.

Advisory locks coordinate this project's queue instances only. External programs
can still allocate GPUs, so telemetry is rechecked after acquiring the locks.
"""
from __future__ import annotations

import csv
import fcntl
import itertools
import json
import math
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

LOCK_ROOT = Path('/data/zjr/screenshot_code_agent/gpu_queue_locks')


@dataclass(frozen=True)
class QueuePolicy:
    candidates: tuple[str,...] = tuple(str(i) for i in range(8))
    poll_seconds: float = 10
    stable_samples: int = 2
    max_used_mib: int = 512
    max_utilization: int = 5
    memory_fraction: float = .8
    headroom_mib: int = 1536
    allow_shared_gpus: bool = False
    max_shared_used_mib: int = 2048

    def validate(self):
        if (len(self.candidates)<4 or len(set(self.candidates))!=len(self.candidates)
                or any(not x.isdecimal() for x in self.candidates)):
            raise ValueError('GPU candidates must contain at least four distinct physical indices')
        if not 1 <= self.poll_seconds <= 60 or not 2 <= self.stable_samples <= 12:
            raise ValueError('Queue interval must be 1..60 seconds; require 2..12 stable samples')
        if not .5 <= self.memory_fraction <= .95 or self.headroom_mib<0:
            raise ValueError('Invalid GPU memory budget')
        if self.max_shared_used_mib < 0:
            raise ValueError('Shared GPU memory ceiling must be nonnegative')
        return self


def query(command):
    return subprocess.check_output(command,text=True,timeout=15)


def snapshot(policy,run=query):
    policy.validate()
    gpu_csv = run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu',
                   '--format=csv,noheader,nounits'])
    process_csv = run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'])
    processes = defaultdict(list)
    for row in csv.reader(process_csv.splitlines(),skipinitialspace=True):
        if not row:
            continue
        if len(row)!=2 or not row[0].startswith('GPU-') or not row[1].strip().isdecimal():
            raise ValueError('Cannot verify GPU compute-process telemetry')
        processes[row[0].strip()].append(int(row[1]))
    rows,seen,uuids = [],set(),set()
    for row in csv.reader(gpu_csv.splitlines(),skipinitialspace=True):
        if not row:
            continue
        if len(row)!=7:
            raise ValueError('Incomplete GPU telemetry')
        index,uuid,name,total,used,free,util = [x.strip() for x in row]
        if not index.isdecimal() or index in seen or uuid in uuids or not uuid.startswith('GPU-'):
            raise ValueError('Invalid GPU identity telemetry')
        seen.add(index)
        uuids.add(uuid)
        total,used,free,util = map(int,(total,used,free,util))
        if total<=0 or min(used,free,util)<0 or used+free>total or util>100:
            raise ValueError('Invalid GPU memory/utilization telemetry')
        if index not in policy.candidates:
            continue
        required = math.ceil(total*policy.memory_fraction)+policy.headroom_mib
        blockers = []
        if processes[uuid] and not policy.allow_shared_gpus:
            blockers.append('compute_process_present')
        allowed_used = policy.max_shared_used_mib if policy.allow_shared_gpus else policy.max_used_mib
        if used>allowed_used:
            blockers.append('memory_in_use')
        if util>policy.max_utilization and not policy.allow_shared_gpus:
            blockers.append('gpu_busy')
        if free<required:
            blockers.append('insufficient_free_memory')
        idle = (not processes[uuid] and used<=policy.max_used_mib
                and util<=policy.max_utilization and free>=required)
        rows.append({'gpu':index,'uuid':uuid,'name':name,'total_mib':total,'used_mib':used,
            'free_mib':free,'utilization':util,'compute_pids':processes[uuid],
            'required_free_mib':required,'idle':idle,'eligible':not blockers,
            'shared':not idle and not blockers,'blockers':blockers})
    if not set(policy.candidates)<=seen:
        raise ValueError('Requested GPU candidates are missing from NVIDIA telemetry')
    return sorted(rows,key=lambda r:int(r['gpu']))


class GPULease:
    def __init__(self,rows,handles):
        self.rows,self.handles = rows,handles
        self.gpus = [r['gpu'] for r in rows]

    def close(self):
        for handle in self.handles:
            handle.close()
        self.handles = []


def try_lease(rows,root=LOCK_ROOT):
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    handles = []
    try:
        for row in sorted(rows,key=lambda r:r['uuid']):
            # Device UUIDs identify locks even when physical indices change.
            uuid = row['uuid']
            if not uuid.startswith('GPU-') or any(not (c.isalnum() or c=='-') for c in uuid):
                raise ValueError('Invalid GPU UUID for lock file')
            handle = (root/(uuid+'.lock')).open('a')
            handles.append(handle)
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return GPULease(rows,handles)
    except BlockingIOError:
        for handle in handles:
            handle.close()
        return None
    except BaseException:
        for handle in handles:
            handle.close()
        raise


def wait_for_idle(c,policy,*,sample=snapshot,sleep=time.sleep,lock_root=LOCK_ROOT):
    policy.validate()
    consecutive = {}
    stage = 'waiting_for_gpu_capacity' if policy.allow_shared_gpus else 'waiting_for_idle_gpus'
    while True:
        try:
            rows = sample(policy)
        except (OSError,ValueError,subprocess.SubprocessError) as error:
            consecutive.clear()
            c.update(stage=stage,selected_gpus=None,
                     gpu_queue={'policy':asdict(policy),'telemetry_error':str(error),
                                'note':'Unknown telemetry is treated as busy; retrying.'})
            sleep(policy.poll_seconds)
            continue
        groups = defaultdict(list)
        for row in rows:
            key = (row['gpu'],row['uuid'])
            consecutive[key] = consecutive.get(key,0)+1 if row.get('eligible',row['idle']) else 0
            row['stable_samples'] = consecutive[key]
            if consecutive[key]>=policy.stable_samples:
                # All four ranks must have matching GPU model and memory capacity.
                groups[(row['name'],row['total_mib'])].append(row)
        c.update(stage=stage,selected_gpus=None,
            gpu_queue={'policy':asdict(policy),'devices':rows,
                       'idle_count':sum(r['idle'] for r in rows),
                       'eligible_count':sum(r.get('eligible',r['idle']) for r in rows),
                       'stable_idle_count':sum(r['idle'] for g in groups.values() for r in g),
                       'stable_eligible_count':sum(len(g) for g in groups.values()),'telemetry_error':None})
        retry_snapshot = False
        for group in groups.values():
            # Prefer idle devices when four are available; share only as needed.
            group = sorted(group,key=lambda r:(not r['idle'],r['used_mib'],int(r['gpu'])))
            for chosen in itertools.combinations(group,4):
                lease = try_lease(list(chosen),lock_root)
                if lease is None:
                    continue
                accepted = False
                try:
                    # Another process may have started during lock acquisition.
                    fresh = {(r['gpu'],r['uuid']):r for r in sample(policy)}
                    if not all(fresh.get((r['gpu'],r['uuid']),{}).get('eligible',
                               fresh.get((r['gpu'],r['uuid']),{}).get('idle',False)) for r in chosen):
                        consecutive.clear()
                        retry_snapshot = True
                        break
                    value = {'time':time.time(),'selected_gpus':lease.gpus,
                             'devices':[fresh[(r['gpu'],r['uuid'])] for r in chosen],
                             'policy':asdict(policy),'coordination':'advisory locks for this project only'}
                    with (c.output/'gpu_allocations.jsonl').open('a') as stream:
                        stream.write(json.dumps(value)+'\n')
                    c.update(stage='gpu_memory_ready',selected_gpus=lease.gpus,gpu_allocation=value)
                    accepted = True
                    return lease
                except (OSError,ValueError,subprocess.SubprocessError) as error:
                    consecutive.clear()
                    retry_snapshot = True
                    c.update(stage=stage,selected_gpus=None,
                             gpu_queue={'policy':asdict(policy),'telemetry_error':str(error)})
                    break
                finally:
                    if not accepted:
                        lease.close()
            if retry_snapshot:
                break
        sleep(policy.poll_seconds)
