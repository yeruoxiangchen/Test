import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import httpx

SOURCE = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('AIGC_DATA_ROOT', '/data/zjr/aigc_studio'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def request(method, base, path, **kwargs):
    with httpx.Client(timeout=30, trust_env=False) as client:
        response = client.request(method, base.rstrip('/') + path, **kwargs)
        response.raise_for_status()
        return response.json() if response.content else {}


def nvidia():
    gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True, timeout=5)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], text=True, timeout=5)
    by_gpu = {a.strip(): {'uuid': b.strip(), 'used_mib': int(c), 'free_mib': int(d), 'utilization': int(e), 'processes': []} for a,b,c,d,e in csv.reader(gpus.splitlines())}
    by_uuid = {g['uuid']: g for g in by_gpu.values()}
    for row in csv.reader(processes.splitlines()):
        if len(row) != 3:
            continue
        uid, pid, memory = (v.strip() for v in row)
        if uid in by_uuid:
            by_uuid[uid]['processes'].append({'pid': int(pid), 'memory_mib': int(memory) if memory.isdigit() else None})
    return by_gpu


def rss(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


class Monitor:
    def __init__(self, root, workers):
        self.root, self.workers = root, workers
        self.rows, self.event = [], threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        with (self.root / 'telemetry.jsonl').open('a') as target:
            while not self.event.is_set():
                begin = time.monotonic()
                try:
                    row = {'time': time.time(), 'gpus': nvidia(), 'rss_mib': {w['id']: rss(w['pid']) for w in self.workers}}
                    self.rows.append(row)
                    target.write(json.dumps(row) + '\n')
                    target.flush()
                except Exception as exc:
                    target.write(json.dumps({'time': time.time(), 'error': str(exc)}) + '\n')
                self.event.wait(max(.1, 1 - (time.monotonic() - begin)))

    def summarize(self, start, end, worker_id):
        worker = next(w for w in self.workers if w['id'] == worker_id)
        rows = [r for r in self.rows if start <= r['time'] <= end]
        observations = [r['gpus'].get(str(worker['gpu']), {}) for r in rows]
        own = [p['memory_mib'] for g in observations for p in g.get('processes', []) if p['pid'] == worker['pid'] and p['memory_mib'] is not None]
        other = sorted({p['pid'] for g in observations for p in g.get('processes', []) if p['pid'] != worker['pid']})
        ram = [r['rss_mib'].get(worker_id) for r in rows if r['rss_mib'].get(worker_id) is not None]
        return {'samples': len(rows), 'observed_process_peak_mib': max(own, default=None),
                'observed_device_peak_mib': max((g['used_mib'] for g in observations if g), default=None),
                'observed_process_rss_peak_mib': max(ram, default=None), 'other_compute_pids': other,
                'clean_device_observed': bool(rows) and not other}

    def start(self):
        self.thread.start()

    def stop(self):
        self.event.set()
        self.thread.join(timeout=10)


def freeze(root, protocol):
    content = (SOURCE / 'evaluation' / 'protocol.json').read_bytes()
    existing = read(root / 'environment.json')
    digest = hashlib.sha256(content).hexdigest()
    hashes = {}
    for folder in ['backend', 'evaluation', 'scripts']:
        for path in (SOURCE / folder).glob('*.py'):
            hashes[str(path.relative_to(SOURCE))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if existing:
        if existing['protocol_sha256'] != digest or existing['source_sha256'] != hashes:
            raise ValueError('Protocol or execution source changed; use a new run id')
        return
    # Read package metadata and the small version file; importing torch would load
    # large shared libraries even while all devices are occupied and we only wait.
    metadata_script = '''import ast,importlib.metadata,json,platform
distribution = importlib.metadata.distribution("torch")
values = {}
for node in ast.parse(distribution.locate_file("torch/version.py").read_text()).body:
    if isinstance(node, ast.Assign):
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        names = [node.target.id]
    else:
        continue
    if "cuda" in names and isinstance(node.value, ast.Constant):
        values["cuda_runtime"] = node.value.value
print(json.dumps({"python":platform.python_version(),"torch":distribution.version,**values}))
'''
    torch_info = subprocess.check_output([str(DATA / 'comfy-venv/bin/python'), '-B', '-c', metadata_script], text=True, timeout=30)
    save(root / 'environment.json', {'protocol_sha256': digest, 'source_sha256': hashes,
         'models': read(DATA / 'models/manifest.json'), 'runtime': json.loads(torch_info),
         'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,driver_version', '--format=csv,noheader'], text=True),
         'comfy_revision': subprocess.check_output(['git', '-C', str(DATA / 'vendor/ComfyUI'), 'rev-parse', 'HEAD'], text=True).strip(),
         'protocol': protocol, 'created': time.time()})


def execution_times(record):
    events = record.get('status', {}).get('messages', [])
    starts = [e[1]['timestamp']/1000 for e in events if e[0] == 'execution_start' and 'timestamp' in e[1]]
    ends = [e[1]['timestamp']/1000 for e in events if e[0] == 'execution_success' and 'timestamp' in e[1]]
    cached = [node for e in events if e[0] == 'execution_cached' for node in e[1].get('nodes', [])]
    return {'execution_start': min(starts) if starts else None, 'execution_end': max(ends) if ends else None, 'cached_nodes': cached}
