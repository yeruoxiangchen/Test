import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / 'pose_aligned_layered_reconstruction'
OLD = PACKAGE / 'outputs/hssd_depth_balance_20261024'
OUT = Path(os.environ.get('HSSD_REPLAY_OUTPUT', PACKAGE / 'outputs/hssd_mesh_replay_20261025')).resolve()
if not OUT.is_relative_to(PACKAGE / 'outputs') or OUT == OLD:
    raise ValueError('Replay output must be a separate project outputs subdirectory')
MODULE = 'pose_aligned_layered_reconstruction.experiments.hssd_mesh_replay_20261025'
PYTHON = '/home/zjr/anaconda3/envs/reconviagen/bin/python'
INDEX, SEED = 2, 42


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def binding(path):
    path = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    before = path.stat()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Changed file during read: ' + str(path))
    return {'path': str(path), 'sha256': h.hexdigest()}


def verify(row):
    if binding(row['path']) != row:
        raise ValueError('Changed bound file: ' + row['path'])
    return Path(row['path'])


def locked(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Changed replay identity: ' + str(path))
    else:
        write(path, value)


def freeze():
    old = read(OLD / 'protocol.json')
    for row in old['sources']:
        verify(row)
    sources = sorted(Path(__file__).parent.glob('*.py')) + [Path(__file__).with_name('run.sh'), Path(__file__).with_name('PROTOCOL.md')]
    # Bind third-party Python too: the original experiment bound only selected files there.
    external = sorted((ROOT / 'ReconViaGen/trellis').rglob('*.py'))
    value = {'format': 'hssd_mesh_replay.v1', 'index': INDEX, 'seed': SEED,
        'object_uid': 'hssd_102344049_i083', 'old_protocol': binding(OLD / 'protocol.json'),
        'sources': [binding(p) for p in sources], 'trellis_sources': [binding(p) for p in external],
        'manifest': old['manifest'], 'slat': binding(OLD / 'mesh/uniform/slat_binding.json'),
        'ss': {a: binding(OLD / 'evaluations' / a / '002/42_ss.json') for a in ['uniform', 'balanced']},
        'old_mesh': {a: binding(OLD / 'mesh' / a / '002/42.json') for a in ['uniform', 'balanced']},
        'plan': ['same_a', 'same_b', 'fixed_decode_a', 'fixed_decode_b', 'fresh'],
        'sampler_steps': 25, 'first_flow_call_internal_trace': True,
        'per_trace_limit_bytes': 2 << 30, 'worker_timeout_seconds': 600,
        'scope': 'one training object/one sampling seed; numerical diagnosis only; no quality or generalization claim'}
    locked(OUT / 'protocol.json', value)
    return value
