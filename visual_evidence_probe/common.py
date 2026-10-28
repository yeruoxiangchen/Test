import datetime
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/correlated_views_20261025'
SOURCE = ROOT.parent / 'pose_aligned_layered_reconstruction/outputs/hssd_pilot_20261020/clean16.json'
ARMS = ('real2', 'repeat8', 'real8', 'gt_render6', 'donor_a6', 'donor_b6', 'mixed_ab6', 'mixed_ba6')
SEEDS = (42, 43)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    p = Path(path).resolve(strict=True)
    return {'path': str(p), 'sha256': sha(p)}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def verify(b):
    path = Path(b['path'])
    if sha(path) != b['sha256']:
        raise ValueError(f'Changed artifact: {path}')
    return path


def lock_json(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Changed protocol: {path}')
    else:
        write(path, value)


def folder(index):
    return OUT / 'objects' / f'{index:03d}'


def emit(index, stage, **kwargs):
    value = {'time': now(), 'index': index, 'stage': stage, **kwargs}
    write(OUT / 'workers' / f'{index:03d}.json', value)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)
