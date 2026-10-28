import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2]
TRACKER = PACKAGE.parent
SOURCE = TRACKER / 'datasets/scene_occlusion_pilot_20261020'
OUT = PACKAGE / 'outputs/hssd_pilot_20261020'
PYTHON = '/home/zjr/anaconda3/envs/reconviagen/bin/python'
BLENDER = '/tmp/blender-3.0.1-linux-x64/blender'
MODULE = 'pose_aligned_layered_reconstruction.experiments.hssd_pilot_20261020'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for data in iter(lambda: f.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def binding(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha(path)}


def verify(b):
    if sha(b['path']) != b['sha256']:
        raise ValueError('Input changed: ' + b['path'])
    return Path(b['path'])


def event(**fields):
    print(json.dumps({'time': now(), **fields}, ensure_ascii=False), flush=True)


def plan():
    data = json.loads((OUT / 'plan.json').read_text())
    if data['identity'] != identity(data['plan']):
        raise ValueError('Pilot plan identity mismatch')
    return data


def object_dir(row):
    return OUT / 'objects' / row['uid']
