import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2]
ROOT = PACKAGE.parent
HERE = Path(__file__).resolve().parent
OUT = PACKAGE / 'outputs/hssd_shared_scene_20261025'
PILOT = PACKAGE / 'outputs/hssd_pilot_20261020'
CELL = PACKAGE / 'outputs/hssd_cell_query_20261023'
SOURCE = ROOT / 'datasets/scene_occlusion_pilot_20261020'
BLEND = SOURCE / 'renders/hssd_102344049/complete_scene.blend'
BLENDER = Path('/tmp/blender-3.0.1-linux-x64/blender')
PYTHON = '/home/zjr/anaconda3/envs/reconviagen/bin/python'
MODULE = 'pose_aligned_layered_reconstruction.experiments.hssd_shared_scene_20261025'
GENA = PACKAGE / 'external/GENAmodal3D'
INDICES = [0, 5, 6, 7]
FRAMES = [0, 4, 7, 11]
SEEDS = [42, 43, 44]
ARMS = ['yxc30k', 'appearance_control']


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha(path)}


def verify(b):
    if sha(b['path']) != b['sha256']:
        raise ValueError('Changed input: ' + b['path'])
    return Path(b['path'])


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def locked(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError('Existing protocol/receipt differs: ' + str(path))
    else:
        write(path, value)


def plan():
    result = read(OUT / 'plan.json')
    if identity(result['plan']) != result['identity']:
        raise ValueError('Shared capture plan changed')
    return result


def event(stage, **kwargs):
    row = {'time': now(), 'stage': stage, **kwargs}
    write(OUT / 'progress.json', row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
