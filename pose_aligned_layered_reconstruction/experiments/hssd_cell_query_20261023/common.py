import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

PACKAGE = Path(__file__).resolve().parents[2]
PILOT = PACKAGE / 'outputs/hssd_pilot_20261020'
OUT = PACKAGE / 'outputs/hssd_cell_query_20261023'
MODULE = 'pose_aligned_layered_reconstruction.experiments.hssd_cell_query_20261023'
PYTHON = '/home/zjr/anaconda3/envs/reconviagen/bin/python'
ARMS = ('appearance_control', 'oracle_point', 'oracle_cell', 'predicted_point', 'predicted_cell')
FORMAT = 'hssd_cell_query_ss_residual.v1'
SEED = 20261021
STEPS = 200
EVAL_SEEDS = (42, 43, 44)


def now():
    return datetime.now(timezone.utc).isoformat()


def binding(path):
    path = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return {'path': str(path), 'sha256': h.hexdigest()}


def verify(value):
    if binding(value['path']) != value:
        raise ValueError('Changed bound file: ' + value['path'])
    return Path(value['path'])


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def sources():
    # Bind the original runtime as well as this independently versioned experiment.
    paths = sorted(set(Path(__file__).parent.glob('*.py')) |
                   set(Path(__file__).parent.with_name('hssd_residual_20261021').glob('*.py')) |
                   set(Path(__file__).parent.with_name('hssd_full_diagnostic_20261021').glob('*.py')) |
                   set((PACKAGE / 'v2').glob('*.py')) | set(PACKAGE.glob('*.py')) |
                   {Path(__file__).with_name('run.sh'), Path(__file__).with_name('PROTOCOL.md')})
    frozen_paths = set(paths)
    for name in ['launch_sources.json', 'frozen_protocol.json', 'smoke.json']:
        frozen_paths.update(Path(row['path']) for row in json.loads((PILOT / name).read_text())['sources'])
    return [binding(p) for p in sorted(frozen_paths)]


def verify_old_sources():
    result = {}
    for name in ['launch_sources.json', 'frozen_protocol.json', 'smoke.json']:
        rows = json.loads((PILOT / name).read_text())['sources']
        for row in rows:
            verify(row)
        result[name] = len(rows)
    return result


def policy(domain, arm, steps=STEPS):
    if domain not in ('clean', 'occluded') or arm not in ARMS or steps < 1:
        raise ValueError('Invalid residual experiment policy')
    return {'format': FORMAT, 'domain': domain, 'arm': arm, 'steps': steps,
        'seed': SEED, 'lr': 1.e-4, 'warmup': 20, 'betas': [.9, .95], 'weight_decay': .01,
        'batch': 1, 'p_uncond': 0., 'weights': 'raw', 'amp': 'bf16',
        'source_code': sources(), 'manifest': binding(PILOT / f'{domain}16.json'),
        'appearance_base': binding(PILOT / f'{domain}_ss_no_layer/checkpoints/last.pt'),
        'fusion': 'frozen appearance weights x frustum; support/confidence stay in features',
        'absolute_tolerance': .02, 'residual_channels': 128, 'projection_bias': False,
        'layer_norm': True, 'query': 'cell_mean_27' if arm.endswith('_cell') else 'point',
        'cell_side': 16, 'cell_offsets': [-1./48., 0., 1./48.],
        'evaluation_seeds': list(EVAL_SEEDS),
        'scope': 'first bounded interface experiment on original16; not heldout'}


def tensor_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        h.update(value.reshape(-1).view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()
