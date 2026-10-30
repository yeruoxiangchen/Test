import os
from functools import lru_cache
from pathlib import Path
from ..hssd_scene_20261028 import common as native

HERE = Path(__file__).resolve().parent
REPO, ROOT, PYTHON, SOURCE = native.REPO, native.ROOT, native.PYTHON, native.SOURCE
OUT = ROOT / 'outputs/hssd_relations192_20261029'
DATA = ROOT / 'datasets/hssd_relations_native_v1'
MODULE = 'pose_constrained_trellis2.experiments.hssd_relations192_20261029'
read, write, immutable, sha, digest, now = native.read, native.write, native.immutable, native.sha, native.digest, native.now
environment, models = native.environment, native.models
CONFIG = {'epochs': 50, 'seed': 42, 'evaluation_seeds': [42, 123],
          'arms': ['independent', 'relational'], 'lora_rank': 8, 'lora_alpha': 16,
          'new_lr': 1e-4, 'lora_lr': 3e-5, 'warmup': 100, 'save_every': 200,
          'p_uncond': .1, 'sampling_steps': 12, 'weights': 'raw',
          'group_size': 2, 'neighbor_pool_grid': 4, 'relation_width': 128,
          'relation_blocks': [5, 11, 17, 23, 29], 'pose_source': 'GT controlled; not RGB-only',
          'selection': 'one group per frozen anchor; neighbor from masks in anchor input frames; no model/GT depth scores',
          'shape_max_tokens': 32768}


@lru_cache(maxsize=1)
def plan():
    d = read(OUT / 'plan.json')
    if digest(d['plan']) != d['identity']:
        raise ValueError('Frozen selection plan changed')
    return d['plan']


def identity():
    files = list(HERE.glob('*.py')) + [HERE / 'run.sh', HERE / 'PROTOCOL.md']
    files += list((REPO / 'pose_constrained_trellis2/scene_relations').glob('*.py'))
    files += list((REPO / 'pose_constrained_trellis2').glob('*.py'))
    files += list(native.HERE.glob('*.py'))
    payload = {'files': {str(p.relative_to(REPO)): sha(p) for p in sorted(files)},
               'plan_identity': read(OUT / 'plan.json')['identity'], 'config': CONFIG, 'models': models()}
    return {'identity': digest(payload), 'runtime': payload}


def codec_folder(uid):
    reuse = plan()['native_reuse']
    return Path(reuse[uid]['folder']) if uid in reuse else DATA / 'objects' / uid


def check_receipt_identity(name, receipt, identity):
    if receipt['runtime_identity'] == identity: return
    migration = read(OUT / 'receipt_compatibility.json')
    assert migration['new_identity'] == identity, 'Migration does not authorize current source'
    expected = migration.get('receipt_identities', {}).get(name, migration['old_identity'])
    assert expected == receipt['runtime_identity'], 'Unknown prior source'
    assert migration['receipt_digests'].get(name) == digest(receipt), 'Unreviewed or changed prior receipt'


def atomic_torch_save(payload, path):
    import torch
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Checkpoint must stay in /data project')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(payload, tmp); tmp.replace(path)
