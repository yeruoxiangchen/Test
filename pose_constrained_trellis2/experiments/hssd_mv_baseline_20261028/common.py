"""Experiment-local paths and frozen comparison contract."""
from pathlib import Path
from ..hssd_scene_20261028 import common as native

HERE = Path(__file__).resolve().parent
REPO, ROOT, PYTHON = native.REPO, native.ROOT, native.PYTHON
OUT = ROOT/'outputs/hssd_mv_baseline_20261028'
DATA = ROOT/'datasets/hssd_mv_baseline_v1'
MODULE = 'pose_constrained_trellis2.experiments.hssd_mv_baseline_20261028'
read, write, immutable, digest, sha, now = native.read, native.write, native.immutable, native.digest, native.sha, native.now
environment, models = native.environment, native.models
CONFIG = {'steps':400, 'warmup':20, 'save_every':200, 'seed':42, 'evaluation_seeds':[42,123],
          'weights':'raw', 'arms':['official_reference','mv_independent'], 'sampling_steps':12,
          'context_mode':'first', 'min_views':2, 'max_views':4, 'p_uncond':.1,
          'new_lr':1e-4, 'lora_lr':3e-5, 'lora_rank':8, 'lora_alpha':16,
          'evaluation':'canonical/world without fitted alignment; train and val separately',
          'pose_source':'GT controlled protocol; not RGB-only',
          'selection':'reuse every Native12 target, no post-result filtering',
          'relation_scope':'prepare input-derived pair proposals and separate GT occlusion qualification; no relation training yet'}

def source_identity():
    files = list(HERE.glob('*.py')) + [HERE/'run.sh',HERE/'PROTOCOL.md']
    files += list((REPO/'pose_constrained_trellis2').glob('*.py'))
    files += list(native.HERE.glob('*.py'))
    files += [REPO/'TRELLIS2/trellis2/renderers/mesh_renderer.py',REPO/'TRELLIS2/trellis2/utils/render_utils.py']
    payload = {'files':{str(p.relative_to(REPO)):sha(p) for p in sorted(files)},
               'plan':read(OUT/'plan.json')['identity'], 'models':models()}
    return {'identity':digest(payload),'runtime':payload}

def plan():
    d=read(OUT/'plan.json')
    if digest(d['plan']) != d['identity']: raise ValueError('Plan identity changed')
    for entry in d['plan']['manifests'].values():
        if sha(entry['path'])!=entry['sha256']:raise ValueError('Frozen manifest changed')
    return d['plan']
