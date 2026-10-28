import json
from pathlib import Path

from ..hssd_cell_query_20261023.common import binding, verify, write, now, tensor_hash, PACKAGE, PYTHON

MODULE = 'pose_aligned_layered_reconstruction.experiments.hssd_depth_balance_20261024'
OUT = PACKAGE / 'outputs/hssd_depth_balance_20261024'
PILOT = PACKAGE / 'outputs/hssd_pilot_20261020'
CELL = PACKAGE / 'outputs/hssd_cell_query_20261023'
CAUSAL = PACKAGE / 'outputs/hssd_lari_causal_20261023'
ARMS = ('uniform', 'balanced')
REFERENCES = ('base', 'appearance_control', 'oracle_point')
SEEDS = (42, 43, 44)
INDICES = (0, 2, 5, 14)
LARI_STEPS, SS_STEPS = 1000, 200
FORMAT = 'hssd_depth_balance.v1'


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    old = read(CAUSAL / 'protocol.json')['sources']
    for row in old:
        verify(row)
    local = sorted(set(Path(__file__).parent.glob('*.py')) |
                   {Path(__file__).with_name(n) for n in ('run.sh', 'PROTOCOL.md')})
    return old + [binding(p) for p in local]


def locked(path, value):
    path = Path(path)
    if path.exists() and read(path) != value:
        raise ValueError('Changed immutable identity: ' + str(path))
    if not path.exists():
        write(path, value)


def freeze():
    old = read(CAUSAL / 'protocol.json')
    config_path = PILOT / 'occluded_lari/run_config.json'
    value = dict(format=FORMAT, sources=sources(), manifest=binding(PILOT / 'occluded16.json'),
        geometry_config=read(config_path)['geometry'], geometry_config_source=binding(config_path),
        appearance_base=binding(PILOT / 'occluded_ss_no_layer/checkpoints/last.pt'),
        reference_identities={a:{'ss':binding(CELL/'evaluations'/f'occluded_{a}'/'identity.json'),
            'ss_complete':binding(CELL/'evaluations'/f'occluded_{a}'/'ss_complete.json'),
            'mesh':binding(CAUSAL/'mesh'/a/'identity.json'),
            'mesh_complete':binding(CAUSAL/'mesh'/a/'complete.json'),
            'slat':binding(CAUSAL/'mesh'/a/'slat_binding.json')} for a in REFERENCES},
        arms=list(ARMS), references=list(REFERENCES), lari_steps=LARI_STEPS, ss_steps=SS_STEPS,
        seeds=list(SEEDS), mesh_indices=list(INDICES), mesh_object_uids=old['object_uids'],
        selection_basis=binding(CAUSAL / 'protocol.json'),
        scope='same16 occluded training objects; selected4 mesh diagnostic; no generalization claim',
        depth_groups=['non_target', 'exposed_target', 'hidden_target'],
        depth_reduction='uniform valid-layer mean vs equal mean of nonempty disjoint group means per object',
        unchanged_losses={'stop': .25, 'ownership': .25, 'confidence': .1, 'cross_view': .1},
        ss_query='point tau=.02; each new frozen LaRI has separately trained residual',
        ss_sampling={'steps':25, 'cfg_strength':5., 'cfg_interval':[.5,1.], 'rescale_t':3.},
        mesh_frame='native canonical; no independently fitted transform')
    rows = read(value['manifest']['path'])['samples']
    if len(rows) != 16 or [rows[i]['object_uid'] for i in INDICES] != old['object_uids']:
        raise ValueError('Changed registered objects')
    locked(OUT / 'protocol.json', value)
    return value


def policy(kind, arm, steps=None, lari_path=None):
    if kind not in ('lari', 'ss') or arm not in ARMS:
        raise ValueError('Unknown training experiment')
    if steps is not None and steps < 1:
        raise ValueError('Steps must be positive')
    protocol = freeze()
    value = dict(format=FORMAT, kind=kind, arm=arm, protocol=protocol,
        steps=steps or (LARI_STEPS if kind == 'lari' else SS_STEPS),
        seed=20261020 if kind == 'lari' else 20261021,
        lr=1e-4, warmup=20, betas=[.9,.95], weight_decay=.01,
        batch=1, views=8, amp='bf16', p_uncond=0., weights='raw',
        manifest=protocol['manifest'], appearance_base=protocol['appearance_base'])
    if kind == 'ss':
        value['lari'] = binding(lari_path or OUT / 'train' / f'lari_{arm}' / 'last.pt')
    return value


def check_record(row, index, seed, arm=None):
    source = read(PILOT / 'occluded16.json')['samples'][index]
    if (row['uid'], row['object_uid'], row['index'], row['seed']) != (source['uid'], source['object_uid'], index, seed):
        raise ValueError('Unpaired record')
    if arm is not None and row['arm'] != arm:
        raise ValueError('Wrong arm')
    for k in ('coords', 'mesh', 'gt', 'source_ss'):
        if k in row:
            verify(row[k])
