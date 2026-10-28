import json
from pathlib import Path

from ..hssd_cell_query_20261023.common import binding, verify, write, now, tensor_hash, PACKAGE, PYTHON

MODULE = 'pose_aligned_layered_reconstruction.experiments.hssd_lari_causal_20261023'
OUT = PACKAGE / 'outputs/hssd_lari_causal_20261023'
CELL = PACKAGE / 'outputs/hssd_cell_query_20261023'
PILOT = PACKAGE / 'outputs/hssd_pilot_20261020'
OLD = PACKAGE / 'outputs/hssd_full_diagnostic_20261021'
INDICES = (0, 2, 5, 14)
UIDS = ('hssd_102344049_i076', 'hssd_102344049_i083', 'hssd_102344049_i075', 'hssd_102344049_i108')
SEEDS = (42, 43, 44)
ADAPTERS = ('predicted_point', 'oracle_point')
ARMS = ('base', 'appearance_control', 'oracle_point', 'oracle_cell', 'predicted_point', 'predicted_cell')
VARIANTS = ('predicted', 'gt_depth', 'gt_existence', 'gt_ownership', 'confidence_one',
            'gt_depth_confidence_one', 'gt_depth_existence_ownership', 'oracle')


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    original = read(CELL / 'smoke.json')['sources']
    for row in original:
        verify(row)
    local = [binding(p) for p in sorted(Path(__file__).parent.glob('*.py'))]
    local += [binding(Path(__file__).with_name(n)) for n in ['run.sh', 'PROTOCOL.md']]
    return original + local


def freeze():
    manifests = {d: binding(PILOT / f'{d}16.json') for d in ['clean', 'occluded']}
    rows = read(manifests['occluded']['path'])['samples']
    assert tuple(rows[i]['object_uid'] for i in INDICES) == UIDS
    identity = {'version': 1, 'sources': sources(), 'manifests': manifests,
        'indices': list(INDICES), 'object_uids': list(UIDS), 'seeds': list(SEEDS),
        'adapters': list(ADAPTERS), 'variants': list(VARIANTS), 'mesh_arms': list(ARMS),
        'selection_reason': ['first original object', 'largest occluded oracle cell degradation',
                             'largest occluded oracle point and predicted-cell gains over appearance',
                             'largest occluded predicted-cell degradation over appearance'],
        'selection_basis': binding(CELL / 'ss_summary.json'),
        'scope': 'post-hoc diagnostic cases, fixed before these interventions; not a heldout benchmark',
        'ss': {'steps': 25, 'cfg': 5., 'interval': [.5, 1.], 'rescale_t': 3., 'amp': 'bf16'},
        'sampling': 'original global object index, not subset index', 'gt_frame': 'native canonical; no fitted transform'}
    path = OUT / 'protocol.json'
    if path.exists() and read(path) != identity:
        raise ValueError('Frozen diagnostic identity changed; do not mix runs')
    if not path.exists():
        write(path, identity)
    return identity


def bind_predictions(domain):
    rows = []
    for i in range(16):
        path = OLD / 'geometry' / domain / f'{i:03d}.json'
        r = read(path)
        for key in ['checkpoint', 'manifest']:
            verify(r['identity'][key])
        verify(r['prediction'])
        rows.append({'record': binding(path), 'prediction': r['prediction']})
    return rows


def locked_identity(path, value):
    path = Path(path)
    if path.exists() and read(path) != value:
        raise ValueError('Changed diagnostic inputs: ' + str(path))
    if not path.exists():
        write(path, value)


def check_record(row, uid, seed):
    if row['uid'] != uid or row['seed'] != seed:
        raise ValueError('Unpaired diagnostic record')
    for key in ['gt', 'coords', 'mesh', 'source_ss']:
        if key in row:
            verify(row[key])
