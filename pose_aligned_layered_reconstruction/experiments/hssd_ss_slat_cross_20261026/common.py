"""Small immutable contracts; safe to import from status without Torch."""
import importlib.metadata
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('_cross_inputadapt_common', HERE.parent/'hssd_input_adaptation_20261025/common.py')
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
ROOT, PACKAGE, BASE_OUT, SHARED, PYTHON = base.ROOT, base.PACKAGE, base.OUT, base.SHARED, base.PYTHON
now, read, write, binding, verify, digest, locked = base.now, base.read, base.write, base.binding, base.verify, base.digest, base.locked
OUT = PACKAGE/'outputs/hssd_ss_slat_cross_20261026'
REPAIR = BASE_OUT/'int32_fix_20261025'
MODULE = 'pose_aligned_layered_reconstruction.experiments.hssd_ss_slat_cross_20261026'
LEVELS, SEEDS = ['518', '2072'], [42, 43, 44]
UIDS = ['hssd_102344049_i075', 'hssd_102344049_i065']
ARM = 'appearance_control'
CALLS = {'positive_calls': 25, 'negative_calls': 19}


def event(stage, **fields):
    row = {'time': now(), 'stage': stage, **fields}
    write(OUT/'progress.json', row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def variant(level):
    if level not in LEVELS:
        raise ValueError('Unsupported input level: '+str(level))
    return 'crop'+level+'_visible'


def case_key(case):
    return (case['uid'], int(case['seed']), case['support_level'], case['condition_level'])


def case_dir(case):
    return OUT/'evaluations'/f'ss{case["support_level"]}_cond{case["condition_level"]}'/case['uid']/str(case['seed'])


def replay_dir(case):
    return OUT/'replays'/f'ss{case["support_level"]}_cond{case["condition_level"]}'/case['uid']/str(case['seed'])


def ensure_plan():
    original = base.plan()['plan']
    ident = base.freeze_runtime()
    completion = read(REPAIR/'completed.json')
    if completion['new_meshes'] != 54 or completion['logical_cases'] != 96:
        raise ValueError('Original InputAdapt experiment is incomplete')
    verify(completion['summary']); verify(completion['report'])
    objects = [next(o for o in original['objects'] if o['uid'] == uid) for uid in UIDS]
    if [o['original_index'] for o in objects] != [5, 7]:
        raise ValueError('Original noise indices changed')
    cases, references = [], []
    slat_binding = None
    for obj in objects:
        for seed in SEEDS:
            for support in LEVELS:
                for condition in LEVELS:
                    folder = base.case_dir(variant(support), ARM, obj['uid'], seed)
                    cond_folder = base.case_dir(variant(condition), ARM, obj['uid'], seed)
                    ss, cond = read(folder/'ss.json'), read(cond_folder/'ss.json')
                    for row, level in [(ss, support), (cond, condition)]:
                        if (row['identity'] != ident or row['object_uid'] != obj['uid'] or row['seed'] != seed
                                or row['arm'] != ARM or row['variant'] != variant(level) or not row['success']):
                            raise ValueError('Wrong archived SS/condition identity')
                    c = {'uid': obj['uid'], 'seed': seed, 'original_index': obj['original_index'],
                        'support_level': support, 'condition_level': condition,
                        'is_reference': support == condition, 'support_record': binding(folder/'ss.json'),
                        'condition_record': binding(cond_folder/'ss.json')}
                    if c['is_reference']:
                        mesh = read(folder/'mesh.json')
                        if (mesh['identity'] != ident or mesh['ss'] != c['support_record'] or not mesh['success']
                                or mesh['fitted_transform'] or mesh['calls'] != CALLS or mesh['inputs'] != cond['inputs']):
                            raise ValueError('Archived diagonal is incomplete or mismatched')
                        if slat_binding is None:
                            slat_binding = mesh['slat_binding']
                        if mesh['slat_binding'] != slat_binding:
                            raise ValueError('Mixed frozen SLat models')
                        c['diagonal_mesh'] = binding(folder/'mesh.json')
                        references.append(c['diagonal_mesh'])
                    cases.append(c)
    value = {'base_identity': ident, 'base_plan': binding(BASE_OUT/'plan.json'),
        'base_evaluation_identity': binding(BASE_OUT/'evaluation_identity.json'),
        'base_completion': binding(REPAIR/'completed.json'), 'base_summary': completion['summary'],
        'runtime_amendment': binding(REPAIR/'runtime_amendment.json'),
        'shared_plan': binding(SHARED/'plan.json'),
        'evaluation_gt': [binding(SHARED/'evaluation_gt'/f'{frame:03d}.npz') for frame in base.FRAMES],
        'objects': objects, 'evaluation_uid_context': [o['uid'] for o in original['objects']],
        'cases': cases, 'references': references, 'slat_binding': slat_binding,
        'levels': LEVELS, 'seeds': SEEDS, 'arm': ARM, 'logical_cases': 24, 'new_meshes': 12,
        'noise': 'frozen master_noise(seed, original_index), lookup by canonical xyz; same coordinates share noise, not equal-length vectors',
        'scope': 'post-hoc factorial diagnostic on two old training objects in one scene; GT-assisted cameras/masks/canonical frame; no training, not heldout, no LaRI contribution claim'}
    if slat_binding['evaluation_uid_count'] != 4:
        raise ValueError('Preserve the historical four-UID SLat loading identity')
    result = {'identity': digest(value), 'plan': value}
    locked(OUT/'plan.json', result)
    return result


def plan():
    value = read(OUT/'plan.json')
    if digest(value['plan']) != value['identity']:
        raise ValueError('Crossover plan changed')
    return value


def logical_cases():
    return plan()['plan']['cases']


def new_jobs():
    return [c for c in logical_cases() if not c['is_reference']]


def smoke_cases():
    selected = [(UIDS[0], 42, '518', '518'), (UIDS[1], 42, '518', '2072')]
    return [next(c for c in logical_cases() if case_key(c) == key) for key in selected]


def object_row(case):
    return next(o for o in plan()['plan']['objects'] if o['uid'] == case['uid'])


def get_result(case):
    mp = verify(case['diagonal_mesh']) if case['is_reference'] else case_dir(case)/'mesh.json'
    sp = verify(case['support_record'])
    return mp, read(mp), sp, read(sp)


def freeze_runtime():
    p = ensure_plan()
    amendment = read(verify(p['plan']['runtime_amendment']))
    for b in amendment['sources'] + amendment['installed_sources']:
        verify(b)
    versions = {name: importlib.metadata.version(name) for name in amendment['versions']}
    if versions != amendment['versions']:
        raise ValueError('Native runtime versions changed')
    regression_receipts = []
    for name in ['kernel_check', 'decoder_small', 'decoder_large']:
        path = REPAIR/(name+'.json')
        row = read(path)
        if not row['passed']:
            raise ValueError('Required Native regression failed')
        for b in row['sources']:
            verify(b)
        regression_receipts.append(binding(path))
    for b in p['plan']['evaluation_gt']:
        verify(b)
    sources = sorted(HERE.glob('*.py')) + [HERE/'run.sh', HERE/'PROTOCOL.md']
    trellis = ROOT/'ReconViaGen/trellis'
    sources += [trellis/'models/structured_latent_vae/decoder_mesh.py',
        trellis/'representations/mesh/cube2mesh.py', trellis/'representations/mesh/flexicube.py',
        trellis/'representations/mesh/utils_cube.py']
    value = {'plan_identity': p['identity'], 'sources': [binding(f) for f in sources],
        'base_identity': p['plan']['base_identity'], 'runtime_amendment': p['plan']['runtime_amendment'],
        'native_regressions': regression_receipts, 'versions': versions,
        'slat_binding': p['plan']['slat_binding'], 'sampler_calls': CALLS,
        'operator': 'LargeInputNative; ordinary inputs delegate exactly to FixedNative'}
    locked(OUT/'evaluation_identity.json', value)
    return digest(value)


def validate_saved_result(record, case, ident, inputs=None):
    """Reject a completed cache whose two independently selected sources differ."""
    p = plan()['plan']
    ss, cond = read(verify(case['support_record'])), read(verify(case['condition_record']))
    expected_support = {'record':case['support_record'], 'coords':ss['coords'], 'inputs':ss['inputs']}
    expected_condition = {'record':case['condition_record'], 'inputs':cond['inputs']}
    expected = {'identity':ident, 'case':case, 'object_uid':case['uid'], 'seed':case['seed'], 'arm':ARM,
        'support_level':case['support_level'], 'condition_level':case['condition_level'],
        'support_source':expected_support, 'condition_source':expected_condition,
        'inputs':cond['inputs'], 'condition_inputs':cond['inputs'], 'ss':case['support_record'],
        'slat_binding':p['slat_binding'], 'calls':CALLS, 'success':True, 'fitted_transform':False}
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError('Completed result contract differs: '+key)
    if inputs is not None and inputs != cond['inputs']:
        raise ValueError('Actual condition differs from completed result')
    for field in ['mesh', 'world_mesh', 'sampled_slat_receipt']:
        verify(record[field])
    cached = read(record['sampled_slat_receipt']['path'])
    verify(cached['latent'])
    contract = cached['contract']
    noise_seed = case['seed']*2000003+case['original_index']*2017+7919
    required = {'identity':ident, 'case':case, 'support_source':expected_support,
        'condition_source':expected_condition, 'slat_binding':p['slat_binding'],
        'original_index':case['original_index'], 'master_seed':noise_seed}
    for key, value in required.items():
        if contract.get(key) != value:
            raise ValueError('Completed latent contract differs: '+key)
    if cached['calls'] != CALLS:
        raise ValueError('Incomplete cached sampler calls')
    n = ss['occupancy']['predicted_voxels']
    for field, shape, dtype in [('initial_coords',[n,4],'torch.int32'), ('initial_noise',[n,8],'torch.float32')]:
        value = contract[field]
        if value['shape'] != shape or value['dtype'] != dtype or len(value['sha256']) != 64:
            raise ValueError('Wrong initial coordinate/noise receipt')
    return record


def check_smoke():
    row = read(OUT/'smoke.json')
    if not row.get('passed') or row['identity'] != digest(read(OUT/'evaluation_identity.json')):
        raise ValueError('Crossover smoke gate has not passed for this identity')
    if not row['diagonal_exact'] or not row['cross_complete'] or not row['coordinate_noise_exact']:
        raise ValueError('Incomplete crossover smoke gate')
    noise = read(verify(row['noise_audit']))
    if not noise['passed'] or noise['identity'] != row['identity'] or len(noise['rows']) != 6:
        raise ValueError('Coordinate-noise audit missing or mismatched')
    expected = smoke_cases()
    if len(row['records']) != len(expected):
        raise ValueError('Expected one diagonal replay and one real crossover')
    for b, case in zip(row['records'], expected):
        r = read(verify(b))
        validate_saved_result(r, case, row['identity'])
        if case['is_reference'] and not r.get('diagonal_exact'):
            raise ValueError('Diagonal replay was not exact')
    return binding(OUT/'smoke.json')
