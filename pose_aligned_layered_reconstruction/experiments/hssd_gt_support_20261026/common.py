"""Frozen sources and oracle-support contracts; status needs no Torch."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('_oracle_cross_common', HERE.parent/'hssd_ss_slat_cross_20261026/common.py')
cross = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cross)
base = cross.base
ROOT, PACKAGE, BASE_OUT, SHARED, PYTHON = cross.ROOT, cross.PACKAGE, cross.BASE_OUT, cross.SHARED, cross.PYTHON
now, read, write, binding, verify, digest, locked = cross.now, cross.read, cross.write, cross.binding, cross.verify, cross.digest, cross.locked
OUT = PACKAGE/'outputs/hssd_gt_support_20261026'
MODULE = 'pose_aligned_layered_reconstruction.experiments.hssd_gt_support_20261026'
ARM, VARIANT = 'appearance_control', 'crop2072_visible'
SEEDS, KINDS = [42,43,44], ['predicted','gt']
UIDS = ['hssd_102344049_i076','hssd_102344049_i075','hssd_102344049_i080','hssd_102344049_i065']
CALLS = {'positive_calls':25,'negative_calls':19}
EXPECTED_PARTITIONS = dict(zip(UIDS, [(9225,9349,1426),(5247,9477,5276),(862,19138,0),(5691,14309,0)]))


def event(stage, **fields):
    row = {'time':now(), 'stage':stage, **fields}
    write(OUT/'progress.json', row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def case_key(case):
    return case['uid'], int(case['seed']), case['support_kind']


def case_dir(case):
    return OUT/'evaluations'/case['support_kind']/case['uid']/str(case['seed'])


def replay_dir(case):
    return OUT/'replays'/case['support_kind']/case['uid']/str(case['seed'])


def canonical_support(coords):
    import numpy as np
    coords = np.asarray(coords)
    if coords.dtype != np.int32 or coords.ndim != 2 or coords.shape[1] not in (3,4) or not len(coords):
        raise ValueError('Expected nonempty int32 canonical support')
    xyz = coords[:,-3:]
    if (np.any(xyz < 0) or np.any(xyz >= 64)
            or not np.array_equal(np.unique(xyz,axis=0),xyz)
            or (coords.shape[1] == 4 and np.any(coords[:,0] != 0))):
        raise ValueError('Support must be unique sorted batch-zero 64-cube coordinates')
    return np.concatenate([np.zeros((len(xyz),1),np.int32),xyz],axis=1)


def oracle_record(obj, seed, condition):
    """Copy only verified official occupancy coordinates, never GT features."""
    import numpy as np
    with np.load(verify(obj['ss'])) as data:
        coords = canonical_support(data['official_coords'])
    folder = OUT/'supports'/obj['uid']
    folder.mkdir(parents=True,exist_ok=True)
    path = folder/'gt_coords.npz'
    if path.exists():
        with np.load(path) as data:
            if not np.array_equal(data['coords'],coords):
                raise ValueError('Existing oracle coordinates changed')
    else:
        np.savez_compressed(path,coords=coords)
    row = {'schema':'shared_scene_gt_support_v1','object_uid':obj['uid'],'seed':seed,
        'arm':ARM,'variant':VARIANT,'support_kind':'gt','oracle_gt_support':True,
        'coords':binding(path),'inputs':condition['inputs'],'source_ss':obj['ss'],
        'source_slat_coordinates':obj['slat'],'source_mesh':obj['mesh'],
        'original_index':obj['original_index'],'canonical_from_world':obj['canonical_from_world'],
        'grid_transform':'identity','features_used':False,'success':True,
        'occupancy':{'iou':1.,'precision':1.,'recall':1.,'f1':1.,
            'predicted_voxels':len(coords),'target_voxels':len(coords),'count_ratio':1.},
        'scope':'privileged GT surface support only; generated SLat features; not an SS prediction or achievable method score'}
    receipt = folder/f'{seed}.json'
    locked(receipt,row)
    return binding(receipt)


def ensure_plan():
    previous_identity = cross.freeze_runtime()
    previous_plan = cross.plan()['plan']
    original = base.plan()['plan']
    archived_summary = read(verify(previous_plan['base_summary']))
    objects = original['objects']
    if [o['uid'] for o in objects] != UIDS or [o['original_index'] for o in objects] != [0,5,6,7]:
        raise ValueError('Keep all four original objects and noise indices')
    cases, references = [], []
    slat_binding = previous_plan['slat_binding']
    for obj in objects:
        for seed in SEEDS:
            folder = base.case_dir(VARIANT,ARM,obj['uid'],seed)
            ss, mesh = read(folder/'ss.json'), read(folder/'mesh.json')
            ssb, mb = binding(folder/'ss.json'), binding(folder/'mesh.json')
            archived = [r for r in archived_summary['records'] if r['variant']==VARIANT
                and r['arm']==ARM and r['uid']==obj['uid'] and r['seed']==seed]
            if (len(archived)!=1 or archived[0]['mesh_record']!=mb
                    or archived[0]['ss_record']!=ssb or archived[0]['mesh']!=mesh):
                raise ValueError('Reference changed since immutable InputAdapt completion')
            if (not ss['success'] or not mesh['success'] or ss['identity'] != previous_plan['base_identity']
                    or mesh['identity'] != previous_plan['base_identity']
                    or ss['object_uid'] != obj['uid'] or mesh['object_uid'] != obj['uid']
                    or ss['seed'] != seed or mesh['seed'] != seed or ss['variant'] != VARIANT
                    or mesh['variant'] != VARIANT or ss['arm'] != ARM or mesh['arm'] != ARM
                    or mesh['ss'] != ssb or mesh['inputs'] != ss['inputs']
                    or mesh['slat_binding'] != slat_binding or mesh['calls'] != CALLS or mesh['fitted_transform']):
                raise ValueError('Archived paired reference is incomplete or mismatched')
            gt = oracle_record(obj,seed,ss)
            for kind, support in [('predicted',ssb),('gt',gt)]:
                case = {'uid':obj['uid'],'seed':seed,'original_index':obj['original_index'],
                    'support_kind':kind,'is_reference':kind=='predicted',
                    'support_record':support,'condition_record':ssb}
                if kind == 'predicted':
                    case['reference_mesh'] = mb
                cases.append(case)
            references.append(mb)
    value = {'base_identity':previous_plan['base_identity'],'base_plan':binding(BASE_OUT/'plan.json'),
        'cross_identity':previous_identity,'cross_evaluation_identity':binding(cross.OUT/'evaluation_identity.json'),
        'base_completion':previous_plan['base_completion'],'base_summary':previous_plan['base_summary'],
        'runtime_amendment':previous_plan['runtime_amendment'],'shared_plan':previous_plan['shared_plan'],
        'evaluation_gt':previous_plan['evaluation_gt'],'objects':objects,'cases':cases,'references':references,
        'evaluation_uid_context':UIDS,'slat_binding':slat_binding,'seeds':SEEDS,'arm':ARM,'variant':VARIANT,
        'support_kinds':KINDS,'logical_cases':24,'new_meshes':12,'new_ss_samples':0,
        'input_choice':'crop2072_visible fixed for every object; not per-object best input selection',
        'noise':'original seed/index master field indexed by canonical xyz; same coordinate same noise',
        'scope':'four old training objects in one shared scene; GT-assisted observations and oracle surface support; diagnostic not training, generalization, LaRI evidence or a guaranteed strict performance upper bound'}
    result = {'identity':digest(value),'plan':value}
    locked(OUT/'plan.json',result)
    return result


def plan():
    value = read(OUT/'plan.json')
    if digest(value['plan']) != value['identity']:
        raise ValueError('Oracle plan changed')
    return value


def logical_cases():
    return plan()['plan']['cases']


def new_jobs():
    return [c for c in logical_cases() if not c['is_reference']]


def object_row(case):
    return next(o for o in plan()['plan']['objects'] if o['uid']==case['uid'])


def smoke_cases():
    keys = [(UIDS[1],42,'predicted'),(UIDS[3],42,'gt')]
    return [next(c for c in logical_cases() if case_key(c)==key) for key in keys]


def get_result(case):
    mp = verify(case['reference_mesh']) if case['is_reference'] else case_dir(case)/'mesh.json'
    sp = verify(case['support_record'])
    return mp, read(mp), sp, read(sp)


def support_source(case):
    ss = read(verify(case['support_record']))
    return {'record':case['support_record'],'coords':ss['coords'],'inputs':ss['inputs'],
        'support_kind':case['support_kind'],'oracle_gt_support':not case['is_reference']}


def freeze_runtime():
    p = ensure_plan()
    for b in p['plan']['evaluation_gt']:
        verify(b)
    sources = sorted(HERE.glob('*.py'))+[HERE/'run.sh',HERE/'PROTOCOL.md']
    value = {'plan_identity':p['identity'],'sources':[binding(f) for f in sources],
        'previous_runtime_identity':p['plan']['cross_identity'],
        'previous_runtime':p['plan']['cross_evaluation_identity'],
        'slat_binding':p['plan']['slat_binding'],'sampler_calls':CALLS,
        'operator':'unchanged LargeInputNative and memory-efficient mesh',
        'oracle':'replace only canonical SS coordinates with verified official_coords; no GT latent features'}
    locked(OUT/'evaluation_identity.json',value)
    return digest(value)


def validate_saved_result(record,case,ident,inputs=None):
    if case not in logical_cases():
        raise ValueError('Case not in frozen plan')
    p = plan()['plan']
    ss, cond = read(verify(case['support_record'])), read(verify(case['condition_record']))
    condition_source = {'record':case['condition_record'],'inputs':cond['inputs']}
    expected = {'identity':ident,'case':case,'object_uid':case['uid'],'seed':case['seed'],
        'arm':ARM,'variant':VARIANT,'support_kind':case['support_kind'],
        'oracle_gt_support':not case['is_reference'],'support_source':support_source(case),
        'condition_source':condition_source,'inputs':cond['inputs'],'condition_inputs':cond['inputs'],
        'ss':case['support_record'],'slat_binding':p['slat_binding'],'calls':CALLS,
        'success':True,'fitted_transform':False,'gt':object_row(case)['mesh'],
        'canonical_from_world':object_row(case)['canonical_from_world']}
    for key,value in expected.items():
        if record.get(key) != value:
            raise ValueError('Saved mesh contract differs: '+key)
    if inputs is not None and inputs != cond['inputs']:
        raise ValueError('Actual observation differs from cached condition')
    for field in ['mesh','world_mesh','sampled_slat_receipt']:
        verify(record[field])
    receipt = read(record['sampled_slat_receipt']['path'])
    verify(receipt['latent'])
    contract = receipt['contract']
    required = {'identity':ident,'case':case,'support_source':support_source(case),
        'condition_source':condition_source,'slat_binding':p['slat_binding'],
        'original_index':case['original_index'],
        'master_seed':case['seed']*2000003+case['original_index']*2017+7919}
    for key,value in required.items():
        if contract.get(key) != value:
            raise ValueError('Saved latent contract differs: '+key)
    if receipt['calls'] != CALLS:
        raise ValueError('Incomplete latent sampling')
    n = ss['occupancy']['predicted_voxels']
    for key,shape,dtype in [('initial_coords',[n,4],'torch.int32'),('initial_noise',[n,8],'torch.float32')]:
        value = contract[key]
        if value['shape']!=shape or value['dtype']!=dtype or len(value['sha256'])!=64:
            raise ValueError('Invalid cached coordinate/noise binding')
    return record


def check_smoke():
    row = read(OUT/'smoke.json')
    ident = digest(read(OUT/'evaluation_identity.json'))
    if (not row.get('passed') or row['identity']!=ident or not row['baseline_replay_exact']
            or not row['oracle_complete'] or not row['coordinate_noise_exact']):
        raise ValueError('Current oracle smoke gate has not passed')
    noise = read(verify(row['noise_audit']))
    if not noise['passed'] or noise['identity']!=ident or len(noise['rows'])!=12:
        raise ValueError('Noise audit incomplete or mismatched')
    if len(row['records'])!=2:
        raise ValueError('Need exact predicted replay and real GT-support sample')
    for b,case in zip(row['records'],smoke_cases()):
        r = validate_saved_result(read(verify(b)),case,ident)
        if case['is_reference'] and not r.get('baseline_replay_exact'):
            raise ValueError('Baseline did not replay exactly')
    return row
