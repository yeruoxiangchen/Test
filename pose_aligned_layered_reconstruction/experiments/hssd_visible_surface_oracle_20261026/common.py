"""Immutable oracle contracts and explicit logical-to-physical data mappings.

Historical bindings retain their original names; payload I/O uses /data copies.
This module imports no ML packages, so status never initializes CUDA.
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACKAGE = ROOT / 'pose_aligned_layered_reconstruction'
OLD = PACKAGE / 'outputs/hssd_shared_ss_20261026'
BASE_OUT = PACKAGE / 'outputs/hssd_input_adaptation_20261025'
SHARED = PACKAGE / 'outputs/hssd_shared_scene_20261025'
PYTHON = Path('/home/zjr/anaconda3/envs/reconviagen/bin/python')
DATA_ROOT = Path('/data/zjr/pose_aligned_layered_reconstruction')
OUT = DATA_ROOT / 'outputs/hssd_visible_surface_oracle_20261026'
MODULE = 'pose_aligned_layered_reconstruction.experiments.hssd_visible_surface_oracle_20261026'
ARMS = ('geometry_oracle_visible',)
ALL_ARMS = ('appearance_control', 'geometry_observed', *ARMS)
UIDS = ['hssd_102344049_i076','hssd_102344049_i075','hssd_102344049_i080','hssd_102344049_i065']
SEEDS = (42,43,44)
STEPS, SEED = 200, 20261026
FORMAT = 'hssd_visible_first_surface_matched_residual.v1'
VARIANT = 'crop2072_visible'
CALLS = {'positive_calls':25,'negative_calls':19}
EXPECTED_PARTITIONS = dict(zip(UIDS,[(9225,9349,1426),(5247,9477,5276),(862,19138,0),(5691,14309,0)]))
FEATURES = ['surface_mean','surface_peak','free_mean','free_multiview_mean','unknown_mean',
    'behind_target_mean','behind_other_mean','other_surface_mean','frustum_coverage_mean',
    'confidence_mean','surface_free_conflict_mean','surface_multiview_mean']
PREVIOUS_IDENTITY = 'bb934bd6554576cf31742a3ea8fa01070ce8dfd766c908dd5cd776de3d867851'
_MAP = None
_HASH = {}


def now(): return datetime.now(timezone.utc).isoformat()
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def mappings():
    global _MAP
    path = OUT/'storage.json'
    if _MAP is None and path.exists():
        value = json.loads(path.read_text())
        if value.get('complete') is not True:
            raise ValueError('Storage staging is incomplete')
        _MAP = value['mapping']
    return _MAP or {}


def physical(path):
    path = Path(path).absolute()
    item = mappings().get(str(path))
    return Path(item['physical']) if item else path


def read(path): return json.loads(physical(path).read_text())


def write(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)


def locked(path,value):
    path = Path(path)
    if path.exists():
        if read(path)!=value: raise ValueError('Frozen record changed: '+str(path))
    else: write(path,value)


def _sha(path):
    path = Path(path); stat = path.stat()
    if str(path).startswith('/home/') and stat.st_size>1024*1024 and path.suffix not in ('.py','.cpp','.cu','.h'):
        raise ValueError('Large home payload must be staged before verification: '+str(path))
    key = (str(path),stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
    if key not in _HASH:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        _HASH[key] = h.hexdigest()
    return _HASH[key]


def binding(path):
    logical = str(Path(path).absolute())
    # References often arrive through verify(), which returns a /data path.
    reverse = {v['physical']:k for k,v in mappings().items()}
    logical = reverse.get(logical,logical)
    return {'path':logical,'sha256':_sha(physical(logical))}


def verify(value):
    path = physical(value['path'])
    if _sha(path)!=value['sha256']:raise ValueError('Binding changed: '+value['path'])
    return path


def event(stage,**fields):
    row={'time':now(),'stage':stage,**fields};write(OUT/'progress.json',row)
    print(json.dumps(row,ensure_ascii=False),flush=True)


def plan():
    value=read(OUT/'plan.json')
    if digest(value['plan'])!=value['identity']:raise ValueError('Oracle plan changed')
    return value


def ensure_plan():
    if (OUT/'plan.json').exists():return plan()
    if not mappings():raise ValueError('Run storage staging first')
    previous=read(OLD/'plan.json')['plan'];summary=read(OLD/'summary.json')
    if (summary['identity']!=PREVIOUS_IDENTITY or summary['new_meshes']!=24
            or summary['logical_cases']!=36):raise ValueError('Completed SharedSS reference required')
    if [o['uid'] for o in previous['objects']]!=UIDS:raise ValueError('Cohort changed')
    cases=[]
    for obj in previous['objects']:
        for seed in SEEDS:
            for arm in ALL_ARMS:
                source_arm=arm if arm not in ARMS else 'geometry_observed'
                rows=[r for r in summary['records'] if (r['uid'],r['seed'],r['arm'])==(obj['uid'],seed,source_arm)]
                if len(rows)!=1:raise ValueError('Nonunique completed reference')
                r=rows[0];c={'uid':obj['uid'],'seed':seed,'arm':arm,'original_index':obj['original_index'],
                    'is_reference':arm not in ARMS,'condition_record':r['condition_record']}
                if c['is_reference']:c.update(reference_ss=r['ss_record'],reference_mesh=r['mesh_record'])
                cases.append(c)
    frozen=next(c for c in previous['cases'] if c['arm']=='frozen' and c['uid']==UIDS[1] and c['seed']==42)
    refs={arm:{key:binding(OLD/'train'/arm/name) for key,name in
        [('run','run.json'),('history','history.jsonl'),('complete','complete.json'),('checkpoint','last.pt')]}
        for arm in ALL_ARMS[:2]}
    p={key:previous[key] for key in ('base_identity','base_summary','base_plan','shared_plan','runtime_amendment',
        'evaluation_gt','objects','parent_checkpoint','parent_training_record','evaluation_uid_context','slat_binding','evidence_policy')}
    p.update(previous_identity=PREVIOUS_IDENTITY,previous_evaluation_identity=binding(OLD/'evaluation_identity.json'),
        previous_summary=binding(OLD/'summary.json'),previous_plan=binding(OLD/'plan.json'),
        previous_visual_audit=binding(OLD/'visual_audit.json'),
        reference_training=refs,reference_initial_check=binding(OLD/'smoke/init_check.json'),
        previous_smoke_training_check=binding(OLD/'smoke/training_check.json'),
        parent_replay_reference={'ss':frozen['reference_ss'],'mesh':frozen['reference_mesh']},
        cases=cases,arms=list(ARMS),all_arms=list(ALL_ARMS),seeds=list(SEEDS),variant=VARIANT,steps=STEPS,
        new_ss_samples=12,new_meshes=12,original_references=24,logical_cases=36,
        oracle_policy={'depth_key':'depth_z_world','valid_confidence':1.,'invalid_confidence':0.,
            'invalid_depth_policy':'unknown','allow_hidden_geometry':False,
            'groups':{'group4':[0,4,7,11],'group2':[0,11]},
            'interpretation':'joint visible-depth and confidence-quality diagnostic; not a pure depth ablation',
            'conditioning_boundary':'stripped first-hit depth and validity only; no instance IDs or hidden/full target geometry'},
        storage_manifest=binding(OUT/'storage.json'),
        scope='four old training objects in one shared scene; one new matched 200-step visible-first-surface GT oracle; '+
            'both depth and confidence replaced; hidden regions remain unknown; privileged cameras/masks/canonical frames; '+
            'not RGB deployment, heldout generalization, or a LaRI contribution claim')
    p['evidence_policy']['source']='GT full-scene visible first-hit depth with valid confidence one; hidden depth/support never conditioning'
    result={'identity':digest(p),'plan':p};locked(OUT/'plan.json',result);return result


def _inherited_sources(record,seen=None):
    seen=set() if seen is None else seen
    key=record['path']
    if key in seen:return []
    seen.add(key);value=read(verify(record));result=[]
    for source in value.get('sources',[]):verify(source);result.append(source)
    for name in ('previous_runtime','runtime_amendment'):
        row=value.get(name)
        if isinstance(row,dict) and 'path' in row:result+=_inherited_sources(row,seen)
    return result


def freeze_runtime():
    p=ensure_plan();value=p['plan']
    if digest(read(verify(value['previous_evaluation_identity'])))!=PREVIOUS_IDENTITY:
        raise ValueError('Previous runtime identity changed')
    inherited=_inherited_sources(value['previous_evaluation_identity'])
    paths=sorted(HERE.glob('*.py'))+[HERE/'run.sh',HERE/'PROTOCOL.md']
    runtime={'plan_identity':p['identity'],'sources':[binding(x) for x in paths],
        'inherited_sources':inherited,'previous_runtime':value['previous_evaluation_identity'],
        'parent_checkpoint':value['parent_checkpoint'],'slat_binding':value['slat_binding'],
        'arms':list(ARMS),'steps':STEPS,'sampling_calls':CALLS,'storage_manifest':value['storage_manifest'],
        'training_target':'unchanged original SS latent; condition only stripped visible first surface and front free space'}
    locked(OUT/'evaluation_identity.json',runtime);return digest(runtime)


def policy(arm,steps=STEPS):
    if arm not in ARMS or steps!=STEPS:raise ValueError('Require matched 200-step contract')
    p=plan()
    return {'format':FORMAT,'arm':arm,'steps':steps,'seed':SEED,'lr':1e-4,'warmup':20,'betas':[.9,.95],
        'weight_decay':.01,'batch':1,'p_uncond':0.,'weights':'raw','amp':'bf16','residual_channels':128,
        'projection_bias':False,'parent_checkpoint':p['plan']['parent_checkpoint'],
        'plan_identity':p['identity'],'runtime_identity':digest(read(OUT/'evaluation_identity.json')),
        'observation_variant':VARIANT,'flow_loss':'same velocity MSE only','view_selection':'same archived visible indices',
        'scope':p['plan']['scope']}


def tensor_hash(state):
    from ..hssd_cell_query_20261023.common import tensor_hash as impl
    return impl(state)


def train_dir(arm):
    if arm not in ARMS:raise ValueError(arm)
    return OUT/'train'/arm


def case_key(c):return c['arm'],c['uid'],int(c['seed'])
def case_dir(c):return OUT/'evaluations'/c['arm']/c['uid']/str(c['seed'])
def logical_cases():return plan()['plan']['cases']
def new_jobs(arm=None):return [c for c in logical_cases() if not c['is_reference'] and (arm is None or c['arm']==arm)]
def object_row(c):return next(o for o in plan()['plan']['objects'] if o['uid']==c['uid'])
def get_result(c):
    mp=verify(c['reference_mesh']) if c['is_reference'] else case_dir(c)/'mesh.json'
    sp=verify(c['reference_ss']) if c['is_reference'] else case_dir(c)/'ss.json'
    return mp,read(mp),sp,read(sp)


def check_smoke():
    value=read(OUT/'smoke.json')
    if not value.get('passed') or value['identity']!=digest(read(OUT/'evaluation_identity.json')):
        raise ValueError('Oracle smoke missing or stale')
    for row in value['artifacts']:verify(row)
    for key in ('parent_ss_mesh_replay_exact','zero_initialization_exact','matched_trainable_capacity',
                'two_step_training_passed','shared_noise_order_exact'):
        if value.get(key) is not True:raise ValueError('Incomplete smoke gate: '+key)
    return value
