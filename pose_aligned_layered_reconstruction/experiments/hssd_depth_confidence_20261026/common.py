"""Frozen four-cell depth/confidence experiment; status uses only stdlib."""
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_depth_conf_prior',HERE.parent/'hssd_visible_surface_oracle_20261026/common.py')
prior=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(prior)
ROOT,PACKAGE,BASE_OUT,SHARED,PYTHON,DATA_ROOT=prior.ROOT,prior.PACKAGE,prior.BASE_OUT,prior.SHARED,prior.PYTHON,prior.DATA_ROOT
OLD=prior.OLD
ORACLE=prior.OUT
OUT=DATA_ROOT/'outputs/hssd_depth_confidence_20261026'
MODULE='pose_aligned_layered_reconstruction.experiments.hssd_depth_confidence_20261026'
ARMS=('geometry_gt_depth_pred_confidence','geometry_pred_depth_unit_confidence')
ALL_ARMS=('appearance_control','geometry_observed',*ARMS,'geometry_oracle_visible')
FACTORIAL_ARMS=('geometry_observed',*ARMS,'geometry_oracle_visible')
UIDS,SEEDS,STEPS,SEED,VARIANT,CALLS,EXPECTED_PARTITIONS,FEATURES=prior.UIDS,prior.SEEDS,prior.STEPS,prior.SEED,prior.VARIANT,prior.CALLS,prior.EXPECTED_PARTITIONS,prior.FEATURES
FORMAT='hssd_depth_confidence_factorial.v1'
PREVIOUS_IDENTITY='c05d02da649f96f5ad75d01acc53c20cc1606e78720d64472b991fc17cb2f4b2'
SHAREDSS_IDENTITY=prior.PREVIOUS_IDENTITY
now,digest,write=prior.now,prior.digest,prior.write
_MAP=None


def tensor_hash(state):
    # prior was loaded as a standalone stdlib module for the monitor. Resolve
    # the unchanged hash implementation by its package name when it is needed.
    from importlib import import_module
    impl=import_module('pose_aligned_layered_reconstruction.experiments.hssd_cell_query_20261023.common').tensor_hash
    return impl(state)


def mappings():
    global _MAP
    path=OUT/'storage.json'
    if _MAP is None and path.exists():
        row=json.loads(path.read_text())
        if row.get('complete') is not True:raise ValueError('Storage staging incomplete')
        _MAP=row['mapping']
    return _MAP if _MAP is not None else prior.mappings()


def physical(path):
    p=Path(path).absolute();row=mappings().get(str(p))
    return Path(row['physical']) if row else p


def read(path):return json.loads(physical(path).read_text())


def binding(path):
    logical=str(Path(path).absolute())
    reverse={row['physical']:key for key,row in mappings().items()}
    logical=reverse.get(logical,logical)
    return {'path':logical,'sha256':prior._sha(physical(logical))}


def verify(b):
    path=physical(b['path'])
    if prior._sha(path)!=b['sha256']:raise ValueError('Changed bound input: '+b['path'])
    return path


def locked(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('Frozen record changed: '+str(path))
    else:write(path,value)


def event(stage,**fields):
    row={'time':now(),'stage':stage,**fields};write(OUT/'progress.json',row)
    print(json.dumps(row,ensure_ascii=False),flush=True)


def plan():
    row=read(OUT/'plan.json')
    if digest(row['plan'])!=row['identity']:raise ValueError('Factorial plan changed')
    return row


def ensure_plan():
    if (OUT/'plan.json').exists():return plan()
    if not (OUT/'storage.json').exists():raise ValueError('Stage additional depth/evidence first')
    old=read(ORACLE/'plan.json')['plan'];summary=read(ORACLE/'summary.json')
    if summary['identity']!=PREVIOUS_IDENTITY or summary['logical_cases']!=36 or summary['new_meshes']!=12:
        raise ValueError('Visible-first-surface oracle must be complete')
    review=read(ORACLE/'review_20261026/review.json')
    if not review['artifacts_verified'] or not review['all_visuals_inspected'] or review['identity']!=PREVIOUS_IDENTITY:
        raise ValueError('Prior result must be reviewed before the final bounded diagnostic')
    if [o['uid'] for o in old['objects']]!=UIDS:raise ValueError('Original cohort changed')
    cases=[]
    for obj in old['objects']:
        for seed in SEEDS:
            for arm in ALL_ARMS:
                source_arm=arm if arm not in ARMS else 'geometry_observed'
                found=[r for r in summary['records'] if (r['uid'],r['seed'],r['arm'])==(obj['uid'],seed,source_arm)]
                if len(found)!=1:raise ValueError('Missing unique archived reference')
                source=found[0]
                case={'uid':obj['uid'],'seed':seed,'arm':arm,'original_index':obj['original_index'],
                    'is_reference':arm not in ARMS,'condition_record':source['condition_record']}
                if case['is_reference']:
                    case.update(reference_ss=source['ss_record'],reference_mesh=source['mesh_record'],
                        expected_identity=PREVIOUS_IDENTITY if arm=='geometry_oracle_visible' else SHAREDSS_IDENTITY)
                cases.append(case)
    refs=dict(old['reference_training'])
    refs['geometry_oracle_visible']={key:binding(ORACLE/'train/geometry_oracle_visible'/name)
        for key,name in [('run','run.json'),('history','history.jsonl'),('complete','complete.json'),('checkpoint','last.pt')]}
    p={key:old[key] for key in ('base_identity','base_summary','base_plan','shared_plan','runtime_amendment',
        'evaluation_gt','objects','parent_checkpoint','parent_training_record','evaluation_uid_context','slat_binding',
        'evidence_policy','parent_replay_reference','reference_initial_check','previous_smoke_training_check')}
    p.update(previous_identity=PREVIOUS_IDENTITY,sharedss_identity=SHAREDSS_IDENTITY,
        previous_evaluation_identity=binding(ORACLE/'evaluation_identity.json'),
        previous_summary=binding(ORACLE/'summary.json'),previous_plan=binding(ORACLE/'plan.json'),
        previous_review=binding(ORACLE/'review_20261026/review.json'),
        reference_visual_audits=[old['previous_visual_audit'],binding(ORACLE/'visual_audit.json')],
        reference_training=refs,cases=cases,arms=list(ARMS),all_arms=list(ALL_ARMS),factorial_arms=list(FACTORIAL_ARMS),
        seeds=list(SEEDS),variant=VARIANT,steps=STEPS,new_ss_samples=24,new_meshes=24,
        original_references=36,logical_cases=60,primary_factorial_cases=48,context_baseline_cases=12,
        predicted_depth={group:binding(OLD/'depth'/f'{group}.json') for group in ('group4','group2')},
        stripped_observations={str(frame):binding(ORACLE/'observations/first_surface'/f'{frame:03d}.json') for frame in (0,4,7,11)},
        reference_evidence={arm:{uid:binding(folder/'evidence'/uid/'receipt.json') for uid in UIDS}
            for arm,folder in [('geometry_observed',OLD),('geometry_oracle_visible',ORACLE)]},
        storage_manifest=binding(OUT/'storage.json'),
        depth_confidence_policy={'arms':{
            'geometry_observed':{'depth':'predicted','confidence':'predicted'},
            ARMS[0]:{'depth':'gt','confidence':'predicted'},
            ARMS[1]:{'depth':'predicted','confidence':'unit'},
            'geometry_oracle_visible':{'depth':'gt','confidence':'unit'}},
            'groups':{'group4':[0,4,7,11],'group2':[0,11]},'invalid_depth_policy':'unknown',
            'allow_hidden_geometry':False,'diagonal_replay':'torch.equal for features, weights, valid on both archived cells',
            'confidence':'original per-view empirical rank, unchanged threshold 0.3; unit removes both rank filtering and rank attenuation',
            'validity':'validity belongs to the selected depth source; predicted-depth cells never use GT valid'},
        scope='last bounded four-old-object/one-scene depth-by-confidence mechanism diagnostic; '+
            'two new matched 200-step arms, two archived diagonal cells, archived appearance context; '+
            'GT first-hit depth cells are privileged; unit confidence is an ablation, not calibrated certainty; '+
            'not heldout generalization or a LaRI contribution claim')
    p['evidence_policy']['source']='factorial crossing of frozen predicted/GT visible first-hit depth and predicted/unit confidence'
    result={'identity':digest(p),'plan':p};locked(OUT/'plan.json',result);return result


def freeze_runtime():
    current=ensure_plan();p=current['plan'];previous=read(verify(p['previous_evaluation_identity']))
    if digest(previous)!=PREVIOUS_IDENTITY:raise ValueError('Previous numerical runtime changed')
    inherited=previous['sources']+previous['inherited_sources']
    for source in inherited:verify(source)
    paths=sorted(HERE.glob('*.py'))+[HERE/'run.sh',HERE/'PROTOCOL.md']
    value={'plan_identity':current['identity'],'sources':[binding(x) for x in paths],
        'inherited_sources':inherited,'previous_runtime':p['previous_evaluation_identity'],
        'parent_checkpoint':p['parent_checkpoint'],'slat_binding':p['slat_binding'],
        'arms':list(ARMS),'steps':STEPS,'sampling_calls':CALLS,'storage_manifest':p['storage_manifest'],
        'training_target':'unchanged archived SS latent; crossed visible depth/confidence conditions only'}
    locked(OUT/'evaluation_identity.json',value);return digest(value)


def policy(arm,steps=STEPS):
    if arm not in ARMS or steps!=STEPS:raise ValueError('Require original matched 200-step budget')
    p=plan()
    return {'format':FORMAT,'arm':arm,'steps':steps,'seed':SEED,'lr':1e-4,'warmup':20,'betas':[.9,.95],
        'weight_decay':.01,'batch':1,'p_uncond':0.,'weights':'raw','amp':'bf16','residual_channels':128,
        'projection_bias':False,'parent_checkpoint':p['plan']['parent_checkpoint'],
        'plan_identity':p['identity'],'runtime_identity':digest(read(OUT/'evaluation_identity.json')),
        'observation_variant':VARIANT,'flow_loss':'same velocity MSE only','view_selection':'same archived visible indices',
        'scope':p['plan']['scope']}


def train_dir(arm):
    if arm not in ARMS:raise ValueError(arm)
    return OUT/'train'/arm


def case_key(case):return case['arm'],case['uid'],int(case['seed'])
def case_dir(case):return OUT/'evaluations'/case['arm']/case['uid']/str(case['seed'])
def logical_cases():return plan()['plan']['cases']
def new_jobs(arm=None):return [case for case in logical_cases() if not case['is_reference'] and (arm is None or case['arm']==arm)]
def object_row(case):return next(obj for obj in plan()['plan']['objects'] if obj['uid']==case['uid'])
def get_result(case):
    mp=verify(case['reference_mesh']) if case['is_reference'] else case_dir(case)/'mesh.json'
    sp=verify(case['reference_ss']) if case['is_reference'] else case_dir(case)/'ss.json'
    return mp,read(mp),sp,read(sp)


def check_smoke():
    row=read(OUT/'smoke.json')
    if (not row.get('passed') or row['identity']!=digest(read(OUT/'evaluation_identity.json'))
            or row.get('arms')!=list(ARMS) or row.get('short_steps_per_arm')!=2):
        raise ValueError('Two-arm smoke missing or stale')
    for item in row['artifacts']:verify(item)
    for key in ('parent_ss_mesh_replay_exact','zero_initialization_exact','matched_trainable_capacity',
                'two_step_training_passed','shared_noise_order_exact','diagonal_evidence_replay_exact'):
        if row.get(key) is not True:raise ValueError('Incomplete smoke: '+key)
    return row
