"""Immutable plans and small contracts; monitor imports no Torch."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_shared_ss_gt_common',HERE.parent/'hssd_gt_support_20261026/common.py')
oracle=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(oracle)
base=oracle.base
ROOT,PACKAGE,BASE_OUT,SHARED,PYTHON=oracle.ROOT,oracle.PACKAGE,oracle.BASE_OUT,oracle.SHARED,oracle.PYTHON
now,read,write,binding,verify,digest,locked=oracle.now,oracle.read,oracle.write,oracle.binding,oracle.verify,oracle.digest,oracle.locked
OUT=PACKAGE/'outputs/hssd_shared_ss_20261026'
MODULE='pose_aligned_layered_reconstruction.experiments.hssd_shared_ss_20261026'
ARMS=('appearance_control','geometry_observed')
ALL_ARMS=('frozen',*ARMS)
UIDS=oracle.UIDS
SEEDS=(42,43,44)
STEPS=200
SEED=20261026
FORMAT='hssd_shared_ss_matched_residual.v1'
VARIANT='crop2072_visible'
CALLS={'positive_calls':25,'negative_calls':19}
EXPECTED_PARTITIONS=oracle.EXPECTED_PARTITIONS
SNAPSHOT=Path('/home/zjr/.cache/huggingface/hub/models--Stable-X--vggt-object-v0-1/snapshots/aac569c280b52ccda1c84a81cd0e27e947bc2cf5')
FEATURES=['surface_mean','surface_peak','free_mean','free_multiview_mean','unknown_mean',
    'behind_target_mean','behind_other_mean','other_surface_mean','frustum_coverage_mean',
    'confidence_mean','surface_free_conflict_mean','surface_multiview_mean']


def event(stage,**fields):
    row={'time':now(),'stage':stage,**fields};write(OUT/'progress.json',row)
    print(json.dumps(row,ensure_ascii=False),flush=True)


def plan():
    value=read(OUT/'plan.json')
    if digest(value['plan'])!=value['identity']:raise ValueError('Shared SS plan changed')
    return value


def ensure_plan():
    if (OUT/'plan.json').exists():
        result=plan();p=result['plan']
        verify(p['previous_evaluation_identity']);verify(p['previous_summary']);verify(p['parent_checkpoint'])
        return result
    previous_identity=oracle.freeze_runtime()
    previous=oracle.plan()['plan']
    summary=read(oracle.OUT/'summary.json')
    if summary['identity']!=previous_identity or summary['new_meshes']!=12 or summary['logical_cases']!=24:
        raise ValueError('GT support diagnostic must be complete')
    objects=previous['objects']
    if [o['uid'] for o in objects]!=UIDS or [o['original_index'] for o in objects]!=[0,5,6,7]:
        raise ValueError('Keep the original complete four-object cohort')
    archived=read(verify(previous['base_summary']))
    cases=[]
    for obj in objects:
        for seed in SEEDS:
            matches=[r for r in archived['records'] if r['variant']==VARIANT and r['arm']=='appearance_control'
                and r['uid']==obj['uid'] and r['seed']==seed]
            if len(matches)!=1:raise ValueError('Missing unique completed old reference')
            old=matches[0]
            verify(old['mesh_record']);verify(old['ss_record'])
            for arm in ALL_ARMS:
                c={'arm':arm,'uid':obj['uid'],'seed':seed,'original_index':obj['original_index'],
                    'is_reference':arm=='frozen','condition_record':old['ss_record']}
                if c['is_reference']:c.update(reference_mesh=old['mesh_record'],reference_ss=old['ss_record'])
                cases.append(c)
    parent=base.CELL/'train/occluded_appearance_control/last.pt'
    if binding(parent)!=read(BASE_OUT/'evaluation_identity.json')['appearance_checkpoint']:
        raise ValueError('Frozen appearance parent changed')
    masks={o['uid']:[{'frame':f,**binding(BASE_OUT/'hires/masks'/o['uid']/f'{f:03d}.png')}
                    for f in base.FRAMES] for o in objects}
    value={'previous_identity':previous_identity,'previous_evaluation_identity':binding(oracle.OUT/'evaluation_identity.json'),
        'previous_summary':binding(oracle.OUT/'summary.json'),'base_identity':previous['base_identity'],
        'base_summary':previous['base_summary'],'base_plan':previous['base_plan'],
        'shared_plan':previous['shared_plan'],'runtime_amendment':previous['runtime_amendment'],
        'evaluation_gt':previous['evaluation_gt'],'objects':objects,'cases':cases,
        'parent_checkpoint':binding(parent),'parent_training_record':binding(parent.parent/'run.json'),
        'evaluation_uid_context':UIDS,'slat_binding':previous['slat_binding'],'arms':list(ARMS),
        'all_arms':list(ALL_ARMS),'seeds':list(SEEDS),'variant':VARIANT,'steps':STEPS,
        'depth_policy':{'snapshot':str(SNAPSHOT),'snapshot_files':[binding(SNAPSHOT/f) for f in ['config.json','model.safetensors']],
            'groups':{'group4':[0,4,7,11],'group2':[0,11]},
            'rgb':[{'frame':f,**binding(BASE_OUT/'hires/rgb'/f'{f:03d}.png')} for f in base.FRAMES],
            'confidence':{'kind':'per_view_empirical_cdf','minimum_rank':0.3},'min_baseline_world':1e-4,
            'scale':'median known-camera-baseline / VGGT-camera-baseline; never fit to GT depth or geometry'},
        'evidence_policy':{'features':FEATURES,'fine_side':64,'token_side':16,'surface_tolerance_canonical':2/64,
            'confidence_minimum_rank':0.3,'full_visible_masks':masks,
            'source':'RGB VGGT-object depth plus known cameras and visible masks; no GT depth/target support',
            'fusion':'per-voxel soft observed surface/free/unknown and coverage, pooled over 4x4x4 cells; independent of DINO weights',
            'supervision':'original SS flow target only; no extra occupancy loss in first matched comparison'},
        'new_ss_samples':24,'new_meshes':24,'original_references':12,'logical_cases':36,
        'scope':'four old objects in one scene; matched shared-input training feasibility, not heldout/generalization; geometry uses predicted RGB depth but known cameras/masks/canonical frames; no LaRI contribution claim'}
    result={'identity':digest(value),'plan':value};locked(OUT/'plan.json',result);return result


def freeze_runtime():
    p=ensure_plan()
    # Recheck all inherited frozen runtime sources without rehashing the 5GB
    # VGGT checkpoint at each training/monitor invocation. depth.py verifies it
    # before an actual model load and binds its unchanged stat on cache reuse.
    if oracle.freeze_runtime()!=p['plan']['previous_identity']:
        raise ValueError('Inherited numerical runtime changed')
    paths=sorted(HERE.glob('*.py'))+[HERE/'run.sh',HERE/'PROTOCOL.md']
    paths+=sorted((ROOT/'ReconViaGen/wheels/vggt/vggt').rglob('*.py'))
    value={'plan_identity':p['identity'],'sources':[binding(x) for x in paths],
        'previous_runtime':p['plan']['previous_evaluation_identity'],'parent_checkpoint':p['plan']['parent_checkpoint'],
        'depth_snapshot_files':p['plan']['depth_policy']['snapshot_files'],'slat_binding':p['plan']['slat_binding'],
        'arms':list(ARMS),'steps':STEPS,'sampling_calls':CALLS,'training_target':'frozen original SS latent; image-only plus separately bound predicted observation evidence'}
    locked(OUT/'evaluation_identity.json',value);return digest(value)


def policy(arm,steps=STEPS):
    if arm not in ARMS or steps!=STEPS:raise ValueError('Require matched 200-step contract; smoke uses max-steps not a new target')
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


def case_key(case):return case['arm'],case['uid'],int(case['seed'])
def case_dir(case):return OUT/'evaluations'/case['arm']/case['uid']/str(case['seed'])
def logical_cases():return plan()['plan']['cases']
def new_jobs(arm=None):return [c for c in logical_cases() if not c['is_reference'] and (arm is None or c['arm']==arm)]
def object_row(case):return next(o for o in plan()['plan']['objects'] if o['uid']==case['uid'])


def get_result(case):
    mp=verify(case['reference_mesh']) if case['is_reference'] else case_dir(case)/'mesh.json'
    sp=verify(case['reference_ss']) if case['is_reference'] else case_dir(case)/'ss.json'
    return mp,read(mp),sp,read(sp)


def check_smoke():
    r=read(OUT/'smoke.json')
    if not r.get('passed') or r['identity']!=digest(read(OUT/'evaluation_identity.json')):
        raise ValueError('Matched shared SS smoke gate missing or stale')
    for b in r['artifacts']:verify(b)
    if not all(r[k] for k in ['parent_ss_mesh_replay_exact','zero_initialization_exact',
                              'matched_trainable_capacity','two_step_training_passed','shared_noise_order_exact']):
        raise ValueError('Required smoke gate incomplete')
    return r
