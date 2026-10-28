import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1]
ROOT=PACKAGE.parent
OUT=PACKAGE/'outputs/hssd_input_adaptation_20261025'
SHARED=PACKAGE/'outputs/hssd_shared_scene_20261025'
SHARED_CODE=HERE.with_name('hssd_shared_scene_20261025')
CELL=PACKAGE/'outputs/hssd_cell_query_20261023'
PYTHON='/home/zjr/anaconda3/envs/reconviagen/bin/python'
BLENDER='/tmp/blender-3.0.1-linux-x64/blender'
MODULE='pose_aligned_layered_reconstruction.experiments.hssd_input_adaptation_20261025'
ARMS=['yxc30k','appearance_control']
SEEDS=[42,43,44]
FRAMES=[0,4,7,11]
VARIANTS=['full_all','full_visible','crop518_visible','crop2072_visible']
HIGH_SCALE=4
SIDE=518

def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def binding(path):
    path=Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':sha(path)}
def verify(value):
    if sha(value['path'])!=value['sha256']:raise ValueError('Changed source: '+value['path'])
    return Path(value['path'])
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def locked(path,value):
    if Path(path).exists():
        if read(path)!=value:raise ValueError('Frozen identity differs: '+str(path))
    else:write(path,value)
def plan():
    value=read(OUT/'plan.json')
    if digest(value['plan'])!=value['identity']:raise ValueError('Plan changed')
    return value
def event(stage,**kwargs):
    row={'time':now(),'stage':stage,**kwargs};write(OUT/'progress.json',row)
    print(json.dumps(row,ensure_ascii=False),flush=True)

def source_domain(variant):
    return 'full' if variant.startswith('full_') else ('crop518' if variant.startswith('crop518') else 'crop2072')
def view_indices(obj,variant):return list(range(4)) if variant=='full_all' else obj['visible_indices']
def owner_variant(obj,variant):
    return 'full_all' if variant=='full_visible' and obj['visible_indices']==list(range(4)) else variant
def cases():
    return [(v,a,o,s) for v in VARIANTS for a in ARMS for o in plan()['plan']['objects'] for s in SEEDS]
def new_jobs():
    return [(v,a,o,s) for v,a,o,s in cases() if owner_variant(o,v)==v and v!='full_all']
def case_dir(variant,arm,uid,seed):return OUT/'evaluations'/variant/arm/uid/str(seed)
def input_path(obj,variant):
    if source_domain(variant)=='full':return verify(obj['source_sample'])
    return OUT/'inputs'/source_domain(variant)/obj['uid']/'sample.pt'
def get_result(variant,arm,obj,seed,kind='mesh'):
    owner=owner_variant(obj,variant)
    path=(SHARED/'evaluations'/arm/obj['uid']/str(seed)/f'{kind}.json' if owner=='full_all'
          else case_dir(owner,arm,obj['uid'],seed)/f'{kind}.json')
    return path,read(path)

def freeze_runtime():
    p=plan()
    original=read(verify(p['plan']['source_evaluation_identity']))
    for value in original['sources']:verify(value)
    sources=set(HERE.glob('*.py'))|{HERE/'run.sh',HERE/'PROTOCOL.md'}
    sources |= set(PACKAGE.glob('*.py'))|set((PACKAGE/'v2').glob('*.py'))
    sources |= {SHARED_CODE/'evaluate.py',SHARED_CODE/'render.py',
        HERE.parent/'hssd_cell_query_20261023/model.py',
        HERE.parent/'hssd_operator_audit_20261025/kernel_policy.py',
        HERE.parent/'round2_v2/memory_efficient_mesh.py',HERE.parent/'round2_v2/render_mesh_audit.py'}
    value={'plan_identity':p['identity'],'sources':[binding(x) for x in sorted(sources)],
        'appearance_checkpoint':binding(CELL/'train/occluded_appearance_control/last.pt'),
        'encoder_identity':binding(OUT/'encoder_identity.json'),
        'encoder_replay':binding(OUT/'encoder_replay.json'),
        'ss_steps':25,'cfg':5.,'cfg_interval':[.5,1.],'rescale_t':3.,
        'noise':'same original_index and seeds as SharedScene4',
        'reuse':'original baseline and bit-identical input/view cases are explicit references, not new replicates'}
    if value['appearance_checkpoint']!=original['checkpoint']:raise ValueError('Appearance weights changed')
    locked(OUT/'evaluation_identity.json',value)
    return digest(value)
