from pathlib import Path
import os
from visual_evidence_probe.experiments.constrained_evidence.config import STORE,PRIOR as ROUND7,OUT as ROUND8,PARAMS as PREVIOUS,read,write,now,digest,setup

OUT=STORE/'outputs/paired_depth_diagnostic_20261028'
ARMS=('estimated_free','estimated_anchored','oracle_free','oracle_anchored')
PARAMS=dict(PREVIOUS,indices=[5,6,10,11],seeds=[42,43],random_seed=20261028,
    normalization='same estimated-target initial loss for all4 arms within a case')

def folder(index,seed=None):
    p=OUT/'objects'/f'{index:03d}'
    return p if seed is None else p/str(seed)

def emit(index,seed,stage,**kw):
    value=dict(time=now(),index=index,seed=seed,stage=stage,**kw)
    write(folder(index,seed)/'status.json',value);print(value,flush=True)

def register(stage):
    files=sorted(Path(__file__).parent.glob('*.py'))+sorted(Path(__file__).parent.glob('*.md'))+[Path(__file__).parents[2]/'run_paired_depth_diagnostic.sh']
    write(OUT/'source_snapshots'/f'{stage}_{now().replace(":","-")}.json',dict(time=now(),stage=stage,params=PARAMS,
        sources=[dict(path=str(p),sha256=digest(p)) for p in files],
        privileged_diagnostic=True,all_arms_gt_pruned_support=True,
        execution=dict(cuda_launch_blocking=os.environ.get('CUDA_LAUNCH_BLOCKING','0')),
        weights_frozen=True,ss_fixed=True,all_runtime_io=str(STORE),
        cohort='four deliberately selected, inspected development objects; not validation'))
