from pathlib import Path
from visual_evidence_probe.experiments.dense_evidence.config import STORE,OUT as PRIOR,PARAMS as OLD,read,write,now,digest,setup

OUT=STORE/'outputs/constrained_evidence_20261028'
ARMS=('dense_reference','anchored','anchored_reliable')
PARAMS=dict(OLD,anchor_count=1024,anchor_neighbors=16,anchor_scale=1/256,
    anchor_weight=.10,anchor_normal_weight=.01,trust_interval=5,
    anchor_mean_limit=.25/256,anchor_p95_limit=1/256,
    min_learning_rate=.03/32,photo_tolerance=.01,reliable_photo_cost=.25,
    reliable_photo_gain=.005,anchor_random_seed=20261028,
    pilot_indices=[4,5,6,10],dev_indices=list(range(4,12)))

def folder(index,seed=None):
    path=OUT/'objects'/f'{index:03d}'
    return path if seed is None else path/str(seed)

def emit(index,seed,stage,**kw):
    value=dict(time=now(),index=index,seed=seed,stage=stage,**kw)
    write(folder(index,seed)/'status.json',value);print(value,flush=True)

def register(stage):
    files=sorted(Path(__file__).parent.glob('*.py'))+sorted(Path(__file__).parent.glob('*.md'))+[Path(__file__).parents[2]/'run_constrained_evidence.sh']
    receipt=dict(time=now(),stage=stage,params=PARAMS,
        sources=[dict(path=str(p),sha256=digest(p)) for p in files],
        source_round7=str(PRIOR),frozen_model_weights=True,original_ss_fixed=True,
        physical_io_root=str(STORE),gt_policy='method never loads GT; audit and evaluation are separate processes')
    write(OUT/'source_snapshots'/f'{stage}_{now().replace(":","-")}.json',receipt)

