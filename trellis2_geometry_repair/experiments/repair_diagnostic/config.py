from pathlib import Path
import hashlib
import os
import fcntl
from contextlib import contextmanager

from ...config import OUT as BASE, ROOT, STORE, read, write, now

OUT=STORE/'outputs/repair_diagnostic_20261028'
ARMS=('latent_input','mesh_input','latent_oracle','mesh_oracle')
PARAMS=dict(indices=[5,6,10,11],seeds=[42,43],steps=40,checkpoints=[0,10,20,40],
            side=256,depth_scale=.03,charbonnier_epsilon=.01,missing_margin=.05,
            latent_lr=.02,feature_cap=.25,feature_prior=.0001,loss_scale=1024.,
            mesh_lr=.02,mesh_cap=.15,mesh_anchors=96,mesh_neighbors=4,
            mesh_taper=.015,mesh_prior=.0001,mesh_smooth=.001,
            min_target_gain=.10,max_local_lost=.005,
            mask_tolerance=.002,max_outside_mean=.001,max_outside_p95=.004)


def case(index,seed):return OUT/'objects'/f'{index:03d}'/str(seed)
def original(index,seed):return BASE/'objects'/f'{index:03d}'/str(seed)


@contextmanager
def case_lock(index,seed):
    path=case(index,seed)/'worker.lock';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot():
    paths=sorted(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'PROTOCOL.md',
          ROOT/'run_diagnostic.sh',ROOT/'runtime.py',ROOT/'compat.py',ROOT/'geometry.py',ROOT/'metrics.py',ROOT/'edit.py',ROOT/'evidence.py',ROOT/'figures.py',
          ROOT.parent/'visual_evidence_probe/experiments/editor_support/objective.py']
    review=Path(__file__).parent/'VISUAL_REVIEW.md'
    if review.exists():paths.append(review)
    records=[dict(path=str(p),sha256=digest(p)) for p in paths]
    identity=hashlib.sha256(''.join(r['sha256'] for r in records).encode()).hexdigest()
    archive=OUT/'source_archives'/identity;archive.mkdir(parents=True,exist_ok=True)
    for p in paths:
        relative=p.relative_to(ROOT.parent)
        dest=archive/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():dest.write_bytes(p.read_bytes())
    write(OUT/'source_snapshot.json',dict(time=now(),sources=records,archive=str(archive),params=PARAMS))


def emit(index,seed,arm,stage,**extra):
    value=dict(time=now(),index=index,seed=seed,arm=arm,stage=stage,**extra)
    write(case(index,seed)/'status.json',value);print(value,flush=True)


def setup():
    import sys
    sys.dont_write_bytecode=True
    from ...config import T2_ROOT
    if str(T2_ROOT) not in sys.path:sys.path.insert(0,str(T2_ROOT))
    OUT.mkdir(parents=True,exist_ok=True)
