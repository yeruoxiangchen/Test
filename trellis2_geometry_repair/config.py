import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
STORE = Path('/data/zjr/trellis2_geometry_repair')
OUT = STORE / 'outputs/native512_pilot_20261028'
MODELS = STORE / 'models'
T2_ROOT = ROOT.parent / 'TRELLIS2'
LEGACY = Path('/data/zjr/visual_evidence_probe/outputs/dense_evidence_20261027')
PARAMS = dict(indices=[5,6,10,11], seeds=[42,43], resolution=512, reference_view=0,
              side=256, image_size=518, steps=40, checkpoints=[0,10,20,40],
              lr=.02, feature_cap=.25, loss_scale=1024., feature_prior=.0001,
              region_fraction=.20, active_radius=2/32, min_error=1/256,
              max_outside_mean=.001, max_outside_p95=.004, mask_tolerance=.002,
              random_seed=20261028, recovery_steps=20)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def folder(index,seed):
    return OUT/'objects'/f'{index:03d}'/str(seed)


def setup():
    sys.dont_write_bytecode=True
    if str(T2_ROOT) not in sys.path:sys.path.insert(0,str(T2_ROOT))
    for path in (OUT,STORE/'tmp',STORE/'cache/torch_extensions',STORE/'cache/matplotlib'):
        path.mkdir(parents=True,exist_ok=True)


def snapshot(stage):
    files=list(ROOT.glob('*.py'))+[ROOT/'README.md',ROOT/'PROTOCOL.md',ROOT/'run.sh']
    files=[p for p in files if p.exists()]
    identity=hashlib.sha256(''.join(p.name+sha256(p) for p in sorted(files)).encode()).hexdigest()
    archive=OUT/'source_archives'/identity
    if not archive.exists():
        archive.mkdir(parents=True,exist_ok=True)
        for p in files:(archive/p.name).write_bytes(p.read_bytes())
    write(OUT/'source_snapshots'/f'{stage}_{now().replace(":","-")}.json',
          dict(time=now(),stage=stage,params=PARAMS,sources=[dict(path=str(p),sha256=sha256(p)) for p in files if p.exists()],
               source_archive=str(archive),
               upstream=[dict(path=str(p),sha256=sha256(p)) for p in (
                   T2_ROOT/'trellis2/models/sc_vaes/fdg_vae.py',
                   T2_ROOT/'trellis2/models/sc_vaes/sparse_unet_vae.py',
                   T2_ROOT/'o-voxel/o_voxel/convert/flexible_dual_grid.py')],
               output=str(OUT),model_weights='frozen official TRELLIS.2 geometry',texture_loaded=False))


def emit(index,seed,stage,**kwargs):
    value=dict(time=now(),index=index,seed=seed,stage=stage,**kwargs)
    write(folder(index,seed)/'status.json',value);print(value,flush=True)
