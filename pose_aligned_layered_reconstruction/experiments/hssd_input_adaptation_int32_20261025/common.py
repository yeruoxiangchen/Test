import importlib.metadata
import importlib.util
from pathlib import Path

# Keep the direct status/runner entry free of package __init__ (Torch imports).
_base_path = Path(__file__).resolve().parent.parent/'hssd_input_adaptation_20261025/common.py'
_spec = importlib.util.spec_from_file_location('_input_adapt_frozen_common', _base_path)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
ROOT, OUT = base.ROOT, base.OUT
now, read, write = base.now, base.read, base.write
binding, verify, digest = base.binding, base.verify, base.digest
freeze_runtime, locked = base.freeze_runtime, base.locked

HERE = Path(__file__).resolve().parent
REPAIR = OUT/'int32_fix_20261025'
MODULE = 'pose_aligned_layered_reconstruction.experiments.hssd_input_adaptation_int32_20261025'
PYTHON = base.PYTHON


def event(stage, **fields):
    import json
    row = {'time': now(), 'stage': stage, **fields}
    write(REPAIR/'progress.json', row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def freeze_amendment():
    ident = freeze_runtime()
    snapshot = REPAIR/'original_results.json'
    if not snapshot.exists():
        meshes, supports, missing = [], [], []
        for v, a, o, s in base.new_jobs():
            folder = base.case_dir(v, a, o['uid'], s)
            supports.append(binding(folder/'ss.json'))
            if (folder/'mesh.json').exists():
                meshes.append(binding(folder/'mesh.json'))
            else:
                missing.append([v, a, o['uid'], s])
        if len(meshes) != 50 or len(supports) != 54 or len(missing) != 4:
            raise ValueError('Expected the diagnosed 50/54 start state')
        write(snapshot, {'time': now(), 'meshes': meshes, 'supports': supports, 'missing': missing,
                         'previous_status': read(OUT/'status.json')})
    for b in read(snapshot)['meshes'] + read(snapshot)['supports']:
        verify(b)
    packages = Path(PYTHON).parent.parent/'lib/python3.10/site-packages'
    installed = ['spconv/pytorch/ops.py', 'spconv/pytorch/cppcore.py',
                 'spconv/csrc/sparse/convops.py', 'cumm/gemm/nvrtc_code.py']
    value = {'base_identity': ident, 'original_results': binding(snapshot),
        'sources': [binding(p) for p in sorted(HERE.glob('*.py'))]
                   + [binding(HERE/'run.sh'), binding(HERE/'PROTOCOL.md')],
        'installed_sources': [binding(packages/p) for p in installed],
        'versions': {p: importlib.metadata.version(p) for p in ['torch', 'spconv-cu120', 'cumm-cu120']},
        'policy': 'native input-row tiling only when A byte size >= INT32_MAX; no support/precision/grid changes',
        'production_tile_rows': 524288, 'center_mm': 'original full torch.mm',
        'required_regressions': ['kernel_check', 'decoder_small', 'decoder_large']}
    locked(REPAIR/'runtime_amendment.json', value)
    return digest(value)


def check_gate(name):
    row = read(REPAIR/(name+'.json'))
    if not row.get('passed'):
        raise ValueError('Numerical regression has not passed: '+name)
    for b in row['sources']:
        verify(b)
    return binding(REPAIR/(name+'.json'))


def gates():
    return {name: check_gate(name) for name in ['kernel_check', 'decoder_small', 'decoder_large']}
