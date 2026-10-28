import os
from pathlib import Path
from ..hssd_mesh_replay_20261025.common import binding, verify, read, write, locked, now, ROOT, PACKAGE, PYTHON

MODULE = 'pose_aligned_layered_reconstruction.experiments.hssd_operator_audit_20261025'
OUT = Path(os.environ.get('HSSD_OPERATOR_OUTPUT', PACKAGE / 'outputs/hssd_operator_audit_20261025')).resolve()
PREVIOUS = PACKAGE / 'outputs/hssd_mesh_replay_20261025'
DEPTH = PACKAGE / 'outputs/hssd_depth_balance_20261024'
ASSETS = Path('/home/zjr/.cache/huggingface/hub/models--Stable-X--trellis-vggt-v0-2/snapshots/647659a5ad5fbf67e22793e7b5e2cee4b30c5d13/ckpts')
SPCONV = Path('/home/zjr/anaconda3/envs/reconviagen/lib/python3.10/site-packages/spconv')
BASE_FILES = ['__init__.py', 'common.py', 'probe.py', 'runner.py', 'run.sh', 'PROTOCOL.md']
if not OUT.is_relative_to(PACKAGE / 'outputs') or OUT in (PREVIOUS, DEPTH):
    raise ValueError('Use an independent project output directory')


def freeze():
    previous = read(PREVIOUS / 'protocol.json')
    older = read(verify(previous['old_protocol']))
    for row in older['sources'] + previous['sources'] + previous['trellis_sources']:
        verify(row)
    result = {'format': 'hssd_operator_audit.v1', 'previous': binding(PREVIOUS / 'summary.json'),
        'sources': [binding(Path(__file__).with_name(f)) for f in BASE_FILES],
        'spconv_sources': [binding(SPCONV / f) for f in ['algo.py', 'constants.py', 'pytorch/ops.py', 'pytorch/conv.py', 'csrc/sparse/convops.py']],
        'block': 'input_blocks.0', 'processes': 3, 'repeats': 2,
        'decoder_processes': 2, 'decoder_repeats': 1, 'gpu_default': 3,
        'source_case': {'object_uid': 'hssd_102344049_i083', 'index': 2, 'seed': 42},
        'precision': 'original fp16 block inside bf16 autocast; original decoder outside autocast',
        'scope': 'operator diagnosis, no training; later repairs and fair evaluation get separate protocol identities'}
    locked(OUT / 'protocol.json', result)
    return result
