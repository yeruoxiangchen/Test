"""Freeze four co-located targets, one shared path, and existing latent bindings."""
import json
import numpy as np
from .common import *
from .geometry import trajectory, canonical_extrinsics, contract_checks


def prepare():
    old = read(PILOT / 'plan.json')
    if old['identity'] != identity(old['plan']):
        raise ValueError('Source pilot identity mismatch')
    rows = []
    for index in INDICES:
        row = dict(old['plan']['objects'][index])
        root = PILOT / 'objects' / row['uid']
        row.update(original_index=index, ss=binding(root / 'ss.npz'), slat=binding(root / 'slat.npz'))
        verify(row['mesh'])
        if row['instance'] not in [76, 75, 80, 65]:
            raise ValueError('Unexpected target cohort')
        rows.append(row)
    eyes, k, w = trajectory()
    frozen = {'format': 'hssd_shared_rgb4.v1', 'scene': '102344049',
        'trajectory_uid': 'hssd_102344049:dining_shared_walk_v1', 'objects': rows,
        'source_plan': binding(PILOT / 'plan.json'), 'blend': binding(BLEND),
        'image_side': 518, 'cycles_samples': 32, 'frame_count': len(eyes),
        'world_eye': eyes.tolist(), 'world_w2c': w.tolist(), 'intrinsics': k.tolist(),
        'input_frames': FRAMES, 'sampling_seeds': SEEDS, 'arms': ARMS,
        'selection': 'all four existing encoded targets in the dining cluster; fixed room walk, no per-target camera search',
        'conditioning': 'calibrated RGB + visible instance mask + GT-derived canonical frame diagnostic; no depth or hidden labels',
        'scope': 'new shared views of old training objects; not heldout, not raw-RGB end-to-end',
        'latent_policy': 'reuse previously audited SS/SLat targets; never re-encode meshes'}
    frozen = {'identity': identity(frozen), 'plan': frozen}
    locked(OUT / 'plan.json', frozen)
    result = {'time': now(), 'plan_identity': frozen['identity'], 'objects': len(rows),
              'frames': len(eyes), 'common_input_frames': FRAMES,
              'projection_checks': contract_checks(rows), 'passed': True}
    write(OUT / 'preflight.json', result)
    event('prepared', objects=len(rows), frames=len(eyes), input_frames=FRAMES)


if __name__ == '__main__':
    prepare()
