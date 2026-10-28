from pathlib import Path

from visual_evidence_probe.common import ROOT, read, write, binding, verify, lock_json, now

BASE = ROOT / 'outputs/correlated_views_cropped_control_20261025'
OUT = ROOT / 'outputs/input_consistency_baselines_20261025'
INDICES = (0, 1, 2, 3)
SEEDS = (42, 43)
PARAMS = dict(steps=100, side=256, lr=.03, rotation_max_degrees=25.,
              translation_max_per_axis=.15, scale_min=.75, scale_max=1.25,
              checkpoints=[0, 20, 40, 60, 80, 100], max_view_regression=.005,
              thresholds=[.80, .85, .90], surface_points=50000, surface_seed=20261025)


def folder(index):
    return OUT / 'objects' / f'{index:03d}'


def emit(stage, **kwargs):
    value = dict(time=now(), stage=stage, **kwargs)
    write(OUT / 'status.json', value)
    print(value, flush=True)


def prepare():
    import numpy as np
    from PIL import Image
    from visual_evidence_probe.render import mask_iou
    from .policy import choose_source, THRESHOLDS

    old = read(BASE / 'protocol.json')
    for source in old['sources'] + old['dependency_sources']:
        verify(source)
    sources = [binding(p) for p in sorted(Path(__file__).parent.glob('*.py'))]
    sources += [binding(Path(__file__).with_name('PROTOCOL.md')), binding(ROOT / 'run_consistency.sh')]
    protocol = dict(format='input_consistency_baselines.v1', base_protocol=binding(BASE / 'protocol.json'),
        sources=sources, inherited_sources=old['sources'] + old['dependency_sources'],
        indices=list(INDICES), seeds=list(SEEDS), params=PARAMS,
        scope='exploratory follow-up on the same four inspected HSSD objects; no held-out claim',
        fitting_inputs='two original masks and calibrated raw cameras; no GT mesh or unseen masks',
        generated_branch='same raw-selected donor before and after alignment; six synthetic views',
        proper_similarity='uniform scale, proper rotation and translation about canonical origin',
        oracle='GT-only candidate ceiling, diagnostic, never an inference policy')
    lock_json(OUT / 'protocol.json', protocol)
    decisions, targets = [], []
    for index in INDICES:
        rec = read(BASE / 'objects' / f'{index:03d}' / 'object.json')
        camera = Path(rec['source']['sample']).parent / 'camera.npz'
        camera_binding = binding(camera)
        with np.load(camera) as arrays:
            k = arrays['intrinsics'][rec['camera_order']].tolist()
            w = arrays['extrinsics'][rec['camera_order']].tolist()
        donors = {key: binding(BASE / 'objects' / f'{index:03d}' / f'results/real2/{seed}.obj')
                  for key, seed in [('a', 42), ('b', 43)]}
        obs = dict(index=index, uid=rec['uid'], intrinsics=k, extrinsics=w,
                   camera_source=camera_binding, camera_order=rec['camera_order'],
                   masks=rec['masks'][:2], donors=donors,
                   original_features=binding(BASE / 'objects' / f'{index:03d}' / 'original_crop.pt'))
        lock_json(folder(index) / 'observations.json', obs)
        observed = np.stack([np.asarray(Image.open(verify(m)).convert('L')) > 127 for m in obs['masks']])
        scores, masks_bindings = {}, {}
        for key in donors:
            source = BASE / 'objects' / f'{index:03d}' / 'renders_raw' / key / 'masks.npz'
            masks_bindings[key] = binding(source)
            with np.load(source) as cache:
                scores[key] = mask_iou(cache['masks'][:2], observed).tolist()
        decisions.append(dict(index=index, uid=rec['uid'], ious=scores, masks=masks_bindings,
            selected=choose_source(scores), gated={str(t): choose_source(scores, t) for t in THRESHOLDS}))
        targets.append(dict(index=index, gt=rec['gt'], gt_coords=rec['gt_coords']))
    # These decisions are sealed before any GT scores are loaded.
    lock_json(OUT / 'raw_decisions.json', dict(uses_gt=False, records=decisions))
    lock_json(OUT / 'evaluation_targets.json', dict(records=targets))
    emit('prepared', objects=len(INDICES), expected_refined_donors=8, expected_new_meshes=8)


def verify_protocol():
    p = read(OUT / 'protocol.json')
    for source in p['sources'] + p['inherited_sources'] + [p['base_protocol']]:
        verify(source)
    if p['params'] != PARAMS:
        raise ValueError('changed parameters')


def observations(index):
    import numpy as np
    import torch
    from PIL import Image
    rec = read(folder(index) / 'observations.json')
    k = torch.tensor(rec['intrinsics'], dtype=torch.float32)
    w = torch.tensor(rec['extrinsics'], dtype=torch.float32)
    masks = np.stack([np.asarray(Image.open(verify(m)).convert('L')) > 127 for m in rec['masks']])
    if masks.shape != (2, 518, 518) or not np.isfinite(k.numpy()).all() or not np.isfinite(w.numpy()).all():
        raise ValueError('unexpected observation contract')
    return rec, k, w, masks
