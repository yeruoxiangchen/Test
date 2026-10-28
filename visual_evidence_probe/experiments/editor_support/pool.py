"""Input-only local view choice and fresh depth proposals without holdout leakage."""
import numpy as np
import torch
import torch.nn.functional as F

from .config import PARAMS, write, folder
from .data import constraints, points_transform
from .objective import select_edit_views, local_edit_region


def build_pool(index, seed, baseline, k, w, images, masks, fit):
    from visual_evidence_probe.experiments.local_repair import photometric
    from visual_evidence_probe.analysis_scripts.local_repair_cropped import propose
    from visual_evidence_probe.experiments.local_repair.config import PARAMS as MATCH
    old = constraints(index, seed)
    if not len(old['target']):
        return dict(status='no_prior_matching_region', points=np.empty((0, 3)), targets=np.empty((0, 3)),
            confidence=np.empty(0), pixel=np.empty((0, 3), np.int64), train_views=[], heldout_views=[])
    source = old['target'] - old['displacement']
    # Fixed input-derived region only. These are not reused as edit directions after holdout selection.
    region_points = torch.tensor(source[np.linspace(0, len(source) - 1, min(24, len(source))).astype(int)],
        device=k.device, dtype=k.dtype)
    selection = select_edit_views(baseline, region_points, k, w, radius=.045,
        min_visible_pixels=PARAMS['validation_min_pixels'], dilation=2)
    eligible = selection['visible_views']
    if len(eligible) >= 3:
        heldout = selection['heldout_views']
        train = [i for i in range(len(k)) if i not in heldout]
        status = 'verified_view_available'
    else:
        heldout = []
        train = list(range(len(k)))
        status = 'unable_validation'
    region = selection['local_region']
    rgb, gray, obs, eroded = photometric.prepare_images(images, masks, PARAMS['side'])
    previous = dict(MATCH)
    try:
        MATCH['train_views'] = train
        proposals = propose(baseline, gray, eroded, k, w)
    finally:
        MATCH.clear(); MATCH.update(previous)
    # Select in stable confidence order, within the SAME fixed input-derived region.
    candidates = []
    for row, view in enumerate(train):
        valid = proposals['valid'][row] & (region[view].cpu().numpy() > .5)
        for y, x in np.argwhere(valid):
            candidates.append((float(proposals['confidence'][row, y, x]), view, int(y), int(x), row))
    candidates.sort(key=lambda r: (-r[0], r[1], r[2], r[3]))
    candidates = candidates[:PARAMS['max_constraints']]
    xyz, targets, confidence, pixels = [], [], [], []
    for weight, view, y, x, row in candidates:
        source_point = baseline['xyz'][view, y, x]
        uv = k.new_tensor([(x + .5) * 518 / PARAMS['side'] - .5,
                           (y + .5) * 518 / PARAMS['side'] - .5, 1.])
        ray = uv @ torch.linalg.inv(k[view]).T @ torch.linalg.inv(w[view])[:3, :3].T
        displacement = ray * float(proposals['delta'][row, y, x])
        displacement *= min(1., .04 / max(float(displacement.norm()), 1.e-8))
        xyz.append(source_point.cpu().numpy()); targets.append((source_point + displacement).cpu().numpy())
        confidence.append(weight); pixels.append([view, y, x])
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    targets = np.asarray(targets, dtype=np.float32).reshape(-1, 3)
    inverse = torch.linalg.inv(fit)
    native_source = points_transform(torch.tensor(xyz, device=k.device), inverse).cpu().numpy()
    native_target = points_transform(torch.tensor(targets, device=k.device), inverse).cpu().numpy()
    supports, baseline_validation = None, None
    if heldout:
        previous = dict(MATCH)
        try:
            MATCH.update(train_views=train, validation_view=heldout[0])
            supports = photometric.validation_reference(baseline, gray, eroded, k, w)
            supports &= region[heldout[0]][None].bool()
            baseline_validation = photometric.validation_cost(baseline, gray, eroded, k, w, supports)
        finally:
            MATCH.clear(); MATCH.update(previous)
        if int(supports.sum()) < PARAMS['validation_min_pixels']:
            status = 'unable_photometric_validation'
    result = dict(status=status, points=xyz, targets=targets, native_points=native_source,
        native_targets=native_target, confidence=np.asarray(confidence, np.float32),
        pixel=np.asarray(pixels, np.int64).reshape(-1, 3), train_views=train, heldout_views=heldout,
        region=region, region_points=region_points, gray=gray, eroded=eroded, rgb=rgb,
        validation_support=supports, baseline_validation_cost=baseline_validation,
        visible_pixels=selection['visible_pixels'], fresh_matching_excludes_holdout=bool(heldout))
    return result


def validation(pred, pool, k, w):
    from visual_evidence_probe.experiments.local_repair import photometric
    from visual_evidence_probe.experiments.local_repair.config import PARAMS as MATCH
    if not pool['heldout_views'] or pool['validation_support'] is None:
        return None
    if int(pool['validation_support'].sum()) < PARAMS['validation_min_pixels']:
        return None
    previous = dict(MATCH)
    try:
        MATCH.update(train_views=pool['train_views'], validation_view=pool['heldout_views'][0])
        return photometric.validation_cost(pred, pool['gray'], pool['eroded'], k, w, pool['validation_support'])
    finally:
        MATCH.clear(); MATCH.update(previous)


def sparse_target(baseline, pool, masks, k, w):
    from .objective import freeze_target
    target = freeze_target(baseline)
    target['depth_valid'] = torch.zeros_like(target['mask'])
    weights = torch.zeros_like(target['depth'])
    for point, confidence, (view, y, x) in zip(pool['targets'], pool['confidence'], pool['pixel']):
        p = torch.tensor(point, device=k.device, dtype=k.dtype)
        depth = p @ w[view, 2, :3] + w[view, 2, 3]
        target['depth'][view, y, x] = depth
        target['depth_valid'][view, y, x] = True
        weights[view, y, x] = float(confidence)
    # Input silhouettes, not invented target silhouettes, supervise the local mask.
    observed = F.interpolate(torch.tensor(masks, device=k.device, dtype=k.dtype)[:, None],
        size=baseline['depth'].shape[-2:], mode='nearest')[:, 0]
    target['softmask'] = observed
    target['silhouette_valid'] = torch.ones_like(target['mask'])
    target['normal_valid'] = torch.zeros_like(target['mask'])
    return target, weights


def save_pool(path, pool):
    path.mkdir(parents=True, exist_ok=True)
    names = ('points', 'targets', 'native_points', 'native_targets', 'confidence', 'pixel')
    values = {name: pool[name] for name in names if name in pool}
    if 'region' in pool: values['region'] = pool['region'].cpu().numpy()
    if pool.get('validation_support') is not None:
        values['validation_support'] = pool['validation_support'].cpu().numpy()
    np.savez_compressed(path / 'input_pool.npz', **values)
    write(path / 'input_pool.json', dict(status=pool['status'], constraints=len(pool['targets']),
        train_views=pool['train_views'], heldout_views=pool['heldout_views'],
        baseline_validation_cost=pool.get('baseline_validation_cost'),
        visible_pixels=pool.get('visible_pixels'), uses_gt=False,
        fresh_matching_excludes_holdout=pool.get('fresh_matching_excludes_holdout', False),
        region_provenance='previous input-only selected local surface; no reuse of old depth correction targets'))
