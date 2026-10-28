"""Input sealing and explicitly privileged, pixel-matched geometric targets."""
import numpy as np
import torch
import torch.nn.functional as F

from .config import PREVIOUS, OLD_INPUT, PARAMS, read, write, digest
from visual_evidence_probe.experiments.editor_support.objective import freeze_target, select_edit_views
from visual_evidence_probe.experiments.editor_support.data import points_transform, active_tokens


def resize_observed(masks, baseline):
    return F.interpolate(torch.as_tensor(masks, device=baseline['depth'].device,
        dtype=baseline['depth'].dtype)[:, None], size=baseline['depth'].shape[-2:], mode='nearest')[:, 0]


def input_target(baseline, pixels, targets, observed, w):
    result = freeze_target(baseline)
    result['depth_valid'] = torch.zeros_like(result['mask'])
    result['normal_valid'] = torch.zeros_like(result['mask'])
    result['softmask'] = observed.detach().clone()
    result['silhouette_valid'] = torch.ones_like(result['mask'])
    for point, (view, y, x) in zip(targets, pixels):
        point = torch.as_tensor(point, device=w.device, dtype=w.dtype)
        result['depth'][view, y, x] = point @ w[view, 2, :3] + w[view, 2, 3]
        result['depth_valid'][view, y, x] = True
        result['mask'][view, y, x] = True
    return result


@torch.no_grad()
def seal_input(index, seed, path, baseline, masks, k, w, fit, z, face_count):
    prior = PREVIOUS / 'factorial' / f'{index:03d}' / str(seed)
    meta = read(prior / 'input_pool.json')
    with np.load(prior / 'input_pool.npz') as arrays:
        saved = {key: arrays[key] for key in arrays.files}
    pixels = saved.get('pixel', np.empty((0, 3), dtype=np.int64)).reshape(-1, 3)
    targets = saved.get('targets', np.empty((0, 3), dtype=np.float32)).reshape(-1, 3)
    if len(pixels) != len(targets) or len(np.unique(pixels, axis=0)) != len(pixels):
        raise ValueError('input pool pixel/target identity mismatch')
    face = baseline['face_id']
    if 'region' in saved:
        region = torch.as_tensor(saved['region'], device=k.device, dtype=k.dtype)
        provenance = 'exact saved fifth-round input-only region'
    else:
        source = OLD_INPUT / 'objects' / f'{index:03d}' / str(seed) / 'automatic_constraints.npz'
        with np.load(source) as arrays:
            selected = arrays['selected_faces']
        if len(selected) != face_count:
            raise ValueError('saved selected face IDs do not match frozen original mesh')
        selected = torch.as_tensor(selected, device=k.device)
        region = (selected[face.clamp_min(0)] & baseline['mask']).to(k.dtype)
        region = F.max_pool2d(region[:, None], 2 * PARAMS['region_dilation'] + 1,
            stride=1, padding=PARAMS['region_dilation'])[:, 0]
        provenance = 'saved fourth-round input-selected faces; no matcher depth needed'
    eligible = (region > .5) & baseline['mask']
    points = baseline['xyz'][eligible]
    if len(points):
        ids = torch.linspace(0, len(points) - 1, min(PARAMS['max_region_points'], len(points)), device=k.device).long()
        points = points[ids]
    selected_faces = torch.zeros(face_count, device=k.device, dtype=torch.bool)
    selected_faces[face[eligible].unique()] = True
    train, heldout = list(meta['train_views']), list(meta['heldout_views'])
    if not train:
        if len(points):
            selection = select_edit_views(baseline, points[::max(1, len(points) // 24)], k, w,
                PARAMS['region_radius'], min_visible_pixels=8)
            heldout = selection['heldout_views']
        train = [i for i in range(len(k)) if i not in heldout]
    if any(int(pixel[0]) not in train for pixel in pixels):
        raise ValueError('saved input proposal uses a heldout view')
    native_points = points_transform(points, torch.linalg.inv(fit)).cpu().numpy()
    active = active_tokens(z, native_points) if len(points) else torch.zeros(len(z.feats), device=z.device, dtype=torch.bool)
    observed = resize_observed(masks, baseline)
    target = input_target(baseline, pixels, targets, observed, w)
    arrays = dict(region=region.cpu().numpy(), region_points=points.cpu().numpy(),
        selected_faces=selected_faces.cpu().numpy(), pixel=pixels, targets=targets,
        active_rows=active.cpu().numpy(), train_views=np.asarray(train), heldout_views=np.asarray(heldout))
    np.savez_compressed(path / 'input_sealed.npz', **arrays)
    torch.save({key: target[key].cpu() for key in
        ('depth', 'mask', 'softmask', 'depth_valid', 'normal_valid', 'silhouette_valid')}, path / 'input_target.pt')
    write(path / 'input_sealed.json', dict(gt_used=False, region_provenance=provenance,
        saved_pool=str(prior / 'input_pool.npz'), saved_pool_sha256=digest(prior / 'input_pool.npz'),
        sealed_arrays_sha256=digest(path / 'input_sealed.npz'), query_count=len(pixels),
        region_pixels=eligible.sum((1, 2)).cpu().tolist(), active_rows=int(active.sum()),
        total_rows=len(active), train_views=train, heldout_views=heldout,
        common_editable_rows_source_only=True, previous_input_status=meta['status']))
    return dict(target=target, pixels=pixels, region=region, region_points=points,
        selected_faces=selected_faces, active=active, observed=observed,
        train=train, heldout=heldout, saved=saved, meta=meta)


def oracle_targets(baseline, gt, input_record):
    """Never mutates the sealed input target; no GT-derived filtering of C."""
    dense = freeze_target(gt)
    dense['softmask'] = input_record['observed'].detach().clone()
    dense['silhouette_valid'] = torch.ones_like(dense['mask'])
    dense['depth_valid'] = gt['mask'].detach().clone()
    depth_only = freeze_target(dense)
    depth_only['normal_valid'] = torch.zeros_like(dense['mask'])
    sparse = freeze_target(input_record['target'])
    sparse['depth_valid'] = torch.zeros_like(sparse['mask'])
    sparse['normal_valid'] = torch.zeros_like(sparse['mask'])
    errors, initial, direction, missing = [], [], [], 0
    for view, y, x in input_record['pixels']:
        hit = bool(gt['mask'][view, y, x])
        sparse['mask'][view, y, x] = hit
        if not hit:
            missing += 1
            continue
        sparse['depth'][view, y, x] = gt['depth'][view, y, x].detach()
        sparse['depth_valid'][view, y, x] = True
        before = float(baseline['depth'][view, y, x] - gt['depth'][view, y, x])
        after = float(input_record['target']['depth'][view, y, x] - gt['depth'][view, y, x])
        errors.append(abs(after)); initial.append(abs(before))
        direction.append(before * (after - before) < 0)
    stats = dict(query_count=len(input_record['pixels']), gt_intersections=len(errors),
        no_gt_intersection=missing, identical_query_pixels=True,
        pure_paired_depth_comparison_valid=missing == 0 and bool(len(errors)),
        input_estimate_depth_mae=float(np.mean(errors)) if errors else None,
        original_depth_mae_at_queries=float(np.mean(initial)) if initial else None,
        correct_direction_fraction=float(np.mean(direction)) if direction else None,
        improved_query_fraction=float(np.mean(np.asarray(errors) < np.asarray(initial))) if errors else None,
        input_target_unchanged=True)
    return dict(oracle_dense_normal=dense, oracle_dense_depth=depth_only, oracle_sparse=sparse), stats


def novel_cameras(k, w, count=8):
    """Farthest-point deterministic cameras, independent of meshes and GT."""
    k, w = np.asarray(k, dtype=np.float64), np.asarray(w, dtype=np.float64)
    centers = np.linalg.inv(w)[:, :3, 3]
    direction = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    radius = float(np.median(np.linalg.norm(centers, axis=1)))
    index = np.arange(96)
    height = -.70 + 1.55 * (index + .5) / len(index)
    angle = index * np.pi * (3 - np.sqrt(5))
    candidates = np.stack((np.sqrt(1 - height ** 2) * np.cos(angle),
        np.sqrt(1 - height ** 2) * np.sin(angle), height), -1)
    chosen, basis = [], direction.copy()
    for _ in range(count):
        scores = np.arccos(np.clip(candidates @ basis.T, -1, 1)).min(1)
        if chosen: scores[chosen] = -1
        selected = int(scores.argmax()); chosen.append(selected)
        basis = np.concatenate((basis, candidates[selected:selected + 1]), axis=0)
    directions = candidates[chosen]
    minimum = np.arccos(np.clip(directions @ direction.T, -1, 1)).min() * 180 / np.pi
    if minimum < 10: raise ValueError('novel evaluation camera too close to an input')
    matrices = []
    for d in directions:
        position = radius * d
        forward = -d
        right = np.cross(forward, [0., 0., 1.]); right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack((right, down, forward))
        matrix = np.eye(4); matrix[:3, :3] = rotation; matrix[:3, 3] = -rotation @ position
        matrices.append(matrix)
    intrinsic = np.array([[PARAMS['novel_focal'], 0, 258.5], [0, PARAMS['novel_focal'], 258.5], [0, 0, 1.]])
    return np.repeat(intrinsic[None], count, 0).astype(np.float32), np.asarray(matrices, np.float32), dict(
        radius=radius, count=count, minimum_input_angle_degrees=float(minimum),
        center=[0., 0., 0.], uses_gt=False, uses_generated_geometry=False,
        optimization_or_selection_uses_novel_views=False)
