"""Differentiable decoded-mesh rendering and fixed-domain edit objectives.

All image tensors use CV image order (row increases downwards). Cameras are the
frozen 518-pixel cameras used by ``input_consistency.refine.project_clip``.
Only ``raster_tensor`` imports nvdiffrast; the objective and view policy run on
CPU. No decoder-provided vertex normals are used: some MeshExtractResult
implementations accumulate those normals with an in-place scatter_add whose
backward fails when the decoder is differentiated through.
"""

from collections.abc import Mapping

import torch
import torch.nn.functional as F


IMAGE_SIZE = 518
DEFAULT_WEIGHTS = dict(depth=1., silhouette=.2, normal=.1, coverage=.2, protect=1.)


def project_clip(vertices, k, w, image_size=IMAGE_SIZE):
    """Project CV world-to-camera matrices with the frozen pixel-center rule."""
    vh = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=-1)
    xyz = vh[None] @ w.transpose(1, 2)
    depth = xyz[..., 2]
    x = 2 * (k[:, None, 0, 0] * xyz[..., 0] + (k[:, None, 0, 2] + .5) * depth) / image_size - depth
    y = 2 * (k[:, None, 1, 1] * xyz[..., 1] + (k[:, None, 1, 2] + .5) * depth) / image_size - depth
    near, far = .001, 100.
    z = (far + near) / (far - near) * depth - 2 * far * near / (far - near)
    return torch.stack((x, y, z, depth), dim=-1).contiguous()


def world_from_depth(depth, k, w, image_size=IMAGE_SIZE):
    """Backproject camera-z with the exact inverse of ``project_clip``.

    A low-resolution pixel represents a center in the original 518 image;
    blindly multiplying K by side/518 introduces a half-pixel discrepancy.
    """
    height, width = depth.shape[-2:]
    yy = (torch.arange(height, device=depth.device, dtype=depth.dtype) + .5) * image_size / height - .5
    xx = (torch.arange(width, device=depth.device, dtype=depth.dtype) + .5) * image_size / width - .5
    yy, xx = torch.meshgrid(yy, xx, indexing='ij')
    x = (xx[None] - k[:, None, None, 0, 2]) * depth / k[:, None, None, 0, 0]
    y = (yy[None] - k[:, None, None, 1, 2]) * depth / k[:, None, None, 1, 1]
    xyz = torch.stack((x, y, depth), dim=-1)
    return torch.einsum('bhwj,bjk->bhwk', xyz - w[:, None, None, :3, 3], w[:, :3, :3])


def face_normals(vertices, faces, eps=1.e-12):
    """Flat world-space normals with a direct cross-product gradient."""
    triangle = vertices[faces.long()]
    cross = torch.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0], dim=-1)
    length = torch.linalg.vector_norm(cross, dim=-1)
    return cross / length[:, None].clamp_min(eps), length > eps


def raster_tensor(vertices, faces, k, w, context, side=128):
    """Render an actual decoded mesh without severing its vertex graph.

    ``vertices`` must be a float32 tensor on the raster context's device.
    ``faces`` may be an integer tensor or array. Depth is perspective-correct
    interpolation of *camera z*. World points are reconstructed from that z.
    Normals are per-face cross products, rotated into camera coordinates. Flat
    normals avoid both unsafe vertex-normal scatter operations and finite
    differences across depth discontinuities. Silhouette coverage is
    antialiased, giving nvdiffrast's boundary gradients back to vertices.

    Returns BxHxW depth/mask/softmask/normal_valid/face_id and BxHxWx3 xyz/normal.
    Background depth, position and normal are zero; background face_id is -1.
    """
    import nvdiffrast.torch as dr

    if not torch.is_tensor(vertices) or vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError('vertices must be a Vx3 tensor; do not detach decoded vertices')
    if vertices.dtype != torch.float32:
        raise ValueError('nvdiffrast vertices must be float32')
    if side <= 0:
        raise ValueError('side must be positive')
    faces = torch.as_tensor(faces, dtype=torch.int32, device=vertices.device).contiguous()
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError('faces must have shape Fx3')
    k = torch.as_tensor(k, dtype=vertices.dtype, device=vertices.device)
    w = torch.as_tensor(w, dtype=vertices.dtype, device=vertices.device)
    if k.ndim == 2:
        k = k[None]
    if w.ndim == 2:
        w = w[None]
    if k.shape[-2:] != (3, 3) or w.shape[-2:] not in ((3, 4), (4, 4)) or len(k) != len(w):
        raise ValueError('expected matching Bx3x3 K and Bx3x4/Bx4x4 world-to-camera matrices')
    if torch.any(k[:, 0, 1] != 0) or torch.any(k[:, 1, 0] != 0):
        raise ValueError('the frozen camera convention requires zero skew')
    batch = len(k)
    if len(vertices) == 0 or len(faces) == 0:
        # Keep an autograd connection even for a topologically empty decode.
        zero = vertices.sum() * 0
        depth = zero + vertices.new_zeros((batch, side, side))
        vector = zero + vertices.new_zeros((batch, side, side, 3))
        return dict(depth=depth, mask=depth.bool(), softmask=depth, normal=vector,
                    normal_valid=depth.bool(), xyz=vector,
                    face_id=torch.full(depth.shape, -1, device=vertices.device, dtype=torch.long))
    clip = project_clip(vertices, k, w)
    rast, _ = dr.rasterize(context, clip, faces, resolution=(side, side))
    hit = rast[..., 3] > 0
    camera_z = (vertices[None] @ w[:, 2, :3, None] + w[:, None, 2, 3, None]).contiguous()
    interpolated_z, _ = dr.interpolate(camera_z, rast, faces)
    depth = interpolated_z[..., 0] * hit
    coverage, _ = dr.interpolate(vertices.new_ones((batch, len(vertices), 1)), rast, faces)
    softmask = dr.antialias(coverage, rast, clip, faces)[..., 0].clamp(0, 1)
    world_normal, face_valid = face_normals(vertices, faces)
    face_id = rast[..., 3].long() - 1
    safe_id = face_id.clamp_min(0)
    normal = torch.einsum('bij,bhwj->bhwi', w[:, :3, :3], world_normal[safe_id])
    normal_valid = hit & face_valid[safe_id]
    normal = normal * normal_valid[..., None]
    xyz = world_from_depth(depth, k, w) * hit[..., None]
    return dict(depth=depth, mask=hit, softmask=softmask, normal=normal,
                normal_valid=normal_valid, xyz=xyz, face_id=face_id)


def freeze_target(render, local_region=None):
    """Copy detached target images so future edits cannot change supervision."""
    result = {name: value.detach().clone() for name, value in render.items() if torch.is_tensor(value)}
    for name in ('depth', 'mask', 'softmask'):
        if name not in result:
            raise ValueError(f'render is missing {name}')
    if local_region is not None:
        result['local_region'] = torch.as_tensor(local_region, device=result['depth'].device,
                                                dtype=result['depth'].dtype).detach().clone()
    return result


def _weighted_mean(value, weight):
    # The denominator only uses the frozen reference/region, never predicted hits.
    return (value * weight).sum() / weight.sum().clamp_min(1)


def _reference_terms(pred, reference, region, depth_scale, missing_depth, unsigned_normals, normal_enabled=True):
    pred_depth = pred['depth']
    ref_depth = reference['depth'].detach()
    ref_hit = reference['mask'].detach().bool() & torch.isfinite(ref_depth) & (ref_depth > 0)
    if 'depth_valid' in reference:
        ref_hit = ref_hit & reference['depth_valid'].detach().bool()
    pred_hit = pred['mask'].bool() & torch.isfinite(pred_depth) & (pred_depth > 0)
    ref_depth = torch.nan_to_num(ref_depth)
    pred_safe = torch.nan_to_num(pred_depth)
    # Every absent ray pays the largest permitted depth error. The same fixed
    # reference rays remain in the denominator when predicted support disappears.
    depth_error = (pred_safe - ref_depth).abs().clamp_max(missing_depth)
    depth_error = torch.where(pred_hit, depth_error, torch.full_like(depth_error, missing_depth))
    valid_weight = region * ref_hit
    depth = _weighted_mean(depth_error / depth_scale, valid_weight)
    # ReLU has zero derivative at equality. clamp_min(0) would choose derivative
    # one there and push an already matched antialiased silhouette outwards.
    coverage = _weighted_mean(F.relu(reference['softmask'].detach() - pred['softmask']), valid_weight)
    silhouette_weight = region
    if 'silhouette_valid' in reference:
        silhouette_weight = silhouette_weight * reference['silhouette_valid'].detach().bool()
    silhouette = _weighted_mean((pred['softmask'] - reference['softmask'].detach()).abs(), silhouette_weight)
    normal = pred_depth.sum() * 0
    if normal_enabled and 'normal' in reference:
        if 'normal' not in pred:
            raise ValueError('prediction must supply normals when the reference supplies normals')
        ref_normal = reference['normal'].detach()
        pred_normal = pred['normal']
        ref_normal_valid = reference.get('normal_valid', reference['mask']).detach().bool()
        ref_normal_valid = ref_normal_valid & torch.isfinite(ref_normal).all(-1)
        ref_normal_valid = ref_normal_valid & (torch.linalg.vector_norm(ref_normal, dim=-1) > 1.e-8)
        pred_normal_valid = pred.get('normal_valid', pred['mask']).bool()
        pred_normal_valid = pred_normal_valid & torch.isfinite(pred_normal).all(-1)
        pred_normal_valid = pred_normal_valid & (torch.linalg.vector_norm(pred_normal, dim=-1) > 1.e-8)
        pn = F.normalize(torch.nan_to_num(pred_normal), dim=-1, eps=1.e-8)
        rn = F.normalize(torch.nan_to_num(ref_normal), dim=-1, eps=1.e-8)
        # For unit vectors this equals 1-cosine (or 1-|cosine|), but squared
        # differences give exactly zero value/gradient for identical normals.
        # A direct dot product can round below one even for identical inputs.
        normal_error = .5 * (pn - rn).square().sum(-1)
        if unsigned_normals:
            normal_error = torch.minimum(normal_error, .5 * (pn + rn).square().sum(-1))
        worst_normal = 1. if unsigned_normals else 2.
        normal_error = torch.where(pred_hit & pred_normal_valid, normal_error,
                                   torch.full_like(normal_error, worst_normal))
        normal = _weighted_mean(normal_error, valid_weight * ref_normal_valid)
    return dict(depth=depth, silhouette=silhouette, normal=normal, coverage=coverage)


def geometry_loss(pred, target, baseline=None, local_region=None, view_indices=None,
                  weights=None, depth_scale=.05, missing_depth=.1, unsigned_normals=True):
    """Supervise a fixed local edit and protect its fixed complement.

    ``target`` is a frozen render of the desired edit; ``baseline`` is a frozen
    render before editing. ``local_region`` is a fixed BxHxW mask/weight in [0,1]
    (or ``target['local_region']``). Without a region, the entire image is local.
    Outside it, baseline depth, silhouette and normals are preserved. Select
    training views with ``view_indices``; holdout views never enter this loss.
    Sparse depth proposals must supply ``target['depth_valid']`` to identify
    actual observations. Optional ``silhouette_valid`` and ``normal_valid``
    likewise mark frozen supervision domains. ``weights`` accepts generic
    component names plus ``local_normal``, ``protect_normal`` etc. overrides,
    allowing sparse depth-only local supervision with normal protection.

    Depth is clipped at missing_depth and normalized by depth_scale. Missing
    target rays pay that same maximum plus a differentiable coverage penalty.
    Thus deleting predicted geometry cannot shrink any supervision denominator
    or improve the per-ray depth/normal penalty. The flat face normals are
    unsigned by default to tolerate decoder triangle winding changes.

    Returns a dictionary of scalar tensors: loss, local, protect and each
    local_{depth,silhouette,normal,coverage}/protect_* component.
    """
    if depth_scale <= 0 or missing_depth <= 0:
        raise ValueError('depth_scale and missing_depth must be positive')
    if pred['depth'].shape != target['depth'].shape:
        raise ValueError('prediction and target must have the same BxHxW shape')
    weight = dict(DEFAULT_WEIGHTS)
    overrides = {f'{domain}_{name}' for domain in ('local', 'protect')
                 for name in ('depth', 'silhouette', 'normal', 'coverage')}
    if weights is not None:
        if not isinstance(weights, Mapping) or set(weights) - (set(weight) | overrides):
            raise ValueError(f'weights may only contain {tuple(weight)} or local_*/protect_* components')
        weight.update(weights)
    if any(float(value) < 0 for value in weight.values()):
        raise ValueError('loss weights must be nonnegative')
    if baseline is None:
        baseline = target
    if baseline['depth'].shape != pred['depth'].shape:
        raise ValueError('baseline and prediction must have the same image shape')
    region = local_region if local_region is not None else target.get('local_region')
    region = torch.ones_like(pred['depth']) if region is None else torch.as_tensor(
        region, device=pred['depth'].device, dtype=pred['depth'].dtype).detach()
    try:
        region = region.expand_as(pred['depth'])
    except RuntimeError as exc:
        raise ValueError('local_region must broadcast to BxHxW') from exc
    if not torch.isfinite(region).all() or torch.any((region < 0) | (region > 1)):
        raise ValueError('local_region must contain finite fixed weights in [0,1]')
    if view_indices is not None:
        index = torch.as_tensor(view_indices, device=pred['depth'].device, dtype=torch.long)
        if index.ndim != 1 or not len(index) or len(index.unique()) != len(index):
            raise ValueError('view_indices must be a nonempty list of unique indices')
        def subset(record):
            return {name: value[index] if torch.is_tensor(value) and value.ndim >= 3 and
                    value.shape[:3] == pred['depth'].shape else value for name, value in record.items()}
        # Subset all three records before changing the shape used by subset().
        pred, target, baseline, region = subset(pred), subset(target), subset(baseline), region[index]
    local_weight = {name: weight.get(f'local_{name}', weight[name])
                    for name in ('depth', 'silhouette', 'normal', 'coverage')}
    protect_weight = {name: weight.get(f'protect_{name}', weight[name]) for name in local_weight}
    local = _reference_terms(pred, target, region, depth_scale, missing_depth, unsigned_normals,
                             normal_enabled=local_weight['normal'] > 0)
    protected = _reference_terms(pred, baseline, 1 - region, depth_scale, missing_depth, unsigned_normals,
                                 normal_enabled=protect_weight['normal'] > 0)
    local_total = sum(local_weight[name] * value for name, value in local.items())
    protect_total = sum(protect_weight[name] * value for name, value in protected.items())
    result = dict(loss=local_total + weight['protect'] * protect_total,
                  local=local_total, protect=protect_total)
    result.update({f'local_{name}': value for name, value in local.items()})
    result.update({f'protect_{name}': value for name, value in protected.items()})
    return result


@torch.no_grad()
def local_edit_region(render, edit_points, radius, dilation=2):
    """Fixed projected local support from current-mesh visibility, without GT."""
    if radius <= 0 or dilation < 0:
        raise ValueError('radius must be positive and dilation nonnegative')
    xyz = render['xyz'].detach()
    points = torch.as_tensor(edit_points, device=xyz.device, dtype=xyz.dtype).reshape(-1, 3)
    if not len(points) or not torch.isfinite(points).all():
        raise ValueError('at least one finite fixed edit point is required')
    distance = torch.full(xyz.shape[:-1], torch.inf, device=xyz.device, dtype=xyz.dtype)
    # Do not allocate pixels x edit_points; edit sets can contain many vertices.
    for point in points:
        distance = torch.minimum(distance, torch.linalg.vector_norm(xyz - point, dim=-1))
    visible = render['mask'].detach().bool() & (distance <= radius)
    region = visible.to(xyz.dtype)
    if dilation:
        region = F.max_pool2d(region[:, None], 2 * dilation + 1, stride=1, padding=dilation)[:, 0]
    return region, visible


@torch.no_grad()
def select_edit_views(render, edit_points, k, w, radius, min_visible_pixels=4, dilation=2):
    """Choose local training/holdout views from current mesh and input cameras.

    Eligible views must see actual baseline surface within the edit sphere.
    The holdout is the eligible camera with the largest minimum angular
    separation from the other eligible cameras, weighted by its visible local
    support. This makes selection dependent on edit visibility and camera
    geometry rather than reserving a fixed numbered view. At least two locally
    visible cameras are required; otherwise status='unable' is returned without
    choosing a holdout camera.
    K is accepted alongside W to make the input-camera provenance explicit;
    visibility itself uses their already-rendered z buffer.
    """
    region, visible = local_edit_region(render, edit_points, radius, dilation)
    if len(k) != len(region) or len(w) != len(region):
        raise ValueError('camera batch and rendered views must match')
    count = visible.sum((-2, -1))
    eligible = torch.nonzero(count >= min_visible_pixels, as_tuple=False).flatten()
    if len(eligible) < 2:
        return dict(status='unable', reason='fewer_than_two_locally_visible_views',
                    train_views=[], heldout_views=[], visible_views=eligible.tolist(),
                    visible_pixels=count.tolist(), local_region=region, uses_gt=False)
    w = torch.as_tensor(w, device=region.device, dtype=render['xyz'].dtype)
    center = -(w[:, :3, :3].transpose(1, 2) @ w[:, :3, 3, None])[..., 0]
    points = torch.as_tensor(edit_points, device=region.device, dtype=center.dtype).reshape(-1, 3)
    direction = F.normalize(center[eligible] - points.mean(0), dim=-1, eps=1.e-8)
    angle = torch.acos((direction @ direction.T).clamp(-1 + 1.e-7, 1 - 1.e-7))
    angle.fill_diagonal_(torch.inf)
    novelty = angle.min(1).values
    support = count[eligible].to(region.dtype)
    score = novelty * (support / support.max()).sqrt()
    holdout = int(eligible[score.argmax()])
    train = [int(index) for index in eligible.tolist() if int(index) != holdout]
    return dict(status='ok', train_views=train, heldout_views=[holdout], visible_views=eligible.tolist(),
                visible_pixels=count.tolist(), local_region=region,
                selection='current_mesh_local_visibility_and_camera_angular_novelty', uses_gt=False)
