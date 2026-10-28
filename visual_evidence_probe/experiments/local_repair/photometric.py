"""Input-only patch matching and local camera-z search. No GT arguments."""
import math
import numpy as np
import torch
import torch.nn.functional as F

from .config import PARAMS


def sample(field, uv):
    grid = 2 * (uv + .5) / 518 - 1
    return F.grid_sample(field[None], grid[None], align_corners=False,
        mode='bilinear', padding_mode='zeros')[0].permute(1, 2, 0)


def project(xyz, k, w):
    camera = xyz @ w[:3, :3].T + w[:3, 3]
    h = camera @ k.T
    return h[..., :2] / h[..., 2:].clamp_min(1.e-6), camera[..., 2]


def pool(image, patch):
    return F.avg_pool2d(image[None, None], patch, stride=1, padding=patch // 2)[0, 0]


def patch_cost(a, b, patch=5):
    ma, mb = pool(a, patch), pool(b, patch)
    va = (pool(a.square(), patch) - ma.square()).clamp_min(0)
    vb = (pool(b.square(), patch) - mb.square()).clamp_min(0)
    corr = (pool(a * b, patch) - ma * mb) / (va * vb).clamp_min(1.e-10).sqrt()
    return (1 - corr.clamp(-1, 1)) / 2, va.sqrt(), vb.sqrt()


def world_from_depth(depth, k, w):
    side = depth.shape[-1]
    axis = (torch.arange(side, device=depth.device, dtype=torch.float32) + .5) * 518 / side - .5
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    rays = torch.stack((xx, yy, torch.ones_like(xx)), -1) @ torch.linalg.inv(k).T
    camera = rays * depth[..., None]
    c2w = torch.linalg.inv(w)
    return camera @ c2w[:3, :3].T + c2w[:3, 3], rays @ c2w[:3, :3].T


def prepare_images(images, masks, side):
    rgb = torch.as_tensor(np.stack([np.asarray(im) for im in images]),
        device='cuda:0', dtype=torch.float32).permute(0, 3, 1, 2) / 255
    rgb = F.interpolate(rgb, size=(side, side), mode='area')
    gray = (rgb * rgb.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1)
    obs = F.interpolate(torch.as_tensor(masks, device='cuda:0', dtype=torch.float32)[:, None],
        size=(side, side), mode='nearest')[:, 0] > .5
    # All patch pixels must be within observed foreground.
    eroded = torch.stack([pool(m.float(), PARAMS['patch']) > .999 for m in obs])
    return rgb, gray, obs, eroded


@torch.no_grad()
def localization_scores(pred, rgb, eroded, k, w):
    scores = []
    for i in PARAMS['train_views']:
        xyz = pred['xyz'][i]
        total = torch.zeros_like(pred['depth'][i])
        count = torch.zeros_like(total)
        for j in PARAMS['train_views']:
            if i == j:
                continue
            uv, z = project(xyz, k[j], w[j])
            ref = sample(rgb[j], uv)
            ref_z = sample(pred['depth'][j][None], uv)[..., 0]
            valid = pred['mask'][i] & eroded[i] & (z > 0)
            valid &= sample(pred['mask'][j][None].float(), uv)[..., 0] > .99
            valid &= sample(eroded[j][None].float(), uv)[..., 0] > .99
            valid &= (z - ref_z).abs() < PARAMS['visibility_tolerance']
            valid &= (uv >= 0).all(-1) & (uv < 518).all(-1)
            total += (rgb[i].permute(1, 2, 0) - ref).abs().mean(-1) * valid
            count += valid
        scores.append(torch.where(count > 0, total / count.clamp_min(1),
            torch.full_like(total, float('nan'))).cpu().numpy())
    return np.stack(scores)


@torch.no_grad()
def pair_volume(pred, gray, eroded, k, w, i, j, offsets):
    z0 = pred['depth'][i]
    xyz0, rays = world_from_depth(z0, k[i], w[i])
    uv0, depth0 = project(xyz0, k[j], w[j])
    zj = sample(pred['depth'][j][None], uv0)[..., 0]
    visible = sample(pred['mask'][j][None].float(), uv0)[..., 0] > .99
    foreground = sample(eroded[j][None].float(), uv0)[..., 0] > .99
    fixed = pred['mask'][i] & eroded[i] & visible & foreground
    fixed &= (depth0 - zj).abs() < PARAMS['visibility_tolerance']
    centers = torch.linalg.inv(w)[:, :3, 3]
    ri = F.normalize(xyz0 - centers[i], dim=-1)
    rj = F.normalize(xyz0 - centers[j], dim=-1)
    angle = torch.acos((ri * rj).sum(-1).clamp(-1, 1)) * 180 / math.pi
    fixed &= (angle >= PARAMS['min_baseline_degrees']) & (angle <= PARAMS['max_baseline_degrees'])
    costs, valids = [], []
    for delta in offsets:
        xyz = xyz0 + rays * delta
        uv, z = project(xyz, k[j], w[j])
        warped = sample(gray[j][None], uv)[..., 0]
        c, sa, sb = patch_cost(gray[i], warped, PARAMS['patch'])
        candidate_foreground = sample(eroded[j][None].float(), uv)[..., 0] > .99
        valid = fixed & candidate_foreground & (z > 0)
        valid &= (uv >= 0).all(-1) & (uv < 518).all(-1)
        candidate_z = sample(pred['depth'][j][None], uv)[..., 0]
        valid &= (z - candidate_z).abs() < PARAMS['candidate_visibility_tolerance']
        valid &= (sa > PARAMS['min_texture_std']) & (sb > PARAMS['min_texture_std'])
        valid &= pool(valid.float(), PARAMS['patch']) > .95
        costs.append(torch.where(valid, c, torch.full_like(c, 2.)))
        valids.append(valid)
    return torch.stack(costs), torch.stack(valids)


@torch.no_grad()
def propose(pred, gray, eroded, k, w):
    offsets = pred['depth'].new_tensor(PARAMS['depth_offsets'])
    zero = len(offsets) // 2
    result = {name: [] for name in ('delta', 'valid', 'confidence', 'baseline_cost', 'best_cost', 'support')}
    for i in PARAMS['train_views']:
        costs, best_depths, supports = [], [], []
        for j in PARAMS['train_views']:
            if i == j:
                continue
            volume, valid = pair_volume(pred, gray, eroded, k, w, i, j, offsets)
            best = volume.argmin(0)
            costs.append(volume)
            best_depths.append(offsets[best])
            supports.append(valid[zero])
        # Fixed reference support across candidates; invalid search candidates pay maximal cost.
        support = torch.stack(supports)
        count = support.sum(0)
        volume = torch.stack(costs)
        average = (volume * support[:, None]).sum(0) / count.clamp_min(1)
        best = average.argmin(0)
        best_cost = average.gather(0, best[None])[0]
        initial_cost = average[zero]
        depths = torch.stack(best_depths)
        agreement = depths.max(0).values - depths.min(0).values
        valid = count >= PARAMS['min_support_views']
        valid &= agreement <= PARAMS['pair_depth_agreement']
        valid &= (initial_cost - best_cost) >= PARAMS['min_ncc_gain']
        valid &= best_cost <= PARAMS['max_ncc_cost']
        valid &= (best > 0) & (best < len(offsets) - 1) & (best != zero)
        confidence = ((initial_cost - best_cost).clamp_min(0) * (1 - best_cost).clamp_min(0)) * valid
        values = dict(delta=offsets[best], valid=valid, confidence=confidence,
            baseline_cost=initial_cost, best_cost=best_cost, support=count)
        for name, value in values.items():
            result[name].append(value.cpu().numpy())
    return {name: np.stack(value) for name, value in result.items()}


@torch.no_grad()
def validation_reference(pred, gray, eroded, k, w):
    i = PARAMS['validation_view']
    supports = []
    for j in PARAMS['train_views']:
        _, valid = pair_volume(pred, gray, eroded, k, w, i, j, [0.])
        supports.append(valid[0])
    return torch.stack(supports)


@torch.no_grad()
def validation_cost(pred, gray, eroded, k, w, supports):
    i = PARAMS['validation_view']
    xyz, _ = world_from_depth(pred['depth'][i], k[i], w[i])
    sums = pred['depth'].new_zeros(())
    count = int(supports.sum())
    if count == 0:
        return None
    for j, fixed in zip(PARAMS['train_views'], supports):
        uv, z = project(xyz, k[j], w[j])
        warped = sample(gray[j][None], uv)[..., 0]
        cost, _, _ = patch_cost(gray[i], warped, PARAMS['patch'])
        valid = pred['mask'][i] & (z > 0) & (uv >= 0).all(-1) & (uv < 518).all(-1)
        valid &= sample(eroded[j][None].float(), uv)[..., 0] > .99
        # Fixed initial support denominator prevents improvements by dropping difficult pixels.
        sums += torch.where(valid, cost, torch.full_like(cost, 2.))[fixed].sum()
    return float(sums / count)
