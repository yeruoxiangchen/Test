"""Deterministic local interventions; no validation GT enters a repair policy.

Synthetic ROI is known from the intervention. This is a repairability experiment,
not a claim of automatic error localization on generated meshes.
"""
from __future__ import annotations
import hashlib
import numpy as np
import torch
import torch.nn.functional as F


KINDS = ('missing', 'displaced', 'spurious', 'identity')


def pair_seed(uid, variant):
    return int(hashlib.sha256(f'repair_v1:{uid}:{variant}'.encode()).hexdigest()[:8], 16)


def select_roi(occupied, seed, resolution=64):
    rng = np.random.default_rng(seed)
    occupied = np.asarray(occupied, dtype=np.int64)
    if occupied.ndim != 2 or occupied.shape[1] != 3 or not len(occupied):
        raise ValueError('ROI selection requires nonempty xyz occupancy')
    center = occupied[int(rng.integers(len(occupied)))]
    half = int(rng.integers(7, 13))
    lo = np.maximum(0, center - half)
    hi = np.minimum(resolution, center + half + 1)
    return np.stack([lo, hi]).astype(np.int32)


def roi_mask(coords, bounds, resolution):
    """Voxel-center inclusion; bounds are in the fixed 64-cell canonical grid."""
    xyz = coords[..., -3:].float() if torch.is_tensor(coords) else np.asarray(coords)[..., -3:].astype(float)
    if torch.is_tensor(xyz):
        b = torch.as_tensor(bounds, device=xyz.device, dtype=xyz.dtype)
        lo = xyz * (64. / resolution)
        hi = (xyz + 1) * (64. / resolution)
        return ((hi > b[0]) & (lo < b[1])).all(-1)
    lo = xyz * (64. / resolution)
    hi = (xyz + 1) * (64. / resolution)
    return ((hi > np.asarray(bounds)[0]) & (lo < np.asarray(bounds)[1])).all(-1)


def corrupt_occupancy(occupied, seed, kind):
    if kind not in KINDS:
        raise ValueError(kind)
    rng = np.random.default_rng(seed)
    occupied = np.asarray(occupied, dtype=np.int64)
    bounds = select_roi(occupied, seed)
    clean = np.zeros((64, 64, 64), dtype=bool)
    clean[tuple(occupied.T)] = True
    bad = clean.copy()
    sl = tuple(slice(int(a), int(b)) for a, b in zip(*bounds))
    region = np.zeros_like(clean)
    region[sl] = True
    if kind == 'missing':
        bad[sl] = False
    elif kind == 'displaced':
        part = bad[sl].copy()
        axis = int(rng.integers(3))
        # Confine the translated/wrapped part to its stated ROI.
        bad[sl] = np.roll(part, int(rng.integers(3, 7)), axis=axis)
    elif kind == 'spurious':
        center = (bounds[0] + bounds[1]) // 2
        radius = int(rng.integers(3, 6))
        grids = np.indices((64, 64, 64))
        ball = ((grids - center[:, None, None, None]) ** 2).sum(0) <= radius ** 2
        # Shell avoids teaching SS a solid volume rather than surface support.
        inner = ((grids - center[:, None, None, None]) ** 2).sum(0) < max(0, radius-1) ** 2
        bad |= ball & ~inner & region
    assert np.array_equal(clean[~region], bad[~region])
    return bad, bounds, region, int(np.count_nonzero(clean != bad))


def corrupt_shape(clean, coords, bounds, seed, kind):
    """Normalized native 32-channel features; outside ROI stays bit-identical.

    The clean support is used only for supervised training / teacher-forced
    diagnostics. Rollout uses predicted SS support and the validity mask below.
    """
    mask = roi_mask(coords, bounds, 32)
    bad = clean.clone()
    present = torch.ones(len(clean), dtype=torch.bool, device=clean.device)
    gen = torch.Generator(device=clean.device).manual_seed(int(seed))
    count = int(mask.sum())
    if count and kind == 'missing':
        bad[mask] = 0
        present[mask] = False
    elif count and kind == 'displaced':
        donor = torch.randperm(len(clean), generator=gen, device=clean.device)[:count]
        bad[mask] = clean[donor] + .15 * torch.randn((count, clean.shape[1]), generator=gen, device=clean.device)
    elif count and kind == 'spurious':
        bad[mask] = clean[mask] + 1.5 * torch.randn((count, clean.shape[1]), generator=gen, device=clean.device)
    elif kind != 'identity' and kind not in KINDS:
        raise ValueError(kind)
    assert torch.equal(bad[~mask], clean[~mask])
    return bad, mask, present


def ss_edit_mask(bounds, device='cpu'):
    axes = torch.arange(16, device=device)
    xyz = torch.stack(torch.meshgrid(axes, axes, axes, indexing='ij'), -1).reshape(-1, 3)
    m = roi_mask(xyz, bounds, 16).reshape(1, 1, 16, 16, 16).float()
    # SS VAE receptive fields cross a cell boundary; the learned SS itself is
    # global. Exact preservation is enforced after decoding at occupancy64.
    return F.max_pool3d(m, 3, 1, 1).bool()


def remap_reference(reference_coords, reference_features, query_coords):
    """Exact sparse lookup. Missing input cells remain missing, no GT fill."""
    ref = reference_coords[:, -3:].long()
    q = query_coords[:, -3:].long()
    key = (ref[:, 0] * 32 + ref[:, 1]) * 32 + ref[:, 2]
    query = (q[:, 0] * 32 + q[:, 1]) * 32 + q[:, 2]
    order = key.argsort()
    sorted_keys = key[order]
    if not len(key):
        return reference_features.new_zeros((len(q), 32)), torch.zeros(len(q), dtype=torch.bool, device=q.device)
    pos = torch.searchsorted(sorted_keys, query)
    valid = (pos < len(key)) & (sorted_keys[pos.clamp(max=len(key)-1)] == query)
    out = reference_features.new_zeros((len(q), 32))
    out[valid] = reference_features[order[pos[valid]]]
    return out, valid
