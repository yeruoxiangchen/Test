"""Complete native Shape-512 support, separate from the pretraining data filter.

The official 512 training config filters samples above 8192 tokens. The VAE and
sparse flow accept variable support on a 32^3 grid. Never crop that support.
Old experiments continue using their original data/runtime modules.
"""
from pathlib import Path
import numpy as np
import torch
from ..constants import SHAPE_LATENT_CHANNELS, SHAPE_LATENT_MEAN, SHAPE_LATENT_STD

NATIVE_SHAPE_MAX_TOKENS = 32 ** 3


def validate_shape(coords, feats, *, resolution=32, max_tokens=NATIVE_SHAPE_MAX_TOKENS):
    coords, feats = np.asarray(coords), np.asarray(feats)
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.issubdtype(coords.dtype, np.integer):
        raise ValueError('Invalid integer native support coordinates')
    if not 0 < max_tokens <= resolution ** 3 or not 0 < len(coords) <= max_tokens:
        raise ValueError(f'Invalid native support count {len(coords)}; capacity {max_tokens}')
    if (coords < 0).any() or (coords >= resolution).any():
        raise ValueError('Native support coordinates outside grid')
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError('Duplicate native support coordinates')
    if feats.shape != (len(coords), SHAPE_LATENT_CHANNELS) or not np.isfinite(feats).all():
        raise ValueError('Invalid native Shape features')


def load_shape_target(path: Path, uid: str, *, max_tokens=NATIVE_SHAPE_MAX_TOKENS):
    with np.load(path, allow_pickle=False) as payload:
        coords, feats = payload['coords'], np.asarray(payload['feats'], dtype=np.float32)
    if coords.ndim == 2 and coords.shape[1] == 4:
        if np.any(coords[:, 0] != 0): raise ValueError(f'{uid}: expected single-object support')
        coords = coords[:, 1:]
    validate_shape(coords, feats, max_tokens=max_tokens)
    mean = np.asarray(SHAPE_LATENT_MEAN, dtype=np.float32)[None]
    std = np.asarray(SHAPE_LATENT_STD, dtype=np.float32)[None]
    return torch.from_numpy(coords.astype(np.int32)), torch.from_numpy((feats - mean) / std)


@torch.no_grad()
def ss_latent_to_shape_coords(latent, decoder, *, max_tokens=NATIVE_SHAPE_MAX_TOKENS):
    if tuple(latent.shape) != (1, 8, 16, 16, 16): raise ValueError('Invalid SS latent shape')
    occupancy = decoder(latent) > 0
    if tuple(occupancy.shape) != (1, 1, 64, 64, 64): raise ValueError('Invalid decoded SS grid')
    occupancy = torch.nn.functional.max_pool3d(occupancy.float(), 2, 2, 0) > .5
    coords = torch.argwhere(occupancy)[:, [0, 2, 3, 4]].int()
    if not len(coords): raise RuntimeError('sampled sparse structure decoded to empty support')
    if len(coords) > max_tokens:
        raise RuntimeError(f'sampled sparse structure has {len(coords)} Shape tokens; contract permits {max_tokens}')
    return coords
