"""Verified native labels and deterministic interventions."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import torch
from .vendor_pose.data import PoseTrellis2Dataset
from .corruptions import corrupt_shape, ss_edit_mask


class RepairDataset(PoseTrellis2Dataset):
    def __init__(self, manifest, stage, cache_size=8):
        super().__init__(manifest, stage=stage)
        self.cached = lru_cache(maxsize=cache_size)(super().__getitem__)

    def __getitem__(self, index):
        # Tensor leaves are immutable throughout training; construct new
        # references before every corruption to keep cached targets intact.
        item = dict(self.cached(index))
        pair_path = Path(item['row']['pair_file'])
        item['pair_pack'] = torch.load(pair_path, map_location='cpu', weights_only=False)
        return item


def intervention(sample, stage, variant, device):
    pair = sample['pair_pack']['pairs'][int(variant)]
    bounds = pair['bounds64'].to(device)
    if stage == 'ss':
        clean = sample['target'].to(device=device, dtype=torch.float32)[None]
        bad = pair['bad_ss'].to(device=device, dtype=torch.float32)[None]
        mask = ss_edit_mask(bounds, device)
        present = torch.ones((1, 1,16,16,16), dtype=torch.bool, device=device)
        coords = None
    else:
        from .vendor_pose.dependency import import_sparse_module
        sp = import_sparse_module()
        xyz = sample['target_coords'].to(device=device, dtype=torch.int32)
        coords = torch.cat([torch.zeros_like(xyz[:,:1]), xyz], -1)
        clean_features = sample['target_feats'].to(device=device, dtype=torch.float32)
        bad_features, mask, present = corrupt_shape(clean_features, coords, bounds, pair['seed'], pair['kind'])
        clean = sp.SparseTensor(clean_features, coords)
        bad = clean.replace(bad_features)
    return {'clean':clean, 'bad':bad, 'mask':mask, 'present':present,
            'bounds64':bounds, 'kind':pair['kind'], 'seed':pair['seed'], 'coords':coords,
            'pair':pair}
