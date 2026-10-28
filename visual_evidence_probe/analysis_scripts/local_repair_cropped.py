"""Explicit preflight amendment: observed object crops and support-aware matching."""
import os
from pathlib import Path

from visual_evidence_probe.experiments.local_repair import config

SOURCE = config.OUT
config.OUT = config.STORE / 'outputs/local_geometry_repair_cropped_20261026'
config.PARAMS.update(min_support_views=1, uniqueness_gap=.01, uniqueness_radius=.01)

import numpy as np
import torch
from PIL import Image
from visual_evidence_probe.experiments.local_repair import runner, photometric


def observation(index):
    p = config.BASE / 'input_crop_resolution_control/objects' / f'{index:03d}'
    rec = config.read(p / 'observations.json')
    images = [Image.open(config.physical(b['path'])).convert('RGB') for b in rec['images']]
    masks = np.stack([np.asarray(Image.open(config.physical(b['path'])).convert('L')) > 127 for b in rec['masks']])
    return rec, torch.tensor(rec['intrinsics'], device='cuda:0', dtype=torch.float32), \
        torch.tensor(rec['extrinsics'], device='cuda:0', dtype=torch.float32), images, masks


@torch.no_grad()
def propose(pred, gray, eroded, k, w):
    p = config.PARAMS
    offsets = pred['depth'].new_tensor(p['depth_offsets'])
    zero = len(offsets) // 2
    result = {name: [] for name in ('delta', 'valid', 'confidence', 'baseline_cost', 'best_cost', 'support')}
    for i in p['train_views']:
        costs, best_depths, supports = [], [], []
        for j in p['train_views']:
            if i == j:
                continue
            volume, valid = photometric.pair_volume(pred, gray, eroded, k, w, i, j, offsets)
            costs.append(volume)
            best_depths.append(offsets[volume.argmin(0)])
            supports.append(valid[zero])
        support = torch.stack(supports)
        count = support.sum(0)
        volume = torch.stack(costs)
        average = (volume * support[:, None]).sum(0) / count.clamp_min(1)
        best = average.argmin(0)
        best_cost = average.gather(0, best[None])[0]
        initial_cost = average[zero]
        depths = torch.stack(best_depths)
        lo = torch.where(support, depths, torch.full_like(depths, 1.)).min(0).values
        hi = torch.where(support, depths, torch.full_like(depths, -1.)).max(0).values
        agreement = hi - lo
        # A distant alternative peak must be worse. Nearby samples of the same basin are allowed.
        far = (offsets[:, None, None] - offsets[best][None]).abs() > p['uniqueness_radius'] + 1.e-6
        alternate = torch.where(far, average, torch.full_like(average, 2.)).min(0).values
        valid = (count >= p['min_support_views']) & (agreement <= p['pair_depth_agreement'])
        valid &= (initial_cost - best_cost) >= p['min_ncc_gain']
        valid &= best_cost <= p['max_ncc_cost']
        valid &= alternate - best_cost >= p['uniqueness_gap']
        valid &= (best > 0) & (best < len(offsets) - 1) & (best != zero)
        confidence = ((initial_cost - best_cost).clamp_min(0) * (1 - best_cost).clamp_min(0)) * valid
        values = dict(delta=offsets[best], valid=valid, confidence=confidence,
            baseline_cost=initial_cost, best_cost=best_cost, support=count)
        for name, value in values.items():
            result[name].append(value.cpu().numpy())
    return {name: np.stack(value) for name, value in result.items()}


original_protocol = runner.protocol


def protocol():
    value = original_protocol()
    p = config.OUT / 'preflight_amendment.json'
    record = dict(format='local_geometry_repair_crop_amendment.v1',
        source=dict(path=str(Path(__file__).resolve()), sha256=config.digest(__file__)),
        strict_protocol=dict(path=str(SOURCE / 'protocol.json'), sha256=config.digest(SOURCE / 'protocol.json')),
        trigger='First strict case index6 seed42 had zero proposals; full cohort strict run retained.',
        differences=['existing input-only 518 object crops and exact crop cameras',
            'minimum one reliable pair; two valid pairs must still agree',
            'ignore unsupported pairs in depth-agreement calculation',
            'reject distant alternative depth minima with NCC gap<0.01'],
        gt_boundary_limitation='Round3 cropped-raster mask resampling discrepancy retained; no new camera fit.',
        no_parameters_tuned_from_gt_repair_metrics=True)
    if p.exists() and config.read(p) != record:
        raise ValueError('crop amendment changed')
    config.write(p, record)
    return value


runner.observation = observation
runner.protocol = protocol
photometric.propose = propose


def main():
    runner.main()


if __name__ == '__main__':
    main()
