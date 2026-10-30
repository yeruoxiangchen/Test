"""Native frozen T2 flow + multiview evidence + an explicit damaged-state input."""
from __future__ import annotations
import torch
from torch import nn
from .vendor_pose.models import (
    DensePoseConditionedFlow, SparsePoseConditionedFlow, apply_attention_lora,
)
from .vendor_pose.runtime import load_official_flow
from .vendor_pose.geometry import canonical_points
from . import common as c


class DamagedStateCondition(nn.Module):
    def __init__(self, channels, output=1024):
        super().__init__()
        self.channels = channels
        self.network = nn.Sequential(nn.Linear(channels+5, 128), nn.SiLU(),
                                     nn.Linear(128, output))

    def forward(self, features, mask, present, positions, enabled=True):
        values = torch.cat([features.float().clamp(-16,16)/4,
                            mask.float().reshape(-1,1), present.float().reshape(-1,1),
                            positions.float()], -1)
        # Full reconstruction control has identical parameter topology, but
        # neither the state nor mask enters its prediction.
        if not enabled:
            values = values * 0
        out = self.network(values)
        return out if enabled else out * 0


class RepairMixin:
    def _pose_condition(self, x, sample, view_indices):
        condition, stats = super()._pose_condition(x, sample, view_indices)
        dense = torch.is_tensor(x)
        coords = self.query_coords if dense else x.coords
        n = len(coords)
        ref = x if sample is None else sample.get('repair_reference', x)
        if dense:
            features = ref.flatten(2).transpose(1,2).reshape(-1,8)
        else:
            if not torch.equal(ref.coords, x.coords):
                raise ValueError('Damaged reference and query support differ')
            features = ref.feats
        mask = torch.zeros(n, device=x.device) if sample is None else sample['repair_mask'].reshape(-1).to(x.device)
        present = torch.zeros(n, device=x.device) if sample is None else sample['repair_present'].reshape(-1).to(x.device)
        pos = canonical_points(coords, resolution=self.resolution).to(x.device)
        geometry = self.damaged_condition(features, mask, present, pos,
                                          enabled=sample is not None and self.arm == 'repair')
        if sample is not None and not sample.get('image_enabled', True):
            condition = condition * 0
        condition = condition + (geometry[None] if dense else geometry)
        stats['repair_condition_rms'] = geometry.float().square().mean().sqrt()
        stats['editable_fraction'] = mask.float().mean()
        return condition, stats


class DenseRepairFlow(RepairMixin, DensePoseConditionedFlow):
    def __init__(self, flow, arm, **kwargs):
        super().__init__(flow, **kwargs)
        self.arm = arm
        self.damaged_condition = DamagedStateCondition(8, self.condition_channels)


class SparseRepairFlow(RepairMixin, SparsePoseConditionedFlow):
    def __init__(self, flow, arm, **kwargs):
        super().__init__(flow, **kwargs)
        self.arm = arm
        self.damaged_condition = DamagedStateCondition(32, self.condition_channels)


def build(stage, arm='repair', device='cuda', checkpointing=True, rank=8):
    if stage not in ['ss','shape'] or arm not in ['repair','full']:
        raise ValueError((stage,arm))
    base = load_official_flow(stage, model_path=c.MODEL_PATHS[stage+'_flow'])
    base.use_checkpoint = checkpointing
    for block in base.blocks:
        block.use_checkpoint = checkpointing
    flow = apply_attention_lora(base, rank=rank, alpha=16)
    cls = DenseRepairFlow if stage == 'ss' else SparseRepairFlow
    model = cls(flow, arm).to(device)
    names = [n for n,p in model.named_parameters() if p.requires_grad]
    if not all('lora_' in n or n.startswith(('aggregator.','block_condition.','damaged_condition.')) for n in names):
        raise ValueError('Unexpected unfrozen backbone weights')
    return model


def conditional_sample(sample, intervention, image_enabled=True):
    # A model condition must not even carry dataset target leaves. Inference
    # accepts the same allowlist and has no access to a dataset plan.
    allowed=('dino_features','intrinsics','extrinsics','extrinsics_type',
             'camera_forward_sign','camera_image_y_sign','grid_transform',
             'image_size','object_to_world')
    evidence={k:sample[k] for k in allowed if k in sample}
    return {**evidence, 'repair_reference':intervention['bad'],
            'repair_mask':intervention['mask'], 'repair_present':intervention['present'],
            'image_enabled':image_enabled}


def features(value):
    return value if torch.is_tensor(value) else value.feats


def replace(value, feats):
    return feats if torch.is_tensor(value) else value.replace(feats)
