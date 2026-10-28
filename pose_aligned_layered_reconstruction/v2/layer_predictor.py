"""Real-ray, depth-axis, target-conditioned multi-view layer prediction."""
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .contracts import LayeredPredictionV2 as LayeredPrediction
from .ray_geometry import CameraBundle, candidate_depths, make_patch_rays
from .multiview_evidence import RayMultiViewEvidence


@dataclass
class GeometryConfig:
    input_channels: int = 1024
    hidden_channels: int = 128
    evidence_channels: int = 64
    layers: int = 5
    candidates: int = 32
    query_chunk: int = 128
    extent: float = .5
    scene_margin: float = 2.
    absolute_tolerance: float = .02
    relative_tolerance: float = 0.


class RayLayerPredictor(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or GeometryConfig()
        c = self.config
        if (min(c.layers, c.hidden_channels, c.evidence_channels, c.query_chunk) < 1 or c.candidates < 2
                or c.extent <= 0 or c.scene_margin <= 0 or c.absolute_tolerance <= 0 or c.relative_tolerance < 0):
            raise ValueError('invalid geometry dimensions')
        self.layers = c.layers
        self.multiview = RayMultiViewEvidence(c.input_channels, c.evidence_channels, c.hidden_channels, c.query_chunk)
        self.ray_encoder = nn.Sequential(nn.Linear(10, c.hidden_channels), nn.SiLU(), nn.Linear(c.hidden_channels, c.hidden_channels))
        self.local_encoder = nn.Linear(c.input_channels * 2 + 1, c.hidden_channels)
        self.target_encoder = nn.Linear(c.input_channels, c.hidden_channels)
        self.depth_encoder = nn.Sequential(nn.Conv1d(c.hidden_channels, c.hidden_channels, 3, padding=1), nn.SiLU(),
                                           nn.Conv1d(c.hidden_channels, c.hidden_channels, 3, padding=1), nn.SiLU())
        # Flattening the candidate axis retains its location; no single-depth softargmax.
        self.depth_readout = nn.Linear(c.hidden_channels * c.candidates, c.hidden_channels)
        self.spatial = nn.Sequential(nn.Conv2d(c.hidden_channels, c.hidden_channels, 3, padding=1), nn.SiLU(),
                                     nn.Conv2d(c.hidden_channels, c.hidden_channels, 3, padding=1), nn.SiLU())
        self.gaps = nn.Conv2d(c.hidden_channels, c.layers + 1, 1)
        self.stopping = nn.Conv2d(c.hidden_channels, c.layers + 1, 1)
        self.ownership = nn.Conv2d(c.hidden_channels, c.layers, 1)
        self.confidence = nn.Conv2d(c.hidden_channels, c.layers, 1)

    def metadata(self):
        return {'format': 'ray_layer_predictor.v2', **asdict(self.config),
                'depth_kind': 'camera_forward_z', 'last_count_class': 'at_least_L',
                'camera_radius_override': False}

    def predict(self, *, raw_patch_features, masked_patch_features, visible_target_mask,
                intrinsics, extrinsics, extrinsics_type, camera_forward_sign, image_size):
        raw, masked = raw_patch_features.float(), masked_patch_features.float()
        views, patches, _ = raw.shape
        side = math.isqrt(patches)
        if side * side != patches or masked.shape != raw.shape:
            raise ValueError('paired feature shapes differ')
        cameras = CameraBundle.from_tensors(intrinsics, extrinsics, extrinsics_type, camera_forward_sign, image_size)
        rays = make_patch_rays(cameras, (side, side))
        depths, hits = candidate_depths(rays, cameras, self.config.candidates, self.config.extent)
        evidence, support = self.multiview(raw, masked, visible_target_mask, cameras, rays, depths)
        mask = visible_target_mask.float()
        if mask.ndim == 3:
            mask = mask[:, None]
        mask = F.interpolate(mask, size=(side, side), mode='bilinear', align_corners=False).flatten(2).transpose(1, 2)
        descriptor = (masked * mask).sum((0, 1)) / mask.sum().clamp_min(1.e-6)
        center = cameras.center_depth[:, None, None].expand(views, patches, 1)
        ray_input = torch.cat((rays.origins, rays.unit_directions, center,
                               depths[..., :1] - center, depths[..., -1:] - center, hits[..., None].float()), -1)
        base = self.ray_encoder(ray_input) + self.local_encoder(torch.cat((raw, masked, mask), -1))
        base = base + self.target_encoder(descriptor)[None, None]
        sequence = evidence + base[:, :, None]
        encoded = self.depth_encoder(sequence.reshape(views * patches, self.config.candidates, -1).transpose(1, 2))
        h = self.depth_readout(encoded.flatten(1)).reshape(views, patches, -1)
        h = self.spatial(h.transpose(1, 2).reshape(views, -1, side, side))
        # Ordered fractions of a real scene interval. The interval covers foreground
        # occluders outside the target cube and moves with actual camera distance.
        gaps = F.softplus(self.gaps(h).float()) + 1.e-4
        # CUDA cumsum(float) is rejected by torch 2.4 deterministic mode used
        # by the frozen SLat runtime. L is small, so explicit prefix additions
        # are inexpensive and preserve a fixed reduction order.
        prefix, running = [], torch.zeros_like(gaps[:, 0])
        for gap in gaps.unbind(1):
            running = running + gap
            prefix.append(running)
        fraction = torch.stack(prefix[:self.layers], 1) / gaps.sum(1, keepdim=True)
        near = (cameras.center_depth - self.config.scene_margin).clamp_min(1.e-4)
        far = torch.maximum(cameras.center_depth + self.config.scene_margin, near + .1)
        predicted_depth = near[:, None, None, None] + fraction * (far - near)[:, None, None, None]
        prediction = LayeredPrediction(predicted_depth, self.stopping(h), self.ownership(h), self.confidence(h)).validate()
        return prediction, {'candidate_depths': depths, 'support': support, 'cameras': cameras, 'rays': rays}

    def forward(self, **kwargs):
        return self.predict(**kwargs)[0]
