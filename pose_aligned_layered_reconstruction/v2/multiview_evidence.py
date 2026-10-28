"""Chunked ray-depth evidence; never builds a V x V x P x D x 1024 tensor."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..projection import sample_maps


class RayMultiViewEvidence(nn.Module):
    def __init__(self, input_channels=1024, channels=64, output_channels=128,
                 query_chunk=128):
        super().__init__()
        self.raw_projection = nn.Linear(input_channels, channels)
        self.masked_projection = nn.Linear(input_channels, channels)
        width = channels * 2
        self.score = nn.Sequential(nn.Linear(width * 2 + 2, channels), nn.SiLU(), nn.Linear(channels, 1))
        self.output = nn.Sequential(nn.Linear(width * 2 + 3, output_channels), nn.LayerNorm(output_channels), nn.SiLU())
        self.query_chunk = query_chunk

    def aggregate_chunk(self, maps, mask, cameras, xyz, ref, query_view):
        views, _, side, _ = maps.shape
        denominator = xyz.new_zeros((len(xyz), 1))
        mean = torch.zeros_like(ref)
        second = torch.zeros_like(ref)
        mask_sum = correlation_sum = count = denominator.clone()
        for s in range(views):
            if s == query_view:
                continue
            projected = cameras.select(slice(s, s + 1)).project(xyz, side)
            feature = sample_maps(maps[s:s + 1], projected['patch_grid'])[0].T
            visible = sample_maps(mask[s:s + 1], projected['image_grid'])[0].T
            valid = projected['valid'][0, :, None].float()
            corr = F.cosine_similarity(feature, ref, dim=-1)[:, None]
            score = self.score(torch.cat((ref, feature, visible, corr), -1)).clamp(-8., 8.)
            weight = score.exp() * valid
            denominator = denominator + weight
            mean = mean + weight * feature
            second = second + weight * feature.square()
            mask_sum = mask_sum + weight * visible
            correlation_sum = correlation_sum + weight * corr
            count = count + valid
        norm = denominator.clamp_min(1.e-8)
        mean = mean / norm
        variance = (second / norm - mean.square()).clamp_min(0.)
        fraction = count / max(views - 1, 1)
        evidence = torch.cat((mean, variance, mask_sum / norm, correlation_sum / norm, fraction), -1)
        result = self.output(evidence) * (count > 0).to(evidence.dtype)
        return result, fraction

    def forward(self, raw_features, masked_features, visible_masks, cameras, rays, depths):
        views, patches, _ = raw_features.shape
        side = math.isqrt(patches)
        if side * side != patches or self.query_chunk < 1:
            raise ValueError('invalid patch grid/chunk')
        source = torch.cat((self.raw_projection(raw_features.float()),
                            self.masked_projection(masked_features.float())), -1)
        maps = source.transpose(1, 2).reshape(views, -1, side, side)
        mask = visible_masks.float()
        if mask.ndim == 3:
            mask = mask[:, None]
        points = rays.points(depths)
        outputs, supports = [], []
        for q in range(views):
            chunks, counts = [], []
            for start in range(0, patches, self.query_chunk):
                end = min(patches, start + self.query_chunk)
                shape = points[q, start:end].shape[:2]
                xyz = points[q, start:end].reshape(-1, 3)
                ref = source[q, start:end, None].expand(-1, depths.shape[-1], -1).reshape(len(xyz), -1)
                if self.training and torch.is_grad_enabled():
                    result, fraction = checkpoint(self.aggregate_chunk, maps, mask, cameras, xyz, ref, q,
                                                  use_reentrant=False)
                else:
                    result, fraction = self.aggregate_chunk(maps, mask, cameras, xyz, ref, q)
                chunks.append(result.reshape(*shape, -1))
                counts.append(fraction.reshape(*shape))
            outputs.append(torch.cat(chunks))
            supports.append(torch.cat(counts))
        return torch.stack(outputs), torch.stack(supports)
