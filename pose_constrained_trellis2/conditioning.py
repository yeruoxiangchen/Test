"""Multiview aggregation and per-DiT-block pose-condition injection."""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn

from .constants import DINO_FEATURE_DIM
from .geometry import split_dinov3_tokens


ContextMode = Literal["first", "concat"]


def select_view_indices(
    view_count: int,
    *,
    minimum: int,
    maximum: int,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Randomly sample an ordered view subset; element zero is the 2D reference."""

    upper = min(int(view_count), int(maximum))
    lower = min(upper, int(minimum))
    if lower <= 0 or upper < lower:
        raise ValueError("invalid view-count bounds")
    count = int(
        torch.randint(
            lower,
            upper + 1,
            (),
            device=device,
            generator=generator,
        ).item()
    )
    return torch.randperm(
        int(view_count), device=device, generator=generator
    )[:count]


def build_2d_context(
    dino_features: torch.Tensor,
    view_indices: torch.Tensor | None = None,
    *,
    mode: ContextMode = "first",
) -> torch.Tensor:
    """Build the native TRELLIS.2 cross-attention context.

    ``first`` keeps the pretrained model's one-image token distribution and is
    the default migration path. ``concat`` is an explicit ablation that keeps
    all selected views in a single longer sequence.
    """

    if mode not in ("first", "concat"):
        raise ValueError(f"unsupported context mode={mode!r}")
    features = dino_features
    if view_indices is not None:
        features = features.index_select(
            0, view_indices.to(device=features.device, dtype=torch.long)
        )
    if int(features.shape[0]) <= 0:
        raise ValueError("2D context selected no views")
    # Validate the spatial token contract while retaining the five prefix
    # tokens in the native cross-attention sequence.
    split_dinov3_tokens(features)
    if mode == "first":
        return features[:1].contiguous()
    return features.reshape(1, -1, int(features.shape[-1])).contiguous()


class MultiviewAggregator(nn.Module):
    """The current method's learned mean/variance view aggregation."""

    def __init__(self, channels: int = DINO_FEATURE_DIM) -> None:
        super().__init__()
        self.channels = int(channels)
        input_channels = 3 * self.channels
        self.feature_mlp = nn.Sequential(
            nn.Linear(input_channels, self.channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels, self.channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels, self.channels),
        )
        self.weight_mlp = nn.Sequential(
            nn.Linear(input_channels, self.channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels, 1),
        )
        nn.init.zeros_(self.feature_mlp[-1].weight)
        nn.init.zeros_(self.feature_mlp[-1].bias)

    def forward(
        self, features: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if features.ndim != 3:
            raise ValueError("features must be [views,queries,channels]")
        views, queries, channels = map(int, features.shape)
        if channels != self.channels or tuple(valid.shape) != (views, queries):
            raise ValueError("feature/visibility schema mismatch")
        mask = valid[..., None]
        count = mask.sum(dim=0, keepdim=True).clamp_min(1)
        masked = features * mask
        mean = masked.sum(dim=0, keepdim=True) / count
        variance = (
            masked.square().sum(dim=0, keepdim=True) / count - mean.square()
        ).clamp_min(0)
        inputs = torch.cat(
            (features, mean.expand_as(features), variance.expand_as(features)), dim=-1
        )
        delta = self.feature_mlp(inputs)
        logits = self.weight_mlp(inputs).masked_fill(~mask, -1.0e4)
        weights = torch.softmax(logits, dim=0)
        unsupported = ~valid.any(dim=0, keepdim=True)[..., None]
        weights = torch.where(unsupported, torch.zeros_like(weights), weights)
        aggregated = mean[0] + (delta * weights).sum(dim=0)
        supported = valid.any(dim=0)
        aggregated = aggregated * supported[:, None].to(aggregated.dtype)
        entropy = -(weights.clamp_min(1.0e-8).log() * weights).sum(dim=0)[..., 0]
        zero = aggregated.new_zeros(())
        stats = {
            "supported_fraction": supported.float().mean(),
            "mean_visible_views": valid.float().sum(dim=0).mean(),
            "aggregation_entropy": (
                entropy[supported].mean() if bool(supported.any().item()) else zero
            ),
            "condition_rms": aggregated.float().square().mean().sqrt(),
        }
        return aggregated, stats


class EveryBlockCondition(nn.Module):
    """Independent zero-initialized condition projection for each DiT block."""

    def __init__(self, condition_channels: int, model_channels: int, blocks: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Linear(int(condition_channels), int(model_channels))
            for _ in range(int(blocks))
        )
        for projection in self.projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, block_index: int, condition: torch.Tensor) -> torch.Tensor:
        return self.projections[int(block_index)](condition)

    def exact_zero(self) -> bool:
        return all(
            int(torch.count_nonzero(parameter.detach()).item()) == 0
            for parameter in self.parameters()
        )


def zero_condition_stats(reference: torch.Tensor, *, queries: int) -> dict[str, torch.Tensor]:
    zero = reference.new_zeros((), dtype=torch.float32)
    return {
        "view_count": zero,
        "visible_fraction": zero,
        "supported_fraction": zero,
        "query_count": zero.new_tensor(float(queries)),
        "mean_visible_views": zero,
        "aggregation_entropy": zero,
        "condition_rms": zero,
    }


def sample_to_device(sample: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    """Move only tensor leaves, retaining string camera-contract metadata."""

    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in sample.items()
    }
