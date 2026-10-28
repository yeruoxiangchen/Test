"""Parent-compatible appearance fusion plus zero-initialized layer adapters."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .layer_matching import LAYER_MATCH_FEATURE_NAMES


class PoseViewAggregator(nn.Module):
    """Exact parameter layout of the frozen 30K mean/variance aggregator."""

    def __init__(self, channels: int = 1024) -> None:
        super().__init__()
        self.channels = int(channels)
        input_dim = 3 * self.channels
        self.feature_mlp = nn.Sequential(
            nn.Linear(input_dim, self.channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels, self.channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels, self.channels),
        )
        self.weight_mlp = nn.Sequential(
            nn.Linear(input_dim, self.channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels, 1),
        )
        nn.init.zeros_(self.feature_mlp[-1].weight)
        nn.init.zeros_(self.feature_mlp[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        valid: torch.Tensor,
        *,
        logit_residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if features.ndim != 3:
            raise ValueError("appearance features must be [V,N,C]")
        views, queries, channels = map(int, features.shape)
        if channels != self.channels or valid.shape != (views, queries):
            raise ValueError("appearance aggregator schema mismatch")
        mask = valid.bool()[..., None]
        count = mask.sum(dim=0, keepdim=True).clamp_min(1)
        masked = features * mask
        mean = masked.sum(dim=0, keepdim=True) / count
        variance = (
            masked.square().sum(dim=0, keepdim=True) / count - mean.square()
        ).clamp_min(0)
        inputs = torch.cat(
            (features, mean.expand_as(features), variance.expand_as(features)), dim=-1
        )
        feature_delta = self.feature_mlp(inputs)
        raw_logits = self.weight_mlp(inputs)[..., 0]
        if logit_residual is not None:
            if logit_residual.shape != raw_logits.shape:
                raise ValueError("layer view-logit residual shape differs")
            raw_logits = raw_logits + logit_residual.to(raw_logits.dtype)
        logits = raw_logits.masked_fill(~valid.bool(), -1.0e4)
        weights = torch.softmax(logits, dim=0)
        all_invalid = ~valid.bool().any(dim=0, keepdim=True)
        weights = torch.where(all_invalid, torch.zeros_like(weights), weights)
        aggregated = mean[0] + (feature_delta * weights[..., None]).sum(dim=0)
        supported = valid.bool().any(dim=0)
        aggregated = aggregated * supported[:, None].to(aggregated.dtype)
        entropy = -(weights.clamp_min(1.0e-8).log() * weights).sum(dim=0)
        stats = {
            "supported_fraction": supported.float().mean(),
            "mean_frustum_views": valid.float().sum(dim=0).mean(),
            "aggregation_entropy": (
                entropy[supported].mean()
                if bool(supported.any().item())
                else entropy.new_zeros(())
            ),
            "appearance_condition_rms": aggregated.float().square().mean().sqrt(),
        }
        details = {
            "view_logits": raw_logits,
            "view_weights": weights,
            "valid": valid.bool(),
        }
        return aggregated[None], stats, details


class ZeroInitViewGate(nn.Module):
    """Learn a residual view score without changing the 30K initialization."""

    def __init__(self, hidden_channels: int = 64) -> None:
        super().__init__()
        input_channels = len(LAYER_MATCH_FEATURE_NAMES)
        self.net = nn.Sequential(
            nn.Linear(input_channels, int(hidden_channels)),
            nn.SiLU(inplace=True),
            nn.Linear(int(hidden_channels), 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or int(features.shape[-1]) != len(
            LAYER_MATCH_FEATURE_NAMES
        ):
            raise ValueError("view-gate feature schema mismatch")
        return self.net(features)[..., 0]

    def exact_zero(self) -> bool:
        return bool(
            torch.count_nonzero(self.net[-1].weight.detach()).item() == 0
            and torch.count_nonzero(self.net[-1].bias.detach()).item() == 0
        )


class LayerEvidenceEncoder(nn.Module):
    def __init__(self, output_channels: int = 128, hidden_channels: int = 128) -> None:
        super().__init__()
        self.output_channels = int(output_channels)
        self.net = nn.Sequential(
            nn.Linear(len(LAYER_MATCH_FEATURE_NAMES), int(hidden_channels)),
            nn.SiLU(inplace=True),
            nn.Linear(int(hidden_channels), self.output_channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class EveryBlockConditionProjection(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, blocks: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Linear(int(input_channels), int(output_channels))
            for _ in range(int(blocks))
        )
        for projection in self.projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, index: int, condition: torch.Tensor) -> torch.Tensor:
        return self.projections[int(index)](condition)

    def exact_zero(self) -> bool:
        return all(
            int(torch.count_nonzero(parameter.detach()).item()) == 0
            for parameter in self.parameters()
        )


def aggregate_layer_evidence(
    encoded: torch.Tensor,
    *,
    appearance_weights: torch.Tensor,
    target_support: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if encoded.ndim != 3:
        raise ValueError("encoded layer evidence must be [V,N,C]")
    if appearance_weights.shape != encoded.shape[:2]:
        raise ValueError("appearance weights do not match layer evidence")
    if target_support.shape != encoded.shape[:2]:
        raise ValueError("target support does not match layer evidence")
    raw_weights = appearance_weights.float() * target_support.float()
    denominator = raw_weights.sum(dim=0, keepdim=True)
    # Retain absolute reliability.  Normalizing these weights to sum to one
    # would amplify a tiny nonzero support into a full-strength condition.
    aggregated = (encoded.float() * raw_weights[..., None]).sum(dim=0)
    supported = denominator[0] > 1.0e-6
    aggregated = aggregated * supported[:, None].to(aggregated.dtype)
    return aggregated[None], {
        "layer_supported_fraction": supported.float().mean(),
        "layer_condition_rms": aggregated.square().mean().sqrt(),
        "layer_view_weight_max_mean": raw_weights.amax(dim=0).mean(),
    }


__all__ = [
    "EveryBlockConditionProjection",
    "LayerEvidenceEncoder",
    "PoseViewAggregator",
    "ZeroInitViewGate",
    "aggregate_layer_evidence",
]
