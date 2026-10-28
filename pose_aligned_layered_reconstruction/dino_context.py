"""DINO-only conditioning utilities copied into the independent package."""

from __future__ import annotations

from typing import Any

import torch


DINO_FEATURE_DIM = 1024
DEFAULT_SS_CONTEXT_TOKENS = 4096


def select_dino_patch_features(features: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(features) or features.ndim != 3:
        raise ValueError("DINO features must be [views,patches,channels]")
    if int(features.shape[0]) <= 0 or int(features.shape[1]) <= 0:
        raise ValueError("DINO features must contain views and patches")
    channels = int(features.shape[-1])
    if channels < DINO_FEATURE_DIM:
        raise ValueError(f"DINO features require >=1024 channels, got {channels}")
    selected = features if channels == DINO_FEATURE_DIM else features[..., -1024:]
    if not bool(torch.isfinite(selected.float()).all().item()):
        raise ValueError("DINO features contain non-finite values")
    return selected.contiguous()


def deterministic_token_indices(
    total: int,
    maximum: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    total = int(total)
    maximum = int(maximum)
    if total <= 0 or maximum <= 0:
        raise ValueError("token counts must be positive")
    if total <= maximum:
        return torch.arange(total, dtype=torch.long, device=device)
    indices = (
        (2 * torch.arange(maximum, dtype=torch.int64, device=device) + 1) * total
    ) // (2 * maximum)
    if int(torch.unique_consecutive(indices).numel()) != maximum:
        raise RuntimeError("deterministic token selection produced duplicates")
    return indices


def build_masked_dino_contexts(
    masked_features: torch.Tensor,
    *,
    ss_context_tokens: int = DEFAULT_SS_CONTEXT_TOKENS,
) -> dict[str, Any]:
    """Build the unchanged SS/SLat context exclusively from masked-target RGB."""

    dino = select_dino_patch_features(masked_features)
    views, patches, channels = map(int, dino.shape)
    flattened = dino.reshape(views * patches, channels)
    indices = deterministic_token_indices(
        len(flattened), int(ss_context_tokens), device=flattened.device
    )
    stock_condition = flattened.index_select(0, indices)[None].contiguous()
    positive = [dino[index][None].contiguous() for index in range(views)]
    return {
        "masked_visual_patch_features": dino,
        "stock_condition": stock_condition,
        "slat_condition": {
            "cond": positive,
            "neg_cond": [torch.zeros_like(value) for value in positive],
        },
        "context_contract": {
            "source": "masked_target_rgb_only",
            "views": views,
            "patches": patches,
            "channels": channels,
            "ss_context_tokens": int(stock_condition.shape[1]),
            "raw_rgb_excluded_from_trellis_context": True,
        },
    }


__all__ = [
    "DEFAULT_SS_CONTEXT_TOKENS",
    "DINO_FEATURE_DIM",
    "build_masked_dino_contexts",
    "deterministic_token_indices",
    "select_dino_patch_features",
]
