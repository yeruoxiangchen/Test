"""Target-conditioned layered-ray predictor over paired DINO feature maps.

This is a project-specific extension of the LaRI representation.  It is not a
copy of, nor checkpoint-compatible with, the official LaRI network.  The
official representation is retained (ordered ray intersections and stopping
index), while target ownership and confidence heads are added for single-
object reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .contracts import LayeredPrediction


@dataclass(frozen=True)
class TargetLaRILossWeights:
    depth: float = 1.0
    stopping: float = 1.0
    target: float = 1.0
    confidence: float = 0.1


class _ConvTower(nn.Module):
    def __init__(self, channels: int, blocks: int) -> None:
        super().__init__()
        groups = max(1, min(32, int(channels) // 8))
        while int(channels) % groups:
            groups -= 1
        layers: list[nn.Module] = []
        for _ in range(int(blocks)):
            layers.extend(
                (
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.GroupNorm(groups, channels),
                    nn.SiLU(inplace=True),
                )
            )
        self.net = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


def camera_condition_vector(
    *,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    extrinsics_type: str,
    camera_forward_sign: float,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized per-view camera metadata and object-origin depth."""

    views = int(intrinsics.shape[0])
    if intrinsics.shape != (views, 3, 3) or extrinsics.shape != (views, 4, 4):
        raise ValueError("camera tensor schema mismatch")
    if extrinsics_type not in {"c2w", "w2c"}:
        raise ValueError("extrinsics_type must be c2w or w2c")
    if float(camera_forward_sign) not in {-1.0, 1.0}:
        raise ValueError("camera_forward_sign must be -1 or +1")
    height, width = map(float, image_size)
    if min(height, width) <= 0:
        raise ValueError("image dimensions must be positive")
    w2c = (
        torch.linalg.inv(extrinsics.float())
        if extrinsics_type == "c2w"
        else extrinsics.float()
    )
    normalized_k = torch.stack(
        (
            intrinsics[:, 0, 0].float() / width,
            intrinsics[:, 1, 1].float() / height,
            intrinsics[:, 0, 2].float() / width,
            intrinsics[:, 1, 2].float() / height,
        ),
        dim=1,
    )
    pose = w2c[:, :3, :4].reshape(views, 12)
    vector = torch.cat((normalized_k, pose), dim=1)
    center_depth = w2c[:, 2, 3] * float(camera_forward_sign)
    return vector, center_depth


class TargetConditionedLaRI(nn.Module):
    """Predict scene layers and target ownership from synchronized view pairs."""

    def __init__(
        self,
        *,
        input_channels: int = 1024,
        hidden_channels: int = 256,
        layers: int = 5,
        decoder_blocks: int = 3,
        minimum_depth_gap: float = 1.0e-3,
    ) -> None:
        super().__init__()
        if min(input_channels, hidden_channels, layers, decoder_blocks) <= 0:
            raise ValueError("Target-LaRI dimensions must be positive")
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.layers = int(layers)
        self.decoder_blocks = int(decoder_blocks)
        self.minimum_depth_gap = float(minimum_depth_gap)
        if not math.isfinite(self.minimum_depth_gap) or self.minimum_depth_gap <= 0:
            raise ValueError("minimum_depth_gap must be finite and positive")

        self.raw_projection = nn.Linear(self.input_channels, self.hidden_channels)
        self.masked_projection = nn.Linear(self.input_channels, self.hidden_channels)
        self.target_projection = nn.Linear(self.input_channels, self.hidden_channels)
        self.camera_projection = nn.Sequential(
            nn.Linear(16, self.hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_channels, self.hidden_channels),
        )
        self.mask_projection = nn.Linear(1, self.hidden_channels)
        self.input_norm = nn.LayerNorm(self.hidden_channels)

        # Separate decoders reduce competition between continuous intersection
        # regression and ordinal/ownership classification.
        self.geometry_tower = _ConvTower(self.hidden_channels, decoder_blocks)
        self.classification_tower = _ConvTower(
            self.hidden_channels, decoder_blocks
        )
        self.first_offset_head = nn.Conv2d(self.hidden_channels, 1, 1)
        self.depth_gap_head = (
            nn.Conv2d(self.hidden_channels, self.layers - 1, 1)
            if self.layers > 1 else nn.Identity()
        )
        self.stopping_head = nn.Conv2d(
            self.hidden_channels, self.layers + 1, 1
        )
        self.target_head = nn.Conv2d(self.hidden_channels, self.layers, 1)
        self.confidence_head = nn.Conv2d(self.hidden_channels, self.layers, 1)

    def _patch_mask(
        self, visible_target_mask: torch.Tensor, *, patch_side: int
    ) -> torch.Tensor:
        mask = visible_target_mask.float()
        if mask.ndim == 3:
            mask = mask[:, None]
        if mask.ndim != 4 or int(mask.shape[1]) != 1:
            raise ValueError("visible target mask must be [V,H,W] or [V,1,H,W]")
        return F.interpolate(
            mask,
            size=(int(patch_side), int(patch_side)),
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)

    def forward(
        self,
        *,
        raw_patch_features: torch.Tensor,
        masked_patch_features: torch.Tensor,
        visible_target_mask: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        extrinsics_type: str,
        camera_forward_sign: float,
        image_size: tuple[int, int],
    ) -> LayeredPrediction:
        if raw_patch_features.ndim != 3:
            raise ValueError("raw DINO features must be [V,P,C]")
        if masked_patch_features.shape != raw_patch_features.shape:
            raise ValueError("raw/masked DINO features must be aligned")
        views, patches, channels = map(int, raw_patch_features.shape)
        if channels != self.input_channels:
            raise ValueError(
                f"Target-LaRI expected {self.input_channels} channels, got {channels}"
            )
        patch_side = int(round(math.sqrt(patches)))
        if patch_side * patch_side != patches:
            raise ValueError("DINO patch count must be square")
        mask = self._patch_mask(visible_target_mask, patch_side=patch_side)
        mask_tokens = mask.flatten(2).transpose(1, 2)

        # A single descriptor pools visible target evidence across all views;
        # it remains available to a view in which the target is fully hidden.
        weights = mask_tokens
        denominator = weights.sum().clamp_min(1.0e-6)
        target_descriptor = (
            masked_patch_features.float() * weights
        ).sum(dim=(0, 1)) / denominator

        camera_vector, center_depth = camera_condition_vector(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            extrinsics_type=str(extrinsics_type),
            camera_forward_sign=float(camera_forward_sign),
            image_size=image_size,
        )
        feature = (
            self.raw_projection(raw_patch_features.float())
            + self.masked_projection(masked_patch_features.float())
            + self.target_projection(target_descriptor)[None, None]
            + self.camera_projection(camera_vector.float())[:, None]
            + self.mask_projection(mask_tokens)
        )
        feature = self.input_norm(feature)
        feature_map = feature.transpose(1, 2).reshape(
            views, self.hidden_channels, patch_side, patch_side
        )
        geometry = self.geometry_tower(feature_map)
        classification = self.classification_tower(feature_map)

        first_offset = self.first_offset_head(geometry)
        # Camera-forward ray depths must remain positive even before the
        # supervised predictor has converged.  Conditioning the softplus by
        # the object-origin depth gives a useful initialization scale while
        # preserving a hard representation invariant.
        first_depth = (
            F.softplus(center_depth[:, None, None, None] + first_offset)
            + self.minimum_depth_gap
        )
        if self.layers > 1:
            gaps = F.softplus(self.depth_gap_head(geometry).float()) + self.minimum_depth_gap
            offsets = torch.cat(
                (first_depth, first_depth + torch.cumsum(gaps, dim=1)), dim=1
            )
        else:
            offsets = first_depth
        depths = offsets
        return LayeredPrediction(
            depths=depths,
            stop_logits=self.stopping_head(classification),
            target_logits=self.target_head(classification),
            confidence_logits=self.confidence_head(classification),
        ).validate()

    def metadata(self) -> dict[str, Any]:
        return {
            "representation": "ordered_camera_depth_plus_stopping_index",
            "target_ownership": True,
            "confidence": True,
            "input_channels": self.input_channels,
            "hidden_channels": self.hidden_channels,
            "layers": self.layers,
            "decoder_blocks": self.decoder_blocks,
            "minimum_depth_gap": self.minimum_depth_gap,
            "raster": "DINO_patch_grid",
            "official_lari_checkpoint_compatible": False,
        }


def _resize_supervision(
    sample: dict[str, torch.Tensor], *, size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    depths = sample["layer_depths_gt"].float()
    stop = sample["ray_stop_index_gt"].long()
    target = sample["layer_target_gt"].float()
    if tuple(depths.shape[-2:]) != tuple(size):
        depths = F.interpolate(depths, size=size, mode="nearest")
        target = F.interpolate(target, size=size, mode="nearest")
        stop = F.interpolate(stop[:, None].float(), size=size, mode="nearest")[:, 0].long()
    return depths, stop, target


def target_lari_loss(
    prediction: LayeredPrediction,
    sample: dict[str, torch.Tensor],
    *,
    weights: TargetLaRILossWeights = TargetLaRILossWeights(),
    confidence_scale: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction.validate()
    depths_gt, stop_gt, target_gt = _resize_supervision(
        sample, size=prediction.spatial_size
    )
    depths_gt = depths_gt.to(prediction.depths.device)
    stop_gt = stop_gt.to(prediction.depths.device)
    target_gt = target_gt.to(prediction.depths.device)
    if depths_gt.shape != prediction.depths.shape:
        raise ValueError("Target-LaRI supervision layer count differs")
    layers = prediction.layer_count
    layer_ids = torch.arange(layers, device=stop_gt.device)[None, :, None, None]
    valid = layer_ids < stop_gt[:, None]
    valid_float = valid.float()
    valid_count = valid_float.sum().clamp_min(1.0)

    depth_error = F.smooth_l1_loss(
        prediction.depths.float(), depths_gt.float(), reduction="none"
    )
    depth_loss = (depth_error * valid_float).sum() / valid_count
    stopping_loss = F.cross_entropy(prediction.stop_logits.float(), stop_gt)
    target_error = F.binary_cross_entropy_with_logits(
        prediction.target_logits.float(), target_gt.float(), reduction="none"
    )
    target_loss = (target_error * valid_float).sum() / valid_count

    absolute_error = (prediction.depths.detach().float() - depths_gt.float()).abs()
    confidence_target = torch.exp(
        -absolute_error / max(float(confidence_scale), 1.0e-6)
    )
    confidence_error = F.binary_cross_entropy_with_logits(
        prediction.confidence_logits.float(), confidence_target, reduction="none"
    )
    confidence_loss = (confidence_error * valid_float).sum() / valid_count
    total = (
        float(weights.depth) * depth_loss
        + float(weights.stopping) * stopping_loss
        + float(weights.target) * target_loss
        + float(weights.confidence) * confidence_loss
    )
    return total, {
        "loss": total.detach(),
        "depth_loss": depth_loss.detach(),
        "stopping_loss": stopping_loss.detach(),
        "target_loss": target_loss.detach(),
        "confidence_loss": confidence_loss.detach(),
        "valid_layer_fraction": valid_float.mean().detach(),
        "target_layer_fraction": (
            (target_gt * valid_float).sum() / valid_count
        ).detach(),
    }


__all__ = [
    "TargetConditionedLaRI",
    "TargetLaRILossWeights",
    "camera_condition_vector",
    "target_lari_loss",
]
