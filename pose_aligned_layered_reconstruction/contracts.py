"""Strict paired-input and layered-ray tensor contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


LAYERED_SAMPLE_FORMAT = "pose_aligned_layered_reconstruction.paired_sample.v1"
LAYERED_SAMPLE_MANIFEST_FORMAT = (
    "pose_aligned_layered_reconstruction.paired_manifest.v1"
)
TARGET_LARI_CHECKPOINT_FORMAT = (
    "pose_aligned_layered_reconstruction.target_lari.v1"
)
LAYERED_SS_CHECKPOINT_FORMAT = (
    "pose_aligned_layered_reconstruction.layered_ss.v1"
)


@dataclass(frozen=True)
class LayeredPrediction:
    """Target-aware layered intersections at one shared raster resolution.

    Shapes:
      depths:            [V,L,H,W], positive camera-forward depth
      stop_logits:       [V,L+1,H,W], classes C=0..L
      target_logits:     [V,L,H,W], target ownership per intersection
      confidence_logits: [V,L,H,W], geometric reliability per intersection
    """

    depths: torch.Tensor
    stop_logits: torch.Tensor
    target_logits: torch.Tensor
    confidence_logits: torch.Tensor

    @property
    def layer_count(self) -> int:
        return int(self.depths.shape[1])

    @property
    def spatial_size(self) -> tuple[int, int]:
        return int(self.depths.shape[-2]), int(self.depths.shape[-1])

    def validate(self) -> "LayeredPrediction":
        if self.depths.ndim != 4:
            raise ValueError("layer depths must be [V,L,H,W]")
        views, layers, height, width = map(int, self.depths.shape)
        if min(views, layers, height, width) <= 0:
            raise ValueError("layer predictions cannot contain empty dimensions")
        if self.stop_logits.shape != (views, layers + 1, height, width):
            raise ValueError("ray-stop logits do not match layer prediction shape")
        expected = (views, layers, height, width)
        if self.target_logits.shape != expected:
            raise ValueError("target logits do not match layer prediction shape")
        if self.confidence_logits.shape != expected:
            raise ValueError("confidence logits do not match layer prediction shape")
        for name, value in (
            ("depths", self.depths),
            ("stop_logits", self.stop_logits),
            ("target_logits", self.target_logits),
            ("confidence_logits", self.confidence_logits),
        ):
            if not bool(torch.isfinite(value.float()).all().item()):
                raise ValueError(f"{name} contains non-finite values")
        if bool((self.depths <= 0).any().item()):
            raise ValueError("layer predictions must use positive camera-forward depth")
        if layers > 1 and bool((self.depths[:, 1:] <= self.depths[:, :-1]).any().item()):
            raise ValueError("layer depths must be strictly increasing along each ray")
        return self

    def valid_probabilities(self) -> torch.Tensor:
        """Return P(C >= layer_index + 1) with shape [V,L,H,W]."""

        self.validate()
        stop_probabilities = torch.softmax(self.stop_logits.float(), dim=1)
        # Reverse cumulative sum over classes C=0..L, then discard C=0.
        return stop_probabilities.flip(1).cumsum(1).flip(1)[:, 1:]

    def select_views(self, indices: torch.Tensor) -> "LayeredPrediction":
        index = indices.to(device=self.depths.device, dtype=torch.long)
        return LayeredPrediction(
            depths=self.depths.index_select(0, index),
            stop_logits=self.stop_logits.index_select(0, index),
            target_logits=self.target_logits.index_select(0, index),
            confidence_logits=self.confidence_logits.index_select(0, index),
        )

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> "LayeredPrediction":
        def move(value: torch.Tensor) -> torch.Tensor:
            target_dtype = value.dtype if dtype is None else dtype
            return value.to(device=device, dtype=target_dtype)

        return LayeredPrediction(
            depths=move(self.depths),
            stop_logits=move(self.stop_logits),
            target_logits=move(self.target_logits),
            confidence_logits=move(self.confidence_logits),
        )


def _require_tensor(sample: dict[str, Any], key: str) -> torch.Tensor:
    value = sample.get(key)
    if not torch.is_tensor(value):
        raise ValueError(f"paired sample requires tensor field {key!r}")
    return value


def ground_truth_prediction(
    sample: dict[str, Any], *, logit_magnitude: float = 20.0
) -> LayeredPrediction:
    """Represent rendered depth-peeling labels as an oracle prediction."""

    depths = _require_tensor(sample, "layer_depths_gt").float()
    stop = _require_tensor(sample, "ray_stop_index_gt").long()
    target = _require_tensor(sample, "layer_target_gt").float()
    if depths.ndim != 4:
        raise ValueError("oracle layer depths must be [V,L,H,W]")
    views, layers, height, width = map(int, depths.shape)
    if stop.shape != (views, height, width):
        raise ValueError("oracle stopping index shape differs")
    if target.shape != depths.shape:
        raise ValueError("oracle target ownership shape differs")
    magnitude = float(logit_magnitude)
    layer_ids = torch.arange(layers, device=stop.device)[None, :, None, None]
    valid = layer_ids < stop[:, None]
    # Renderers conventionally store zero in invalid slots.  A prediction must
    # remain ordered everywhere for stable bilinear sampling, so extend only
    # invalid suffixes without changing any supervised intersection.
    sanitized = depths.clone()
    if layers:
        fallback_first = torch.ones_like(sanitized[:, 0])
        sanitized[:, 0] = torch.where(valid[:, 0], sanitized[:, 0], fallback_first)
        for layer in range(1, layers):
            fallback = sanitized[:, layer - 1] + 1.0
            sanitized[:, layer] = torch.where(
                valid[:, layer], sanitized[:, layer], fallback
            )
    stop_logits = depths.new_full((views, layers + 1, height, width), -magnitude)
    stop_logits.scatter_(1, stop[:, None], magnitude)
    target_logits = torch.where(target > 0.5, magnitude, -magnitude)
    confidence_logits = torch.full_like(depths, magnitude)
    return LayeredPrediction(
        depths=sanitized,
        stop_logits=stop_logits,
        target_logits=target_logits,
        confidence_logits=confidence_logits,
    ).validate()


def validate_paired_sample(
    sample: dict[str, Any],
    *,
    require_labels: bool = False,
    require_ss_target: bool = False,
    expected_layers: int | None = None,
) -> dict[str, int]:
    """Fail closed on cross-branch view, camera, and raster mismatches."""

    if sample.get("format") != LAYERED_SAMPLE_FORMAT:
        raise ValueError(f"unexpected paired sample format={sample.get('format')!r}")
    if float(sample.get("layer_camera_y_sign", 1.0)) not in (-1.0, 1.0):
        raise ValueError("layer_camera_y_sign must be -1 or +1")
    masked = _require_tensor(sample, "masked_visual_patch_features")
    raw = _require_tensor(sample, "raw_visual_patch_features")
    visible = _require_tensor(sample, "visible_target_mask")
    intrinsics = _require_tensor(sample, "intrinsics")
    extrinsics = _require_tensor(sample, "extrinsics")
    if masked.ndim != 3 or raw.shape != masked.shape:
        raise ValueError("raw/masked DINO features must share [V,P,C]")
    views, patches, channels = map(int, masked.shape)
    patch_side = int(round(math.sqrt(patches)))
    if views <= 0 or patch_side * patch_side != patches or channels < 1:
        raise ValueError("paired DINO feature dimensions are invalid")
    if visible.ndim == 4 and int(visible.shape[1]) == 1:
        visible = visible[:, 0]
    if visible.ndim != 3 or int(visible.shape[0]) != views:
        raise ValueError("visible target mask must be [V,H,W] or [V,1,H,W]")
    height, width = map(int, visible.shape[-2:])
    if min(height, width) <= 0:
        raise ValueError("visible target masks cannot be empty rasters")
    if intrinsics.shape != (views, 3, 3) or extrinsics.shape != (views, 4, 4):
        raise ValueError("camera tensors do not match paired view count")
    image_size = tuple(int(value) for value in sample.get("image_size", ()))
    if image_size != (height, width):
        raise ValueError(
            f"image_size must equal mask raster {(height, width)}, got {image_size}"
        )
    if str(sample.get("extrinsics_type")) not in {"c2w", "w2c"}:
        raise ValueError("extrinsics_type must be c2w or w2c")
    forward = float(sample.get("camera_forward_sign", 0.0))
    if forward not in {-1.0, 1.0}:
        raise ValueError("camera_forward_sign must be exactly -1 or +1")
    if str(sample.get("grid_transform")) not in {"identity", "pixal3d_rotation"}:
        raise ValueError("unsupported grid_transform")
    for name, value in (
        ("masked DINO", masked),
        ("raw DINO", raw),
        ("visible mask", visible),
        ("intrinsics", intrinsics),
        ("extrinsics", extrinsics),
    ):
        if not bool(torch.isfinite(value.float()).all().item()):
            raise ValueError(f"{name} contains non-finite values")
    if bool(((visible.float() < 0.0) | (visible.float() > 1.0)).any().item()):
        raise ValueError("visible target mask values must be in [0,1]")
    if float(visible.float().sum().item()) <= 0.0:
        raise ValueError("at least one view must expose target pixels")

    layers = 0
    if require_labels or "layer_depths_gt" in sample:
        depths = _require_tensor(sample, "layer_depths_gt")
        stop = _require_tensor(sample, "ray_stop_index_gt")
        target = _require_tensor(sample, "layer_target_gt")
        if depths.ndim != 4 or int(depths.shape[0]) != views:
            raise ValueError("layer_depths_gt must be [V,L,Hl,Wl]")
        layers = int(depths.shape[1])
        layer_height, layer_width = map(int, depths.shape[-2:])
        if stop.shape != (views, layer_height, layer_width):
            raise ValueError("ray_stop_index_gt shape differs from layer depths")
        if layers < 1 or min(layer_height, layer_width) < 1:
            raise ValueError("ground-truth layer tensors cannot have empty dimensions")
        if stop.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("ray stopping indices must use an integer dtype")
        if target.shape != depths.shape:
            raise ValueError("layer_target_gt shape differs from layer depths")
        if expected_layers is not None and layers != int(expected_layers):
            raise ValueError(f"expected {expected_layers} layers, got {layers}")
        if bool(((stop < 0) | (stop > layers)).any().item()):
            raise ValueError("ray stopping indices must be between 0 and L")
        if not bool(torch.isfinite(depths.float()).all().item()):
            raise ValueError("layer depth labels contain non-finite values")
        if not bool(torch.isfinite(target.float()).all().item()):
            raise ValueError("layer target labels contain non-finite values")
        layer_ids = torch.arange(layers, device=stop.device)[None, :, None, None]
        valid = layer_ids < stop[:, None]
        if bool((depths[valid] <= 0).any().item()):
            raise ValueError("valid layer depths must be positive")
        if layers > 1:
            adjacent_valid = valid[:, 1:] & valid[:, :-1]
            if bool((depths[:, 1:][adjacent_valid] <= depths[:, :-1][adjacent_valid]).any().item()):
                raise ValueError("valid ground-truth layer depths are not ordered")
        if bool(((target.float() < 0.0) | (target.float() > 1.0)).any().item()):
            raise ValueError("target ownership labels must be in [0,1]")
        if bool((target.bool() & ~valid).any().item()):
            raise ValueError("invalid ray layers cannot own the target")

    if require_ss_target:
        ss_target = _require_tensor(sample, "target")
        if ss_target.shape != (8, 16, 16, 16):
            raise ValueError("SS target must be [8,16,16,16]")
        if not bool(torch.isfinite(ss_target.float()).all().item()):
            raise ValueError("SS target contains non-finite values")

    return {
        "views": views,
        "patches": patches,
        "patch_side": patch_side,
        "channels": channels,
        "image_height": height,
        "image_width": width,
        "layers": layers,
    }


__all__ = [
    "LAYERED_SAMPLE_FORMAT",
    "LAYERED_SAMPLE_MANIFEST_FORMAT",
    "LAYERED_SS_CHECKPOINT_FORMAT",
    "TARGET_LARI_CHECKPOINT_FORMAT",
    "LayeredPrediction",
    "ground_truth_prediction",
    "validate_paired_sample",
]
