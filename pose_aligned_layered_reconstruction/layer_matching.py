"""Match ordered target-aware ray intersections to posed 3D queries."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .contracts import LayeredPrediction
from .projection import sample_maps


LAYER_MATCH_FEATURE_NAMES = (
    "target_surface_support",
    "non_target_surface_support",
    "any_surface_support",
    "visible_target_support",
    "hidden_target_support",
    "occluder_before",
    "free_space_probability",
    "signed_target_depth_residual",
    "absolute_target_depth_residual",
    "matched_layer_rank",
    "matched_confidence",
    "visible_target_mask",
)


@dataclass(frozen=True)
class QueryLayerMatch:
    features: torch.Tensor
    target_support: torch.Tensor
    visible_target_support: torch.Tensor
    hidden_target_support: torch.Tensor
    occluder_before: torch.Tensor
    matched_target_depth: torch.Tensor
    matched_layer_rank: torch.Tensor

    def validate(self) -> "QueryLayerMatch":
        if self.features.ndim != 3:
            raise ValueError("query-layer features must be [V,N,F]")
        views, queries, channels = map(int, self.features.shape)
        if channels != len(LAYER_MATCH_FEATURE_NAMES):
            raise ValueError("query-layer feature schema differs")
        for name, value in (
            ("target_support", self.target_support),
            ("visible_target_support", self.visible_target_support),
            ("hidden_target_support", self.hidden_target_support),
            ("occluder_before", self.occluder_before),
            ("matched_target_depth", self.matched_target_depth),
            ("matched_layer_rank", self.matched_layer_rank),
        ):
            if value.shape != (views, queries):
                raise ValueError(f"{name} query shape differs")
        if not bool(torch.isfinite(self.features).all().item()):
            raise ValueError("query-layer match contains non-finite values")
        return self


def corrupt_layer_prediction(
    prediction: LayeredPrediction, mode: str
) -> LayeredPrediction:
    if mode == "correct":
        return prediction
    if mode == "layer_view_cyclic1":
        return LayeredPrediction(
            depths=torch.roll(prediction.depths, 1, 0),
            stop_logits=torch.roll(prediction.stop_logits, 1, 0),
            target_logits=torch.roll(prediction.target_logits, 1, 0),
            confidence_logits=torch.roll(prediction.confidence_logits, 1, 0),
        )
    if mode == "target_view_cyclic1":
        return LayeredPrediction(
            depths=prediction.depths,
            stop_logits=prediction.stop_logits,
            target_logits=torch.roll(prediction.target_logits, 1, 0),
            confidence_logits=prediction.confidence_logits,
        )
    raise ValueError(f"unsupported layer corruption mode={mode!r}")


def _sample_prediction(
    prediction: LayeredPrediction, image_grid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction.validate()
    valid_maps = prediction.valid_probabilities().to(image_grid.device)
    sampled_valid = sample_maps(valid_maps, image_grid)
    # Invalid suffix depths are arbitrary placeholders.  Their interpolation
    # must not move a valid intersection near a silhouette boundary.
    def sample_valid(value: torch.Tensor) -> torch.Tensor:
        numerator = sample_maps(value.float() * valid_maps, image_grid)
        return (numerator / sampled_valid.clamp_min(1.0e-8)).permute(0, 2, 1)

    depths = sample_valid(prediction.depths)
    valid = sampled_valid.permute(0, 2, 1)
    target = sample_valid(torch.sigmoid(prediction.target_logits.float()))
    confidence = sample_valid(torch.sigmoid(prediction.confidence_logits.float()))
    return depths, valid.clamp(0.0, 1.0), target.clamp(0.0, 1.0), confidence.clamp(0.0, 1.0)


def match_queries_to_layers(
    prediction: LayeredPrediction,
    *,
    image_grid: torch.Tensor,
    camera_depth: torch.Tensor,
    frustum_valid: torch.Tensor,
    visible_target_mask: torch.Tensor,
    layer_grid: torch.Tensor | None = None,
    absolute_tolerance: float = 0.02,
    relative_tolerance: float = 0.01,
) -> QueryLayerMatch:
    """Create view/query evidence without copying occluder appearance."""

    if image_grid.ndim != 3 or int(image_grid.shape[-1]) != 2:
        raise ValueError("image_grid must be [V,N,2]")
    if camera_depth.shape != image_grid.shape[:2]:
        raise ValueError("camera depth/query grid shape differs")
    if frustum_valid.shape != camera_depth.shape:
        raise ValueError("frustum validity shape differs")
    if float(absolute_tolerance) <= 0 or float(relative_tolerance) < 0:
        raise ValueError("layer matching tolerances are invalid")
    depths, layer_valid, target_probability, confidence = _sample_prediction(
        prediction, image_grid if layer_grid is None else layer_grid
    )
    query_depth = camera_depth.float()[..., None]
    tolerance = (
        float(absolute_tolerance)
        + float(relative_tolerance) * query_depth.abs()
    ).clamp_min(1.0e-6)
    signed = query_depth - depths
    normalized_distance = signed.abs() / tolerance
    proximity = torch.exp(-normalized_distance)
    common = layer_valid * confidence * proximity
    target_scores = common * target_probability
    non_target_scores = common * (1.0 - target_probability)
    target_support = target_scores.amax(dim=-1)
    non_target_support = non_target_scores.amax(dim=-1)
    any_surface_support = common.amax(dim=-1)

    target_weights = target_scores / target_scores.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    matched_depth = (target_weights * depths).sum(dim=-1)
    matched_confidence = (target_weights * confidence).sum(dim=-1)
    layers = int(depths.shape[-1])
    ranks = torch.linspace(
        0.0, 1.0, layers, device=depths.device, dtype=depths.dtype
    )
    matched_rank = (target_weights * ranks).sum(dim=-1)
    signed_target_residual = (camera_depth.float() - matched_depth) / tolerance[..., 0]
    signed_target_residual = signed_target_residual.clamp(-4.0, 4.0) / 4.0
    absolute_target_residual = signed_target_residual.abs()

    definitely_before = torch.sigmoid((query_depth - depths) / tolerance)
    before_non_target = (
        layer_valid * confidence * (1.0 - target_probability) * definitely_before
    ).clamp(0.0, 1.0)
    occluder_before = 1.0 - torch.prod(1.0 - before_non_target, dim=-1)
    before_any = (layer_valid * confidence * definitely_before).clamp(0.0, 1.0)
    free_space_probability = torch.prod(1.0 - before_any, dim=-1)

    mask = visible_target_mask.float()
    if mask.ndim == 3:
        mask = mask[:, None]
    visible_at_query = sample_maps(mask, image_grid)[:, 0].clamp(0.0, 1.0)
    visible_target_support = (
        target_support * visible_at_query * (1.0 - occluder_before)
    )
    hidden_probability = torch.maximum(1.0 - visible_at_query, occluder_before)
    hidden_target_support = target_support * hidden_probability
    frustum = frustum_valid.float()
    target_support = target_support * frustum
    visible_target_support = visible_target_support * frustum
    hidden_target_support = hidden_target_support * frustum
    occluder_before = occluder_before * frustum
    non_target_support = non_target_support * frustum
    any_surface_support = any_surface_support * frustum
    free_space_probability = free_space_probability * frustum
    visible_at_query = visible_at_query * frustum
    matched_confidence = matched_confidence * frustum
    matched_rank = matched_rank * frustum
    signed_target_residual = signed_target_residual * frustum
    absolute_target_residual = absolute_target_residual * frustum
    matched_depth = matched_depth * frustum

    features = torch.stack(
        (
            target_support,
            non_target_support,
            any_surface_support,
            visible_target_support,
            hidden_target_support,
            occluder_before,
            free_space_probability,
            signed_target_residual,
            absolute_target_residual,
            matched_rank,
            matched_confidence,
            visible_at_query,
        ),
        dim=-1,
    )
    return QueryLayerMatch(
        features=features,
        target_support=target_support,
        visible_target_support=visible_target_support,
        hidden_target_support=hidden_target_support,
        occluder_before=occluder_before,
        matched_target_depth=matched_depth,
        matched_layer_rank=matched_rank,
    ).validate()


__all__ = [
    "LAYER_MATCH_FEATURE_NAMES",
    "QueryLayerMatch",
    "corrupt_layer_prediction",
    "match_queries_to_layers",
]
