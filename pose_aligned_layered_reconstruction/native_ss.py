"""Frozen-30K-compatible SS Flow with target-aware layered ray conditioning."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ._bootstrap import bootstrap_external_paths
from .contracts import LayeredPrediction, ground_truth_prediction, validate_paired_sample
from .dino_context import select_dino_patch_features
from .layer_matching import corrupt_layer_prediction, match_queries_to_layers
from .projection import build_ss_projection, project_masked_dino, resize_sampling_grid
from .target_lari import TargetConditionedLaRI
from .view_fusion import (
    EveryBlockConditionProjection,
    LayerEvidenceEncoder,
    PoseViewAggregator,
    ZeroInitViewGate,
    aggregate_layer_evidence,
)


LAYERED_SS_MODEL_VERSION = "pose_aligned_layered_reconstruction.native_ss.v1"
PARENT_SS_MODEL_FORMAT = "pose_point_depth_mv.native_ss_genrecon_no_vggt.v1"


def volume_xyz_to_flow_tokens(volume: torch.Tensor) -> torch.Tensor:
    if volume.ndim != 5 or tuple(volume.shape[-3:]) != (16, 16, 16):
        raise ValueError("SS volume must be [B,C,16,16,16]")
    return volume.flatten(2).transpose(1, 2).contiguous()


def flow_tokens_to_volume_xyz(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3 or int(tokens.shape[1]) != 16**3:
        raise ValueError("SS flow tokens must be [B,4096,C]")
    return tokens.transpose(1, 2).reshape(
        int(tokens.shape[0]), int(tokens.shape[2]), 16, 16, 16
    ).contiguous()


def _lora_disabled(module: nn.Module):
    disable = getattr(module, "disable_adapter", None)
    if disable is None:
        raise RuntimeError("PEFT flow does not expose disable_adapter()")
    return disable()


class LayeredNativeSSFlow(nn.Module):
    """Current 30K appearance path plus zero-init layered geometry adapters."""

    def __init__(
        self,
        flow: nn.Module,
        *,
        condition_channels: int = 1024,
        layer_channels: int = 128,
        lari_hidden_channels: int = 256,
        lari_layers: int = 5,
        lari_decoder_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.flow = flow
        self.condition_channels = int(condition_channels)
        self.layer_channels = int(layer_channels)
        core = self.flow_core
        schema = (
            int(core.resolution),
            int(core.in_channels),
            int(core.out_channels),
            int(core.patch_size),
        )
        if schema != (16, 8, 8, 1):
            raise ValueError(f"unsupported stock SS Flow schema={schema}")

        # Names and shapes of these two modules intentionally match SS30K.
        self.aggregator = PoseViewAggregator(self.condition_channels)
        self.block_condition = EveryBlockConditionProjection(
            self.condition_channels, int(core.model_channels), len(core.blocks)
        )

        # New modules are isolated under distinct names and start as an exact
        # functional no-op through their final zero-initialized adapters.
        self.target_lari = TargetConditionedLaRI(
            input_channels=self.condition_channels,
            hidden_channels=int(lari_hidden_channels),
            layers=int(lari_layers),
            decoder_blocks=int(lari_decoder_blocks),
        )
        self.layer_encoder = LayerEvidenceEncoder(
            output_channels=self.layer_channels
        )
        self.view_gate = ZeroInitViewGate()
        self.layer_block_condition = EveryBlockConditionProjection(
            self.layer_channels, int(core.model_channels), len(core.blocks)
        )

    @property
    def flow_core(self) -> nn.Module:
        base = getattr(self.flow, "base_model", None)
        core = getattr(base, "model", None)
        return core if isinstance(core, nn.Module) else self.flow

    def stock_prediction(
        self, x: torch.Tensor, t: torch.Tensor, condition: Any
    ) -> torch.Tensor:
        with _lora_disabled(self.flow):
            return self.flow(x, t, condition)

    def _core_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Any,
        appearance_condition: torch.Tensor,
        layer_condition: torch.Tensor | None,
    ) -> torch.Tensor:
        bootstrap_external_paths()
        from trellis.modules.spatial import patchify, unpatchify

        core = self.flow_core
        h = volume_xyz_to_flow_tokens(patchify(x, core.patch_size))
        h = core.input_layer(h) + core.pos_emb[None]
        t_embedding = core.t_embedder(t)
        if core.share_mod:
            t_embedding = core.adaLN_modulation(t_embedding)
        h = h.type(core.dtype)
        t_embedding = t_embedding.type(core.dtype)
        contexts = condition if isinstance(condition, list) else [condition]
        for context_value in contexts:
            context = context_value.type(core.dtype)
            for block_index, block in enumerate(core.blocks):
                h = h + self.block_condition(
                    block_index, appearance_condition
                ).to(h.dtype)
                if layer_condition is not None:
                    h = h + self.layer_block_condition(
                        block_index, layer_condition
                    ).to(h.dtype)
                h = block(h, t_embedding, context)
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = core.out_layer(h)
        h = flow_tokens_to_volume_xyz(h)
        return unpatchify(h, core.patch_size).contiguous()

    def predict_layers(
        self,
        sample: dict[str, Any],
        *,
        device: torch.device,
        view_indices: torch.Tensor | None,
        source: str,
        raw_feature_mode: str = "correct",
    ) -> LayeredPrediction:
        if source not in {"predicted", "oracle"}:
            raise ValueError("layer source must be predicted or oracle")
        views = int(sample["masked_visual_patch_features"].shape[0])
        indices = (
            torch.arange(views, device=device)
            if view_indices is None
            else view_indices.to(device=device, dtype=torch.long)
        )
        if source == "oracle":
            return ground_truth_prediction(sample).to(device).select_views(indices)

        raw = select_dino_patch_features(sample["raw_visual_patch_features"]).to(
            device=device, dtype=torch.float32
        ).index_select(0, indices)
        if raw_feature_mode == "raw_feature_view_cyclic1":
            raw = torch.roll(raw, shifts=1, dims=0)
        elif raw_feature_mode != "correct":
            raise ValueError(f"unsupported raw feature mode={raw_feature_mode!r}")
        masked = select_dino_patch_features(
            sample["masked_visual_patch_features"]
        ).to(device=device, dtype=torch.float32).index_select(0, indices)
        visible = sample["visible_target_mask"].to(
            device=device, dtype=torch.float32
        ).index_select(0, indices)
        intrinsics = sample["intrinsics"].to(
            device=device, dtype=torch.float32
        ).index_select(0, indices)
        intrinsics = intrinsics.clone()
        intrinsics[:, 1, 1] *= float(sample.get("layer_camera_y_sign", 1.0))
        extrinsics = sample["extrinsics"].to(
            device=device, dtype=torch.float32
        ).index_select(0, indices)
        return self.target_lari(
            raw_patch_features=raw,
            masked_patch_features=masked,
            visible_target_mask=visible,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            extrinsics_type=str(sample["extrinsics_type"]),
            camera_forward_sign=float(sample["camera_forward_sign"]),
            image_size=tuple(int(value) for value in sample["image_size"]),
        )

    def build_conditions(
        self,
        sample: dict[str, Any],
        *,
        device: torch.device,
        view_indices: torch.Tensor | None = None,
        use_layer_branch: bool = True,
        layer_source: str = "predicted",
        projection_mode: str = "correct",
        layer_corruption: str = "correct",
        raw_feature_mode: str = "correct",
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        metadata = validate_paired_sample(
            sample,
            require_labels=layer_source == "oracle",
            expected_layers=(self.target_lari.layers if layer_source == "oracle" else None),
        )
        if int(metadata["channels"]) != self.condition_channels:
            raise ValueError("SS layered path requires 1024-D DINO features")
        geometry = build_ss_projection(
            sample,
            device=device,
            view_indices=view_indices,
            projection_mode=str(projection_mode),
        )
        appearance = project_masked_dino(
            sample, geometry, device=device, view_indices=view_indices
        )
        if not use_layer_branch:
            aggregated, stats, _ = self.aggregator(
                appearance, geometry["valid"]
            )
            return aggregated, None, stats

        prediction = self.predict_layers(
            sample,
            device=device,
            view_indices=view_indices,
            source=str(layer_source),
            raw_feature_mode=str(raw_feature_mode),
        )
        prediction = corrupt_layer_prediction(prediction, str(layer_corruption))
        layer_geometry = build_ss_projection(
            sample, device=device, view_indices=view_indices,
            projection_mode=str(projection_mode), for_layer_branch=True,
        )
        visible = sample["visible_target_mask"].to(
            device=device, dtype=torch.float32
        )
        if view_indices is not None:
            visible = visible.index_select(
                0, view_indices.to(device=device, dtype=torch.long)
            )
        match = match_queries_to_layers(
            prediction,
            image_grid=layer_geometry["image_grid"],
            camera_depth=layer_geometry["camera_depth"],
            frustum_valid=layer_geometry["valid"],
            visible_target_mask=visible,
            layer_grid=resize_sampling_grid(
                layer_geometry["image_grid"],
                image_size=tuple(sample["image_size"]),
                map_size=prediction.spatial_size,
            ),
        )
        view_logit_residual = self.view_gate(match.features)
        appearance_condition, appearance_stats, details = self.aggregator(
            appearance,
            geometry["valid"],
            logit_residual=view_logit_residual,
        )
        per_view_layer = self.layer_encoder(match.features)
        layer_condition, layer_stats = aggregate_layer_evidence(
            per_view_layer,
            appearance_weights=details["view_weights"],
            target_support=match.target_support,
        )
        stats = {
            **appearance_stats,
            **layer_stats,
            "target_support_mean": match.target_support.mean(),
            "visible_target_support_mean": match.visible_target_support.mean(),
            "hidden_target_support_mean": match.hidden_target_support.mean(),
            "occluder_before_mean": match.occluder_before.mean(),
            "view_logit_residual_rms": view_logit_residual.square().mean().sqrt(),
        }
        return appearance_condition, layer_condition, stats

    def adapted_prediction(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Any,
        sample: dict[str, Any] | None,
        *,
        view_indices: torch.Tensor | None = None,
        use_layer_branch: bool = True,
        layer_source: str = "predicted",
        projection_mode: str = "correct",
        layer_corruption: str = "correct",
        raw_feature_mode: str = "correct",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if sample is None:
            appearance_condition = x.new_zeros(
                (int(x.shape[0]), 16**3, self.condition_channels),
                dtype=torch.float32,
            )
            layer_condition = None
            zero = x.new_zeros((), dtype=torch.float32)
            stats = {
                "supported_fraction": zero,
                "appearance_condition_rms": zero,
                "layer_supported_fraction": zero,
                "layer_condition_rms": zero,
            }
        else:
            appearance_condition, layer_condition, stats = self.build_conditions(
                sample,
                device=x.device,
                view_indices=view_indices,
                use_layer_branch=bool(use_layer_branch),
                layer_source=str(layer_source),
                projection_mode=str(projection_mode),
                layer_corruption=str(layer_corruption),
                raw_feature_mode=str(raw_feature_mode),
            )
        prediction = self._core_forward(
            x, t, condition, appearance_condition, layer_condition
        )
        stats["condition_present"] = x.new_tensor(float(sample is not None))
        stats["layer_branch_present"] = x.new_tensor(
            float(sample is not None and use_layer_branch)
        )
        return prediction, stats

    def parent_prediction(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Any,
        sample: dict[str, Any] | None,
        *,
        view_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prediction, _ = self.adapted_prediction(
            x,
            t,
            condition,
            sample,
            view_indices=view_indices,
            use_layer_branch=False,
        )
        return prediction

    def new_adapters_exact_zero(self) -> bool:
        return bool(
            self.view_gate.exact_zero()
            and self.layer_block_condition.exact_zero()
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "format": LAYERED_SS_MODEL_VERSION,
            "parent": "no-VGGT Native-SS step30000 EMA",
            "parent_parameter_layout_preserved": True,
            "appearance_input": "masked_target_dino",
            "layer_input": "raw_rgb_dino_plus_visible_target_cue",
            "query_grid": "dense_16_cubed",
            "target_lari": self.target_lari.metadata(),
            "layer_match_features": 12,
            "layer_channels": self.layer_channels,
            "new_view_gate": "zero_init_residual_logits",
            "new_block_condition": "zero_init_independent_projection_every_block",
            "layer_raster_sampling": "half_pixel_centers",
            "layer_strength": "appearance_weights_times_absolute_target_support",
            "stock_content_identity": getattr(self, "stock_binding", None),
        }


class LayeredSSCFGFlow(nn.Module):
    """Apply the new layer branch only to the positive CFG branch."""

    def __init__(
        self,
        model: LayeredNativeSSFlow,
        positive_condition: Any,
        sample: dict[str, Any],
        *,
        layer_source: str = "predicted",
        use_layer_branch: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.positive_condition = positive_condition
        self.sample = sample
        self.layer_source = str(layer_source)
        self.use_layer_branch = bool(use_layer_branch)
        self.positive_calls = 0
        self.negative_calls = 0
        self.cached_conditions = None

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: Any) -> torch.Tensor:
        positive = condition is self.positive_condition or (
            torch.is_tensor(condition)
            and torch.is_tensor(self.positive_condition)
            and condition.data_ptr() == self.positive_condition.data_ptr()
        )
        self.positive_calls += int(positive)
        self.negative_calls += int(not positive)
        if positive and not torch.is_grad_enabled():
            if self.cached_conditions is None:
                appearance, layer, _ = self.model.build_conditions(
                    self.sample, device=x.device, layer_source=self.layer_source,
                    use_layer_branch=self.use_layer_branch,
                )
                self.cached_conditions = (appearance, layer)
            return self.model._core_forward(
                x, t, condition, *self.cached_conditions
            )
        prediction, _ = self.model.adapted_prediction(
            x,
            t,
            condition,
            self.sample if positive else None,
            use_layer_branch=positive and self.use_layer_branch,
            layer_source=self.layer_source,
        )
        return prediction


def build_layered_ss_components(
    *,
    pretrained: str = "Stable-X/trellis-vggt-v0-2",
    lora_rank: int = 8,
    lora_alpha: int = 16,
    condition_channels: int = 1024,
    layer_channels: int = 128,
    lari_hidden_channels: int = 256,
    lari_layers: int = 5,
    lari_decoder_blocks: int = 3,
    gradient_checkpointing: bool = False,
    device: torch.device,
) -> tuple[Any, LayeredNativeSSFlow, dict[str, Any]]:
    """Load only the stock SS Flow and sampler, not the full TRELLIS pipeline."""

    bootstrap_external_paths()
    from peft import LoraConfig, get_peft_model
    from trellis.pipelines import samplers
    from .pretrained import load_trellis_model

    base_flow, stock_binding = load_trellis_model(
        pretrained, "ss_flow_img_dit_L_16l8_fp16"
    )
    base_flow = base_flow.to(device).eval()
    for parameter in base_flow.parameters():
        parameter.requires_grad_(False)
    base_flow.use_checkpoint = bool(gradient_checkpointing)
    for block in base_flow.blocks:
        block.use_checkpoint = bool(gradient_checkpointing)
    flow = get_peft_model(
        base_flow,
        LoraConfig(
            r=int(lora_rank),
            lora_alpha=int(lora_alpha),
            lora_dropout=0.0,
            bias="none",
            target_modules=["to_q", "to_kv", "to_out", "to_qkv"],
        ),
    )
    model = LayeredNativeSSFlow(
        flow,
        condition_channels=int(condition_channels),
        layer_channels=int(layer_channels),
        lari_hidden_channels=int(lari_hidden_channels),
        lari_layers=int(lari_layers),
        lari_decoder_blocks=int(lari_decoder_blocks),
    ).to(device)
    model.stock_binding = stock_binding
    sampler = samplers.FlowEulerGuidanceIntervalSampler(sigma_min=1.0e-5)
    sampler_params = {
        "steps": 25,
        "cfg_strength": 5.0,
        "cfg_interval": [0.5, 1.0],
        "rescale_t": 3.0,
    }
    summary = {
        **model.metadata(),
        "pretrained": str(pretrained),
        "lora_rank": int(lora_rank),
        "lora_alpha": int(lora_alpha),
        "sampler": type(sampler).__name__,
        "sampler_params": sampler_params,
    }
    return sampler, model, summary


__all__ = [
    "LAYERED_SS_MODEL_VERSION",
    "LayeredNativeSSFlow",
    "LayeredSSCFGFlow",
    "PARENT_SS_MODEL_FORMAT",
    "build_layered_ss_components",
    "flow_tokens_to_volume_xyz",
    "volume_xyz_to_flow_tokens",
]
