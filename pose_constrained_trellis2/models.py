"""TRELLIS.2 Flow adapters implementing the current pose-constrained method."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Iterator, Literal

import torch
import torch.nn.functional as F
from torch import nn

from .conditioning import EveryBlockCondition, MultiviewAggregator
from .constants import (
    FLOW_BLOCKS,
    FLOW_MODEL_CHANNELS,
    SHAPE_LATENT_CHANNELS,
    SHAPE_LATENT_RESOLUTION,
    SS_LATENT_CHANNELS,
    SS_LATENT_RESOLUTION,
)
from .geometry import dense_coords, project_pose_features


Stage = Literal["ss", "shape"]


def _manual_cast(value: Any, dtype: torch.dtype) -> Any:
    return value if torch.is_autocast_enabled() else value.type(dtype)


def _flow_core(flow: nn.Module) -> nn.Module:
    """Return the original TRELLIS.2 module through an optional PEFT wrapper."""

    get_base_model = getattr(flow, "get_base_model", None)
    if callable(get_base_model):
        core = get_base_model()
        if isinstance(core, nn.Module):
            return core
    base = getattr(flow, "base_model", None)
    core = getattr(base, "model", None)
    return core if isinstance(core, nn.Module) else flow


def _lora_disabled(flow: nn.Module):
    disable = getattr(flow, "disable_adapter", None)
    return disable() if callable(disable) else nullcontext()


def _lora_outputs_exact_zero(flow: nn.Module) -> bool:
    values = [
        parameter
        for name, parameter in flow.named_parameters()
        if "lora_B" in name
    ]
    return not values or all(
        int(torch.count_nonzero(value.detach()).item()) == 0 for value in values
    )


def _straight_through_dense(
    prediction: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != reference.shape:
        raise ValueError("dense straight-through tensors differ in shape")
    return reference.detach() + (prediction - prediction.detach())


def _straight_through_sparse(prediction: Any, reference: Any) -> Any:
    if not torch.equal(prediction.coords, reference.coords):
        raise ValueError("sparse straight-through coordinates differ")
    feats = reference.feats.detach() + (
        prediction.feats - prediction.feats.detach()
    )
    return prediction.replace(feats)


def _same_tensor_identity(value: Any, reference: Any) -> bool:
    if value is reference:
        return True
    return bool(
        torch.is_tensor(value)
        and torch.is_tensor(reference)
        and value.shape == reference.shape
        and value.data_ptr() == reference.data_ptr()
    )


class DensePoseConditionedFlow(nn.Module):
    """TRELLIS.2 SS Flow with current-method pose conditioning and LoRA."""

    stage: Stage = "ss"

    def __init__(
        self,
        flow: nn.Module,
        *,
        condition_channels: int = 1024,
        strict_schema: bool = True,
    ) -> None:
        super().__init__()
        self.flow = flow
        self.condition_channels = int(condition_channels)
        core = self.flow_core
        schema = (
            int(core.resolution),
            int(core.in_channels),
            int(core.out_channels),
            int(core.model_channels),
            len(core.blocks),
        )
        expected = (
            SS_LATENT_RESOLUTION,
            SS_LATENT_CHANNELS,
            SS_LATENT_CHANNELS,
            FLOW_MODEL_CHANNELS,
            FLOW_BLOCKS,
        )
        if strict_schema and schema != expected:
            raise ValueError(f"unsupported TRELLIS.2 SS Flow schema={schema}")
        self.resolution = int(core.resolution)
        self.aggregator = MultiviewAggregator(self.condition_channels)
        self.block_condition = EveryBlockCondition(
            self.condition_channels, int(core.model_channels), len(core.blocks)
        )
        self.register_buffer(
            "query_coords",
            dense_coords(self.resolution, with_batch=True),
            persistent=False,
        )

    @property
    def flow_core(self) -> nn.Module:
        return _flow_core(self.flow)

    def stock_prediction(
        self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        with _lora_disabled(self.flow):
            return self.flow(x, t, context)

    def _pose_condition(
        self,
        x: torch.Tensor,
        sample: dict[str, Any] | None,
        view_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if int(x.shape[0]) != 1:
            raise ValueError("pose-conditioned SS currently requires batch size one")
        if sample is None:
            condition = x.new_zeros(
                (1, self.resolution**3, self.condition_channels),
                dtype=torch.float32,
            )
            zero = x.new_zeros((), dtype=torch.float32)
            return condition, {
                "view_count": zero,
                "visible_fraction": zero,
                "supported_fraction": zero,
                "query_count": zero.new_tensor(float(self.resolution**3)),
                "mean_visible_views": zero,
                "aggregation_entropy": zero,
                "condition_rms": zero,
            }
        projected, valid, projection_stats = project_pose_features(
            sample,
            self.query_coords,
            resolution=self.resolution,
            device=x.device,
            view_indices=view_indices,
        )
        condition, aggregation_stats = self.aggregator(projected, valid)
        return condition[None], {**projection_stats, **aggregation_stats}

    def _adapted_core_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        condition_3d: torch.Tensor,
    ) -> torch.Tensor:
        core = self.flow_core
        h = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1).contiguous()
        h = core.input_layer(h)
        if core.pe_mode == "ape":
            h = h + core.pos_emb[None]
        t_embedding = core.t_embedder(t)
        if core.share_mod:
            t_embedding = core.adaLN_modulation(t_embedding)
        t_embedding = _manual_cast(t_embedding, core.dtype)
        h = _manual_cast(h, core.dtype)
        context = _manual_cast(context, core.dtype)
        for block_index, block in enumerate(core.blocks):
            residual = self.block_condition(block_index, condition_3d)
            h = h + residual.to(dtype=h.dtype)
            h = block(h, t_embedding, context, core.rope_phases)
        h = _manual_cast(h, x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = core.out_layer(h)
        return h.permute(0, 2, 1).view(
            h.shape[0], h.shape[2], *([core.resolution] * 3)
        ).contiguous()

    def adapted_prediction(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        sample: dict[str, Any] | None,
        *,
        view_indices: torch.Tensor | None = None,
        stock_velocity: torch.Tensor | None = None,
        precomputed_condition: tuple[
            torch.Tensor, dict[str, torch.Tensor]
        ] | None = None,
        compare_stock: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if precomputed_condition is None:
            condition_3d, base_stats = self._pose_condition(
                x, sample, view_indices
            )
        else:
            condition_3d, base_stats = precomputed_condition
        stats = dict(base_stats)
        prediction = self._adapted_core_forward(
            x, t, context, condition_3d
        )
        zero_anchor = (
            self.block_condition.exact_zero() and _lora_outputs_exact_zero(self.flow)
        )
        stock = stock_velocity
        if stock is None and (bool(compare_stock) or zero_anchor):
            stock = self.stock_prediction(x, t, context)
        if zero_anchor:
            if stock is None:
                raise RuntimeError("zero initialization requires a stock reference")
            prediction = _straight_through_dense(prediction, stock)
        reference = prediction.float()
        stats.update(
            {
                "condition_present": reference.new_tensor(float(sample is not None)),
                "zero_init_stock_anchor": reference.new_tensor(float(zero_anchor)),
                "stock_comparison": reference.new_tensor(float(stock is not None)),
                "prediction_rms": reference.square().mean().sqrt(),
            }
        )
        if stock is not None:
            delta = reference - stock.float()
            stats.update(
                {
                    "flow_delta_rms": delta.square().mean().sqrt(),
                    "flow_delta_abs_max": delta.abs().amax(),
                }
            )
        return prediction, stats

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        sample: dict[str, Any] | None = None,
        *,
        view_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prediction, _ = self.adapted_prediction(
            x,
            t,
            context,
            sample,
            view_indices=view_indices,
            compare_stock=False,
        )
        return prediction


class SparsePoseConditionedFlow(nn.Module):
    """TRELLIS.2 Shape-SLat 512 Flow with current-method pose conditioning."""

    stage: Stage = "shape"

    def __init__(
        self,
        flow: nn.Module,
        *,
        condition_channels: int = 1024,
        strict_schema: bool = True,
    ) -> None:
        super().__init__()
        self.flow = flow
        self.condition_channels = int(condition_channels)
        core = self.flow_core
        schema = (
            int(core.resolution),
            int(core.in_channels),
            int(core.out_channels),
            int(core.model_channels),
            len(core.blocks),
        )
        expected = (
            SHAPE_LATENT_RESOLUTION,
            SHAPE_LATENT_CHANNELS,
            SHAPE_LATENT_CHANNELS,
            FLOW_MODEL_CHANNELS,
            FLOW_BLOCKS,
        )
        if strict_schema and schema != expected:
            raise ValueError(f"unsupported TRELLIS.2 Shape Flow schema={schema}")
        self.resolution = int(core.resolution)
        self.aggregator = MultiviewAggregator(self.condition_channels)
        self.block_condition = EveryBlockCondition(
            self.condition_channels, int(core.model_channels), len(core.blocks)
        )

    @property
    def flow_core(self) -> nn.Module:
        return _flow_core(self.flow)

    def stock_prediction(self, x: Any, t: torch.Tensor, context: Any) -> Any:
        with _lora_disabled(self.flow):
            return self.flow(x, t, context)

    def _pose_condition(
        self,
        x: Any,
        sample: dict[str, Any] | None,
        view_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if int(x.shape[0]) != 1:
            raise ValueError("pose-conditioned Shape Flow currently requires batch size one")
        queries = int(x.feats.shape[0])
        if sample is None:
            condition = x.feats.new_zeros(
                (queries, self.condition_channels), dtype=torch.float32
            )
            zero = x.feats.new_zeros((), dtype=torch.float32)
            return condition, {
                "view_count": zero,
                "visible_fraction": zero,
                "supported_fraction": zero,
                "query_count": zero.new_tensor(float(queries)),
                "mean_visible_views": zero,
                "aggregation_entropy": zero,
                "condition_rms": zero,
            }
        projected, valid, projection_stats = project_pose_features(
            sample,
            x.coords,
            resolution=self.resolution,
            device=x.device,
            view_indices=view_indices,
        )
        condition, aggregation_stats = self.aggregator(projected, valid)
        return condition, {**projection_stats, **aggregation_stats}

    def _adapted_core_forward(
        self,
        x: Any,
        t: torch.Tensor,
        context: Any,
        condition_3d: torch.Tensor,
    ) -> Any:
        core = self.flow_core
        if isinstance(context, list):
            from .dependency import import_sparse_module

            context = import_sparse_module().VarLenTensor.from_tensor_list(context)
        h = core.input_layer(x)
        h = _manual_cast(h, core.dtype)
        t_embedding = core.t_embedder(t)
        if core.share_mod:
            t_embedding = core.adaLN_modulation(t_embedding)
        t_embedding = _manual_cast(t_embedding, core.dtype)
        context = _manual_cast(context, core.dtype)
        if core.pe_mode == "ape":
            positional = core.pos_embedder(h.coords[:, 1:])
            h = h + _manual_cast(positional, core.dtype)
        if int(condition_3d.shape[0]) != int(h.feats.shape[0]):
            raise ValueError("Shape condition and active SLat coordinates differ")
        for block_index, block in enumerate(core.blocks):
            residual = self.block_condition(block_index, condition_3d)
            h = h.replace(h.feats + residual.to(dtype=h.dtype))
            h = block(h, t_embedding, context)
        h = _manual_cast(h, x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        return core.out_layer(h)

    def adapted_prediction(
        self,
        x: Any,
        t: torch.Tensor,
        context: Any,
        sample: dict[str, Any] | None,
        *,
        view_indices: torch.Tensor | None = None,
        stock_velocity: Any | None = None,
        precomputed_condition: tuple[
            torch.Tensor, dict[str, torch.Tensor]
        ] | None = None,
        compare_stock: bool = True,
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        if precomputed_condition is None:
            condition_3d, base_stats = self._pose_condition(
                x, sample, view_indices
            )
        else:
            condition_3d, base_stats = precomputed_condition
        stats = dict(base_stats)
        prediction = self._adapted_core_forward(x, t, context, condition_3d)
        zero_anchor = (
            self.block_condition.exact_zero() and _lora_outputs_exact_zero(self.flow)
        )
        stock = stock_velocity
        if stock is None and (bool(compare_stock) or zero_anchor):
            stock = self.stock_prediction(x, t, context)
        if stock is not None and not torch.equal(prediction.coords, stock.coords):
            raise RuntimeError("pose adaptation changed Shape-SLat coordinates")
        if zero_anchor:
            if stock is None:
                raise RuntimeError("zero initialization requires a stock reference")
            prediction = _straight_through_sparse(prediction, stock)
        reference = prediction.feats.float()
        stats.update(
            {
                "condition_present": reference.new_tensor(float(sample is not None)),
                "zero_init_stock_anchor": reference.new_tensor(float(zero_anchor)),
                "stock_comparison": reference.new_tensor(float(stock is not None)),
                "prediction_rms": reference.square().mean().sqrt(),
            }
        )
        if stock is not None:
            delta = reference - stock.feats.float()
            stats.update(
                {
                    "flow_delta_rms": delta.square().mean().sqrt(),
                    "flow_delta_abs_max": delta.abs().amax(),
                }
            )
        return prediction, stats

    def forward(
        self,
        x: Any,
        t: torch.Tensor,
        context: Any,
        sample: dict[str, Any] | None = None,
        *,
        view_indices: torch.Tensor | None = None,
    ) -> Any:
        prediction, _ = self.adapted_prediction(
            x,
            t,
            context,
            sample,
            view_indices=view_indices,
            compare_stock=False,
        )
        return prediction


class BoundPoseCFGFlow(nn.Module):
    """Bind pose evidence only to the positive branch of a stock T2 CFG sampler."""

    def __init__(
        self,
        model: DensePoseConditionedFlow | SparsePoseConditionedFlow,
        *,
        positive_context: Any,
        sample: dict[str, Any],
        view_indices: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.positive_context = positive_context
        self.sample = sample
        self.view_indices = view_indices
        self.positive_calls = 0
        self.negative_calls = 0
        self.condition_builds = 0
        self._cached_positive_condition: tuple[
            torch.Tensor, dict[str, torch.Tensor]
        ] | None = None
        self._cached_sparse_coords: torch.Tensor | None = None
        self.stats: list[dict[str, float]] = []

    def forward(self, x: Any, t: torch.Tensor, context: Any) -> Any:
        positive = _same_tensor_identity(context, self.positive_context)
        precomputed = None
        if positive:
            if self._cached_positive_condition is None:
                self._cached_positive_condition = self.model._pose_condition(
                    x, self.sample, self.view_indices
                )
                self.condition_builds += 1
                if not torch.is_tensor(x):
                    self._cached_sparse_coords = x.coords.detach().clone()
            elif self._cached_sparse_coords is not None and not torch.equal(
                x.coords, self._cached_sparse_coords
            ):
                raise RuntimeError("Shape sampler changed coordinates after pose caching")
            precomputed = self._cached_positive_condition
        prediction, stats = self.model.adapted_prediction(
            x,
            t,
            context,
            self.sample if positive else None,
            view_indices=self.view_indices if positive else None,
            precomputed_condition=precomputed,
            compare_stock=False,
        )
        if positive:
            self.positive_calls += 1
        else:
            self.negative_calls += 1
        self.stats.append(
            {
                key: float(value.detach().float().item())
                for key, value in stats.items()
                if torch.is_tensor(value) and value.numel() == 1
            }
        )
        return prediction


def apply_attention_lora(
    base_flow: nn.Module,
    *,
    rank: int = 8,
    alpha: int = 16,
) -> nn.Module:
    """Freeze the official backbone and add LoRA to every attention projection."""

    for parameter in base_flow.parameters():
        parameter.requires_grad_(False)
    if int(rank) == 0:
        return base_flow
    if int(rank) < 0 or int(alpha) <= 0:
        raise ValueError("LoRA rank must be non-negative and alpha positive")
    from peft import LoraConfig, get_peft_model

    return get_peft_model(
        base_flow,
        LoraConfig(
            r=int(rank),
            lora_alpha=int(alpha),
            lora_dropout=0.0,
            bias="none",
            target_modules=["to_q", "to_kv", "to_out", "to_qkv"],
        ),
    )


def build_pose_conditioned_flow(
    base_flow: nn.Module,
    *,
    stage: Stage,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    gradient_checkpointing: bool = True,
) -> DensePoseConditionedFlow | SparsePoseConditionedFlow:
    """Adapt an official TRELLIS.2 geometry Flow without altering its weights."""

    core = base_flow
    core.use_checkpoint = bool(gradient_checkpointing)
    for block in core.blocks:
        block.use_checkpoint = bool(gradient_checkpointing)
    flow = apply_attention_lora(
        base_flow, rank=int(lora_rank), alpha=int(lora_alpha)
    )
    if stage == "ss":
        model: DensePoseConditionedFlow | SparsePoseConditionedFlow = (
            DensePoseConditionedFlow(flow)
        )
    elif stage == "shape":
        model = SparsePoseConditionedFlow(flow)
    else:
        raise ValueError(f"unsupported stage={stage!r}")
    validate_trainable_surface(model, require_lora=int(lora_rank) > 0)
    return model


def trainable_named_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in trainable_named_parameters(model).items()
    }


def load_trainable_state_dict(
    model: nn.Module, state: dict[str, torch.Tensor]
) -> None:
    expected = trainable_named_parameters(model)
    if set(state) != set(expected):
        raise ValueError(
            "trainable state differs: "
            f"missing={sorted(set(expected) - set(state))[:8]} "
            f"unexpected={sorted(set(state) - set(expected))[:8]}"
        )
    with torch.no_grad():
        for name, value in state.items():
            parameter = expected[name]
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(f"checkpoint shape mismatch for {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def validate_trainable_surface(
    model: nn.Module, *, require_lora: bool = True
) -> dict[str, Any]:
    names = list(trainable_named_parameters(model))
    unexpected = [
        name
        for name in names
        if "lora_" not in name
        and not name.startswith("aggregator.")
        and not name.startswith("block_condition.")
    ]
    lora_names = [name for name in names if "lora_" in name]
    if unexpected or not names or (require_lora and not lora_names):
        raise RuntimeError(
            f"invalid trainable surface: unexpected={unexpected[:8]} "
            f"lora_count={len(lora_names)}"
        )
    return {
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "lora_tensor_count": len(lora_names),
        "condition_tensor_count": len(names) - len(lora_names),
        "block_count": len(model.block_condition.projections),
    }


def optimizer_parameter_groups(
    model: nn.Module,
    *,
    new_lr: float,
    lora_lr: float,
    new_weight_decay: float,
) -> list[dict[str, Any]]:
    lora: list[nn.Parameter] = []
    condition: list[nn.Parameter] = []
    for name, parameter in trainable_named_parameters(model).items():
        (lora if "lora_" in name else condition).append(parameter)
    if not condition or (not lora and float(lora_lr) != 0.0):
        raise RuntimeError("optimizer requires condition parameters and configured LoRA")
    groups: list[dict[str, Any]] = [
        {
            "params": condition,
            "lr": float(new_lr),
            "weight_decay": float(new_weight_decay),
            "name": "pose_condition",
        }
    ]
    if lora:
        groups.insert(
            0,
            {
                "params": lora,
                "lr": float(lora_lr),
                "weight_decay": 0.0,
                "name": "lora",
            },
        )
    return groups


def iter_trainable_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


__all__ = [
    "BoundPoseCFGFlow",
    "DensePoseConditionedFlow",
    "SparsePoseConditionedFlow",
    "apply_attention_lora",
    "build_pose_conditioned_flow",
    "load_trainable_state_dict",
    "optimizer_parameter_groups",
    "trainable_named_parameters",
    "trainable_state_dict",
    "validate_trainable_surface",
]
