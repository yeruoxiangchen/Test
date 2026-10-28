"""Content-bound parent initialization and compact research checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .contracts import LAYERED_SS_CHECKPOINT_FORMAT, TARGET_LARI_CHECKPOINT_FORMAT
from .native_ss import PARENT_SS_MODEL_FORMAT, LayeredNativeSSFlow
from .target_lari import TargetConditionedLaRI


DEFAULT_PARENT_SS30K = Path(
    "/data/zjr/proobjaverse_official_30k_checkpoint_archives/"
    "ProObjaverse_30K_noVGGT_SS_SLat_numbered_checkpoints_20260818_v1/"
    "ss/checkpoints/step_030000.pt"
)
DEFAULT_PARENT_SS30K_SHA256 = (
    "042a1b5467b05975584aeb571dec6ffaed5096edcc6abe4aa88600c9c9506b7f"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parent_parameter_names(model: LayeredNativeSSFlow) -> set[str]:
    return {
        name
        for name, _ in model.named_parameters()
        if "lora_" in name
        or name.startswith("aggregator.")
        or name.startswith("block_condition.")
    }


def new_parameter_names(model: LayeredNativeSSFlow) -> set[str]:
    prefixes = (
        "target_lari.",
        "layer_encoder.",
        "view_gate.",
        "layer_block_condition.",
    )
    return {
        name for name, _ in model.named_parameters() if name.startswith(prefixes)
    }


def load_parent_ss30k(
    model: LayeredNativeSSFlow,
    checkpoint_path: str | Path = DEFAULT_PARENT_SS30K,
    *,
    weights: str = "ema",
    verify_hash: bool = True,
    freeze_parent: bool = True,
) -> dict[str, Any]:
    """Load every and only parent-trainable tensor into matching module names."""

    path = Path(checkpoint_path).expanduser().resolve(strict=True)
    # Always bind adapters to the actual parent content.  verify_hash controls
    # comparison with the published local SS30K digest, not identity recording.
    observed_hash = sha256_file(path)
    if (
        verify_hash
        and path == DEFAULT_PARENT_SS30K.resolve()
        and observed_hash != DEFAULT_PARENT_SS30K_SHA256
    ):
        raise RuntimeError("default SS30K checkpoint SHA-256 differs")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != PARENT_SS_MODEL_FORMAT:
        raise ValueError(
            f"unexpected parent SS format={checkpoint.get('format')!r}"
        )
    if int(checkpoint.get("step", -1)) != 30_000:
        raise ValueError("parent initialization must be the frozen step 30000")
    if weights not in {"ema", "raw"}:
        raise ValueError("parent weights must be ema or raw")
    state_key = "ema_trainable_state" if weights == "ema" else "model_trainable_state"
    state = checkpoint.get(state_key)
    if not isinstance(state, dict):
        raise ValueError(f"parent checkpoint lacks {state_key}")
    expected = _parent_parameter_names(model)
    if set(state) != expected:
        raise RuntimeError(
            "parent parameter layout differs: "
            f"missing={sorted(expected - set(state))[:8]} "
            f"unexpected={sorted(set(state) - expected)[:8]}"
        )
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameter = named[name]
            if parameter.shape != value.shape:
                raise RuntimeError(f"parent tensor shape differs for {name}")
            parameter.copy_(value.to(parameter.device, parameter.dtype))

    if freeze_parent:
        new_names = new_parameter_names(model)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in new_names)
        trainable = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        if trainable != new_names or not trainable:
            raise RuntimeError("failed to isolate new layered parameters")
    return {
        "path": str(path),
        "sha256": observed_hash,
        "format": str(checkpoint["format"]),
        "step": int(checkpoint["step"]),
        "weights": str(weights),
        "loaded_tensor_count": len(state),
        "freeze_parent": bool(freeze_parent),
        "new_trainable_tensor_count": len(new_parameter_names(model)),
    }


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not state:
        raise RuntimeError("refusing to save an empty trainable state")
    return state


def load_exact_state(
    model: nn.Module, state: dict[str, torch.Tensor], *, trainable_only: bool
) -> None:
    named = dict(model.named_parameters())
    expected = {
        name
        for name, parameter in named.items()
        if (parameter.requires_grad or not trainable_only)
    }
    if set(state) != expected:
        raise ValueError(
            f"checkpoint state differs: missing={sorted(expected - set(state))[:8]} "
            f"unexpected={sorted(set(state) - expected)[:8]}"
        )
    with torch.no_grad():
        for name, value in state.items():
            parameter = named[name]
            if parameter.shape != value.shape:
                raise ValueError(f"checkpoint tensor shape differs for {name}")
            parameter.copy_(value.to(parameter.device, parameter.dtype))


def save_target_lari_checkpoint(
    path: str | Path,
    *,
    model: TargetConditionedLaRI,
    step: int,
    optimizer: torch.optim.Optimizer | None,
    data_identity: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": TARGET_LARI_CHECKPOINT_FORMAT,
            "step": int(step),
            "model": model.state_dict(),
            "model_metadata": model.metadata(),
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "data_identity": data_identity,
            "history": history,
        },
        destination,
    )


def load_target_lari_checkpoint(
    model: TargetConditionedLaRI, path: str | Path
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != TARGET_LARI_CHECKPOINT_FORMAT:
        raise ValueError("unexpected Target-LaRI checkpoint format")
    if checkpoint.get("model_metadata") != model.metadata():
        raise ValueError("Target-LaRI checkpoint architecture differs")
    model.load_state_dict(checkpoint["model"], strict=True)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "step": int(checkpoint["step"]),
        "data_identity": checkpoint.get("data_identity"),
    }


def save_layered_ss_checkpoint(
    path: str | Path,
    *,
    model: LayeredNativeSSFlow,
    step: int,
    parent_binding: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    data_identity: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": LAYERED_SS_CHECKPOINT_FORMAT,
            "step": int(step),
            "parent_binding": parent_binding,
            "model_metadata": model.metadata(),
            "new_trainable_state": trainable_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "data_identity": data_identity,
            "history": history,
        },
        destination,
    )


def load_layered_ss_checkpoint(
    model: LayeredNativeSSFlow,
    path: str | Path,
    *,
    parent_binding: dict[str, Any],
    target_lari_binding: dict[str, Any] | None = None,
    layer_source: str | None = None,
    allow_layer_source_change: bool = False,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != LAYERED_SS_CHECKPOINT_FORMAT:
        raise ValueError("unexpected layered SS checkpoint format")
    if checkpoint.get("model_metadata") != model.metadata():
        raise ValueError("layered SS checkpoint architecture differs")
    training_data = checkpoint.get("data_identity", {})
    trained_source = training_data.get("layer_source")
    if (
        layer_source is not None
        and trained_source != layer_source
        and not allow_layer_source_change
    ):
        raise ValueError(
            f"adapter trained with {trained_source!r}, requested {layer_source!r}; "
            "use the explicit layer-source-change flag only for an ablation"
        )
    if trained_source == "predicted":
        expected_lari = training_data.get("target_lari") or {}
        if not target_lari_binding or expected_lari.get("sha256") != target_lari_binding.get("sha256"):
            raise ValueError("layered SS requires the exact Target-LaRI checkpoint used in training")
    frozen_parent = checkpoint.get("parent_binding")
    if not isinstance(frozen_parent, dict):
        raise ValueError("layered SS checkpoint lacks parent binding")
    for key in ("sha256", "format", "step", "weights"):
        if frozen_parent.get(key) != parent_binding.get(key):
            raise RuntimeError(f"layered SS parent binding differs for {key}")
    state = checkpoint.get("new_trainable_state")
    if not isinstance(state, dict) or not state:
        raise ValueError("layered SS checkpoint lacks adapter state")
    allowed = ("layer_encoder.", "view_gate.", "layer_block_condition.")
    if any(not str(name).startswith(allowed) for name in state):
        raise ValueError("layered SS checkpoint contains non-adapter tensors")
    named = dict(model.named_parameters())
    expected = {
        name
        for name in named
        if name.startswith(allowed)
    }
    if set(state) != expected:
        raise ValueError(
            "layered SS adapter state differs: "
            f"missing={sorted(expected - set(state))[:8]} "
            f"unexpected={sorted(set(state) - expected)[:8]}"
        )
    with torch.no_grad():
        for name, value in state.items():
            parameter = named[name]
            if parameter.shape != value.shape:
                raise ValueError(f"layered SS tensor shape differs for {name}")
            parameter.copy_(value.to(parameter.device, parameter.dtype))
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "step": int(checkpoint.get("step", 0)),
        "data_identity": checkpoint.get("data_identity"),
    }


__all__ = [
    "DEFAULT_PARENT_SS30K",
    "DEFAULT_PARENT_SS30K_SHA256",
    "load_exact_state",
    "load_layered_ss_checkpoint",
    "load_parent_ss30k",
    "load_target_lari_checkpoint",
    "new_parameter_names",
    "save_layered_ss_checkpoint",
    "save_target_lari_checkpoint",
    "sha256_file",
    "trainable_state_dict",
]
