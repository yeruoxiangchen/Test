"""Adapter-only checkpointing with a strict TRELLIS.2 stage contract."""

from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn

from .constants import CHECKPOINT_FORMAT
from .models import (
    load_trainable_state_dict,
    trainable_named_parameters,
    trainable_state_dict,
)


def initialize_ema(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().float().clone()
        for name, parameter in trainable_named_parameters(model).items()
    }


@torch.no_grad()
def update_ema(
    model: nn.Module,
    ema: dict[str, torch.Tensor],
    *,
    decay: float,
) -> None:
    named = trainable_named_parameters(model)
    if set(named) != set(ema):
        raise ValueError("EMA/trainable parameter names differ")
    for name, parameter in named.items():
        ema[name].mul_(float(decay)).add_(
            parameter.detach().float(), alpha=1.0 - float(decay)
        )


def ema_ramp_decay(target: float, update: int) -> float:
    if int(update) <= 0 or not 0.0 < float(target) < 1.0:
        raise ValueError("invalid EMA update/target")
    return min(float(target), float(1 + int(update)) / float(10 + int(update)))


def _cpu_ema(ema: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in ema.items()}


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    ema: dict[str, torch.Tensor],
    stage: str,
    step: int,
    micro_step: int,
    args: dict[str, Any],
    data_identity: dict[str, Any],
    model_identity: dict[str, Any],
    history: list[dict[str, Any]],
    rng_by_rank: list[dict[str, Any]] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    trainable = list(trainable_named_parameters(model).values())
    if not trainable or not all(
        bool(torch.isfinite(value.detach()).all().item()) for value in trainable
    ):
        raise RuntimeError("refusing to save empty or non-finite trainable state")
    if set(ema) != set(trainable_named_parameters(model)) or not all(
        bool(torch.isfinite(value).all().item()) for value in ema.values()
    ):
        raise RuntimeError("refusing to save invalid EMA state")
    saved_rng = list(rng_by_rank) if rng_by_rank is not None else [rng_state()]
    if not saved_rng or not all(isinstance(value, dict) for value in saved_rng):
        raise ValueError("rng_by_rank must contain one state object per rank")
    payload = {
        "format": CHECKPOINT_FORMAT,
        "stage": str(stage),
        "step": int(step),
        "micro_step": int(micro_step),
        "model_trainable_state": trainable_state_dict(model),
        "ema_trainable_state": _cpu_ema(ema),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "args": dict(args),
        "data_identity": dict(data_identity),
        "model_identity": dict(model_identity),
        "history": list(history),
        "rng": saved_rng[0],
        "rng_by_rank": saved_rng,
    }
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def update_checkpoint_alias(source: str | Path, alias: str | Path) -> None:
    """Atomically make ``alias`` a hard link to a completed checkpoint."""

    source_path = Path(source).resolve(strict=True)
    alias_path = Path(alias)
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = alias_path.with_name(f".{alias_path.name}.tmp-{os.getpid()}")
    os.link(source_path, temporary)
    os.replace(temporary, alias_path)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    stage: str,
    weights: str = "raw",
) -> dict[str, Any]:
    if weights not in ("raw", "ema"):
        raise ValueError("weights must be raw or ema")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unexpected checkpoint format={payload.get('format')!r}")
    if payload.get("stage") != str(stage):
        raise ValueError(
            f"checkpoint stage={payload.get('stage')!r} does not match {stage!r}"
        )
    state_key = (
        "ema_trainable_state" if weights == "ema" else "model_trainable_state"
    )
    state = payload.get(state_key)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint lacks {state_key}")
    load_trainable_state_dict(model, state)
    return payload


def ema_to_device(
    model: nn.Module, state: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    named = trainable_named_parameters(model)
    if set(named) != set(state):
        raise ValueError("checkpoint EMA/trainable names differ")
    return {
        name: state[name].to(device=parameter.device, dtype=torch.float32).clone()
        for name, parameter in named.items()
    }


__all__ = [
    "ema_ramp_decay",
    "ema_to_device",
    "initialize_ema",
    "load_checkpoint",
    "restore_rng_state",
    "rng_state",
    "save_checkpoint",
    "update_checkpoint_alias",
    "update_ema",
]
