"""Run the independent two-stage pose-constrained TRELLIS.2 geometry path."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from .conditioning import build_2d_context, sample_to_device
from .constants import (
    CHECKPOINT_FORMAT,
    SHAPE_FLOW_MODEL,
    SHAPE_LATENT_RESOLUTION,
    SS_FLOW_MODEL,
    SS_LATENT_CHANNELS,
    SS_LATENT_RESOLUTION,
)
from .data import PoseTrellis2Dataset
from .models import BoundPoseCFGFlow, load_trainable_state_dict
from .runtime import (
    decode_shape_latent,
    export_geometry_mesh,
    load_pose_conditioned_flow,
    load_shape_decoder,
    load_ss_decoder,
    make_sampler,
    make_shape_noise,
    ss_latent_to_shape_coords,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--ss-checkpoint", required=True)
    parser.add_argument("--shape-checkpoint", required=True)
    parser.add_argument("--output", required=True, help="untextured .ply or .obj")
    parser.add_argument("--report", default="")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--views", default="all", help="ordered view indices, e.g. 0,3,5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ss-steps", type=int, default=0)
    parser.add_argument("--shape-steps", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser


def parse_views(specification: str, view_count: int) -> torch.Tensor:
    text = str(specification).strip().lower()
    if text in ("", "all"):
        values = list(range(int(view_count)))
    else:
        values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values or len(set(values)) != len(values):
        raise ValueError("views must be a non-empty ordered set")
    invalid = [value for value in values if value < 0 or value >= int(view_count)]
    if invalid:
        raise IndexError(f"view indices outside [0,{view_count}): {invalid}")
    return torch.tensor(values, dtype=torch.long)


def _checkpoint_configuration(path: str | Path, stage: str) -> dict[str, Any]:
    payload = torch.load(
        Path(path).expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a pose-constrained TRELLIS.2 checkpoint")
    if payload.get("stage") != stage:
        raise ValueError(
            f"checkpoint {path} has stage={payload.get('stage')!r}, expected {stage!r}"
        )
    arguments = payload.get("args")
    identity = payload.get("model_identity")
    if not isinstance(arguments, dict) or not isinstance(identity, dict):
        raise ValueError(f"checkpoint {path} lacks its frozen model configuration")
    if (
        identity.get("old_30k_checkpoint_loaded") is not False
        or identity.get("distillation") is not False
    ):
        raise RuntimeError("checkpoint is not a clean official-TRELLIS.2 initialization")
    required = ("lora_rank", "lora_alpha", "context_mode")
    missing = [name for name in required if name not in arguments]
    if missing:
        raise ValueError(f"checkpoint {path} lacks arguments={missing}")
    step = int(payload.get("step", -1))
    if step <= 0:
        raise ValueError(f"checkpoint {path} has invalid optimizer step={step}")
    model_path = str(arguments.get("model_path", ""))
    expected_base = model_path or (
        SS_FLOW_MODEL if stage == "ss" else SHAPE_FLOW_MODEL
    )
    if identity.get("base_model") != expected_base:
        raise RuntimeError(f"checkpoint {path} base-model identity differs")
    data_identity = payload.get("data_identity")
    if not isinstance(data_identity, dict):
        raise ValueError(f"checkpoint {path} lacks its data identity")
    return {
        "payload": payload,
        "step": step,
        "lora_rank": int(arguments["lora_rank"]),
        "lora_alpha": int(arguments["lora_alpha"]),
        "context_mode": str(arguments["context_mode"]),
        "model_path": model_path,
        "model_identity": identity,
        "data_identity": data_identity,
    }


def _validate_manifest_binding(
    configuration: dict[str, Any],
    manifest_identity: dict[str, Any],
    stage: str,
) -> None:
    checkpoint_identity = configuration["data_identity"]
    expected_stage = str(checkpoint_identity.get("stage"))
    if expected_stage != stage:
        raise RuntimeError(
            f"{stage} checkpoint data stage differs: {expected_stage!r}"
        )
    fields = ("manifest_sha256", "sample_count", "uid_sha256", "target_family")
    mismatch = {
        field: (checkpoint_identity.get(field), manifest_identity.get(field))
        for field in fields
        if checkpoint_identity.get(field) != manifest_identity.get(field)
    }
    if mismatch:
        raise RuntimeError(f"{stage} checkpoint/manifest identity differs={mismatch}")


def _load_stage_model(
    path: str | Path,
    stage: str,
    *,
    device: torch.device,
    weights: str,
    manifest_identity: dict[str, Any],
) -> tuple[torch.nn.Module, dict[str, Any]]:
    configuration = _checkpoint_configuration(path, stage)
    _validate_manifest_binding(configuration, manifest_identity, stage)
    model = load_pose_conditioned_flow(
        stage,
        device=device,
        model_path=configuration["model_path"] or None,
        lora_rank=configuration["lora_rank"],
        lora_alpha=configuration["lora_alpha"],
        gradient_checkpointing=False,
    ).eval()
    # Reuse the already memory-mapped payload instead of reading a multi-GB
    # training checkpoint a second time.
    payload = configuration.pop("payload")
    state_key = (
        "ema_trainable_state" if weights == "ema" else "model_trainable_state"
    )
    state = payload.get(state_key)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint {path} lacks {state_key}")
    load_trainable_state_dict(model, state)
    del state, payload
    gc.collect()
    return model, configuration


def _release_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sample_stage(
    stage: str,
    model: torch.nn.Module,
    noise: Any,
    *,
    sample: dict[str, Any],
    view_indices: torch.Tensor,
    context_mode: str,
    steps_override: int,
    quiet: bool,
) -> tuple[Any, dict[str, Any]]:
    positive = build_2d_context(
        sample["dino_features"],
        view_indices,
        mode=context_mode,  # type: ignore[arg-type]
    ).to(device=noise.device)
    negative = torch.zeros_like(positive)
    bound = BoundPoseCFGFlow(
        model,  # type: ignore[arg-type]
        positive_context=positive,
        sample=sample,
        view_indices=view_indices,
    )
    sampler, parameters = make_sampler(stage)
    if int(steps_override) > 0:
        parameters["steps"] = int(steps_override)
    result = sampler.sample(
        bound,
        noise,
        cond=positive,
        neg_cond=negative,
        **parameters,
        verbose=not quiet,
        tqdm_desc=f"Pose-conditioned TRELLIS.2 {stage}",
    )
    positive_stats = [
        row for row in bound.stats if row.get("condition_present") == 1.0
    ]
    summary = {
        "sampler": parameters,
        "positive_calls": int(bound.positive_calls),
        "negative_calls": int(bound.negative_calls),
        "pose_condition_builds": int(bound.condition_builds),
        "mean_supported_fraction": float(
            np.mean([row["supported_fraction"] for row in positive_stats])
        )
        if positive_stats
        else 0.0,
        "mean_prediction_rms": float(
            np.mean([row["prediction_rms"] for row in positive_stats])
        ) if positive_stats else 0.0,
    }
    return result.samples, summary


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.ss_steps) < 0 or int(args.shape_steps) < 0:
        raise ValueError("step overrides must be non-negative")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("TRELLIS.2 geometry inference requires a CUDA device")
    output = Path(args.output).expanduser().resolve()
    if output.suffix.lower() not in (".ply", ".obj"):
        raise ValueError("--output must end in .ply or .obj")
    dataset = PoseTrellis2Dataset(args.manifest, stage="inference")
    manifest_identity = dataset.identity()
    if int(args.sample_index) < 0 or int(args.sample_index) >= len(dataset):
        raise IndexError("sample-index lies outside the manifest")
    sample = sample_to_device(dataset[int(args.sample_index)], device)
    view_indices = parse_views(
        args.views, int(sample["dino_features"].shape[0])
    )
    view_indices_device = view_indices.to(device)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    started = time.time()

    ss_model, ss_configuration = _load_stage_model(
        args.ss_checkpoint,
        "ss",
        device=device,
        weights=args.weights,
        manifest_identity=manifest_identity,
    )
    ss_noise = torch.randn(
        (1, SS_LATENT_CHANNELS) + (SS_LATENT_RESOLUTION,) * 3,
        device=device,
    )
    ss_latent, ss_stats = _sample_stage(
        "ss",
        ss_model,
        ss_noise,
        sample=sample,
        view_indices=view_indices_device,
        context_mode=ss_configuration["context_mode"],
        steps_override=int(args.ss_steps),
        quiet=bool(args.quiet),
    )
    del ss_model, ss_noise
    _release_cuda_cache()

    ss_decoder = load_ss_decoder(device=device)
    shape_coords = ss_latent_to_shape_coords(
        ss_latent, ss_decoder, shape_resolution=SHAPE_LATENT_RESOLUTION
    )
    del ss_decoder, ss_latent
    _release_cuda_cache()

    shape_model, shape_configuration = _load_stage_model(
        args.shape_checkpoint,
        "shape",
        device=device,
        weights=args.weights,
        manifest_identity=manifest_identity,
    )
    shape_noise = make_shape_noise(shape_coords, device=device)
    shape_latent, shape_stats = _sample_stage(
        "shape",
        shape_model,
        shape_noise,
        sample=sample,
        view_indices=view_indices_device,
        context_mode=shape_configuration["context_mode"],
        steps_override=int(args.shape_steps),
        quiet=bool(args.quiet),
    )
    del shape_model, shape_noise
    _release_cuda_cache()

    shape_decoder = load_shape_decoder(device=device)
    mesh = decode_shape_latent(shape_latent, shape_decoder)
    export_geometry_mesh(mesh, output)
    vertex_count = int(mesh.vertices.shape[0])
    face_count = int(mesh.faces.shape[0])
    del shape_decoder, shape_latent, mesh
    _release_cuda_cache()

    report = {
        "format": "pose_constrained_trellis2.inference.v1",
        "geometry_only": True,
        "texture_flow_loaded": False,
        "uid": str(sample["uid"]),
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "sample_index": int(args.sample_index),
        "selected_views": view_indices.tolist(),
        "reference_view": int(view_indices[0]),
        "seed": int(args.seed),
        "weights": str(args.weights),
        "ss_checkpoint": str(Path(args.ss_checkpoint).expanduser().resolve()),
        "shape_checkpoint": str(Path(args.shape_checkpoint).expanduser().resolve()),
        "ss_checkpoint_step": ss_configuration["step"],
        "shape_checkpoint_step": shape_configuration["step"],
        "ss_context_mode": ss_configuration["context_mode"],
        "shape_context_mode": shape_configuration["context_mode"],
        "shape_active_tokens": int(shape_coords.shape[0]),
        "mesh": str(output),
        "vertices": vertex_count,
        "faces": face_count,
        "ss": ss_stats,
        "shape": shape_stats,
        "elapsed_seconds": time.time() - started,
    }
    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else output.with_suffix(output.suffix + ".json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    report = run(make_parser().parse_args())
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
