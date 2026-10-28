#!/usr/bin/env python3
"""Load the real SS30K parent and audit zero-init compatibility on CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from .checkpoints import DEFAULT_PARENT_SS30K, load_parent_ss30k
from .dino_context import build_masked_dino_contexts
from .native_ss import build_layered_ss_components
from .synthetic_fixture import make_synthetic_paired_sample


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parent_ss30k", default=str(DEFAULT_PARENT_SS30K))
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--condition_tokens", type=int, default=32)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--skip_parent_hash", action="store_true")
    parser.add_argument("--run_backward", action="store_true")
    parser.add_argument("--output_report", default="")
    return parser


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = make_parser().parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the real SS30K smoke requires CUDA")
    if int(args.condition_tokens) <= 0:
        raise ValueError("condition_tokens must be positive")
    torch.cuda.set_device(device)
    torch.manual_seed(20260909)
    torch.cuda.manual_seed_all(20260909)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    print("[smoke] building stock SS Flow and independent adapters", flush=True)
    sampler, model, summary = build_layered_ss_components(
        pretrained=str(args.pretrained),
        gradient_checkpointing=bool(args.run_backward),
        device=device,
    )
    built_seconds = time.time() - started
    print(f"[smoke] built in {built_seconds:.1f}s; loading SS30K EMA", flush=True)
    parent = load_parent_ss30k(
        model,
        args.parent_ss30k,
        weights="ema",
        verify_hash=not bool(args.skip_parent_hash),
        freeze_parent=True,
    )
    loaded_seconds = time.time() - started
    print(f"[smoke] loaded {parent['loaded_tensor_count']} parent tensors; running forward", flush=True)
    sample = make_synthetic_paired_sample(views=int(args.views))
    condition = build_masked_dino_contexts(
        sample["masked_visual_patch_features"],
        ss_context_tokens=int(args.condition_tokens),
    )["stock_condition"].to(device)
    target = sample["target"].to(device)[None]
    noise = torch.randn_like(target)
    x_t, gt_velocity = sampler._get_model_gt(target, 0.5, noise)
    timestep = torch.full((1,), 500.0, device=device)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        parent_prediction = model.parent_prediction(
            x_t, timestep, condition, sample
        )
        layered_prediction, stats = model.adapted_prediction(
            x_t,
            timestep,
            condition,
            sample,
            use_layer_branch=True,
            layer_source="oracle",
        )
    _synchronize(device)
    difference = (layered_prediction.float() - parent_prediction.float()).abs()
    forward_report = {
        "new_adapters_exact_zero": model.new_adapters_exact_zero(),
        "max_abs_difference": float(difference.amax().item()),
        "mean_abs_difference": float(difference.mean().item()),
        "parent_finite": bool(torch.isfinite(parent_prediction).all().item()),
        "layered_finite": bool(torch.isfinite(layered_prediction).all().item()),
        "target_support_mean": float(stats["target_support_mean"].item()),
    }
    forward_report["passed"] = bool(
        forward_report["new_adapters_exact_zero"]
        and forward_report["max_abs_difference"] == 0.0
        and forward_report["parent_finite"]
        and forward_report["layered_finite"]
    )
    if not forward_report["passed"]:
        raise RuntimeError(f"real-parent forward audit failed: {forward_report}")
    print(f"[smoke] forward={json.dumps(forward_report)}", flush=True)

    backward_report = None
    if args.run_backward:
        print("[smoke] running backward and optimizer step", flush=True)
        del parent_prediction, layered_prediction, difference
        torch.cuda.empty_cache()
        for parameter in model.target_lari.parameters():
            parameter.requires_grad_(False)
        trainable = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(trainable, lr=1.0e-4)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, _ = model.adapted_prediction(
                x_t,
                timestep,
                condition,
                sample,
                use_layer_branch=True,
                layer_source="oracle",
            )
            loss = F.mse_loss(prediction.float(), gt_velocity.float())
        loss.backward()
        finite_gradients = all(
            parameter.grad is None
            or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in trainable
        )
        nonzero_gradient_tensors = sum(
            int(
                parameter.grad is not None
                and bool(torch.count_nonzero(parameter.grad).item())
            )
            for parameter in trainable
        )
        optimizer.step()
        finite_parameters = all(
            bool(torch.isfinite(parameter).all().item()) for parameter in trainable
        )
        backward_report = {
            "loss": float(loss.detach().item()),
            "trainable_tensor_count": len(trainable),
            "nonzero_gradient_tensor_count": nonzero_gradient_tensors,
            "finite_gradients": finite_gradients,
            "finite_parameters_after_step": finite_parameters,
            "layer_projection_remains_zero": model.layer_block_condition.exact_zero(),
        }
        backward_report["passed"] = bool(
            finite_gradients
            and finite_parameters
            and nonzero_gradient_tensors > 0
            and not backward_report["layer_projection_remains_zero"]
        )
        if not backward_report["passed"]:
            raise RuntimeError(f"real-parent backward audit failed: {backward_report}")
    _synchronize(device)
    result = {
        "passed": True,
        "scope": "real SS30K EMA load plus oracle-layer adapter forward",
        "device": str(device),
        "torch_version": torch.__version__,
        "parent": parent,
        "model": summary,
        "build_seconds": built_seconds,
        "parent_loaded_seconds": loaded_seconds,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "condition_tokens": int(args.condition_tokens),
        "actual_condition_tokens": int(condition.shape[1]),
        "views": int(args.views),
        "forward": forward_report,
        "backward": backward_report,
    }
    if args.output_report:
        destination = Path(args.output_report).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
