#!/usr/bin/env python3
"""Train zero-init layered SS adapters on top of frozen SS30K EMA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .checkpoints import (
    DEFAULT_PARENT_SS30K,
    load_parent_ss30k,
    load_target_lari_checkpoint,
    save_layered_ss_checkpoint,
)
from .native_ss import build_layered_ss_components
from .paired_dataset import PairedLayerDataset, collate_one


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--parent_ss30k", default=str(DEFAULT_PARENT_SS30K))
    parser.add_argument("--parent_weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--skip_parent_hash", action="store_true")
    parser.add_argument("--target_lari_checkpoint", default="")
    parser.add_argument("--layer_source", choices=("oracle", "predicted"), default="oracle")
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_steps", type=int, default=1_000)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_condition_views", type=int, default=1)
    parser.add_argument("--max_condition_views", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--layer_channels", type=int, default=128)
    parser.add_argument("--lari_hidden_channels", type=int, default=256)
    parser.add_argument("--lari_layers", type=int, default=5)
    parser.add_argument("--lari_decoder_blocks", type=int, default=3)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        "max_steps",
        "save_every",
        "log_every",
        "grad_accum",
        "min_condition_views",
        "max_condition_views",
        "learning_rate",
        "grad_clip",
        "layer_channels",
        "lari_hidden_channels",
        "lari_layers",
        "lari_decoder_blocks",
    )
    bad = [name for name in positive if float(getattr(args, name)) <= 0]
    if bad:
        raise ValueError(f"arguments must be positive: {bad}")
    if int(args.min_condition_views) > int(args.max_condition_views):
        raise ValueError("min_condition_views exceeds max_condition_views")
    if int(args.num_workers) < 0 or float(args.weight_decay) < 0:
        raise ValueError("num_workers/weight_decay must be non-negative")
    if args.layer_source == "predicted" and not args.target_lari_checkpoint:
        raise ValueError("predicted layer training requires --target_lari_checkpoint")


def _sample_views(sample: dict, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    views = int(sample["masked_visual_patch_features"].shape[0])
    upper = min(views, int(args.max_condition_views))
    lower = min(upper, int(args.min_condition_views))
    count = int(torch.randint(lower, upper + 1, (), device=device).item())
    indices = torch.randperm(views, device=device)[:count]
    exposed = sample["visible_target_mask"].float().reshape(views, -1).sum(-1) > 0
    exposed = exposed.to(device)
    if not bool(exposed[indices].any()):
        candidates = torch.nonzero(exposed, as_tuple=False).flatten()
        if not len(candidates):
            raise ValueError("view sampling needs at least one visible target cue")
        indices[0] = candidates[torch.randint(len(candidates), (), device=device)]
    return indices


def _sample_t(device: torch.device) -> float:
    return float(torch.sigmoid(torch.randn((), device=device) + 1.0).item())


@torch.no_grad()
def _zero_init_audit(
    *,
    model,
    sampler,
    sample: dict,
    device: torch.device,
    layer_source: str,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> dict:
    target = sample["target"].to(device=device, dtype=torch.float32)[None]
    x_t, _ = sampler._get_model_gt(target, 0.5, torch.zeros_like(target))
    t = torch.full((1,), 500.0, device=device)
    condition = sample["stock_condition"].to(device)
    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
        parent = model.parent_prediction(x_t, t, condition, sample)
        layered, _ = model.adapted_prediction(
            x_t,
            t,
            condition,
            sample,
            use_layer_branch=True,
            layer_source=layer_source,
        )
    maximum = float((layered.float() - parent.float()).abs().amax().item())
    report = {
        "new_adapters_exact_zero": model.new_adapters_exact_zero(),
        "layered_minus_parent_max_abs": maximum,
        "passed": bool(model.new_adapters_exact_zero() and maximum == 0.0),
    }
    if not report["passed"]:
        raise RuntimeError(f"zero-init parent preservation failed: {report}")
    return report


def main() -> None:
    args = make_parser().parse_args()
    _validate_args(args)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("full SS Flow training currently requires a CUDA device")
    torch.cuda.set_device(device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    dataset = PairedLayerDataset(
        args.manifest,
        require_labels=args.layer_source == "oracle",
        require_ss_target=True,
    )
    first = dataset[0]
    observed_layers = (
        int(first["layer_depths_gt"].shape[1])
        if "layer_depths_gt" in first else int(args.lari_layers)
    )
    if observed_layers != int(args.lari_layers):
        raise ValueError("dataset/model layer counts differ")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=collate_one,
        pin_memory=True,
        generator=torch.Generator().manual_seed(int(args.seed)),
    )
    sampler, model, model_summary = build_layered_ss_components(
        pretrained=str(args.pretrained),
        layer_channels=int(args.layer_channels),
        lari_hidden_channels=int(args.lari_hidden_channels),
        lari_layers=int(args.lari_layers),
        lari_decoder_blocks=int(args.lari_decoder_blocks),
        gradient_checkpointing=bool(args.gradient_checkpointing),
        device=device,
    )
    parent_binding = load_parent_ss30k(
        model,
        args.parent_ss30k,
        weights=str(args.parent_weights),
        verify_hash=not bool(args.skip_parent_hash),
        freeze_parent=True,
    )
    lari_binding = None
    if args.target_lari_checkpoint:
        lari_binding = load_target_lari_checkpoint(
            model.target_lari, args.target_lari_checkpoint
        )
    # Stage-B SS adapter training treats the independently supervised layer
    # predictor as frozen.  Oracle runs do not execute it at all.
    for parameter in model.target_lari.parameters():
        parameter.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    allowed = ("layer_encoder.", "view_gate.", "layer_block_condition.")
    if not trainable or any(not name.startswith(allowed) for name in trainable_names):
        raise RuntimeError(f"unexpected SS adapter trainable parameters={trainable_names[:8]}")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.95),
    )
    amp_enabled = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp_dtype == "fp16", init_scale=8192.0
    )
    model.eval()
    initial_audit = _zero_init_audit(
        model=model,
        sampler=sampler,
        sample=first,
        device=device,
        layer_source=str(args.layer_source),
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
    )
    model.train()
    model.flow.eval()
    model.aggregator.eval()
    model.block_condition.eval()
    model.target_lari.eval()

    data_identity = {
        **dataset.identity(),
        "layer_source": str(args.layer_source),
        "parent": parent_binding,
        "target_lari": lari_binding,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    history: list[dict] = []
    accumulator: list[dict] = []
    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro_step = 0
    epoch = 0
    started = time.time()
    while step < int(args.max_steps):
        for sample in loader:
            if step >= int(args.max_steps):
                break
            target = sample["target"].to(device=device, dtype=torch.float32)[None]
            condition = sample["stock_condition"].to(device)
            view_indices = _sample_views(sample, args, device)
            t_value = _sample_t(device)
            noise = torch.randn_like(target)
            x_t, gt_velocity = sampler._get_model_gt(target, t_value, noise)
            t = torch.full((1,), 1000.0 * t_value, device=device)
            with torch.no_grad():
                with torch.autocast(
                    device.type, dtype=amp_dtype, enabled=amp_enabled
                ):
                    parent = model.parent_prediction(
                        x_t, t, condition, sample, view_indices=view_indices
                    )
                parent_loss = F.mse_loss(parent.float(), gt_velocity.float())
            sync_step = (micro_step + 1) % int(args.grad_accum) == 0
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                prediction, stats = model.adapted_prediction(
                    x_t,
                    t,
                    condition,
                    sample,
                    view_indices=view_indices,
                    use_layer_branch=True,
                    layer_source=str(args.layer_source),
                )
                flow_loss = F.mse_loss(prediction.float(), gt_velocity.float())
                loss = flow_loss / float(args.grad_accum)
            scaler.scale(loss).backward()
            micro_step += 1
            accumulator.append(
                {
                    "flow_loss": float(flow_loss.detach().item()),
                    "parent_loss": float(parent_loss.item()),
                    "gain": float((parent_loss - flow_loss.detach()).item()),
                    "views": float(len(view_indices)),
                    "target_support": float(stats["target_support_mean"].detach().item()),
                    "hidden_support": float(
                        stats["hidden_target_support_mean"].detach().item()
                    ),
                }
            )
            if not sync_step:
                continue
            scaler.unscale_(optimizer)
            if not all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in trainable
            ):
                raise RuntimeError("layered SS gradients became non-finite")
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip)).item()
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if not all(bool(torch.isfinite(parameter).all()) for parameter in trainable):
                raise RuntimeError("layered SS parameters became non-finite")
            step += 1
            row = {
                "step": step,
                "micro_step": micro_step,
                "flow_loss": float(np.mean([item["flow_loss"] for item in accumulator])),
                "parent_loss": float(np.mean([item["parent_loss"] for item in accumulator])),
                "gain": float(np.mean([item["gain"] for item in accumulator])),
                "view_count": float(np.mean([item["views"] for item in accumulator])),
                "target_support": float(
                    np.mean([item["target_support"] for item in accumulator])
                ),
                "hidden_support": float(
                    np.mean([item["hidden_support"] for item in accumulator])
                ),
                "gradient_norm_before_clip": gradient_norm,
                "elapsed_seconds": float(time.time() - started),
            }
            accumulator.clear()
            history.append(row)
            with (output_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            if step == 1 or step % int(args.log_every) == 0:
                print(f"[layered_ss] {json.dumps(row, ensure_ascii=False)}", flush=True)
            if step % int(args.save_every) == 0 or step == int(args.max_steps):
                for name in (f"step_{step:06d}.pt", "last.pt"):
                    save_layered_ss_checkpoint(
                        output_dir / "checkpoints" / name,
                        model=model,
                        step=step,
                        parent_binding=parent_binding,
                        optimizer=optimizer,
                        data_identity=data_identity,
                        history=history,
                    )
        epoch += 1
    report = {
        "passed": True,
        "model": model_summary,
        "initial_zero_audit": initial_audit,
        "data_identity": data_identity,
        "trainable_names": trainable_names,
        "last": history[-1],
        "checkpoint": str(output_dir / "checkpoints" / "last.pt"),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
