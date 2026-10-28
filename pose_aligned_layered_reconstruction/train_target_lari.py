#!/usr/bin/env python3
"""Supervise Target-Conditioned LaRI before coupling it to SS Flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoints import save_target_lari_checkpoint
from .paired_dataset import PairedLayerDataset, collate_one
from .target_lari import TargetConditionedLaRI, target_lari_loss


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument("--save_every", type=int, default=1_000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--decoder_blocks", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        "max_steps",
        "save_every",
        "log_every",
        "layers",
        "hidden_channels",
        "decoder_blocks",
        "learning_rate",
        "grad_clip",
    )
    bad = [name for name in positive if float(getattr(args, name)) <= 0]
    if bad:
        raise ValueError(f"arguments must be positive: {bad}")
    if int(args.num_workers) < 0 or float(args.weight_decay) < 0:
        raise ValueError("num_workers/weight_decay must be non-negative")


def _model_forward(
    model: TargetConditionedLaRI,
    sample: dict,
    device: torch.device,
):
    intrinsics = sample["intrinsics"].to(device).float().clone()
    intrinsics[:, 1, 1] *= float(sample.get("layer_camera_y_sign", 1.0))
    return model(
        raw_patch_features=sample["raw_visual_patch_features"].to(device).float(),
        masked_patch_features=sample["masked_visual_patch_features"].to(device).float(),
        visible_target_mask=sample["visible_target_mask"].to(device).float(),
        intrinsics=intrinsics,
        extrinsics=sample["extrinsics"].to(device).float(),
        extrinsics_type=str(sample["extrinsics_type"]),
        camera_forward_sign=float(sample["camera_forward_sign"]),
        image_size=tuple(int(value) for value in sample["image_size"]),
    )


def main() -> None:
    args = make_parser().parse_args()
    _validate_args(args)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset = PairedLayerDataset(
        args.manifest,
        require_labels=True,
        require_ss_target=False,
        build_trellis_contexts=False,
    )
    first = dataset[0]
    input_channels = int(first["raw_visual_patch_features"].shape[-1])
    observed_layers = int(first["layer_depths_gt"].shape[1])
    if observed_layers != int(args.layers):
        raise ValueError(
            f"dataset/model layer count differs: {observed_layers} != {args.layers}"
        )
    model = TargetConditionedLaRI(
        input_channels=input_channels,
        hidden_channels=int(args.hidden_channels),
        layers=int(args.layers),
        decoder_blocks=int(args.decoder_blocks),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.95),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=collate_one,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(int(args.seed)),
    )
    data_identity = dataset.identity()
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    history: list[dict] = []
    step = 0
    epoch = 0
    started = time.time()
    save_target_lari_checkpoint(
        output_dir / "checkpoints" / "step_000000.pt", model=model, step=0,
        optimizer=optimizer, data_identity=data_identity, history=history,
    )
    model.train()
    while step < int(args.max_steps):
        for sample in loader:
            if step >= int(args.max_steps):
                break
            optimizer.zero_grad(set_to_none=True)
            prediction = _model_forward(model, sample, device)
            loss, metrics = target_lari_loss(prediction, sample)
            loss.backward()
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            if not all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in trainable
            ):
                raise RuntimeError("Target-LaRI gradients became non-finite")
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip)).item()
            )
            optimizer.step()
            if not all(bool(torch.isfinite(parameter).all()) for parameter in trainable):
                raise RuntimeError("Target-LaRI parameters became non-finite")
            step += 1
            row = {
                "step": step,
                **{key: float(value.item()) for key, value in metrics.items()},
                "gradient_norm_before_clip": gradient_norm,
                "elapsed_seconds": float(time.time() - started),
            }
            history.append(row)
            with (output_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            if step == 1 or step % int(args.log_every) == 0:
                print(f"[target_lari] {json.dumps(row, ensure_ascii=False)}", flush=True)
            if step % int(args.save_every) == 0 or step == int(args.max_steps):
                save_target_lari_checkpoint(
                    output_dir / "checkpoints" / f"step_{step:06d}.pt",
                    model=model,
                    step=step,
                    optimizer=optimizer,
                    data_identity=data_identity,
                    history=history,
                )
                save_target_lari_checkpoint(
                    output_dir / "checkpoints" / "last.pt",
                    model=model,
                    step=step,
                    optimizer=optimizer,
                    data_identity=data_identity,
                    history=history,
                )
        epoch += 1
    report = {
        "passed": True,
        "step": step,
        "epoch": epoch,
        "model": model.metadata(),
        "data_identity": data_identity,
        "last": history[-1],
        "checkpoint": str(output_dir / "checkpoints" / "last.pt"),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
