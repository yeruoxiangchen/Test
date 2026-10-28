#!/usr/bin/env python3
"""Sample layered SS and decode 64-cube occupancy coordinates.

This intentionally stops at sparse structure.  The first research stage keeps
the frozen SLat30K and mesh decoder outside the experimental variable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoints import (
    DEFAULT_PARENT_SS30K,
    load_layered_ss_checkpoint,
    load_parent_ss30k,
    load_target_lari_checkpoint,
)
from .native_ss import LayeredSSCFGFlow, build_layered_ss_components
from .paired_dataset import PairedLayerDataset
from .pretrained import load_trellis_model


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--layered_ss_checkpoint", required=True)
    parser.add_argument("--target_lari_checkpoint", default="")
    parser.add_argument("--parent_ss30k", default=str(DEFAULT_PARENT_SS30K))
    parser.add_argument("--parent_weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layer_source", choices=("predicted", "oracle"), default="predicted")
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--lari_layers", type=int, default=5)
    parser.add_argument("--lari_hidden_channels", type=int, default=256)
    parser.add_argument("--lari_decoder_blocks", type=int, default=3)
    parser.add_argument("--allow_layer_source_change", action="store_true")
    parser.add_argument("--layer_channels", type=int, default=128)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg_strength", type=float, default=5.0)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    return parser


@torch.no_grad()
def main() -> None:
    args = make_parser().parse_args()
    if int(args.steps) <= 0 or int(args.max_samples) < 0:
        raise ValueError("steps must be positive and max_samples non-negative")
    if args.layer_source == "predicted" and not args.target_lari_checkpoint:
        raise ValueError("predicted inference requires --target_lari_checkpoint")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("SS sampling requires CUDA")
    torch.cuda.set_device(device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset = PairedLayerDataset(
        args.manifest,
        require_labels=args.layer_source == "oracle",
        require_ss_target=False,
    )
    sampler, model, summary = build_layered_ss_components(
        pretrained=str(args.pretrained),
        layer_channels=int(args.layer_channels),
        lari_hidden_channels=int(args.lari_hidden_channels),
        lari_layers=int(args.lari_layers),
        lari_decoder_blocks=int(args.lari_decoder_blocks),
        device=device,
    )
    parent = load_parent_ss30k(model, args.parent_ss30k, weights=args.parent_weights)
    lari = None
    if args.target_lari_checkpoint:
        lari = load_target_lari_checkpoint(
            model.target_lari, args.target_lari_checkpoint
        )
    candidate = load_layered_ss_checkpoint(
        model,
        args.layered_ss_checkpoint,
        parent_binding=parent,
        target_lari_binding=lari,
        layer_source=str(args.layer_source),
        allow_layer_source_change=bool(args.allow_layer_source_change),
    )
    decoder, decoder_binding = load_trellis_model(
        args.pretrained, "ss_dec_conv3d_16l8_fp16"
    )
    decoder = decoder.to(device).eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    model.eval()
    amp_enabled = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    limit = len(dataset) if int(args.max_samples) <= 0 else min(
        len(dataset), int(args.max_samples)
    )
    records: list[dict] = []
    for index in range(limit):
        sample = dataset[index]
        positive = sample["stock_condition"].to(device)
        negative = torch.zeros_like(positive)
        generator = torch.Generator(device=device).manual_seed(
            int(args.seed) * 1_000_003 + index * 1_009
        )
        initial = torch.randn(
            (1, 8, 16, 16, 16), generator=generator, device=device
        )
        flow = LayeredSSCFGFlow(
            model,
            positive,
            sample,
            layer_source=str(args.layer_source),
        )
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
            latent = sampler.sample(
                flow,
                initial,
                cond=positive,
                neg_cond=negative,
                steps=int(args.steps),
                cfg_strength=float(args.cfg_strength),
                cfg_interval=[0.5, 1.0],
                rescale_t=3.0,
                verbose=False,
            ).samples
            logits = decoder(latent.to(dtype=next(decoder.parameters()).dtype)).float()
        if not bool(torch.isfinite(latent).all()) or not bool(torch.isfinite(logits).all()):
            raise RuntimeError("SS sampling/decoding produced non-finite values")
        coords = (
            torch.argwhere(logits > 0)[:, [0, 2, 3, 4]]
            .cpu()
            .numpy()
            .astype(np.int32)
        )
        uid = str(sample["uid"])
        destination = output_dir / f"{index:05d}.npz"
        np.savez_compressed(
            destination,
            latent=latent.float().cpu().numpy(),
            coords=coords,
            uid=np.asarray(uid),
        )
        records.append(
            {
                "index": index,
                "uid": uid,
                "output": str(destination),
                "active_coordinates": int(len(coords)),
                "finite_latent_and_logits": True,
                "cfg_positive_calls": flow.positive_calls,
                "cfg_negative_calls": flow.negative_calls,
            }
        )
    report = {
        "passed": True,
        "scope": "SS latent and decoded occupancy only",
        "model": summary,
        "parent": parent,
        "target_lari": lari,
        "layered_ss": candidate,
        "layer_source": str(args.layer_source),
        "records": records,
        "args": vars(args),
        "decoder_content_identity": decoder_binding,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
