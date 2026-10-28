#!/usr/bin/env python3
"""Create a deterministic paired-cache fixture for smoke testing only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .contracts import LAYERED_SAMPLE_FORMAT, validate_paired_sample
from .paired_dataset import sha256_file, write_manifest


def make_synthetic_paired_sample(
    *,
    channels: int = 1024,
    patch_side: int = 37,
    image_side: int = 518,
    layers: int = 5,
    views: int = 2,
    include_ss_target: bool = True,
) -> dict:
    """Return a schema-valid fixture; it is not meaningful training data."""

    if min(channels, patch_side, image_side, layers) <= 0:
        raise ValueError("synthetic dimensions must be positive")
    if int(views) < 2:
        raise ValueError("synthetic fixture requires at least two views")
    generator = torch.Generator().manual_seed(20260909)
    c2w = torch.eye(4).repeat(views, 1, 1)
    c2w[:, 2, 3] = -2.0
    c2w[1, 0, 3] = 0.1
    focal = 0.8 * float(image_side)
    principal = (float(image_side) - 1.0) / 2.0
    intrinsics = torch.tensor(
        ((focal, 0.0, principal), (0.0, focal, principal), (0.0, 0.0, 1.0))
    ).repeat(views, 1, 1)
    visible = torch.zeros((views, image_side, image_side), dtype=torch.float32)
    lo, hi = image_side // 4, 3 * image_side // 4
    visible[:, lo:hi, lo:hi] = 1.0
    patch_count = patch_side * patch_side
    masked = torch.randn((views, patch_count, channels), generator=generator)
    # Correlated target evidence plus a deterministic raw-only occluder signal.
    raw = masked.clone()
    raw[0, : patch_count // 3, : min(32, channels)] += 1.5
    raw[1, patch_count // 2 :, : min(32, channels)] -= 1.0
    depth_values = torch.linspace(1.25, 2.75, layers)
    depths = depth_values[None, :, None, None].expand(
        views, layers, patch_side, patch_side
    ).clone()
    stopping = torch.full(
        (views, patch_side, patch_side), layers, dtype=torch.long
    )
    ownership = torch.zeros_like(depths)
    ownership[:, min(2, layers - 1)] = 1.0
    sample = {
        "format": LAYERED_SAMPLE_FORMAT,
        "uid": "synthetic-0000",
        "object_uid": "synthetic-object-0000",
        "masked_visual_patch_features": masked,
        "raw_visual_patch_features": raw,
        "visible_target_mask": visible,
        "intrinsics": intrinsics,
        "extrinsics": c2w,
        "image_size": (image_side, image_side),
        "grid_transform": "identity",
        "extrinsics_type": "c2w",
        "camera_forward_sign": 1.0,
        "layer_depths_gt": depths,
        "ray_stop_index_gt": stopping,
        "layer_target_gt": ownership,
        "synthetic_fixture_only": True,
    }
    if include_ss_target:
        sample["target"] = torch.randn(
            (8, 16, 16, 16), generator=generator
        )
    validate_paired_sample(
        sample,
        require_labels=True,
        require_ss_target=include_ss_target,
        expected_layers=layers,
    )
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--channels", type=int, default=1024)
    parser.add_argument("--patch_side", type=int, default=37)
    parser.add_argument("--image_side", type=int, default=518)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--without_ss_target", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    sample = make_synthetic_paired_sample(
        channels=int(args.channels),
        patch_side=int(args.patch_side),
        image_side=int(args.image_side),
        layers=int(args.layers),
        views=int(args.views),
        include_ss_target=not bool(args.without_ss_target),
    )
    sample_path = output_dir / "synthetic-0000.pt"
    torch.save(sample, sample_path)
    manifest = write_manifest(
        output_dir / "manifest.json",
        rows=(
            {
                "uid": sample["uid"],
                "object_uid": sample["object_uid"],
                "sample": sample_path.name,
                "sha256": sha256_file(sample_path),
            },
        ),
        config={
            "synthetic_fixture_only": True,
            "channels": int(args.channels),
            "patch_side": int(args.patch_side),
            "image_side": int(args.image_side),
            "layers": int(args.layers),
            "views": int(args.views),
        },
        root=output_dir,
    )
    print(
        json.dumps(
            {
                "passed": True,
                "manifest": str(output_dir / "manifest.json"),
                "manifest_identity": manifest["manifest_identity"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
