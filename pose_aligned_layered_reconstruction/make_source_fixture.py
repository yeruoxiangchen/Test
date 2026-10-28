#!/usr/bin/env python3
"""Create two analytic box-scene views for the image-to-cache smoke test.

Ray-box intersections provide consistent visible masks and target/non-target
depth peeling.  The optional SS latent is random and ONLY tests trainer wiring;
it must never be used as a reconstruction-quality benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .prepare_paired_cache import SOURCE_MANIFEST_FORMAT


def ray_box(origin, directions, lower, upper):
    safe = np.where(np.abs(directions) < 1e-8, 1e-8, directions)
    t0 = (np.asarray(lower) - origin) / safe
    t1 = (np.asarray(upper) - origin) / safe
    entry = np.minimum(t0, t1).max(axis=-1)
    leave = np.maximum(t0, t1).min(axis=-1)
    valid = (leave > entry) & (entry > 0)
    return np.stack((np.where(valid, entry, 0), np.where(valid, leave, 0))).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--side", type=int, default=128)
    args = parser.parse_args()
    side = int(args.side)
    if side < 16:
        raise ValueError("fixture side must be >=16")
    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    ys, xs = np.mgrid[:side, :side].astype(np.float32)
    focal, center = side * 0.95, (side - 1) / 2
    directions = np.stack(((xs - center) / focal, (ys - center) / focal, np.ones_like(xs)), axis=-1)
    k = np.asarray(((focal, 0, center), (0, focal, center), (0, 0, 1)), dtype=np.float32)
    ks, poses, target_layers, other_layers, raw_paths, mask_paths = [], [], [], [], [], []
    visibility = []
    for index, camera_x in enumerate((-0.1, 0.12)):
        origin = np.asarray((camera_x, 0, -1.3), dtype=np.float32)
        target = ray_box(origin, directions, (-0.3, -0.3, -0.22), (0.3, 0.3, 0.22))
        other = ray_box(origin, directions, (-0.03, -0.45, -0.60), (0.20, 0.14, -0.46))
        target_exists = target[0] > 0
        other_exists = other[0] > 0
        visible = target_exists & (~other_exists | (target[0] < other[0]))
        occluder_visible = other_exists & (~target_exists | (other[0] < target[0]))
        rgb = np.full((side, side, 3), 205, dtype=np.uint8)
        rgb[visible] = (175, 65, 40)
        rgb[occluder_visible] = (35, 130, 70)
        # Fixed raster texture makes the paired DINO difference non-degenerate.
        rgb[visible & ((xs.astype(int) + ys.astype(int)) % 12 < 3)] = (225, 160, 85)
        raw_name, mask_name = f"raw_{index}.png", f"mask_{index}.png"
        Image.fromarray(rgb).save(root / raw_name)
        Image.fromarray(visible.astype(np.uint8) * 255).save(root / mask_name)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = origin
        poses.append(pose)
        ks.append(k)
        target_layers.append(target)
        other_layers.append(other)
        raw_paths.append(raw_name)
        mask_paths.append(mask_name)
        visibility.append(float(visible.sum() / max(1, target_exists.sum())))
    np.savez_compressed(root / "cameras.npz", intrinsics=np.stack(ks), extrinsics=np.stack(poses))
    np.savez_compressed(root / "layers.npz", target_depths=np.stack(target_layers), non_target_depths=np.stack(other_layers))
    random_target = np.random.default_rng(20260909).standard_normal((8, 16, 16, 16)).astype(np.float32)
    np.savez_compressed(root / "random_ss_target_for_smoke_only.npz", target=random_target)
    row = {
        "uid": "box_scene_smoke",
        "object_uid": "box_smoke_only",
        "raw_images": raw_paths,
        "visible_masks": mask_paths,
        "camera": "cameras.npz",
        "tracking_boxes_xyxy": [[side // 5, side // 5, 4 * side // 5, 4 * side // 5]] * 2,
        "extrinsics_type": "c2w",
        "camera_forward_sign": 1.0,
        "grid_transform": "identity",
    }
    for name, labels in (("source_train.json", True), ("source_inference.json", False)):
        current = dict(row)
        if labels:
            current.update(layers="layers.npz", ss_target="random_ss_target_for_smoke_only.npz")
        payload = {
            "format": SOURCE_MANIFEST_FORMAT,
            "root": str(root),
            "synthetic_fixture_only": True,
            "ss_target_is_random_not_geometry_ground_truth": True,
            "samples": [current],
        }
        (root / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "output_dir": str(root), "visible_target_ratios": visibility}), flush=True)


if __name__ == "__main__":
    main()
