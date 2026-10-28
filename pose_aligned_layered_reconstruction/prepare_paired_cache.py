#!/usr/bin/env python3
"""Build content-bound paired DINO and target-aware layered-ray caches.

The source manifest deliberately keeps raw RGB and the visible target mask as
two paths for every view.  Both are transformed by one crop affine.  Raw RGB
retains occluders; the masked copy is derived here and is the only image source
used to construct the inherited TRELLIS condition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contracts import LAYERED_SAMPLE_FORMAT, validate_paired_sample
from .depth_peeling import merge_target_and_occluder_layers
from .paired_dataset import sha256_file, write_manifest
from .paired_encoder import build_frozen_dino_encoder, encode_paired_views
from .paired_preprocessing import prepare_paired_paths


SOURCE_MANIFEST_FORMAT = (
    "pose_aligned_layered_reconstruction.source_manifest.v1"
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--patch_side", type=int, default=37)
    parser.add_argument("--crop_margin", type=float, default=1.35)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--dino_model", default="dinov2_vitl14_reg")
    parser.add_argument("--max_samples", type=int, default=0)
    return parser


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve(strict=True) if path.is_absolute() else (root / path).resolve(strict=True)


def _load_npz(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"{path} lacks arrays {missing}")
        return {key: np.asarray(payload[key]) for key in required}


def _load_ss_target(path: Path) -> torch.Tensor:
    if path.suffix.lower() == ".npz":
        value = _load_npz(path, ("target",))["target"]
        target = torch.from_numpy(np.asarray(value))
    elif path.suffix.lower() == ".npy":
        target = torch.from_numpy(np.load(path, allow_pickle=False))
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        target = payload["target"] if isinstance(payload, dict) else payload
    if not torch.is_tensor(target) or target.shape != (8, 16, 16, 16):
        raise ValueError(f"SS target must be [8,16,16,16]: {path}")
    target = target.float().contiguous()
    if not bool(torch.isfinite(target).all().item()):
        raise ValueError(f"SS target contains non-finite values: {path}")
    return target


def _validate_source(payload: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    if payload.get("format") != SOURCE_MANIFEST_FORMAT:
        raise ValueError(f"unexpected source manifest format={payload.get('format')!r}")
    root = Path(str(payload.get("root", "."))).expanduser().resolve(strict=True)
    rows = payload.get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source manifest must contain a non-empty samples list")
    uids = [str(row.get("uid", "")) for row in rows if isinstance(row, dict)]
    if len(uids) != len(rows) or not all(uids) or len(set(uids)) != len(uids):
        raise ValueError("source sample UIDs must be non-empty and unique")
    return root, [dict(row) for row in rows]


def _path_list(root: Path, row: dict[str, Any], key: str) -> list[Path]:
    values = row.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"source row requires non-empty {key}")
    return [_resolve(root, str(value)) for value in values]


def _source_bindings(paths: list[Path]) -> list[dict[str, str]]:
    return [{"path": str(path), "sha256": sha256_file(path)} for path in paths]


@torch.no_grad()
def main() -> None:
    args = make_parser().parse_args()
    if min(int(args.resolution), int(args.patch_side), int(args.layers)) <= 0:
        raise ValueError("resolution, patch_side, and layers must be positive")
    if (int(args.resolution), int(args.patch_side)) != (518, 37):
        raise ValueError("the SS30K DINO cache requires resolution=518, patch_side=37")
    if str(args.dino_model) != "dinov2_vitl14_reg":
        raise ValueError("the SS30K encoder contract requires dinov2_vitl14_reg")
    if float(args.crop_margin) < 1.0:
        raise ValueError("crop_margin must be >= 1")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    source_path = Path(args.source_manifest).expanduser().resolve(strict=True)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    source_root, rows = _validate_source(payload)
    if int(args.max_samples) > 0:
        rows = rows[: int(args.max_samples)]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    sample_dir = output_dir / "samples"
    sample_dir.mkdir()

    encoder = build_frozen_dino_encoder(
        dino_model=str(args.dino_model), device=device
    )
    output_rows: list[dict[str, Any]] = []
    for sample_index, row in enumerate(rows):
        uid = str(row["uid"])
        object_uid = str(row.get("object_uid", uid))
        raw_paths = _path_list(source_root, row, "raw_images")
        mask_paths = _path_list(source_root, row, "visible_masks")
        if len(raw_paths) != len(mask_paths):
            raise ValueError(f"raw/mask view count differs for {uid}")
        camera_path = _resolve(source_root, str(row.get("camera", "")))
        camera = _load_npz(camera_path, ("intrinsics", "extrinsics"))
        tracking_boxes = row.get("tracking_boxes_xyxy")

        prepared = prepare_paired_paths(
            raw_paths,
            mask_paths,
            resolution=int(args.resolution),
            crop_margin=float(args.crop_margin),
            tracking_boxes_xyxy=tracking_boxes,
        )
        intrinsics = prepared.transform_intrinsics(camera["intrinsics"])
        extrinsics = np.asarray(camera["extrinsics"], dtype=np.float32)
        views = len(raw_paths)
        if extrinsics.shape != (views, 4, 4):
            raise ValueError(f"camera extrinsics/view count differs for {uid}")
        encoded = encode_paired_views(
            encoder,
            raw_images=prepared.raw_images,
            masked_images=prepared.masked_images,
        )
        sample: dict[str, Any] = {
            "format": LAYERED_SAMPLE_FORMAT,
            "uid": uid,
            "object_uid": object_uid,
            "synthetic_fixture_only": bool(payload.get("synthetic_fixture_only", False)),
            "raw_visual_patch_features": encoded[
                "raw_visual_patch_features"
            ].half().cpu(),
            "masked_visual_patch_features": encoded[
                "masked_visual_patch_features"
            ].half().cpu(),
            "visible_target_mask": torch.from_numpy(
                prepared.visible_masks
            ).half(),
            "intrinsics": torch.from_numpy(intrinsics).float(),
            "extrinsics": torch.from_numpy(extrinsics).float(),
            "image_size": (int(args.resolution), int(args.resolution)),
            "grid_transform": str(row.get("grid_transform", "identity")),
            "extrinsics_type": str(row.get("extrinsics_type", "c2w")),
            "camera_forward_sign": float(row.get("camera_forward_sign", 1.0)),
            "paired_encoding_contract": encoded["paired_encoding_contract"],
            "preprocessing": {
                "resolution": int(args.resolution),
                "patch_side": int(args.patch_side),
                "crop_margin": float(args.crop_margin),
                "crop_boxes_xyxy": [list(box) for box in prepared.crop_boxes_xyxy],
                "source_sizes_wh": [list(size) for size in prepared.source_sizes_wh],
                "intrinsics_rule": "K_feature=source_to_feature_affine@K_source",
            },
            "source_binding": {
                "raw_images": _source_bindings(raw_paths),
                "visible_masks": _source_bindings(mask_paths),
                "camera": {"path": str(camera_path), "sha256": sha256_file(camera_path)},
            },
        }
        # Real inference requires no ground-truth layers or hidden target
        # image.  Only supervised/oracle datasets supply this optional file.
        if row.get("layers"):
            layer_path = _resolve(source_root, str(row["layers"]))
            peeled = _load_npz(layer_path, ("target_depths", "non_target_depths"))
            target_depths = prepared.transform_view_layer_stack(
                peeled["target_depths"], output_resolution=int(args.patch_side)
            )
            non_target_depths = prepared.transform_view_layer_stack(
                peeled["non_target_depths"], output_resolution=int(args.patch_side)
            )
            layer_depths, stopping, ownership = merge_target_and_occluder_layers(
                torch.from_numpy(target_depths),
                torch.from_numpy(non_target_depths),
                output_layers=int(args.layers),
            )
            sample.update({
                "layer_depths_gt": layer_depths.float(),
                "ray_stop_index_gt": stopping.long(),
                "layer_target_gt": ownership.float(),
            })
            sample["source_binding"]["layers"] = {
                "path": str(layer_path), "sha256": sha256_file(layer_path)
            }
        ss_target_value = row.get("ss_target")
        if ss_target_value:
            target_path = _resolve(source_root, str(ss_target_value))
            sample["target"] = _load_ss_target(target_path)
            sample["source_binding"]["ss_target"] = {
                "path": str(target_path),
                "sha256": sha256_file(target_path),
            }
        validate_paired_sample(
            sample,
            require_labels=bool(row.get("layers")),
            require_ss_target=bool(ss_target_value),
            expected_layers=int(args.layers),
        )
        destination = sample_dir / f"{sample_index:06d}.pt"
        torch.save(sample, destination)
        output_rows.append(
            {
                "uid": uid,
                "object_uid": object_uid,
                "sample": str(destination.relative_to(output_dir)),
                "sha256": sha256_file(destination),
                "view_count": views,
                "has_ss_target": bool(ss_target_value),
                "has_layer_labels": bool(row.get("layers")),
            }
        )
        print(
            json.dumps(
                {"index": sample_index, "uid": uid, "views": views},
                ensure_ascii=False,
            ),
            flush=True,
        )

    config = {
        "synthetic_fixture_only": bool(payload.get("synthetic_fixture_only", False)),
        "ss_target_is_random_not_geometry_ground_truth": bool(
            payload.get("ss_target_is_random_not_geometry_ground_truth", False)
        ),
        "source_manifest": str(source_path),
        "source_manifest_sha256": sha256_file(source_path),
        "paired_versions": {
            "raw": "occluders_and_scene_context_retained",
            "masked": "visible_target_only_derived_after_shared_crop",
        },
        "resolution": int(args.resolution),
        "patch_side": int(args.patch_side),
        "crop_margin": float(args.crop_margin),
        "layers": int(args.layers),
        "layer_semantics": "positive_camera_depth_sorted_with_target_ownership",
        "dino_model": str(args.dino_model),
    }
    manifest = write_manifest(
        output_dir / "manifest.json",
        rows=output_rows,
        config=config,
        root=output_dir,
    )
    print(
        json.dumps(
            {
                "passed": True,
                "manifest": str(output_dir / "manifest.json"),
                "manifest_identity": manifest["manifest_identity"],
                "samples": len(output_rows),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
