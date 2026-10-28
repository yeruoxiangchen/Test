"""Encode posed multiview images into a strict TRELLIS.2 training manifest."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from .constants import (
    CACHE_FORMAT,
    DINO_FEATURE_DIM,
    DINO_FULL_TOKENS,
    DINO_IMAGE_SIZE,
    DINO_MODEL,
    SAMPLE_FORMAT,
)
from .data import (
    PoseTrellis2Dataset,
    load_shape_target,
    load_ss_target,
    sha256_file,
)
from .runtime import load_dinov3


RAW_SPEC_FORMAT = "pose_constrained_trellis2.raw_views.v1"
PREPROCESS_FORMAT = "pose_constrained_trellis2.mask_square_512.v1"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="raw-view JSON specification")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dino-model", default=DINO_MODEL)
    parser.add_argument("--crop-margin", type=float, default=0.10)
    parser.add_argument("--mask-threshold", type=float, default=0.80)
    parser.add_argument("--max-views", type=int, default=0, help="0 retains every view")
    parser.add_argument("--max-samples", type=int, default=0, help="0 processes every sample")
    parser.add_argument("--ss-target-dir", default="")
    parser.add_argument("--shape-target-dir", default="")
    parser.add_argument("--ss-target-template", default="{uid}.npz")
    parser.add_argument("--shape-target-template", default="{uid}.npz")
    parser.add_argument("--skip-hashes", action="store_true")
    return parser


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _read_spec(path: str | Path) -> tuple[dict[str, Any], Path]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("format") != RAW_SPEC_FORMAT:
        raise ValueError(f"unsupported raw-view spec={payload.get('format')!r}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("raw-view spec contains no samples")
    root = _resolve(source.parent, payload.get("root", "."))
    return payload, root


def _as_camera_tensor(
    value: Any, *, views: int, shape: tuple[int, int], label: str
) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float32)
    expected = (int(views),) + tuple(shape)
    if tuple(result.shape) != expected or not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must be finite {expected}, got {tuple(result.shape)}")
    return result


def _mask_from_file(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    if tuple(mask.shape) != shape:
        raise ValueError(f"mask {path} shape={mask.shape} differs from image={shape}")
    return mask


def _prepare_view(
    image_path: Path,
    mask_path: Path | None,
    intrinsic: torch.Tensor,
    *,
    margin: float,
    threshold: float,
) -> tuple[Image.Image, torch.Tensor, dict[str, Any]]:
    source = Image.open(image_path).convert("RGBA")
    rgba = np.asarray(source, dtype=np.uint8)
    mask = None
    if mask_path is not None:
        mask = _mask_from_file(mask_path, tuple(rgba.shape[:2]))
    return prepare_rgba_view(
        rgba,
        intrinsic,
        mask=mask,
        margin=margin,
        threshold=threshold,
    )


def prepare_rgba_view(
    rgba: np.ndarray,
    intrinsic: torch.Tensor,
    *,
    mask: np.ndarray | None = None,
    margin: float,
    threshold: float,
    require_foreground: bool = False,
) -> tuple[Image.Image, torch.Tensor, dict[str, Any]]:
    """Apply the shared object crop to an in-memory RGBA view and its K.

    The tar-backed ProObjaverse cache uses this public in-memory entry point;
    the generic file-backed builder above deliberately goes through the same
    implementation so their RGB/K geometry cannot drift.
    """

    rgba = np.asarray(rgba, dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"RGBA input must be [H,W,4], got {rgba.shape}")
    height, width = map(int, rgba.shape[:2])
    alpha = rgba[..., 3].astype(np.float32) / 255.0
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float32)
        if tuple(mask.shape) != (height, width):
            raise ValueError(
                f"mask shape={mask.shape} differs from image={(height, width)}"
            )
        if float(mask.max(initial=0.0)) > 1.0:
            mask = mask / 255.0
        alpha = alpha * np.clip(mask, 0.0, 1.0)
    foreground = np.argwhere(alpha > float(threshold))
    if foreground.size:
        y0, x0 = foreground.min(axis=0)
        y1, x1 = foreground.max(axis=0) + 1
        object_width = max(1, int(x1 - x0))
        object_height = max(1, int(y1 - y0))
        side = max(
            2,
            int(
                math.ceil(
                    max(object_width, object_height) * (1.0 + margin)
                )
            ),
        )
        center_x = 0.5 * float(x0 + x1)
        center_y = 0.5 * float(y0 + y1)
    else:
        if require_foreground:
            raise ValueError("RGBA/mask contains no foreground above threshold")
        side = max(width, height)
        center_x = 0.5 * float(width)
        center_y = 0.5 * float(height)
    left = int(math.floor(center_x - 0.5 * side))
    top = int(math.floor(center_y - 0.5 * side))
    combined = rgba.copy()
    combined[..., 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    cropped = Image.fromarray(combined, mode="RGBA").crop(
        (left, top, left + side, top + side)
    )
    resized = cropped.resize(
        (DINO_IMAGE_SIZE, DINO_IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    resized_array = np.asarray(resized, dtype=np.uint8)
    masked_rgb = resized_array[..., :3] * (
        resized_array[..., 3:4] > float(threshold) * 255.0
    ).astype(np.uint8)
    prepared = Image.fromarray(masked_rgb, mode="RGB")
    scale = float(DINO_IMAGE_SIZE) / float(side)
    affine = intrinsic.new_tensor(
        ((scale, 0.0, -scale * left), (0.0, scale, -scale * top), (0.0, 0.0, 1.0))
    )
    transformed = affine @ intrinsic.float()
    metadata = {
        "source_size": [height, width],
        "crop_xywh": [left, top, side, side],
        "source_to_feature": affine.tolist(),
        "foreground_fraction": float((alpha > float(threshold)).mean()),
    }
    return prepared, transformed, metadata


def _target_path(
    row: dict[str, Any],
    field: str,
    *,
    uid: str,
    source_root: Path,
    target_dir: str,
    template: str,
) -> Path | None:
    if row.get(field):
        path = _resolve(source_root, row[field])
    elif target_dir:
        relative = template.format(uid=uid)
        path = _resolve(Path(target_dir).expanduser().resolve(), relative)
    else:
        return None
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _safe_sample_name(index: int, uid: str) -> str:
    import hashlib

    suffix = hashlib.sha256(uid.encode("utf-8")).hexdigest()[:16]
    return f"{index:06d}_{suffix}.pt"


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 <= float(args.crop_margin) <= 1.0:
        raise ValueError("crop-margin must be in [0,1]")
    if not 0.0 <= float(args.mask_threshold) <= 1.0:
        raise ValueError("mask-threshold must be in [0,1]")
    if int(args.max_views) < 0 or int(args.max_samples) < 0:
        raise ValueError("max-views/max-samples must be non-negative")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("DINOv3 cache preparation requires CUDA")
    if device.index is not None:
        torch.cuda.set_device(device.index)
    specification, source_root = _read_spec(args.spec)
    rows = list(specification["samples"])
    if int(args.max_samples) > 0:
        rows = rows[: int(args.max_samples)]
    output = Path(args.output_dir).expanduser().resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing cache manifest: {manifest_path}"
        )
    output.mkdir(parents=True, exist_ok=True)
    extractor = load_dinov3(device=device, model_name=str(args.dino_model))
    manifest_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"sample {index} is not an object")
        uid = str(row.get("uid", ""))
        if not uid or uid in seen:
            raise ValueError(f"empty or duplicate uid={uid!r}")
        seen.add(uid)
        images_raw = row.get("images")
        if not isinstance(images_raw, list) or not images_raw:
            raise ValueError(f"uid={uid} has no images")
        images = [_resolve(source_root, value) for value in images_raw]
        if not all(path.is_file() for path in images):
            missing = [str(path) for path in images if not path.is_file()]
            raise FileNotFoundError(f"uid={uid} missing images={missing[:4]}")
        masks_raw = row.get("masks")
        if masks_raw is None:
            masks: list[Path | None] = [None] * len(images)
        elif isinstance(masks_raw, list) and len(masks_raw) == len(images):
            masks = [_resolve(source_root, value) if value else None for value in masks_raw]
        else:
            raise ValueError(f"uid={uid} masks must be absent or match images")
        if any(path is not None and not path.is_file() for path in masks):
            raise FileNotFoundError(f"uid={uid} contains a missing mask")
        view_count = len(images)
        intrinsics = _as_camera_tensor(
            row.get("intrinsics"), views=view_count, shape=(3, 3), label="intrinsics"
        )
        extrinsics = _as_camera_tensor(
            row.get("extrinsics"), views=view_count, shape=(4, 4), label="extrinsics"
        )
        extrinsics_type = str(row.get("extrinsics_type", "w2c"))
        camera_forward_sign = float(row.get("camera_forward_sign", 1.0))
        camera_image_y_sign = float(row.get("camera_image_y_sign", 1.0))
        grid_transform = str(row.get("grid_transform", "identity"))
        if extrinsics_type not in ("w2c", "c2w"):
            raise ValueError(f"uid={uid} extrinsics_type must be w2c or c2w")
        if camera_forward_sign not in (-1.0, 1.0):
            raise ValueError(f"uid={uid} camera_forward_sign must be -1 or +1")
        if camera_image_y_sign not in (-1.0, 1.0):
            raise ValueError(f"uid={uid} camera_image_y_sign must be -1 or +1")
        if grid_transform not in ("identity", "pixal3d_rotation"):
            raise ValueError(f"uid={uid} unsupported grid_transform={grid_transform!r}")
        if bool((intrinsics[:, (0, 1), (0, 1)] <= 0).any()):
            raise ValueError(f"uid={uid} camera focal lengths must be positive")
        determinants = torch.linalg.det(extrinsics[:, :3, :3])
        if bool((determinants.abs() < 1.0e-8).any()):
            raise ValueError(f"uid={uid} camera rotations are singular")
        if int(args.max_views) > 0:
            retained = min(view_count, int(args.max_views))
            images, masks = images[:retained], masks[:retained]
            intrinsics, extrinsics = intrinsics[:retained], extrinsics[:retained]
            view_count = retained
        prepared: list[Image.Image] = []
        transformed_intrinsics: list[torch.Tensor] = []
        view_metadata: list[dict[str, Any]] = []
        for image_path, mask_path, intrinsic in zip(images, masks, intrinsics):
            image, transformed, metadata = _prepare_view(
                image_path,
                mask_path,
                intrinsic,
                margin=float(args.crop_margin),
                threshold=float(args.mask_threshold),
            )
            prepared.append(image)
            transformed_intrinsics.append(transformed)
            view_metadata.append(metadata)
        features = extractor(prepared).detach().to(dtype=torch.float16, device="cpu")
        expected = (view_count, DINO_FULL_TOKENS, DINO_FEATURE_DIM)
        if tuple(features.shape) != expected or not bool(torch.isfinite(features).all()):
            raise RuntimeError(f"uid={uid} DINOv3 output={tuple(features.shape)}, expected={expected}")
        pose_payload: dict[str, Any] = {
            "format": SAMPLE_FORMAT,
            "uid": uid,
            "dino_features": features.contiguous(),
            "intrinsics": torch.stack(transformed_intrinsics).float(),
            "extrinsics": extrinsics.float(),
            "extrinsics_type": extrinsics_type,
            "camera_forward_sign": camera_forward_sign,
            "camera_image_y_sign": camera_image_y_sign,
            "grid_transform": grid_transform,
            "image_size": [DINO_IMAGE_SIZE, DINO_IMAGE_SIZE],
            "source_images": [str(path) for path in images],
            "view_preprocessing": view_metadata,
            "preprocessing": {
                "format": PREPROCESS_FORMAT,
                "crop_margin": float(args.crop_margin),
                "mask_threshold": float(args.mask_threshold),
                "background": "black_after_resized_alpha_threshold",
                "rgb_and_mask_resampling": "bilinear",
                "intrinsics_rule": "K_feature=source_to_feature_affine@K_source",
            },
        }
        if row.get("object_to_world") is not None:
            object_to_world = torch.as_tensor(row["object_to_world"], dtype=torch.float32)
            if tuple(object_to_world.shape) != (4, 4) or not bool(torch.isfinite(object_to_world).all()):
                raise ValueError(f"uid={uid} object_to_world must be finite [4,4]")
            pose_payload["object_to_world"] = object_to_world
        relative_pose = Path("samples") / _safe_sample_name(index, uid)
        pose_path = output / relative_pose
        _atomic_torch_save(pose_payload, pose_path)
        manifest_row: dict[str, Any] = {
            "uid": uid,
            "pose_file": str(relative_pose),
            "view_count": view_count,
        }
        ss_target = _target_path(
            row,
            "ss_target_file",
            uid=uid,
            source_root=source_root,
            target_dir=str(args.ss_target_dir),
            template=str(args.ss_target_template),
        )
        shape_target = _target_path(
            row,
            "shape_target_file",
            uid=uid,
            source_root=source_root,
            target_dir=str(args.shape_target_dir),
            template=str(args.shape_target_template),
        )
        if ss_target is not None:
            load_ss_target(ss_target, uid)
            manifest_row["ss_target_file"] = str(ss_target)
        if shape_target is not None:
            load_shape_target(shape_target, uid)
            manifest_row["shape_target_file"] = str(shape_target)
        if not args.skip_hashes:
            manifest_row["pose_file_sha256"] = sha256_file(pose_path)
            if ss_target is not None:
                manifest_row["ss_target_file_sha256"] = sha256_file(ss_target)
            if shape_target is not None:
                manifest_row["shape_target_file_sha256"] = sha256_file(shape_target)
        manifest_rows.append(manifest_row)
        print(json.dumps({"cached": index + 1, "uid": uid, "views": view_count}), flush=True)
    del extractor
    torch.cuda.empty_cache()
    manifest = {
        "format": CACHE_FORMAT,
        "root": ".",
        "source_spec": str(Path(args.spec).expanduser().resolve()),
        "source_spec_sha256": sha256_file(args.spec),
        "dino_model": str(args.dino_model),
        "dino_image_size": DINO_IMAGE_SIZE,
        "dino_tokens": DINO_FULL_TOKENS,
        "geometry_targets": "native_trellis2_geometry",
        "samples": manifest_rows,
    }
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, manifest_path)
    # Load every target family that is present.  This catches incompatible
    # native T2 latent dimensions before a long 30K run starts.
    PoseTrellis2Dataset(manifest_path, stage="inference", verify_hashes=not args.skip_hashes)
    if all("ss_target_file" in row for row in manifest_rows):
        PoseTrellis2Dataset(manifest_path, stage="ss", verify_hashes=not args.skip_hashes)
    if all("shape_target_file" in row for row in manifest_rows):
        PoseTrellis2Dataset(manifest_path, stage="shape", verify_hashes=not args.skip_hashes)
    return manifest


def main() -> None:
    args = make_parser().parse_args()
    manifest = run(args)
    print(
        json.dumps(
            {
                "manifest": str(Path(args.output_dir).expanduser().resolve() / "manifest.json"),
                "samples": len(manifest["samples"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
