"""Strict manifest loader for pose-conditioned TRELLIS.2 geometry training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import (
    CACHE_FORMAT,
    DINO_FEATURE_DIM,
    DINO_FULL_TOKENS,
    SAMPLE_FORMAT,
    SHAPE_LATENT_CHANNELS,
    SHAPE_MAX_TOKENS,
    SHAPE_LATENT_MEAN,
    SHAPE_LATENT_RESOLUTION,
    SHAPE_LATENT_STD,
    SS_LATENT_CHANNELS,
    SS_LATENT_RESOLUTION,
)
from .geometry import split_dinov3_tokens


Stage = Literal["ss", "shape", "inference"]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_indices(specification: str, size: int) -> list[int]:
    text = str(specification).strip().lower()
    if text in ("", "all"):
        return list(range(int(size)))
    result: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(item))
    invalid = [index for index in result if index < 0 or index >= int(size)]
    if invalid:
        raise IndexError(f"indices outside dataset size {size}: {invalid[:8]}")
    return result


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def load_pose_payload(path: Path, uid: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != SAMPLE_FORMAT:
        raise ValueError(f"uid={uid} invalid pose payload format: {path}")
    if str(payload.get("uid")) != uid:
        raise ValueError(f"pose payload UID differs: {payload.get('uid')} != {uid}")
    features = payload.get("dino_features")
    intrinsics = payload.get("intrinsics")
    extrinsics = payload.get("extrinsics")
    if not torch.is_tensor(features) or tuple(features.shape[1:]) != (
        DINO_FULL_TOKENS,
        DINO_FEATURE_DIM,
    ):
        raise ValueError(
            f"uid={uid} DINOv3 features must be [V,{DINO_FULL_TOKENS},1024]"
        )
    views = int(features.shape[0])
    if (
        not torch.is_tensor(intrinsics)
        or not torch.is_tensor(extrinsics)
        or tuple(intrinsics.shape) != (views, 3, 3)
        or tuple(extrinsics.shape) != (views, 4, 4)
    ):
        raise ValueError(f"uid={uid} camera tensors do not match {views} views")
    split_dinov3_tokens(features, allow_patch_only=False)
    finite = (features, intrinsics, extrinsics)
    if not all(bool(torch.isfinite(value.float()).all().item()) for value in finite):
        raise ValueError(f"uid={uid} pose payload contains non-finite tensors")
    image_size = payload.get("image_size")
    if not isinstance(image_size, (list, tuple)) or list(map(int, image_size)) != [512, 512]:
        raise ValueError(f"uid={uid} pose payload image_size must be [512,512]")
    if payload.get("extrinsics_type") not in ("w2c", "c2w"):
        raise ValueError(f"uid={uid} invalid extrinsics_type")
    if float(payload.get("camera_forward_sign", 0.0)) not in (-1.0, 1.0):
        raise ValueError(f"uid={uid} invalid camera_forward_sign")
    if float(payload.get("camera_image_y_sign", 0.0)) not in (-1.0, 1.0):
        raise ValueError(f"uid={uid} invalid camera_image_y_sign")
    return payload


def load_ss_target(path: Path, uid: str) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as payload:
        if "z" not in payload:
            raise ValueError(f"uid={uid} SS target lacks z: {path}")
        target = np.asarray(payload["z"], dtype=np.float32)
    if target.ndim == 5 and target.shape[0] == 1:
        target = target[0]
    expected = (SS_LATENT_CHANNELS,) + (SS_LATENT_RESOLUTION,) * 3
    if tuple(target.shape) != expected or not np.isfinite(target).all():
        raise ValueError(f"uid={uid} invalid SS target {target.shape}: {path}")
    return torch.from_numpy(target)


def load_shape_target(path: Path, uid: str) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path, allow_pickle=False) as payload:
        if not {"coords", "feats"}.issubset(payload.files):
            raise ValueError(f"uid={uid} Shape target lacks coords/feats: {path}")
        coords = np.asarray(payload["coords"], dtype=np.int32)
        feats = np.asarray(payload["feats"], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] not in (3, 4):
        raise ValueError(f"uid={uid} invalid Shape coords={coords.shape}")
    coords = coords[:, -3:]
    if (
        feats.ndim != 2
        or feats.shape != (len(coords), SHAPE_LATENT_CHANNELS)
        or not np.isfinite(feats).all()
        or (coords < 0).any()
        or (coords >= SHAPE_LATENT_RESOLUTION).any()
    ):
        raise ValueError(
            f"uid={uid} invalid native TRELLIS.2 Shape target "
            f"coords={coords.shape} feats={feats.shape}"
        )
    if len(coords) == 0:
        raise ValueError(f"uid={uid} Shape target has no active coordinates")
    if len(coords) > SHAPE_MAX_TOKENS:
        raise ValueError(
            f"uid={uid} Shape target has {len(coords)} active tokens; "
            f"official Shape-512 maximum is {SHAPE_MAX_TOKENS}"
        )
    # Duplicate coordinates make sparse attention and decoder semantics ambiguous.
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError(f"uid={uid} Shape target contains duplicate coordinates")
    mean = np.asarray(SHAPE_LATENT_MEAN, dtype=np.float32)[None]
    std = np.asarray(SHAPE_LATENT_STD, dtype=np.float32)[None]
    normalized = (feats - mean) / std
    return torch.from_numpy(coords), torch.from_numpy(normalized)


class PoseTrellis2Dataset(Dataset):
    """Join DINOv3/K/T evidence with native TRELLIS.2 SS or Shape targets."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        stage: Stage,
        indices: str = "all",
        verify_hashes: bool = False,
    ) -> None:
        if stage not in ("ss", "shape", "inference"):
            raise ValueError(f"unsupported dataset stage={stage!r}")
        self.stage = stage
        self.manifest_path = Path(manifest).expanduser().resolve()
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("format") != CACHE_FORMAT:
            raise ValueError(f"unsupported cache format={payload.get('format')!r}")
        rows = payload.get("samples")
        if not isinstance(rows, list) or not rows:
            raise ValueError("cache manifest has no samples")
        selected = parse_indices(indices, len(rows))
        self.rows = [rows[index] for index in selected]
        self.root = _resolve(
            self.manifest_path.parent,
            payload.get("root", self.manifest_path.parent),
        )
        self.payload = payload
        uids = [str(row.get("uid", "")) for row in self.rows]
        if not all(uids) or len(set(uids)) != len(uids):
            raise ValueError("cache manifest contains empty or duplicate UIDs")
        required = ["pose_file"]
        if stage == "ss":
            required.append("ss_target_file")
        elif stage == "shape":
            required.append("shape_target_file")
        for row in self.rows:
            missing = [name for name in required if not row.get(name)]
            if missing:
                raise ValueError(f"uid={row.get('uid')} missing fields={missing}")
        if verify_hashes:
            self.verify_hashes()

    def __len__(self) -> int:
        return len(self.rows)

    def resolve_row_path(self, row: dict[str, Any], field: str) -> Path:
        return _resolve(self.root, row[field])

    def verify_hashes(self) -> None:
        for row in self.rows:
            for field in ("pose_file", "ss_target_file", "shape_target_file"):
                if not row.get(field):
                    continue
                path = self.resolve_row_path(row, field)
                if not path.is_file():
                    raise FileNotFoundError(path)
                expected = str(row.get(f"{field}_sha256", ""))
                if expected and sha256_file(path) != expected:
                    raise RuntimeError(f"uid={row['uid']} hash differs for {field}")

    def identity(self) -> dict[str, Any]:
        uids = [str(row["uid"]) for row in self.rows]
        return {
            "manifest": str(self.manifest_path),
            "manifest_sha256": sha256_file(self.manifest_path),
            "stage": self.stage,
            "sample_count": len(self.rows),
            "uid_sha256": canonical_json_sha256(uids),
            "target_family": "native_trellis2_geometry",
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[index])
        uid = str(row["uid"])
        pose_path = self.resolve_row_path(row, "pose_file")
        result = load_pose_payload(pose_path, uid)
        result.update({"uid": uid, "pose_path": str(pose_path), "row": row})
        if self.stage == "ss":
            target_path = self.resolve_row_path(row, "ss_target_file")
            result["target"] = load_ss_target(target_path, uid)
            result["target_path"] = str(target_path)
        elif self.stage == "shape":
            target_path = self.resolve_row_path(row, "shape_target_file")
            coords, feats = load_shape_target(target_path, uid)
            result["target_coords"] = coords
            result["target_feats"] = feats
            result["target_path"] = str(target_path)
        return result


def collate_one(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) != 1:
        raise ValueError("pose-conditioned training uses per-rank batch size one")
    return samples[0]


__all__ = [
    "PoseTrellis2Dataset",
    "canonical_json_sha256",
    "collate_one",
    "load_shape_target",
    "load_pose_payload",
    "load_ss_target",
    "parse_indices",
    "sha256_file",
]
