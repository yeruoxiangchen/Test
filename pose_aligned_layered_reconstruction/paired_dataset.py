"""Manifest-backed paired DINO/layer dataset with content identity checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from .contracts import (
    LAYERED_SAMPLE_FORMAT,
    LAYERED_SAMPLE_MANIFEST_FORMAT,
    validate_paired_sample,
)
from .dino_context import build_masked_dino_contexts


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_manifest(
    path: str | Path,
    *,
    rows: Iterable[dict[str, Any]],
    config: dict[str, Any],
    root: str | Path | None = None,
) -> dict[str, Any]:
    destination = Path(path).expanduser().resolve()
    normalized = [dict(row) for row in rows]
    if not normalized:
        raise ValueError("paired manifest requires at least one sample")
    uids = [str(row.get("uid", "")) for row in normalized]
    if not all(uids) or len(uids) != len(set(uids)):
        raise ValueError("paired manifest UIDs must be non-empty and unique")
    object_uids = sorted(
        {str(row.get("object_uid", row["uid"])) for row in normalized}
    )
    payload = {
        "format": LAYERED_SAMPLE_MANIFEST_FORMAT,
        "root": str(Path(root).expanduser().resolve() if root else destination.parent),
        "sample_count": len(normalized),
        "object_count": len(object_uids),
        "config": dict(config),
        "config_hash": canonical_json_sha256(config),
        "samples": normalized,
    }
    payload["manifest_identity"] = canonical_json_sha256(
        {
            "format": payload["format"],
            "config_hash": payload["config_hash"],
            "samples": normalized,
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


class PairedLayerDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        *,
        require_labels: bool = True,
        require_ss_target: bool = False,
        verify_hashes: bool = True,
        build_trellis_contexts: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest).expanduser().resolve(strict=True)
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("format") != LAYERED_SAMPLE_MANIFEST_FORMAT:
            raise ValueError("unexpected paired manifest format")
        self.root = Path(payload.get("root", self.manifest_path.parent)).resolve()
        self.config = dict(payload.get("config", {}))
        self.config_hash = str(payload.get("config_hash", ""))
        if canonical_json_sha256(self.config) != self.config_hash:
            raise ValueError("paired manifest config hash differs")
        rows = payload.get("samples")
        if not isinstance(rows, list) or not rows:
            raise ValueError("paired manifest contains no samples")
        if int(payload.get("sample_count", -1)) != len(rows):
            raise ValueError("paired manifest sample_count differs")
        expected_identity = canonical_json_sha256(
            {
                "format": payload["format"],
                "config_hash": self.config_hash,
                "samples": rows,
            }
        )
        if payload.get("manifest_identity") != expected_identity:
            raise ValueError("paired manifest identity differs")
        self.rows = [dict(row) for row in rows]
        self.require_labels = bool(require_labels)
        self.require_ss_target = bool(require_ss_target)
        self.build_trellis_contexts = bool(build_trellis_contexts)
        self.verify_hashes = bool(verify_hashes)
        seen: set[str] = set()
        for row in self.rows:
            uid = str(row.get("uid", ""))
            relative = str(row.get("sample", ""))
            if not uid or uid in seen or not relative:
                raise ValueError("paired rows contain empty/duplicate identities")
            seen.add(uid)
            path = self._resolve(relative)
            if not path.is_file():
                raise FileNotFoundError(path)
            expected_hash = str(row.get("sha256", ""))
            if verify_hashes and (not expected_hash or sha256_file(path) != expected_hash):
                raise RuntimeError(f"paired sample hash differs: {path}")

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[int(index)]
        path = self._resolve(row["sample"])
        sample = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(sample, dict):
            raise ValueError(f"paired sample root must be a dict: {path}")
        if sample.get("format") != LAYERED_SAMPLE_FORMAT:
            raise ValueError(f"paired sample format differs: {path}")
        if str(sample.get("uid", "")) != str(row["uid"]):
            raise RuntimeError(f"paired sample UID differs: {path}")
        if str(sample.get("object_uid", sample["uid"])) != str(
            row.get("object_uid", row["uid"])
        ):
            raise RuntimeError(f"paired sample object UID differs: {path}")
        validate_paired_sample(
            sample,
            require_labels=self.require_labels,
            require_ss_target=self.require_ss_target,
        )
        contexts = (
            build_masked_dino_contexts(
                sample["masked_visual_patch_features"]
            )
            if self.build_trellis_contexts
            else {}
        )
        return {
            **sample,
            **(
                {
                    "stock_condition": contexts["stock_condition"],
                    "slat_condition": contexts["slat_condition"],
                }
                if contexts
                else {}
            ),
            "sample_path": str(path),
        }

    def identity(self) -> dict[str, Any]:
        uids = [str(row["uid"]) for row in self.rows]
        objects = sorted(
            {str(row.get("object_uid", row["uid"])) for row in self.rows}
        )
        return {
            "manifest": str(self.manifest_path),
            "manifest_sha256": sha256_file(self.manifest_path),
            "config_hash": self.config_hash,
            "sample_count": len(self.rows),
            "object_count": len(objects),
            "uid_hash": canonical_json_sha256(uids),
            "object_uid_hash": canonical_json_sha256(objects),
            "object_uids": objects,
            "paired_raw_and_masked": True,
            "build_trellis_contexts": self.build_trellis_contexts,
            "sample_hashes_verified_at_open": self.verify_hashes,
            "synthetic_fixture_only": bool(self.config.get("synthetic_fixture_only", False)),
            "ss_target_is_random_not_geometry_ground_truth": bool(
                self.config.get("ss_target_is_random_not_geometry_ground_truth", False)
            ),
        }


def collate_one(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("variable-view paired dataset requires batch_size=1")
    return batch[0]


__all__ = [
    "PairedLayerDataset",
    "canonical_json_sha256",
    "collate_one",
    "sha256_file",
    "write_manifest",
]
