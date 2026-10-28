"""Recover clean view-0 previews with the original frozen pixel preprocessing.

CPU only. No visual features, mesh, latent, training asset or old preview is
changed. The displayed 512px images are reduced from the exact 518px input.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
from PIL import Image, __version__ as pillow_version
import torch

from ...prepare_official_experiment import paired_pixels, RECIPE
from ...paired_dataset import canonical_json_sha256


PACKAGE = Path(__file__).resolve().parents[2]
SOURCE = PACKAGE / "outputs/round2_v2_diagnostics_20261018/mesh"
OUTPUT = PACKAGE / "outputs/round2_v2_mesh_visual_audit_20261018"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha(path)}


def verify(item):
    if sha(item["path"]) != item["sha256"]:
        raise ValueError("changed source: " + item["path"])
    return item["path"]


def main():
    torch.set_num_threads(2)
    plan_path = SOURCE / "plan.json"
    payload = json.loads(plan_path.read_text())
    if canonical_json_sha256(payload["plan"]) != payload["identity_hash"]:
        raise ValueError("mesh plan identity mismatch")
    rows = sorted((r for r in payload["plan"]["rows"] if r["domain"] == "legacy_clean"),
                  key=lambda r: r["index"])
    if [r["index"] for r in rows] != list(range(32)):
        raise ValueError("expected exactly the 32 clean inputs")
    destination = OUTPUT / "visuals/inputs"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for row in rows:
        sample = torch.load(verify(row["sample"]), map_location="cpu", mmap=True, weights_only=False)
        source = sample["source_binding"]
        if sample["uid"] != row["object_uid"] or source["uid"] != sample["uid"]:
            raise ValueError("sample/source object identity mismatch")
        cache_binding = {"path": source["cache"], "sha256": source["cache_sha256"]}
        cached = torch.load(verify(cache_binding), map_location="cpu", mmap=True, weights_only=False)
        if cached["uid"] != sample["uid"]:
            raise ValueError("source cache UID mismatch")
        for key in ("view_ids", "intrinsics", "extrinsics"):
            if not torch.equal(cached[key], sample[key]):
                raise ValueError("source cache/sample mismatch: " + key)
        if cached["preprocessing"] != sample["preprocessing"]:
            raise ValueError("preprocessing binding mismatch")
        if tuple(sample["image_size"]) != (518, 518):
            raise ValueError("unexpected source image size")
        subset = {key: cached[key][:1] for key in
                  ("view_ids", "intrinsics", "extrinsics", "source_to_feature_affines", "masks")}
        raw, _, masks, pixel_audit = paired_pixels(source, subset)
        mask_difference = float((masks - sample["visible_target_mask"][:1]).abs().max())
        if mask_difference != 0:
            raise ValueError("recovered alpha differs from frozen sample")
        original_pixels = np.asarray(raw[0].convert("RGB"), np.uint8)
        existing_checks = []
        for name in ("raw_view0.png", "qa_raw_view0.png"):
            path = Path(row["sample"]["path"]).parent / name
            if path.exists():
                with Image.open(path) as image:
                    reference = np.asarray(image.convert("RGB"), np.uint8)
                if reference.shape != original_pixels.shape:
                    raise ValueError("existing 518px preview has a different size")
                error = int(np.abs(reference.astype(np.int16) - original_pixels.astype(np.int16)).max())
                if error:
                    raise ValueError("recovered pixels differ from existing preview")
                existing_checks.append({"reference": binding(path), "max_abs_rgb_error": error,
                                        "pixel_exact": True})
        if row["index"] == 0 and not any(c["reference"]["path"].endswith("/raw_view0.png") for c in existing_checks):
            raise ValueError("object 0 reference preview must be available")
        target = destination / f"{row['index']:03d}.png"
        display = raw[0].convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        display.save(target)
        affine = subset["source_to_feature_affines"][0].numpy()
        scale = float(affine[0, 0])
        side = int(round(518 / scale))
        left, top = np.rint(-affine[:2, 2] / scale).astype(int)
        source_record = Path(row["sample"]["path"]).parent / "record.json"
        historical_member_checks = []
        if source_record.exists():
            previous = json.loads(source_record.read_text()).get("pixel_audit", {})
            hashes = previous.get("selected_render_member_sha256", {})
            for member, value in pixel_audit["selected_render_member_sha256"].items():
                if member in hashes:
                    if value != hashes[member]:
                        raise ValueError("source render member differs from original materialization")
                    historical_member_checks.append(member)
        record = {
            "index": row["index"], "object_index": row["object_index"], "object_uid": row["object_uid"],
            "displayed_view_position": 0, "original_view_id": int(subset["view_ids"][0]),
            "sample": row["sample"], "cache": cache_binding, "render_tar": binding(source["render"]),
            "render_tar_size_bytes": Path(source["render"]).stat().st_size,
            "source_to_feature_affine": affine.tolist(),
            "source_crop_xyxy": [int(left), int(top), int(left + side), int(top + side)],
            "preprocessing": sample["preprocessing"], "source_recipe_sha256": canonical_json_sha256(RECIPE),
            "pixel_audit": pixel_audit, "sample_alpha_max_abs_error": mask_difference,
            "historical_member_hashes_checked": historical_member_checks,
            "existing_preview_checks": existing_checks,
            "original_518_rgb_sha256": hashlib.sha256(original_pixels.tobytes()).hexdigest(),
            "output": binding(target), "output_size": [512, 512],
        }
        records.append(record)
        print(json.dumps({"recovered": len(records), "index": row["index"],
                          "view_id": record["original_view_id"], "camera_error": pixel_audit["camera_max_abs_error"],
                          "mask_error": mask_difference}), flush=True)
    report = {
        "format": "palr.mesh_clean_input_recovery.v1", "complete": True, "record_count": len(records),
        "source_mesh_plan": binding(plan_path), "source_mesh_plan_identity": payload["identity_hash"],
        "protocol": "Original prepare_official_experiment.paired_pixels on frozen view-0 affine: RGBA crop, 518x518 PIL BILINEAR, alpha-composite onto black; preview only resized to 512x512 with PIL LANCZOS.",
        "no_feature_extraction": True, "no_gpu": True, "original_assets_unchanged": True,
        "pillow_version": pillow_version,
        "code": [binding(__file__), binding(PACKAGE / "prepare_official_experiment.py"),
                 binding(PACKAGE / "paired_dataset.py")],
        "records": records,
    }
    path = OUTPUT / "clean_input_recovery.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": True, "record_count": len(records), "manifest": str(path)}), flush=True)


if __name__ == "__main__":
    main()
