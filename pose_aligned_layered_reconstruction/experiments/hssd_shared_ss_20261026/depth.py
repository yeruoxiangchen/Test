"""RGB-only VGGT-object scene-depth diagnostic, with camera-baseline scale.

The two frame groups are inferred independently. No target depth, occupancy,
latent, mesh, or evaluation geometry is read here. Raw VGGT confidence is
1 + exp(logit), not a probability; ranks are within each individual view.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
SCHEMA = "shared_rgb_vggt_depth_v1"
EXPECTED_GROUPS = {"group4": [0, 4, 7, 11], "group2": [0, 11]}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def empirical_rank(values):
    """Average empirical ranks, scaled to [0,1]; all ties receive 0.5.

    Ties are not broken by pixel order. This is a relative ordering of the
    model's confidence, not a calibrated probability of correct geometry.
    """
    values = np.asarray(values)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Confidence must be one finite 2D view")
    _, inverse, counts = np.unique(values.reshape(-1), return_inverse=True, return_counts=True)
    if len(counts) == 1:
        return np.full(values.shape, 0.5, dtype=np.float32)
    middle = np.cumsum(counts) - (counts + 1) / 2.0
    ranks = middle / max(values.size - 1, 1)
    return ranks[inverse].reshape(values.shape).astype(np.float32)


def _angle_degrees(rotation):
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))


def _centers(w2c):
    return -np.einsum("vji,vj->vi", w2c[:, :3, :3], w2c[:, :3, 3])


def _proper_rotation(matrix):
    u, _, vt = np.linalg.svd(matrix)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    return u @ correction @ vt


def camera_scale_diagnostic(pred_w2c, known_w2c, frames, min_baseline_world=1e-4):
    """Only median pairwise camera-baseline ratios set the depth scale.

    Sim(3) and orientation fits below are diagnostics only. Two cameras cannot
    identify a center-only 3D rotation, so that fit is explicitly unavailable.
    """
    pred = np.asarray(pred_w2c, dtype=np.float64)
    known = np.asarray(known_w2c, dtype=np.float64)
    if (pred.shape != (len(frames), 3, 4) or known.shape != (len(frames), 4, 4)
            or not np.isfinite(pred).all() or not np.isfinite(known).all()):
        raise ValueError("Camera matrices or frame counts are invalid")
    for label, mats in (("predicted", pred), ("known", known)):
        rotations = mats[:, :3, :3]
        if (np.max(np.abs(rotations @ rotations.transpose(0, 2, 1) - np.eye(3))) > 1e-3
                or np.max(np.abs(np.linalg.det(rotations) - 1)) > 1e-3):
            raise ValueError(f"Invalid {label} camera rotations")
    pc, kc = _centers(pred), _centers(known)
    pairs, ratios = [], []
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            dp, dk = float(np.linalg.norm(pc[j] - pc[i])), float(np.linalg.norm(kc[j] - kc[i]))
            valid = dp > 1e-8 and dk >= min_baseline_world
            ratio = dk / dp if valid else None
            rp = pred[j, :3, :3] @ pred[i, :3, :3].T
            rk = known[j, :3, :3] @ known[i, :3, :3].T
            pairs.append({"frames": [int(frames[i]), int(frames[j])],
                          "known_baseline_world": dk, "predicted_baseline": dp,
                          "valid_for_scale": valid, "world_per_predicted": ratio,
                          "relative_rotation_error_degrees": _angle_degrees(rk.T @ rp)})
            if valid:
                ratios.append(ratio)
    if not ratios:
        raise ValueError("No nondegenerate camera-baseline pair; depth scale is undefined")
    scale = float(np.median(ratios))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Nonpositive or nonfinite camera-derived world scale")
    px, kx = pc - pc.mean(0), kc - kc.mean(0)
    pred_rank, known_rank = int(np.linalg.matrix_rank(px)), int(np.linalg.matrix_rank(kx))
    # Independent orientation alignment remains identifiable with two views.
    orientation_rotation = _proper_rotation(sum(
        known[i, :3, :3].T @ pred[i, :3, :3] for i in range(len(frames))))
    orientation_translation = kc.mean(0) - scale * orientation_rotation @ pc.mean(0)
    aligned = scale * (pc @ orientation_rotation.T) + orientation_translation
    orientation_errors = [_angle_degrees(
        known[i, :3, :3] @ orientation_rotation @ pred[i, :3, :3].T)
        for i in range(len(frames))]
    sim3 = {"available": False, "reason": "center-only rotation underdetermined",
            "predicted_center_rank": pred_rank, "known_center_rank": known_rank}
    if len(frames) >= 3 and min(pred_rank, known_rank) >= 2:
        cov = kx.T @ px / len(frames)
        u, singular, vt = np.linalg.svd(cov)
        signs = np.ones(3)
        signs[-1] = np.linalg.det(u @ vt)
        rotation = u @ np.diag(signs) @ vt
        fitted_scale = float(np.sum(singular * signs) / np.mean(np.sum(px * px, axis=1)))
        translation = kc.mean(0) - fitted_scale * rotation @ pc.mean(0)
        residuals = np.linalg.norm(fitted_scale * (pc @ rotation.T) + translation - kc, axis=1)
        sim3 = {"available": True, "diagnostic_only_not_used_for_depth": True,
                "scale": fitted_scale, "rotation": rotation.tolist(), "translation": translation.tolist(),
                "center_residual_world": residuals.tolist(),
                "center_rmse_world": float(np.sqrt(np.mean(residuals ** 2))),
                "camera_rotation_error_degrees": [_angle_degrees(
                    known[i, :3, :3] @ rotation @ pred[i, :3, :3].T) for i in range(len(frames))],
                "predicted_center_rank": pred_rank, "known_center_rank": known_rank}
    return {"method": "median_known_baseline_divided_by_predicted_baseline",
            "world_per_predicted": scale, "pairs": pairs,
            "ratio_min": float(min(ratios)), "ratio_max": float(max(ratios)),
            "ratio_relative_mad": float(np.median(np.abs(np.asarray(ratios) - scale)) / scale),
            "ratio_relative_range": float((max(ratios) - min(ratios)) / scale),
            "sim3_diagnostic": sim3,
            "orientation_alignment_diagnostic": {
                "rotation": orientation_rotation.tolist(), "translation": orientation_translation.tolist(),
                "scale": scale, "camera_rotation_error_degrees": orientation_errors,
                "center_residual_world": np.linalg.norm(aligned - kc, axis=1).tolist(),
                "diagnostic_only_not_applied_to_depth": True},
            "depth_conversion": "depth_z_world = predicted_camera_z * world_per_predicted; no shift or spatial warp",
            "gt_geometry_used": False}


def _stat(path):
    path = Path(path).resolve(strict=True)
    s = path.stat()
    return {"path": str(path), "size": s.st_size, "mtime_ns": s.st_mtime_ns,
            "ctime_ns": s.st_ctime_ns, "inode": s.st_ino, "device": s.st_dev}


def _quantiles(array):
    return dict(zip(["min", "p01", "p10", "p50", "p90", "p99", "max"],
                    np.quantile(array, [0, .01, .1, .5, .9, .99, 1]).astype(float).tolist()))


def _validate_npz(path, frames, metadata):
    with np.load(path, allow_pickle=False) as data:
        if data["metadata_json"].item() != json.dumps(metadata, sort_keys=True, allow_nan=False):
            raise ValueError("Depth NPZ and JSON metadata disagree")
        if not np.array_equal(data["frames"], frames):
            raise ValueError("Depth frame indices changed")
        for key in ("depth_z_world", "depth_z_model", "raw_conf", "rank_conf"):
            value = data[key]
            if value.shape != (len(frames), 518, 518) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"Invalid depth-cache field: {key}")
        if np.any(data["depth_z_world"] <= 0) or np.any(data["rank_conf"] < 0) or np.any(data["rank_conf"] > 1):
            raise ValueError("Depth or confidence range invalid")
        if not np.allclose(data["depth_z_world"], data["depth_z_model"] * float(data["scale_world_per_model"]), rtol=1e-6):
            raise ValueError("Depth scale metadata inconsistent")


def run():
    # Keep CPU diagnostics importable without Torch or the experiment package.
    if __package__:
        from . import common
    else:
        import common
    p = common.plan()
    policy = p["plan"]["depth_policy"]
    if policy["groups"] != EXPECTED_GROUPS:
        raise ValueError("Depth groups must preserve the exact four-view and stool two-view inputs")
    if policy["confidence"]["kind"] != "per_view_empirical_cdf":
        raise ValueError("This depth implementation requires per-view empirical confidence ranks")
    snapshot = Path(policy["snapshot"]).resolve(strict=True)
    model_files = policy["snapshot_files"]
    expected_files = {str((snapshot / name).resolve(strict=True)) for name in ("config.json", "model.safetensors")}
    if {str(Path(b["path"]).resolve(strict=True)) for b in model_files} != expected_files:
        raise ValueError("Both local snapshot config and weights must be frozen in the plan")
    shared_binding = p["plan"]["shared_plan"]
    shared = common.read(common.verify(shared_binding))["plan"]
    rgb = {int(row["frame"]): row for row in policy["rgb"]}
    if set(rgb) != set(EXPECTED_GROUPS["group4"]) or len(rgb) != len(policy["rgb"]):
        raise ValueError("Expected exactly four uniquely bound shared RGB frames")
    for row in rgb.values():
        common.verify(row)
    vggt_root = Path(common.ROOT) / "ReconViaGen/wheels/vggt"
    source_files = sorted((vggt_root / "vggt").rglob("*.py"))
    contract = {"schema": SCHEMA, "plan_identity": p["identity"], "depth_policy": policy,
                "shared_plan": shared_binding, "implementation": common.binding(__file__),
                "vggt_source": [common.binding(path) for path in source_files],
                "scope": "VGGT-object on full-scene RGB diagnostic; not a scene-depth accuracy claim"}
    model_stats = [_stat(row["path"]) for row in model_files]
    out = Path(common.OUT) / "depth"
    out.mkdir(parents=True, exist_ok=True)
    missing, receipts = [], []
    for name, frames in EXPECTED_GROUPS.items():
        receipt_path = out / f"{name}.json"
        if receipt_path.exists():
            receipt = common.read(receipt_path)
            if (receipt["source_contract"] != contract or receipt["model_files_stat"] != model_stats
                    or receipt["group"] != name or receipt["frames"] != frames or not receipt["success"]):
                raise ValueError(f"Stale or changed depth cache: {receipt_path}")
            _validate_npz(common.verify(receipt["npz"]), frames, receipt["metadata"])
            receipts.append(common.binding(receipt_path))
            common.event("depth_cache_verified", group=name, frames=frames)
        else:
            missing.append((name, frames))
    if missing:
        # Offline is mandatory; a missing/mismatched local file must never fall
        # back to downloading a checkpoint. Full 5GB hash is done once per run
        # that actually needs inference, never during status or cache-only use.
        for row in model_files:
            common.verify(row)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        sys.path.insert(0, str(vggt_root))
        import torch
        from PIL import Image
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        if not torch.cuda.is_available():
            raise RuntimeError("Depth inference requires the explicitly selected CUDA device")
        common.event("depth_model_load", snapshot=str(snapshot), groups=[name for name, _ in missing])
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        model = VGGT.from_pretrained(str(snapshot), local_files_only=True, strict=True).eval()
        model.point_head = None
        model.track_head = None
        model = model.to("cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        for name, frames in missing:
            started = time.monotonic()
            common.event("depth_group_start", group=name, frames=frames)
            source_images = []
            for frame in frames:
                with Image.open(common.verify(rgb[frame])) as original:
                    if original.size != (2072, 2072):
                        raise ValueError("Depth must use the original 2072-square shared photo")
                    # This local official-repository loader expects singleton
                    # tuples of PIL images and unconditionally stacks alphas.
                    source_images.append((original.convert("RGB").convert("RGBA"),))
            images, alphas = load_and_preprocess_images(source_images)
            if images.shape != (len(frames), 3, 518, 518) or not torch.all(alphas == 1):
                raise ValueError("Unexpected VGGT preprocessing geometry or nonopaque alpha")
            del alphas, source_images
            images = images.unsqueeze(0).to("cuda")
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                prediction = model(images)
                with torch.autocast("cuda", enabled=False):
                    extrinsics, intrinsics = pose_encoding_to_extri_intri(
                        prediction["pose_enc"].float(), image_size_hw=(518, 518))
            depth = prediction["depth"][0, ..., 0].float().cpu().numpy()
            conf = prediction["depth_conf"][0].float().cpu().numpy()
            pred_w2c = extrinsics[0].float().cpu().numpy()
            pred_k = intrinsics[0].float().cpu().numpy()
            if (not np.isfinite(depth).all() or not np.isfinite(conf).all()
                    or np.any(depth <= 0) or np.any(conf < 1) or not np.isfinite(pred_k).all()):
                raise ValueError("Nonfinite or invalid VGGT depth/confidence/camera output")
            known_w2c = np.asarray(shared["world_w2c"], dtype=np.float64)[frames]
            all_k = np.asarray(shared["intrinsics"], dtype=np.float64)
            known_k = np.repeat(all_k[None], len(frames), axis=0) if all_k.shape == (3, 3) else all_k[frames]
            camera = camera_scale_diagnostic(pred_w2c, known_w2c, frames, policy["min_baseline_world"])
            scale = camera["world_per_predicted"]
            world = (depth * scale).astype(np.float32)
            rank = np.stack([empirical_rank(view) for view in conf])
            metadata = {"schema": SCHEMA, "source_contract_sha256": _digest(contract),
                        "group": name, "frames": frames, "image_side": 518,
                        "camera_convention": "OpenCV camera_from_world; x right, y down, z forward",
                        "depth_semantics": "predicted camera-forward z, scaled to known-camera world units",
                        "confidence": {"raw": "1 + exp(logit), not a probability",
                                       "rank": "per-view average empirical ranks/(pixel_count-1); ties share rank; all tied -> 0.5",
                                       "not_calibrated_probability": True,
                                       "minimum_rank_for_downstream_evidence": policy["confidence"]["minimum_rank"],
                                       "raw_per_view_quantiles": [_quantiles(view) for view in conf]},
                        "depth_world_per_view_quantiles": [_quantiles(view) for view in world],
                        "camera_scale": camera,
                        "intrinsics_relative_focal_error": ((pred_k[:, (0, 1), (0, 1)] /
                                                              known_k[:, (0, 1), (0, 1)]) - 1).tolist(),
                        "preprocessing": "bound 2072 RGB -> opaque RGBA -> local VGGT load_and_preprocess_images -> bicubic 518 square",
                        "independent_group_inference": True, "gt_geometry_used": False}
            path = out / f"{name}.npz"
            temp = path.with_name(path.name + f".{os.getpid()}.tmp")
            with temp.open("wb") as stream:
                np.savez_compressed(stream, frames=np.asarray(frames, np.int32), depth_z_world=world,
                                    depth_z_model=depth, raw_conf=conf, rank_conf=rank,
                                    pred_w2c=pred_w2c, pred_intrinsics=pred_k,
                                    known_w2c=known_w2c, known_intrinsics=known_k,
                                    scale_world_per_model=np.asarray(scale, np.float64),
                                    metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
            temp.replace(path)
            _validate_npz(path, frames, metadata)
            if [_stat(row["path"]) for row in model_files] != model_stats:
                raise ValueError("Model files changed during inference")
            receipt_path = out / f"{name}.json"
            common.write(receipt_path, {"schema": SCHEMA, "success": True, "group": name,
                "frames": frames, "source_contract": contract, "model_files_stat": model_stats,
                "npz": common.binding(path), "metadata": metadata,
                "runtime": {"torch_version": torch.__version__, "autocast_dtype": str(dtype),
                            "cuda_device_name": torch.cuda.get_device_name(),
                            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                            "elapsed_seconds": time.monotonic() - started}})
            receipts.append(common.binding(receipt_path))
            common.event("depth_group_complete", group=name, frames=frames,
                         scale=scale, ratio_relative_mad=camera["ratio_relative_mad"])
            del prediction, extrinsics, intrinsics, images
            torch.cuda.empty_cache()
        del model
    common.write(out / "complete.json", {"schema": SCHEMA, "success": True,
        "source_contract_sha256": _digest(contract), "groups": sorted(receipts, key=lambda row: row["path"]),
        "gt_geometry_used": False, "scope": contract["scope"]})
    common.event("depth_complete", groups=list(EXPECTED_GROUPS), gt_geometry_used=False)


def self_test():
    """CPU-only checks of scale/gauge recovery, two-view degeneracy and ties."""
    theta = .4
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0], [0, 0, 1.]])
    centers = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 1.]])
    shift, scale = np.array([3., -2., .5]), 2.5
    pred = np.repeat(np.eye(4)[None], 4, axis=0)
    pred[:, :3, 3] = -centers
    known = np.repeat(np.eye(4)[None], 4, axis=0)
    known[:, :3, :3] = rotation.T
    known[:, :3, 3] = -np.einsum("ij,vj->vi", rotation.T, scale * (centers @ rotation.T) + shift)
    result = camera_scale_diagnostic(pred[:, :3], known, [0, 4, 7, 11])
    assert abs(result["world_per_predicted"] - scale) < 1e-12
    assert result["sim3_diagnostic"]["center_rmse_world"] < 1e-12
    assert max(result["orientation_alignment_diagnostic"]["camera_rotation_error_degrees"]) < 1e-5
    two = camera_scale_diagnostic(pred[:2, :3], known[:2], [0, 11])
    assert not two["sim3_diagnostic"]["available"]
    assert np.allclose(empirical_rank(np.array([[1, 2], [2, 3.]])), [[0, .5], [.5, 1]])
    assert np.all(empirical_rank(np.ones((2, 2))) == .5)
    try:
        camera_scale_diagnostic(pred[:2, :3] * 0, known[:2], [0, 11])
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid rotations must fail")
    print("depth CPU checks passed: scale, Sim3, camera rotations, two-view degeneracy, tied ranks", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="CPU-only math checks; no model load")
    args = parser.parse_args()
    self_test() if args.self_test else run()
