"""CPU-only soft scene observations for the matched shared-input SS experiment.

Inputs are independently inferred RGB depth, known camera/canonical frames and
visible instance masks. Geometry behind the first observed surface remains
unknown. These are soft conditioning features, never hard occupancy carving.
No evaluation depth, target coordinates, latent, or mesh is opened here.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True
SCHEMA = "shared_predicted_observation_evidence_v1"
FEATURE_NAMES = ["surface_mean", "surface_peak", "free_mean", "free_multiview_mean", "unknown_mean",
                 "behind_target_mean", "behind_other_mean", "other_surface_mean", "frustum_coverage_mean",
                 "confidence_mean", "surface_free_conflict_mean", "surface_multiview_mean"]
GROUPS = {"group4": [0, 4, 7, 11], "group2": [0, 11]}


def _common():
    if __package__:
        from . import common
    else:
        import common
    return common


def fine_evidence(delta, rank, target_mask, frustum, *, tau=2 / 64, minimum_rank=.3):
    """Fuse [views, xyz] ray observations before any spatial pooling."""
    if delta.ndim != 2 or any(x.shape != delta.shape for x in (rank, target_mask, frustum)):
        raise ValueError("Expected corresponding [views, points] observation arrays")
    if (tau <= 0 or not 0 <= minimum_rank <= 1 or not torch.isfinite(delta).all()
            or not torch.isfinite(rank).all() or torch.any(rank < 0) or torch.any(rank > 1)
            or target_mask.dtype != torch.bool or frustum.dtype != torch.bool):
        raise ValueError("Invalid ray observations")
    observed = frustum & (rank >= minimum_rank)
    conf = rank * observed.float()
    near = torch.exp(-.5 * (delta / tau).square()) * (delta.abs() <= 3 * tau)
    surface_v = conf * target_mask.float() * near
    free_v = conf * ((-delta - tau) / (2 * tau)).clamp(0, 1)
    surface = surface_v.amax(0)
    free = free_v.amax(0)
    unknown = 1 - torch.maximum(surface, free)
    behind_target = (conf * target_mask.float() * (delta > tau)).mean(0)
    behind_other = (conf * (~target_mask).float() * (delta > tau)).mean(0)
    other_surface = (conf * (~target_mask).float() * near).amax(0)
    coverage = frustum.float().mean(0)
    confidence = conf.sum(0) / frustum.sum(0).clamp_min(1)
    strong_surface = (conf * target_mask.float() * (delta.abs() <= tau)).amax(0)
    strong_free = (conf * (delta < -3 * tau)).amax(0)
    conflict = torch.minimum(strong_surface, strong_free)
    # The second channel is replaced by a cell maximum during pooling.
    values = torch.stack([surface, surface, free, free_v.mean(0), unknown,
                          behind_target, behind_other, other_surface, coverage,
                          confidence, conflict, surface_v.mean(0)], dim=-1)
    return values, frustum.any(0)


def pool_xyz(fine, valid, *, fine_side=64, token_side=16):
    """Pool contiguous xyz cells, preserving x-major/y/z-fast token order."""
    if (fine_side % token_side or fine.shape != (fine_side**3, 12)
            or valid.shape != (fine_side**3,) or valid.dtype != torch.bool):
        raise ValueError("Invalid fine-grid pooling contract")
    cell = fine_side // token_side
    blocks = fine.reshape(token_side, cell, token_side, cell, token_side, cell, 12)
    pooled = blocks.mean(dim=(1, 3, 5))
    pooled[..., 1] = blocks[..., 0].amax(dim=(1, 3, 5))
    coarse_valid = valid.reshape(token_side, cell, token_side, cell, token_side, cell).any(dim=(1, 3, 5))
    pooled[~coarse_valid] = 0
    return {"features": pooled.reshape(1, token_side**3, 12).contiguous(),
            "weights": coarse_valid.reshape(1, token_side**3).float().contiguous(),
            "valid": coarse_valid.reshape(1, token_side**3).contiguous()}


def project_observations(points, canonical_from_world, known_w2c, known_k, depth, rank, masks):
    """Sample full-frame observations at canonical query points, on CPU."""
    matrices = torch.as_tensor(canonical_from_world, dtype=torch.float64)
    if matrices.shape != (4, 4) or not torch.isfinite(matrices).all():
        raise ValueError("Invalid known canonical transform")
    if not torch.allclose(matrices[3], torch.tensor([0., 0., 0., 1.], dtype=torch.float64), atol=1e-9, rtol=0):
        raise ValueError("Canonical transform must be affine")
    scale = torch.linalg.norm(matrices[:3, 0]).item()
    if scale <= 0 or not torch.allclose(matrices[:3, :3].T @ matrices[:3, :3],
                                      torch.eye(3, dtype=torch.float64) * scale**2, atol=1e-7, rtol=1e-7):
        raise ValueError("Canonical transform must have one uniform scale")
    views, height, width = depth.shape
    if (points.ndim != 2 or points.shape[1] != 3 or rank.shape != depth.shape or masks.shape != depth.shape
            or known_w2c.shape != (views, 4, 4) or known_k.shape != (views, 3, 3)
            or any(t.device.type != "cpu" for t in (points, depth, rank, masks, known_w2c, known_k))):
        raise ValueError("Expected corresponding CPU points, full-image maps and known cameras")
    point_h = torch.cat([points.double(), torch.ones((len(points), 1), dtype=torch.float64)], dim=1)
    world = point_h @ torch.linalg.inv(matrices).T
    camera = torch.einsum("vij,nj->vni", known_w2c.double(), world)[..., :3]
    uvw = torch.einsum("vij,vnj->vni", known_k.double(), camera)
    z = camera[..., 2]
    safe_denominator = uvw[..., 2].clamp_min(1e-12)
    uv = uvw[..., :2] / safe_denominator[..., None]
    frustum = ((z > 1e-6) & (uv[..., 0] >= 0) & (uv[..., 0] <= width - 1)
               & (uv[..., 1] >= 0) & (uv[..., 1] <= height - 1))
    # Out-of-frustum coordinates need not be evaluated; zero confidence below
    # removes them. Clamping also avoids nonfinite sampling grids behind cameras.
    grid = torch.stack([2 * uv[..., 0] / (width - 1) - 1,
                        2 * uv[..., 1] / (height - 1) - 1], dim=-1).float().clamp(-2, 2)
    grid = grid.unsqueeze(2)
    maps = torch.stack([depth.float(), rank.float()], dim=1)
    sampled = F.grid_sample(maps, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[..., 0]
    sampled_depth, sampled_rank = sampled[:, 0], sampled[:, 1].clamp(0, 1)
    sampled_mask = F.grid_sample(masks.float().unsqueeze(1), grid, mode="nearest",
                                 padding_mode="zeros", align_corners=True)[:, 0, :, 0] > .5
    if torch.any(frustum & ((sampled_depth <= 0) | ~torch.isfinite(sampled_depth))):
        raise ValueError("Invalid predicted depth inside the known-camera frustum")
    delta = ((z - sampled_depth.double()) * scale).float()
    return delta, sampled_rank, sampled_mask, frustum, scale


def _inputs(obj):
    """Validate observation bindings only; never open evaluation targets."""
    common = _common()
    p = common.plan()
    expected = [row for row in p["plan"]["objects"] if row["uid"] == obj["uid"]]
    if len(expected) != 1:
        raise ValueError("Object is absent or duplicated in the frozen cohort")
    expected = expected[0]
    for key in ("uid", "canonical_from_world", "visible_indices"):
        if obj[key] != expected[key]:
            raise ValueError("Object observation metadata changed: " + key)
    policy = p["plan"]["evidence_policy"]
    if (policy["features"] != FEATURE_NAMES or common.FEATURES != FEATURE_NAMES
            or policy["fine_side"] != 64 or policy["token_side"] != 16
            or policy["surface_tolerance_canonical"] != 2 / 64):
        raise ValueError("Unexpected frozen evidence feature/grid contract")
    frames = [GROUPS["group4"][int(i)] for i in obj["visible_indices"]]
    matching = [name for name, group in GROUPS.items() if frames == group]
    if len(matching) != 1:
        raise ValueError("Visible RGB set must exactly match an independently inferred depth group")
    group = matching[0]
    depth_receipt_path = Path(common.OUT) / "depth" / f"{group}.json"
    depth_binding = common.binding(depth_receipt_path)
    depth_receipt = common.read(common.verify(depth_binding))
    dc = depth_receipt["source_contract"]
    if (not depth_receipt["success"] or depth_receipt["group"] != group
            or depth_receipt["frames"] != frames or dc["plan_identity"] != p["identity"]
            or dc["depth_policy"] != p["plan"]["depth_policy"]
            or dc["shared_plan"] != p["plan"]["shared_plan"]
            or depth_receipt["metadata"]["gt_geometry_used"]
            or depth_receipt["metadata"]["source_contract_sha256"] != common.digest(dc)):
        raise ValueError("Depth receipt is stale or uses different observations")
    common.verify(depth_receipt["npz"])
    mask_rows = policy["full_visible_masks"][obj["uid"]]
    mask_by_frame = {int(row["frame"]): row for row in mask_rows}
    if len(mask_by_frame) != len(mask_rows):
        raise ValueError("Duplicate frame mask")
    selected_masks = [mask_by_frame[f] for f in frames]
    for row in selected_masks:
        common.verify(row)
    shared = common.read(common.verify(p["plan"]["shared_plan"]))["plan"]
    known_w2c = np.asarray(shared["world_w2c"], dtype=np.float64)[frames]
    known_k = np.asarray(shared["intrinsics"], dtype=np.float64)[frames]
    projection_path = Path(common.PACKAGE) / "projection.py"
    source = {"schema": SCHEMA, "plan_identity": p["identity"], "uid": obj["uid"],
              "selected_frames": frames, "visible_indices": obj["visible_indices"], "depth_group": group,
              "depth_receipt": depth_binding, "depth_npz": depth_receipt["npz"], "selected_masks": selected_masks,
              "shared_camera_plan": p["plan"]["shared_plan"],
              "canonical_from_world": obj["canonical_from_world"],
              "known_w2c": known_w2c.tolist(), "known_intrinsics": known_k.tolist(),
              "policy": {key: value for key, value in policy.items() if key != "full_visible_masks"},
              "implementation": common.binding(__file__), "projection_implementation": common.binding(projection_path),
              "torch_version": torch.__version__, "grid_transform": "identity",
              "sampling": "full-frame grid_sample; align_corners=True; depth/rank bilinear, mask nearest",
              "gt_geometry_used": False}
    return common, source, depth_receipt


def _validate_tensors(value):
    if not isinstance(value, dict) or set(value) != {"features", "weights", "valid"}:
        raise ValueError("Evidence payload must contain exactly three tensors")
    features, weights, valid = (value[key] for key in ("features", "weights", "valid"))
    if (any(not torch.is_tensor(t) or t.device.type != "cpu" or t.requires_grad for t in value.values())
            or features.shape != (1, 4096, 12) or weights.shape != (1, 4096) or valid.shape != (1, 4096)
            or features.dtype != torch.float32 or weights.dtype != torch.float32 or valid.dtype != torch.bool
            or not torch.isfinite(features).all() or (features < 0).any() or (features > 1).any()
            or not torch.equal(weights, valid.float()) or torch.any(features[~valid] != 0)):
        raise ValueError("Invalid evidence tensors, ranges, weights or xyz layout")
    return value


def load_evidence(obj):
    """Load a verified cache only; never generate observations during training."""
    common, source, _ = _inputs(obj)
    receipt_path = Path(common.OUT) / "evidence" / obj["uid"] / "receipt.json"
    receipt_binding = common.binding(receipt_path)
    receipt = common.read(common.verify(receipt_binding))
    if (not receipt["success"] or receipt["schema"] != SCHEMA or receipt["sources"] != source
            or receipt["feature_names"] != FEATURE_NAMES or receipt["gt_geometry_used"]):
        raise ValueError("Evidence receipt is stale or uses different observations")
    value = torch.load(common.verify(receipt["tensors"]), map_location="cpu", weights_only=True)
    return _validate_tensors(value), receipt_binding


def _statistics(features, valid):
    rows = features[0]
    active = valid[0]
    summaries = {}
    for i, name in enumerate(FEATURE_NAMES):
        channel = rows[:, i]
        summaries[name] = {"min": float(channel.min()), "mean": float(channel.mean()), "max": float(channel.max()),
                           "mean_valid": float(channel[active].mean()) if active.any() else None}
    return {"features": summaries, "valid_cell_fraction": float(active.float().mean()),
            "fractions_over_all_cells": {
                "surface_mean_gt_0_1": float((rows[:, 0] > .1).float().mean()),
                "surface_peak_gt_0_3": float((rows[:, 1] > .3).float().mean()),
                "free_mean_gt_0_3": float((rows[:, 2] > .3).float().mean()),
                "unknown_mean_gt_0_9_and_valid": float(((rows[:, 4] > .9) & active).float().mean()),
                "surface_free_conflict_mean_gt_0_1": float((rows[:, 10] > .1).float().mean())}}


def build_evidence(obj):
    common, source, depth_receipt = _inputs(obj)
    receipt_path = Path(common.OUT) / "evidence" / obj["uid"] / "receipt.json"
    if receipt_path.exists():
        _, receipt_binding = load_evidence(obj)
        common.event("evidence_cache_verified", uid=obj["uid"], frames=source["selected_frames"])
        return receipt_binding
    from PIL import Image
    # Load only the self-contained query-grid implementation, even for direct
    # CLI execution; it does not construct a model or load target geometry.
    spec = importlib.util.spec_from_file_location("_shared_evidence_projection", source["projection_implementation"]["path"])
    projection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(projection)
    points = projection.canonical_query_points(64, "identity", device=torch.device("cpu"))
    with np.load(common.verify(depth_receipt["npz"]), allow_pickle=False) as arrays:
        metadata = json.loads(arrays["metadata_json"].item())
        if (metadata != depth_receipt["metadata"] or arrays["frames"].tolist() != source["selected_frames"]
                or not np.array_equal(arrays["known_w2c"], np.asarray(source["known_w2c"]))
                or not np.array_equal(arrays["known_intrinsics"], np.asarray(source["known_intrinsics"]))):
            raise ValueError("Depth maps and their known-camera/metadata bindings disagree")
        depth = torch.from_numpy(arrays["depth_z_world"].copy())
        rank = torch.from_numpy(arrays["rank_conf"].copy())
        known_w2c = torch.from_numpy(arrays["known_w2c"].copy())
        known_k = torch.from_numpy(arrays["known_intrinsics"].copy())
    views = len(source["selected_frames"])
    if (depth.shape != (views, 518, 518) or rank.shape != depth.shape
            or not torch.isfinite(depth).all() or (depth <= 0).any()
            or not torch.isfinite(rank).all() or (rank < 0).any() or (rank > 1).any()):
        raise ValueError("Invalid predicted depth map")
    mask_arrays = []
    for row in source["selected_masks"]:
        with Image.open(common.verify(row)) as image:
            if image.size != (2072, 2072):
                raise ValueError("Visible mask must correspond to the original 2072 photo")
            resized = image.convert("L").resize((518, 518), Image.Resampling.NEAREST)
            mask_arrays.append(torch.from_numpy((np.asarray(resized) > 127).copy()))
    masks = torch.stack(mask_arrays)
    delta, sampled_rank, sampled_mask, frustum, scale = project_observations(
        points, source["canonical_from_world"], known_w2c, known_k, depth, rank, masks)
    fine, fine_valid = fine_evidence(delta, sampled_rank, sampled_mask, frustum,
        tau=source["policy"]["surface_tolerance_canonical"],
        minimum_rank=source["policy"]["confidence_minimum_rank"])
    value = _validate_tensors(pool_xyz(fine, fine_valid))
    tensor_path = Path(common.OUT) / "evidence" / f"{obj['uid']}.pt"
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tensor_path.with_name(tensor_path.name + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(tensor_path)
    stats = _statistics(value["features"], value["valid"])
    common.write(receipt_path, {"schema": SCHEMA, "success": True, "uid": obj["uid"],
        "sources": source, "tensors": common.binding(tensor_path), "feature_names": FEATURE_NAMES,
        "canonical_scale": scale, "statistics": stats, "gt_geometry_used": False,
        "fine_diagnostics": {"frustum_coverage_fraction": float(frustum.float().mean()),
                             "evidence_confident_fraction": float((frustum & (sampled_rank >= .3)).float().mean()),
                             "any_frustum_point_fraction": float(fine_valid.float().mean()),
                             "surface_mean": float(fine[:, 0].mean()), "free_mean": float(fine[:, 2].mean()),
                             "unknown_mean": float(fine[:, 4].mean()), "conflict_mean": float(fine[:, 10].mean())},
        "scope": "soft RGB-predicted first-surface/free/unknown observations; no hard carving; known cameras, visible masks and canonical frame remain privileged"})
    _, receipt_binding = load_evidence(obj)
    common.event("evidence_object_complete", uid=obj["uid"], frames=source["selected_frames"],
                 valid_fraction=stats["valid_cell_fraction"])
    return receipt_binding


def run():
    common = _common()
    torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
    p = common.plan()
    if len(p["plan"]["objects"]) != 4:
        raise ValueError("Keep the complete frozen four-object cohort")
    receipts = [build_evidence(obj) for obj in p["plan"]["objects"]]
    common.write(Path(common.OUT) / "evidence/complete.json", {"schema": SCHEMA, "success": True,
        "plan_identity": p["identity"], "objects": receipts, "feature_names": FEATURE_NAMES,
        "gt_geometry_used": False, "hard_carving": False,
        "statistics": {common.read(row["path"])["uid"]: common.read(row["path"])["statistics"] for row in receipts}})
    common.event("evidence_complete", objects=len(receipts), gt_geometry_used=False)


def self_test():
    """CPU semantic tests: ray ordering, occlusion, low confidence and xyz order."""
    torch.set_num_threads(1)
    tau = 2 / 64
    # Queries respectively before depth, at target surface, behind an occluder,
    # at a non-target surface and at a low-confidence target surface.
    delta = torch.tensor([[-4 * tau, 0., 4 * tau, 0., 0.]])
    rank = torch.tensor([[1., 1., 1., 1., .1]])
    target = torch.tensor([[True, True, False, False, True]])
    frustum = torch.ones_like(target)
    fine, valid = fine_evidence(delta, rank, target, frustum)
    assert fine[0, 2] == 1 and fine[0, 0] == 0 and fine[0, 4] == 0
    assert fine[1, 0] == 1 and fine[1, 2] == 0 and fine[1, 4] == 0
    assert fine[2, 2] == 0 and fine[2, 4] == 1 and fine[2, 6] == 1
    assert fine[3, 7] == 1 and fine[3, 0] == 0 and fine[3, 2] == 0 and fine[3, 4] == 1
    assert fine[4, 0] == 0 and fine[4, 2] == 0 and fine[4, 4] == 1 and valid[4]
    # The same point receiving incompatible first surfaces is flagged, not
    # silently hard-carved or converted into a GT occupancy target.
    conflicting, _ = fine_evidence(torch.tensor([[0.], [-4 * tau]]), torch.ones(2, 1),
                                  torch.ones(2, 1, dtype=torch.bool), torch.ones(2, 1, dtype=torch.bool))
    assert conflicting[0, 10] == 1
    xyz = torch.stack(torch.meshgrid(*(torch.arange(4) for _ in range(3)), indexing="ij"), -1)
    code = 100 * xyz[..., 0] + 10 * xyz[..., 1] + xyz[..., 2]
    tagged = code.float().reshape(-1, 1).repeat(1, 12)
    pooled = pool_xyz(tagged, torch.ones(64, dtype=torch.bool), fine_side=4, token_side=2)
    for x in range(2):
        for y in range(2):
            for z in range(2):
                index = x * 4 + y * 2 + z
                block = code[2*x:2*x+2, 2*y:2*y+2, 2*z:2*z+2].float()
                assert pooled["features"][0, index, 0] == block.mean()
                assert pooled["features"][0, index, 1] == block.max()
    # Known projection + canonical scale: z_world=1.5,2,2.5 relative to a
    # flat observed depth 2.0 maps to canonical deltas -1,0,+1 with scale 2.
    canonical = torch.eye(4, dtype=torch.float64)
    canonical[:3, :3] *= 2
    points = torch.tensor([[0., 0., 3.], [0., 0., 4.], [0., 0., 5.]])
    w2c = torch.eye(4).unsqueeze(0)
    k = torch.tensor([[[1., 0., 1.], [0., 1., 1.], [0., 0., 1.]]])
    result = project_observations(points, canonical, w2c, k, torch.full((1, 3, 3), 2.),
                                  torch.ones(1, 3, 3), torch.ones(1, 3, 3, dtype=torch.bool))
    assert torch.equal(result[0], torch.tensor([[-1., 0., 1.]])) and result[3].all()
    print("evidence CPU checks passed: free/surface/occluded-unknown, low confidence, conflict, xyz pooling, camera/world scale", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="CPU-only semantic tests, no cache generation")
    args = parser.parse_args()
    self_test() if args.self_test else run()
