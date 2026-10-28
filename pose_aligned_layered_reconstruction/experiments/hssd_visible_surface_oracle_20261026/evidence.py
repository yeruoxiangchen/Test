"""Visible first-surface oracle for the frozen SharedSS geometry operator.

This privileged diagnostic removes BOTH inferred depth and confidence error.
It is not a depth-only ablation or an RGB-only result. The extractor alone may
read ``depth_z_world`` from the four frozen scene archives; evidence generation
and training read only the separately stripped depth/valid caches. No instance
ID, hidden target surface, complete object geometry or latent is read here.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np
import torch


SCHEMA = "shared_visible_surface_oracle_evidence_v1"
OBS_SCHEMA = "stripped_visible_first_surface_v1"
FEATURE_NAMES = ["surface_mean", "surface_peak", "free_mean", "free_multiview_mean", "unknown_mean",
                 "behind_target_mean", "behind_other_mean", "other_surface_mean", "frustum_coverage_mean",
                 "confidence_mean", "surface_free_conflict_mean", "surface_multiview_mean"]
GROUPS = {"group4": [0, 4, 7, 11], "group2": [0, 11]}
STOOL_UID = "hssd_102344049_i065"
COHORT = {"hssd_102344049_i076", "hssd_102344049_i075", "hssd_102344049_i080", STOOL_UID}
OPERATOR_PATH = Path(__file__).resolve().parents[1] / "hssd_shared_ss_20261026/evidence.py"
_OPERATORS = {}


def _common():
    if __package__:
        from . import common
    else:
        import common
    return common


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _operator(common=None, source_binding=None):
    """Reuse the frozen operator, rather than reimplement its channel semantics."""
    if common is None:  # Pure CPU self-test: no experiment plan or data access.
        path, key = OPERATOR_PATH, "self_test"
    else:
        source_binding = source_binding or common.binding(OPERATOR_PATH)
        path, key = common.verify(source_binding), source_binding["sha256"]
    if key not in _OPERATORS:
        module = _module("_visible_oracle_shared_operator_" + key, path)
        if module.FEATURE_NAMES != FEATURE_NAMES or module.GROUPS != GROUPS:
            raise ValueError("The frozen SharedSS operator/channel order changed")
        _OPERATORS[key] = module
    return _OPERATORS[key]


def _policy(p):
    policy = p["plan"]["oracle_policy"]
    required = {"depth_key": "depth_z_world", "valid_confidence": 1.0,
                "invalid_confidence": 0.0, "invalid_depth_policy": "unknown",
                "allow_hidden_geometry": False, "groups": GROUPS}
    if any(policy.get(key) != value for key, value in required.items()):
        raise ValueError("Unexpected frozen visible-oracle policy")
    if {obj["uid"] for obj in p["plan"]["objects"]} != COHORT or len(p["plan"]["objects"]) != 4:
        raise ValueError("Keep the complete frozen four-object cohort")
    return policy


def selected_frames(obj):
    """The stool may never acquire the two frames absent from its RGB input."""
    indices = obj["visible_indices"]
    expected = [0, 3] if obj["uid"] == STOOL_UID else [0, 1, 2, 3]
    if obj["uid"] not in COHORT or indices != expected:
        raise ValueError("Object RGB/frame subset differs from the frozen SharedSS input")
    return [GROUPS["group4"][i] for i in indices]


def extract_visible_depth(archive):
    """Deliberately request exactly one NPZ field, even if others are present."""
    depth = np.asarray(archive["depth_z_world"])
    if depth.shape != (518, 518) or depth.dtype != np.float32:
        raise ValueError("Expected the frozen 518 x 518 float32 world-camera z depth")
    valid = np.isfinite(depth) & (depth > 0)
    # Zero is only a storage sentinel. It is never used as an observed surface.
    return {"depth_z_world": np.where(valid, depth, 0).astype(np.float32),
            "valid": valid.astype(np.bool_)}


def _validate_observation(value):
    if set(value) != {"depth_z_world", "valid"}:
        raise ValueError("Observation cache must contain only depth_z_world and valid")
    depth, valid = value["depth_z_world"], value["valid"]
    if (depth.shape != (518, 518) or valid.shape != depth.shape
            or depth.dtype != np.float32 or valid.dtype != np.bool_
            or not np.isfinite(depth).all() or np.any(depth[valid] <= 0)
            or np.any(depth[~valid] != 0)):
        raise ValueError("Invalid stripped first-surface observation")
    return value


def _load_stripped_archive(archive):
    # Reject an unstripped original archive before requesting any array. This
    # makes the boundary explicit even if a receipt points at the wrong file.
    if set(archive.files) != {"depth_z_world", "valid"}:
        raise ValueError("Refuse an unstripped archive at the evidence boundary")
    return _validate_observation({key: archive[key] for key in ("depth_z_world", "valid")})


def _raw_bindings(p):
    rows = p["plan"]["evaluation_gt"]
    by_frame = {int(Path(row["path"]).stem): row for row in rows}
    if len(rows) != 4 or sorted(by_frame) != GROUPS["group4"]:
        raise ValueError("Oracle archives must be exactly the four existing shared scene frames")
    return by_frame


def _observation_source(common, p, frame):
    return {"schema": OBS_SCHEMA, "plan_identity": p["identity"], "frame": frame,
            "source": _raw_bindings(p)[frame], "extracted_fields": ["depth_z_world"],
            "stored_fields": ["depth_z_world", "valid"], "oracle_policy": _policy(p),
            "implementation": common.binding(__file__)}


def _observation_receipt(common, p, frame):
    """Validate bindings without opening or hashing the original GT archive."""
    receipt_path = Path(common.OUT) / "observations/first_surface" / f"{frame:03d}.json"
    receipt_binding = common.binding(receipt_path)
    receipt = common.read(common.verify(receipt_binding))
    if (receipt.get("schema") != OBS_SCHEMA or receipt.get("success") is not True
            or receipt.get("source_contract") != _observation_source(common, p, frame)
            or receipt.get("hidden_geometry_used") is not False
            or receipt.get("confidence_error_removed") is not True):
        raise ValueError("Stale, incompatible or unstripped observation receipt")
    common.verify(receipt["npz"])
    return receipt, receipt_binding


def prepare_observations():
    """Independent extraction stage; the only function opening original GT NPZs."""
    common = _common()
    p = common.plan()
    _policy(p)
    rows = []
    for frame, raw_binding in sorted(_raw_bindings(p).items()):
        receipt_path = Path(common.OUT) / "observations/first_surface" / f"{frame:03d}.json"
        if receipt_path.exists():
            receipt, rb = _observation_receipt(common, p, frame)
            with np.load(common.verify(receipt["npz"]), allow_pickle=False) as arrays:
                _load_stripped_archive(arrays)
            rows.append(rb)
            continue
        # No other archive member is requested. Source verification binds this
        # extraction to the previously frozen full-scene renderer output.
        source_path = common.verify(raw_binding)
        with np.load(source_path, allow_pickle=False) as archive:
            value = _validate_observation(extract_visible_depth(archive))
        target = receipt_path.with_suffix(".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **value)
        temporary.replace(target)
        valid, depth = value["valid"], value["depth_z_world"]
        common.write(receipt_path, {"schema": OBS_SCHEMA, "success": True,
            "source_contract": _observation_source(common, p, frame),
            "npz": common.binding(target), "shape": list(depth.shape),
            "valid_pixels": int(valid.sum()), "invalid_pixels": int((~valid).sum()),
            "valid_depth_range_world": [float(depth[valid].min()), float(depth[valid].max())] if valid.any() else None,
            "visible_gt_geometry_used": True, "hidden_geometry_used": False,
            "confidence_error_removed": True, "depth_only_ablation": False,
            "physical_source": str(source_path),
            "scope": "privileged observed first surface only; invalid/no-hit remains unknown"})
        _, rb = _observation_receipt(common, p, frame)
        rows.append(rb)
        common.event("oracle_observation_extracted", frame=frame, valid=int(valid.sum()), invalid=int((~valid).sum()))
    common.write(Path(common.OUT) / "observations/complete.json", {
        "schema": OBS_SCHEMA, "success": True, "plan_identity": p["identity"],
        "frames": GROUPS["group4"], "receipts": rows, "hidden_geometry_used": False,
        "confidence_error_removed": True, "depth_only_ablation": False})


def project_visible_oracle(operator, points, canonical_from_world, known_w2c, known_k, depth, valid, masks):
    """Keep old projection; any nonzero invalid interpolation weight is unknown.

    Sampling the INVALID indicator separately avoids a tolerance that could
    mistakenly accept a tiny invalid contribution. A positive placeholder depth
    satisfies the old projector's finite check, but its ray gets confidence zero
    whenever that placeholder contributes. It cannot produce free/surface data.
    """
    if (valid.dtype != torch.bool or depth.shape != valid.shape or masks.shape != depth.shape
            or not torch.isfinite(depth).all() or torch.any(depth[valid] <= 0)
            or torch.any(depth[~valid] != 0)):
        raise ValueError("Invalid depth/valid contract at oracle projection")
    safe_depth = torch.where(valid, depth, torch.ones_like(depth))
    delta, invalid_mass, target, frustum, scale = operator.project_observations(
        points, canonical_from_world, known_w2c, known_k, safe_depth, (~valid).float(), masks)
    # Logical observed confidence is exactly one; undefined observations have
    # zero confidence. Frustum coverage itself remains the same geometric test.
    confidence = (invalid_mass == 0).float()
    return delta, confidence, target, frustum, scale, invalid_mass


def _inputs(obj):
    common = _common()
    p = common.plan()
    oracle_policy = _policy(p)
    expected = [row for row in p["plan"]["objects"] if row["uid"] == obj["uid"]]
    if len(expected) != 1 or any(obj[key] != expected[0][key]
                               for key in ("uid", "canonical_from_world", "visible_indices")):
        raise ValueError("Object differs from the frozen observation cohort")
    frames = selected_frames(obj)
    policy = p["plan"]["evidence_policy"]
    if (policy["features"] != FEATURE_NAMES or common.FEATURES != FEATURE_NAMES
            or policy["fine_side"] != 64 or policy["token_side"] != 16
            or policy["surface_tolerance_canonical"] != 2 / 64
            or policy["confidence_minimum_rank"] != .3):
        raise ValueError("Frozen SharedSS evidence/grid/pooling policy changed")
    observation_receipts, observation_bindings = [], []
    for frame in frames:
        receipt, rb = _observation_receipt(common, p, frame)
        observation_receipts.append(receipt)
        observation_bindings.append(rb)
    masks = policy["full_visible_masks"][obj["uid"]]
    by_frame = {int(row["frame"]): row for row in masks}
    if len(by_frame) != len(masks) or sorted(by_frame) != GROUPS["group4"]:
        raise ValueError("Unexpected frozen visible-mask set")
    selected_masks = [by_frame[frame] for frame in frames]
    for mask in selected_masks:
        common.verify(mask)
    shared = common.read(common.verify(p["plan"]["shared_plan"]))["plan"]
    w2c = np.asarray(shared["world_w2c"], dtype=np.float64)[frames]
    intrinsics = np.asarray(shared["intrinsics"], dtype=np.float64)[frames]
    if w2c.shape != (len(frames), 4, 4) or intrinsics.shape != (len(frames), 3, 3):
        raise ValueError("Invalid frozen known cameras")
    source = {"schema": SCHEMA, "plan_identity": p["identity"], "uid": obj["uid"],
        "selected_frames": frames, "visible_indices": obj["visible_indices"],
        "observations": observation_bindings, "observation_npzs": [row["npz"] for row in observation_receipts],
        "selected_masks": selected_masks, "shared_camera_plan": p["plan"]["shared_plan"],
        "canonical_from_world": obj["canonical_from_world"],
        "known_w2c": w2c.tolist(), "known_intrinsics": intrinsics.tolist(),
        "policy": {key: value for key, value in policy.items() if key != "full_visible_masks"},
        "oracle_policy": oracle_policy, "implementation": common.binding(__file__),
        "operator_implementation": common.binding(OPERATOR_PATH),
        "projection_implementation": common.binding(Path(common.PACKAGE) / "projection.py"),
        "torch_version": str(torch.__version__), "grid_transform": "identity",
        "sampling": "old full-frame projection; align_corners=True; depth bilinear, mask nearest; any invalid contribution sets confidence=0",
        "visible_gt_geometry_used": True, "hidden_geometry_used": False,
        "confidence_error_removed": True, "depth_only_ablation": False, "hard_carving": False}
    return common, source, observation_receipts


def load_evidence(obj):
    """Load verified tensors only; do not extract, render or access original GT."""
    common, source, _ = _inputs(obj)
    path = Path(common.OUT) / "evidence" / obj["uid"] / "receipt.json"
    rb = common.binding(path)
    receipt = common.read(common.verify(rb))
    if (receipt.get("schema") != SCHEMA or receipt.get("success") is not True
            or receipt.get("sources") != source or receipt.get("feature_names") != FEATURE_NAMES
            or receipt.get("hidden_geometry_used") is not False
            or receipt.get("confidence_error_removed") is not True):
        raise ValueError("Stale or incompatible oracle evidence receipt")
    value = torch.load(common.verify(receipt["tensors"]), map_location="cpu", weights_only=True)
    operator = _operator(common, source["operator_implementation"])
    return operator._validate_tensors(value), rb


def build_evidence(obj):
    common, source, receipts = _inputs(obj)
    receipt_path = Path(common.OUT) / "evidence" / obj["uid"] / "receipt.json"
    if receipt_path.exists():
        _, rb = load_evidence(obj)
        common.event("oracle_evidence_cache_verified", uid=obj["uid"], frames=source["selected_frames"])
        return rb
    from PIL import Image
    operator = _operator(common, source["operator_implementation"])
    projection = _module("_visible_oracle_query_grid", common.verify(source["projection_implementation"]))
    points = projection.canonical_query_points(64, "identity", device=torch.device("cpu"))
    depths, valids = [], []
    for receipt in receipts:
        with np.load(common.verify(receipt["npz"]), allow_pickle=False) as arrays:
            value = _load_stripped_archive(arrays)
        depths.append(torch.from_numpy(value["depth_z_world"].copy()))
        valids.append(torch.from_numpy(value["valid"].copy()))
    masks = []
    for row in source["selected_masks"]:
        with Image.open(common.verify(row)) as image:
            if image.size != (2072, 2072):
                raise ValueError("Visible masks must use the frozen 2072-to-518 nearest resize")
            masks.append(torch.from_numpy((np.asarray(image.convert("L").resize(
                (518, 518), Image.Resampling.NEAREST)) > 127).copy()))
    delta, confidence, target, frustum, scale, invalid_mass = project_visible_oracle(
        operator, points, source["canonical_from_world"], torch.tensor(source["known_w2c"], dtype=torch.float64),
        torch.tensor(source["known_intrinsics"], dtype=torch.float64), torch.stack(depths),
        torch.stack(valids), torch.stack(masks))
    fine, fine_valid = operator.fine_evidence(delta, confidence, target, frustum,
        tau=source["policy"]["surface_tolerance_canonical"], minimum_rank=source["policy"]["confidence_minimum_rank"])
    value = operator._validate_tensors(operator.pool_xyz(fine, fine_valid))
    tensor_path = Path(common.OUT) / "evidence" / f"{obj['uid']}.pt"
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tensor_path.with_name(tensor_path.name + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(tensor_path)
    statistics = operator._statistics(value["features"], value["valid"])
    common.write(receipt_path, {"schema": SCHEMA, "success": True, "uid": obj["uid"],
        "sources": source, "tensors": common.binding(tensor_path), "feature_names": FEATURE_NAMES,
        "canonical_scale": scale, "statistics": statistics,
        "visible_gt_geometry_used": True, "hidden_geometry_used": False,
        "confidence_error_removed": True, "depth_only_ablation": False, "hard_carving": False,
        "fine_diagnostics": {"frustum_coverage_fraction": float(frustum.float().mean()),
            "invalid_interpolation_fraction_in_frustum": float((frustum & (invalid_mass > 0)).float().sum() / frustum.sum().clamp_min(1)),
            "evidence_confident_fraction": float((frustum & (confidence >= .3)).float().mean()),
            "any_frustum_point_fraction": float(fine_valid.float().mean()),
            "surface_mean": float(fine[:, 0].mean()), "free_mean": float(fine[:, 2].mean()),
            "unknown_mean": float(fine[:, 4].mean()), "conflict_mean": float(fine[:, 10].mean())},
        "scope": "privileged visible scene depth plus valid=1 confidence; behind first surface stays unknown; same soft SharedSS operator"})
    _, rb = load_evidence(obj)
    common.event("oracle_evidence_object_complete", uid=obj["uid"], frames=source["selected_frames"],
                 valid_fraction=statistics["valid_cell_fraction"])
    return rb


def run():
    common = _common()
    torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
    p = common.plan()
    _policy(p)
    # Extraction is explicit and separate. This stage never opens original GT.
    receipts = [build_evidence(obj) for obj in p["plan"]["objects"]]
    statistics = {}
    for rb in receipts:
        row = common.read(common.verify(rb))
        statistics[row["uid"]] = row["statistics"]
    common.write(Path(common.OUT) / "evidence/complete.json", {"schema": SCHEMA, "success": True,
        "plan_identity": p["identity"], "objects": receipts, "feature_names": FEATURE_NAMES,
        "visible_gt_geometry_used": True, "hidden_geometry_used": False, "hard_carving": False,
        "confidence_error_removed": True, "depth_only_ablation": False, "statistics": statistics})
    common.event("oracle_evidence_complete", objects=len(receipts), hidden_geometry_used=False)


def self_test():
    """No plan, GPU, archive, cache or output access; synthetic CPU contracts."""
    torch.set_num_threads(1)
    operator = _operator()
    operator.self_test()  # Includes behind/occluder/conflict and x-major xyz tests.
    calls = []

    class FieldGuard:
        def __getitem__(self, key):
            calls.append(key)
            if key != "depth_z_world":
                raise AssertionError("Hidden/instance field requested")
            value = np.ones((518, 518), dtype=np.float32)
            value[0, :4] = [0, -1, np.nan, np.inf]
            return value

    stripped = _validate_observation(extract_visible_depth(FieldGuard()))
    assert calls == ["depth_z_world"] and not stripped["valid"][0, :4].any()
    assert (stripped["depth_z_world"][0, :4] == 0).all()
    try:
        _validate_observation(dict(stripped, instance_id=np.zeros((518, 518))))
    except ValueError:
        pass
    else:
        raise AssertionError("Unexpected fields were accepted into stripped cache")
    class WrongArchive:
        files = ["depth_z_world", "instance_id"]

        def __getitem__(self, key):
            raise AssertionError("Unstripped archive was accessed before rejection")

    try:
        _load_stripped_archive(WrongArchive())
    except ValueError:
        pass
    else:
        raise AssertionError("An original archive crossed the stripped-cache boundary")
    assert selected_frames({"uid": STOOL_UID, "visible_indices": [0, 3]}) == [0, 11]
    for bad in ([0, 1, 2, 3], [0, 1], [0, 3, 3]):
        try:
            selected_frames({"uid": STOOL_UID, "visible_indices": bad})
        except ValueError:
            pass
        else:
            raise AssertionError("The stool gained extra/wrong frames")
    eye = torch.eye(4, dtype=torch.float64)
    k = torch.eye(3, dtype=torch.float64)[None]
    depth = torch.full((1, 3, 3), 2.)
    valid = torch.ones_like(depth, dtype=torch.bool)
    masks = torch.ones_like(valid)
    # Pixel projections: (0,0), (0.5,0.5), (2,2); all at z=2.
    points = torch.tensor([[0., 0., 2.], [1., 1., 2.], [4., 4., 2.]])
    old = operator.project_observations(points, eye, eye[None], k, depth, valid.float(), masks)
    got = project_visible_oracle(operator, points, eye, eye[None], k, depth, valid, masks)
    assert torch.equal(old[0], got[0]) and torch.allclose(old[1], got[1], atol=1e-7, rtol=0)
    assert torch.equal(old[2], got[2]) and torch.equal(old[3], got[3]) and old[4] == got[4]
    old_fine, _ = operator.fine_evidence(*old[:4])
    new_fine, _ = operator.fine_evidence(*got[:4])
    assert torch.allclose(old_fine, new_fine, atol=1e-7, rtol=0)
    valid[0, 1, 1] = False
    depth[0, 1, 1] = 0
    got = project_visible_oracle(operator, points, eye, eye[None], k, depth, valid, masks)
    assert got[1].tolist() == [[1., 0., 1.]] and got[-1][0, 1] == .25
    fine, fine_valid = operator.fine_evidence(*got[:4])
    assert fine_valid.all() and fine[1, 0] == 0 and fine[1, 2] == 0 and fine[1, 4] == 1
    # All no-hit rays remain unknown even when the placeholder would imply free.
    empty = torch.zeros_like(valid)
    got = project_visible_oracle(operator, points * .1, eye, eye[None], k,
                                 torch.zeros_like(depth), empty, masks)
    fine, _ = operator.fine_evidence(*got[:4])
    assert (fine[:, 0] == 0).all() and (fine[:, 2] == 0).all() and (fine[:, 4] == 1).all()
    assert not torch.cuda.is_initialized()
    report = {"success": True, "test": "visible_first_surface_oracle_cpu",
        "checked": ["depth_only_archive_access", "cache_allowlist", "invalid_depth_unknown",
                    "invalid_bilinear_contributor_unknown", "all_valid_old_projection_parity",
                    "behind_unknown", "occluder_not_target", "stool_no_extra_frames", "xyz_pool_order"],
        "gpu_initialized": False}
    print(json.dumps(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("extract", "build", "self-test"), nargs="?", default="build")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test or args.action == "self-test":
        self_test()
    elif args.action == "extract":
        prepare_observations()
    else:
        run()


if __name__ == "__main__":
    main()
