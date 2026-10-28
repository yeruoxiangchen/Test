"""Factor observed depth and confidence without changing the geometry operator.

Only the two off-diagonal cells are new. Before writing them, reconstruct both
historical diagonals and require bit-exact features/weights/valid parity. GT
observations are already-stripped visible-first-hit depth/valid caches. The
predicted-depth/unit-confidence path never opens those GT observations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import torch

SCHEMA = "depth_confidence_cross_evidence_v1"
DIAGONAL_SCHEMA = "depth_confidence_diagonal_exact_v1"
FEATURE_NAMES = ["surface_mean", "surface_peak", "free_mean", "free_multiview_mean", "unknown_mean",
                 "behind_target_mean", "behind_other_mean", "other_surface_mean", "frustum_coverage_mean",
                 "confidence_mean", "surface_free_conflict_mean", "surface_multiview_mean"]
GROUPS = {"group4": [0, 4, 7, 11], "group2": [0, 11]}
ARMS = ("geometry_gt_depth_pred_confidence", "geometry_pred_depth_unit_confidence")
DIAGONALS = ("geometry_observed", "geometry_oracle_visible")
FACTORS = {"geometry_observed": {"depth": "predicted", "confidence": "predicted"},
           "geometry_gt_depth_pred_confidence": {"depth": "gt", "confidence": "predicted"},
           "geometry_pred_depth_unit_confidence": {"depth": "predicted", "confidence": "unit"},
           "geometry_oracle_visible": {"depth": "gt", "confidence": "unit"}}
UIDS = ("hssd_102344049_i076", "hssd_102344049_i075", "hssd_102344049_i080", "hssd_102344049_i065")
OPERATOR_PATH = Path(__file__).resolve().parents[1] / "hssd_shared_ss_20261026/evidence.py"
ORACLE_PATH = Path(__file__).resolve().parents[1] / "hssd_visible_surface_oracle_20261026/evidence.py"
_MODULES = {}


def _common():
    if __package__:
        from . import common
    else:
        import common
    return common


def _module(name, path):
    key = (name, str(path))
    if key not in _MODULES:
        spec = importlib.util.spec_from_file_location(name, str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULES[key] = module
    return _MODULES[key]


def _operator(common=None):
    path = OPERATOR_PATH if common is None else common.verify(common.binding(OPERATOR_PATH))
    module = _module("_depth_confidence_old_operator", path)
    if module.FEATURE_NAMES != FEATURE_NAMES or module.GROUPS != GROUPS:
        raise ValueError("Frozen operator features/groups changed")
    return module


def selected_frames(obj):
    selected = [0, 3] if obj["uid"] == UIDS[-1] else [0, 1, 2, 3]
    if obj["uid"] not in UIDS or obj["visible_indices"] != selected:
        raise ValueError("Frozen object/view subset changed")
    return [GROUPS["group4"][i] for i in selected]


def _policy(p):
    policy = p["plan"]["depth_confidence_policy"]
    if (policy.get("arms") != FACTORS or policy.get("invalid_depth_policy") != "unknown"
            or policy.get("allow_hidden_geometry") is not False or policy.get("groups") != GROUPS):
        raise ValueError("Unexpected frozen depth/confidence factorization")
    if [o["uid"] for o in p["plan"]["objects"]] != list(UIDS):
        raise ValueError("Keep all four objects in their original order")
    return policy


def _stripped(archive):
    # An original GT archive is rejected before any array is accessed.
    if set(archive.files) != {"depth_z_world", "valid"}:
        raise ValueError("Only stripped first-hit depth/valid caches may be read")
    depth, valid = archive["depth_z_world"], archive["valid"]
    if (depth.dtype != np.float32 or valid.dtype != np.bool_ or depth.shape != (518, 518)
            or valid.shape != depth.shape or not np.isfinite(depth).all()
            or np.any(depth[valid] <= 0) or np.any(depth[~valid] != 0)):
        raise ValueError("Invalid stripped observation")
    return depth, valid


def _inputs(obj, arm):
    if arm not in FACTORS:
        raise ValueError("Unknown geometry factorial cell")
    c = _common(); p = c.plan(); _policy(p)
    expected = [x for x in p["plan"]["objects"] if x["uid"] == obj["uid"]]
    if len(expected) != 1 or any(obj[k] != expected[0][k] for k in ("uid", "canonical_from_world", "visible_indices")):
        raise ValueError("Object observation contract changed")
    factors = FACTORS[arm]; frames = selected_frames(obj)
    group = next(k for k, values in GROUPS.items() if values == frames)
    policy = p["plan"]["evidence_policy"]
    if (policy["features"] != FEATURE_NAMES or c.FEATURES != FEATURE_NAMES or policy["fine_side"] != 64
            or policy["token_side"] != 16 or policy["surface_tolerance_canonical"] != 2/64
            or policy["confidence_minimum_rank"] != .3):
        raise ValueError("Geometry operator grid/tolerance/channel policy changed")
    masks = policy["full_visible_masks"][obj["uid"]]
    by_frame = {int(row["frame"]): row for row in masks}
    if len(by_frame) != len(masks) or sorted(by_frame) != GROUPS["group4"]:
        raise ValueError("Unexpected visible-mask frame set")
    selected_masks = [by_frame[f] for f in frames]
    for row in selected_masks:c.verify(row)
    shared = c.read(c.verify(p["plan"]["shared_plan"]))["plan"]
    w2c = np.asarray(shared["world_w2c"], np.float64)[frames]
    intrinsics = np.asarray(shared["intrinsics"], np.float64)[frames]
    prediction = None; prediction_binding = None
    if factors["depth"] == "predicted" or factors["confidence"] == "predicted":
        prediction_binding = p["plan"]["predicted_depth"][group]
        prediction = c.read(c.verify(prediction_binding))
        contract = prediction["source_contract"]
        if (prediction.get("success") is not True or prediction["group"] != group
                or prediction["frames"] != frames or contract["shared_plan"] != p["plan"]["shared_plan"]
                or contract["depth_policy"]["groups"] != GROUPS
                or contract["depth_policy"]["confidence"] != {"kind": "per_view_empirical_cdf", "minimum_rank": .3}
                or prediction["metadata"]["gt_geometry_used"] is not False
                or prediction["metadata"]["source_contract_sha256"] != c.digest(contract)):
            raise ValueError("Predicted depth/rank provenance differs from the frozen RGB inference")
        c.verify(prediction["npz"])
    observations = []; observation_bindings = []
    # Deliberately no GT receipt/data access for either predicted-depth cell.
    if factors["depth"] == "gt":
        for frame in frames:
            b = p["plan"]["stripped_observations"][str(frame)]
            row = c.read(c.verify(b)); source = row["source_contract"]
            if (row.get("schema") != "stripped_visible_first_surface_v1" or row.get("success") is not True
                    or source["frame"] != frame or source["extracted_fields"] != ["depth_z_world"]
                    or source["stored_fields"] != ["depth_z_world", "valid"]
                    or row["hidden_geometry_used"] is not False
                    or source["oracle_policy"]["allow_hidden_geometry"] is not False):
                raise ValueError("GT observation is not a stripped visible first surface")
            c.verify(row["npz"]);observations.append(row);observation_bindings.append(b)
    source = {"schema": SCHEMA, "plan_identity": p["identity"], "uid": obj["uid"], "arm": arm,
        "factors": factors, "selected_frames": frames, "visible_indices": obj["visible_indices"],
        "depth_group": group, "predicted_receipt": prediction_binding,
        "predicted_npz": None if prediction is None else prediction["npz"],
        "stripped_receipts": observation_bindings, "stripped_npzs": [row["npz"] for row in observations],
        "selected_masks": selected_masks, "shared_camera_plan": p["plan"]["shared_plan"],
        "canonical_from_world": obj["canonical_from_world"], "known_w2c": w2c.tolist(),
        "known_intrinsics": intrinsics.tolist(),
        "numeric_policy": {k: policy[k] for k in ("features", "fine_side", "token_side", "surface_tolerance_canonical", "confidence_minimum_rank")},
        "factorial_policy": p["plan"]["depth_confidence_policy"],
        "implementation": c.binding(__file__), "operator_implementation": c.binding(OPERATOR_PATH),
        "projection_implementation": c.binding(Path(c.PACKAGE)/"projection.py"),
        "torch_version": str(torch.__version__), "grid_transform": "identity",
        "sampling": "old depth/rank bilinear, target mask nearest; invalid depth contribution forces unknown",
        "visible_gt_geometry_used": factors["depth"] == "gt", "hidden_geometry_used": False,
        "unit_confidence_uses_only_own_depth_validity": factors["confidence"] == "unit", "hard_carving": False}
    return c, source, prediction, observations


def project_factor(operator, points, canonical, w2c, intrinsics, depth, valid, rank, masks):
    """Own-depth validity gates either unchanged predicted rank or exact unit confidence."""
    if (valid.dtype != torch.bool or masks.dtype != torch.bool or valid.shape != depth.shape
            or masks.shape != depth.shape or not torch.isfinite(depth).all()
            or torch.any(depth[valid] <= 0) or torch.any(depth[~valid] != 0)):
        raise ValueError("Invalid own-depth validity contract")
    if rank is not None and (rank.shape != depth.shape or not torch.isfinite(rank).all()
                             or torch.any(rank < 0) or torch.any(rank > 1)):
        raise ValueError("Predicted confidence must retain its [0,1] full-frame rank")
    safe_depth = torch.where(valid, depth, torch.ones_like(depth))
    invalid_projection = operator.project_observations(points, canonical, w2c, intrinsics,
        safe_depth, (~valid).float(), masks)
    invalid_mass = invalid_projection[1]
    if rank is None:
        delta, _, target, frustum, scale = invalid_projection
        confidence = (invalid_mass == 0).float()
    else:
        delta, confidence, target, frustum, scale = operator.project_observations(
            points, canonical, w2c, intrinsics, safe_depth, rank, masks)
        confidence = torch.where(invalid_mass == 0, confidence, torch.zeros_like(confidence))
    return delta, confidence, target, frustum, scale, invalid_mass


def _calculate(obj, arm):
    c, source, prediction, observations = _inputs(obj, arm)
    operator = _operator(c)
    projection = _module("_depth_confidence_query_grid", c.verify(source["projection_implementation"]))
    points = projection.canonical_query_points(64, "identity", device=torch.device("cpu"))
    factors = FACTORS[arm]; predicted_depth = rank = None
    if prediction is not None:
        with np.load(c.verify(prediction["npz"]), allow_pickle=False) as arrays:
            metadata = json.loads(arrays["metadata_json"].item())
            if (metadata != prediction["metadata"] or arrays["frames"].tolist() != source["selected_frames"]
                    or not np.array_equal(arrays["known_w2c"], np.asarray(source["known_w2c"]))
                    or not np.array_equal(arrays["known_intrinsics"], np.asarray(source["known_intrinsics"]))):
                raise ValueError("Predicted maps/cameras/metadata differ")
            if factors["depth"] == "predicted":
                predicted_depth = torch.from_numpy(arrays["depth_z_world"].copy())
            if factors["confidence"] == "predicted":
                rank = torch.from_numpy(arrays["rank_conf"].copy())
    shape = (len(source["selected_frames"]), 518, 518)
    if factors["depth"] == "predicted":
        depth = predicted_depth
        if depth.shape != shape or not torch.isfinite(depth).all() or torch.any(depth <= 0):
            raise ValueError("Historical predicted depth must remain finite and positive")
        valid = torch.isfinite(depth) & (depth > 0)
    else:
        depths, valids = [], []
        for row in observations:
            with np.load(c.verify(row["npz"]), allow_pickle=False) as arrays:
                d, v = _stripped(arrays)
            depths.append(torch.from_numpy(d.copy()));valids.append(torch.from_numpy(v.copy()))
        depth, valid = torch.stack(depths), torch.stack(valids)
    if depth.dtype != torch.float32 or depth.shape != shape or (rank is not None and rank.shape != shape):
        raise ValueError("Unexpected frozen observation shape/dtype")
    from PIL import Image
    masks = []
    for row in source["selected_masks"]:
        with Image.open(c.verify(row)) as image:
            if image.size != (2072, 2072):raise ValueError("Expected original 2072 visible masks")
            masks.append(torch.from_numpy((np.asarray(image.convert("L").resize(
                (518, 518), Image.Resampling.NEAREST)) > 127).copy()))
    result = project_factor(operator, points, source["canonical_from_world"],
        torch.tensor(source["known_w2c"], dtype=torch.float64),
        torch.tensor(source["known_intrinsics"], dtype=torch.float64), depth, valid, rank, torch.stack(masks))
    delta, confidence, target, frustum, scale, invalid_mass = result
    fine, fine_valid = operator.fine_evidence(delta, confidence, target, frustum,
        tau=source["numeric_policy"]["surface_tolerance_canonical"],
        minimum_rank=source["numeric_policy"]["confidence_minimum_rank"])
    value = operator._validate_tensors(operator.pool_xyz(fine, fine_valid))
    diagnostics = {"frustum_coverage_fraction": float(frustum.float().mean()),
        "invalid_interpolation_fraction_in_frustum": float((frustum & (invalid_mass > 0)).float().sum()/frustum.sum().clamp_min(1)),
        "evidence_confident_fraction": float((frustum & (confidence >= .3)).float().mean()),
        "any_frustum_point_fraction": float(fine_valid.float().mean()),
        "surface_mean": float(fine[:, 0].mean()), "free_mean": float(fine[:, 2].mean()),
        "unknown_mean": float(fine[:, 4].mean()), "conflict_mean": float(fine[:, 10].mean())}
    return value, source, scale, diagnostics


def tensor_receipt(value):
    value = value.detach().contiguous().cpu()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def _diagonal_contract(c):
    return {"schema": DIAGONAL_SCHEMA, "plan_identity": c.plan()["identity"],
        "implementation": c.binding(__file__), "operator_implementation": c.binding(OPERATOR_PATH),
        "projection_implementation": c.binding(Path(c.PACKAGE)/"projection.py"),
        "torch_version": str(torch.__version__), "comparison": "torch.equal features/weights/valid; no tolerance"}


def diagonal_receipt(*, verify_inputs=False):
    c = _common();p = c.plan();path = Path(c.OUT)/"evidence/diagonal_check.json"
    bound = c.binding(path);record = c.read(c.verify(bound))
    if (record.get("source_contract") != _diagonal_contract(c) or record.get("passed") is not True
            or record.get("diagonal_evidence_replay_exact") is not True or len(record["checks"]) != 8):
        raise ValueError("Missing/stale exact diagonal evidence replay")
    seen = set()
    for row in record["checks"]:
        uid, arm = row["uid"], row["arm"]
        if (uid not in UIDS or arm not in DIAGONALS or (uid, arm) in seen
                or row.get("all_tensors_exact") is not True
                or row["reference_receipt"] != p["plan"]["reference_evidence"][arm][uid]):
            raise ValueError("Incomplete/changed diagonal source")
        seen.add((uid, arm))
        # Full replay-source validation is a preparation concern. In training,
        # the exact receipt itself is bound by each cross evidence artifact;
        # the predicted/unit path must not open GT observation caches even as
        # an incidental diagonal-source validation side effect.
        if verify_inputs:
            reference = c.read(c.verify(row["reference_receipt"]))
            if row["reference_tensors"] != reference["tensors"]:raise ValueError("Diagonal cached tensor source changed")
            c.verify(row["reference_tensors"])
            obj = next(x for x in p["plan"]["objects"] if x["uid"] == uid)
            if row["sources"] != _inputs(obj, arm)[1]:raise ValueError("Diagonal observations changed")
    return bound


def check_diagonals():
    c = _common();p = c.plan();path = Path(c.OUT)/"evidence/diagonal_check.json"
    if path.exists():return diagonal_receipt(verify_inputs=True)
    checks = []
    for arm in DIAGONALS:
        for obj in p["plan"]["objects"]:
            value, source, _, _ = _calculate(obj, arm)
            rb = p["plan"]["reference_evidence"][arm][obj["uid"]]
            row = c.read(c.verify(rb));old_source = row["sources"]
            if (row.get("success") is not True or row["uid"] != obj["uid"]
                    or row["feature_names"] != FEATURE_NAMES
                    or any(old_source[key] != source[key] for key in ("selected_frames", "visible_indices",
                        "canonical_from_world", "known_w2c", "known_intrinsics", "selected_masks"))):
                raise ValueError("Reference diagonal observation contract differs")
            expected = _operator(c)._validate_tensors(torch.load(c.verify(row["tensors"]),
                map_location="cpu", weights_only=True))
            if not all(torch.equal(value[key], expected[key]) for key in ("features", "weights", "valid")):
                raise ValueError("Factorial diagonal is not bit-exact: " + arm + "/" + obj["uid"])
            checks.append({"uid": obj["uid"], "arm": arm, "sources": source,
                "reference_receipt": rb, "reference_tensors": row["tensors"], "all_tensors_exact": True,
                "tensors": {key: tensor_receipt(value[key]) for key in ("features", "weights", "valid")}})
            c.event("diagonal_evidence_exact", arm=arm, uid=obj["uid"])
    c.write(path, {"schema": DIAGONAL_SCHEMA, "success": True, "passed": True,
        "plan_identity": p["identity"], "source_contract": _diagonal_contract(c),
        "diagonal_evidence_replay_exact": True, "checks": checks, "gpu_initialized": torch.cuda.is_initialized()})
    return diagonal_receipt(verify_inputs=True)


def load_evidence(obj, arm):
    if arm not in ARMS:raise ValueError("Training only loads the two new cross cells")
    c, source, _, _ = _inputs(obj, arm)
    rb = c.binding(Path(c.OUT)/"evidence"/arm/obj["uid"]/"receipt.json")
    row = c.read(c.verify(rb))
    if (row.get("schema") != SCHEMA or row.get("success") is not True or row["sources"] != source
            or row["feature_names"] != FEATURE_NAMES or row["hidden_geometry_used"] is not False
            or row["diagonal_check"] != diagonal_receipt()):
        raise ValueError("Stale or incompatible cross-cell evidence cache")
    value = torch.load(c.verify(row["tensors"]), map_location="cpu", weights_only=True)
    return _operator(c)._validate_tensors(value), rb


def build_evidence(obj, arm, diagonal):
    c = _common(); path = Path(c.OUT)/"evidence"/arm/obj["uid"]/"receipt.json"
    if path.exists():return load_evidence(obj, arm)[1]
    value, source, scale, diagnostics = _calculate(obj, arm)
    tensor = Path(c.OUT)/"evidence"/arm/(obj["uid"]+".pt")
    tensor.parent.mkdir(parents=True, exist_ok=True)
    temporary = tensor.with_name(tensor.name+f".{os.getpid()}.tmp")
    torch.save(value, temporary);temporary.replace(tensor)
    c.write(path, {"schema": SCHEMA, "success": True, "uid": obj["uid"], "arm": arm,
        "sources": source, "tensors": c.binding(tensor), "feature_names": FEATURE_NAMES,
        "canonical_scale": scale, "diagonal_check": diagonal,
        "statistics": _operator(c)._statistics(value["features"], value["valid"]),
        "fine_diagnostics": diagnostics, "visible_gt_geometry_used": FACTORS[arm]["depth"] == "gt",
        "hidden_geometry_used": False, "confidence_is_unit": FACTORS[arm]["confidence"] == "unit",
        "hard_carving": False, "scope": "matched depth/confidence cross cell; unchanged soft geometry operator"})
    rb = load_evidence(obj, arm)[1]
    c.event("cross_evidence_complete", arm=arm, uid=obj["uid"])
    return rb


def run():
    c = _common();torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
    p = c.plan();_policy(p);diagonal = check_diagonals()
    receipts = [build_evidence(obj, arm, diagonal) for arm in ARMS for obj in p["plan"]["objects"]]
    if torch.cuda.is_initialized():raise ValueError("Evidence generation must stay on CPU")
    c.write(Path(c.OUT)/"evidence/complete.json", {"schema": SCHEMA, "success": True,
        "plan_identity": p["identity"], "arms": list(ARMS), "objects": receipts,
        "diagonal_check": diagonal, "diagonal_evidence_replay_exact": True,
        "hidden_geometry_used": False, "hard_carving": False, "gpu_initialized": False})
    c.event("cross_evidence_all_complete", cells=2, objects_per_cell=4, diagonal_replays=8)


def self_test():
    torch.set_num_threads(1);operator = _operator();operator.self_test()
    old_oracle = _module("_depth_confidence_old_oracle", ORACLE_PATH)
    eye = torch.eye(4, dtype=torch.float64);k = torch.eye(3, dtype=torch.float64)[None]
    depth = torch.full((1, 3, 3), 2.);valid = torch.ones_like(depth, dtype=torch.bool)
    masks = torch.ones_like(valid);rank = torch.linspace(0, 1, 9).reshape(1, 3, 3)
    points = torch.tensor([[0., 0., 2.], [1., 1., 2.], [4., 4., 2.]])
    old = operator.project_observations(points, eye, eye[None], k, depth, rank, masks)
    got = project_factor(operator, points, eye, eye[None], k, depth, valid, rank, masks)
    assert all(torch.equal(old[i], got[i]) for i in range(4)) and old[4] == got[4]
    oracle = old_oracle.project_visible_oracle(operator, points, eye, eye[None], k, depth, valid, masks)
    unit = project_factor(operator, points, eye, eye[None], k, depth, valid, None, masks)
    assert all(torch.equal(oracle[i], unit[i]) for i in (0, 1, 2, 3, 5)) and oracle[4] == unit[4]
    # Each intervention leaves the other factor and all projection geometry
    # unchanged when both observed depths are valid.
    deeper = project_factor(operator, points, eye, eye[None], k, depth+.2, valid, rank, masks)
    assert not torch.equal(deeper[0], got[0]) and torch.equal(deeper[1], got[1])
    assert all(torch.equal(deeper[i], got[i]) for i in (2, 3, 5))
    assert torch.equal(unit[0], got[0]) and not torch.equal(unit[1], got[1])
    valid[0, 1, 1] = False;depth[0, 1, 1] = 0
    for confidence in (rank, None):
        result = project_factor(operator, points, eye, eye[None], k, depth, valid, confidence, masks)
        assert result[1][0, 1] == 0 and result[5][0, 1] == .25
        fine, _ = operator.fine_evidence(*result[:4])
        assert fine[1, 0] == fine[1, 2] == 0 and fine[1, 4] == 1
    empty = torch.zeros_like(valid)
    for confidence in (rank, None):
        result = project_factor(operator, points*.1, eye, eye[None], k, torch.zeros_like(depth), empty, confidence, masks)
        fine, _ = operator.fine_evidence(*result[:4])
        assert torch.all(fine[:, 0] == 0) and torch.all(fine[:, 2] == 0) and torch.all(fine[:, 4] == 1)
    class OriginalArchive:
        files = ["depth_z_world", "instance_id"]
        def __getitem__(self, key):raise AssertionError("Unstripped archive was accessed")
    try:_stripped(OriginalArchive())
    except ValueError:pass
    else:raise AssertionError("Original GT archive accepted")
    assert selected_frames({"uid": UIDS[-1], "visible_indices": [0, 3]}) == [0, 11]
    try:selected_frames({"uid": UIDS[-1], "visible_indices": [0, 1, 2, 3]})
    except ValueError:pass
    else:raise AssertionError("Stool gained extra views")
    # Exercise the actual source-selection path with a forbidden GT mapping.
    # Looking at GT receipts or validity from the predicted/unit path fails.
    from types import SimpleNamespace
    class ForbiddenGT:
        def __getitem__(self, key):raise AssertionError("Predicted/unit opened a GT observation")
    def fake_binding(path):return {"path": str(path), "sha256": "0"*64}
    def fake_digest(value):return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    shared_binding = fake_binding("shared.json")
    objects = [{"uid": uid, "visible_indices": [0, 3] if uid == UIDS[-1] else [0, 1, 2, 3],
                "canonical_from_world": eye.tolist()} for uid in UIDS]
    numeric = {"features": FEATURE_NAMES, "fine_side": 64, "token_side": 16,
        "surface_tolerance_canonical": 2/64, "confidence_minimum_rank": .3,
        "full_visible_masks": {uid: [{"frame": frame, **fake_binding(f"{uid}/{frame}.png")}
            for frame in GROUPS["group4"]] for uid in UIDS}}
    fake_plan = {"identity": "test", "plan": {"objects": objects, "shared_plan": shared_binding,
        "evidence_policy": numeric, "depth_confidence_policy": {"arms": FACTORS, "groups": GROUPS,
            "invalid_depth_policy": "unknown", "allow_hidden_geometry": False},
        "predicted_depth": {group: fake_binding(group+".json") for group in GROUPS},
        "stripped_observations": ForbiddenGT()}}
    rows = {"shared.json": {"plan": {"world_w2c": [eye.tolist()]*12,
        "intrinsics": [torch.eye(3, dtype=torch.float64).tolist()]*12}}}
    for group, frames in GROUPS.items():
        contract = {"shared_plan": shared_binding, "depth_policy": {"groups": GROUPS,
            "confidence": {"kind": "per_view_empirical_cdf", "minimum_rank": .3}}}
        rows[group+".json"] = {"success": True, "group": group, "frames": frames,
            "source_contract": contract, "npz": fake_binding(group+".npz"),
            "metadata": {"gt_geometry_used": False, "source_contract_sha256": fake_digest(contract)}}
    reads = []
    def fake_read(path):
        reads.append(str(path));return rows[str(path)]
    fake = SimpleNamespace(plan=lambda: fake_plan, FEATURES=FEATURE_NAMES, PACKAGE=Path("/not_read"),
        binding=fake_binding, verify=lambda b: b["path"], read=fake_read, digest=fake_digest)
    original_common = globals()["_common"]
    try:
        globals()["_common"] = lambda: fake
        for obj in objects:
            _, source, prediction, gt = _inputs(obj, ARMS[1])
            assert gt == [] and source["stripped_receipts"] == source["stripped_npzs"] == []
            assert prediction is not None and not source["visible_gt_geometry_used"]
        assert set(reads) == {"shared.json", "group4.json", "group2.json"}
    finally:
        globals()["_common"] = original_common
    assert not torch.cuda.is_initialized()
    report = {"success": True, "test": "depth_confidence_cpu", "gpu_initialized": False,
        "checks": ["predicted_diagonal_projection_exact", "unit_diagonal_projection_exact",
            "invalid_bilinear_contribution_unknown_both_confidences", "no_hit_unknown",
            "unstripped_gt_archive_rejected", "stool_frame_subset", "factors_change_independently",
            "predicted_unit_source_path_never_reads_gt", "old_surface_free_unknown_xyz_semantics"]}
    print(json.dumps(report));return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="build", choices=("build", "self-test"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test or args.action == "self-test":self_test()
    else:run()
