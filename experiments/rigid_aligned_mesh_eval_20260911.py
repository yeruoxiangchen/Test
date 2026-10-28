"""Recompute native and rigid-aligned mesh metrics without scale fitting.

The alignment is a diagnostic SE(3) registration: proper rotations and
translation are allowed, while reflection, scale, deformation, component
filtering, and mesh refinement are forbidden.  Existing metric files are read
for identity checks and are never overwritten.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pose_aligned_layered_reconstruction.mesh_metrics import sample_surface, surface_metrics


PALR = ROOT / "pose_aligned_layered_reconstruction"
YCB = PALR / "outputs/ycbv_drill_occlusion_20260910"
YCB_ADAPTED = YCB / "camera_adapted_lari"
DEV64 = PALR / "outputs/mesh_comparison_20260910/dev64_v2"
VERSION = "palr.se3_aligned_surface_metrics.v1"
THRESHOLDS = (0.01, 0.02, 0.05)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def proper_axis_transforms(source: np.ndarray, target: np.ndarray) -> list[np.ndarray]:
    """Return identity plus all proper PCA-axis permutation/sign hypotheses."""

    def basis(points: np.ndarray) -> np.ndarray:
        centered = points - points.mean(axis=0)
        values, vectors = np.linalg.eigh(centered.T @ centered)
        return vectors[:, np.argsort(values)[::-1]]

    source_basis, target_basis = basis(source), basis(target)
    rotations = [np.eye(3)]
    for permutation in itertools.permutations(range(3)):
        permutation_matrix = np.eye(3)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            signed = permutation_matrix @ np.diag(signs)
            rotation = target_basis @ signed @ source_basis.T
            if np.linalg.det(rotation) > 0.999999:
                rotations.append(rotation)
    unique = []
    for rotation in rotations:
        if not any(np.max(np.abs(rotation - old)) < 1.0e-7 for old in unique):
            unique.append(rotation)
    return unique


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    covariance = (source - source_mean).T @ (target - target_mean)
    left, _, right = np.linalg.svd(covariance)
    row_rotation = left @ right
    if np.linalg.det(row_rotation) < 0:
        left[:, -1] *= -1
        row_rotation = left @ right
    translation = target_mean - source_mean @ row_rotation
    return row_rotation, translation


def symmetric_chamfer(source: np.ndarray, target: np.ndarray) -> float:
    source_tree, target_tree = cKDTree(source), cKDTree(target)
    return float(
        0.5
        * (
            target_tree.query(source, workers=1)[0].mean()
            + source_tree.query(target, workers=1)[0].mean()
        )
    )


def symmetric_icp(
    source: np.ndarray,
    target: np.ndarray,
    initial_rotation: np.ndarray,
    *,
    iterations: int = 35,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit one global proper SE(3) transform using bidirectional matches."""

    row_rotation = np.asarray(initial_rotation, np.float64).T
    translation = target.mean(axis=0) - source.mean(axis=0) @ row_rotation
    target_tree = cKDTree(target)
    previous = math.inf
    for _ in range(int(iterations)):
        transformed = source @ row_rotation + translation
        forward = target_tree.query(transformed, workers=1)[1]
        reverse = cKDTree(transformed).query(target, workers=1)[1]
        paired_source = np.concatenate((source, source[reverse]), axis=0)
        paired_target = np.concatenate((target[forward], target), axis=0)
        row_rotation, translation = kabsch(paired_source, paired_target)
        transformed = source @ row_rotation + translation
        score = symmetric_chamfer(transformed, target)
        if previous - score < 1.0e-7:
            break
        previous = score
    return row_rotation, translation, score


def fit_se3(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, int]:
    best = None
    hypotheses = proper_axis_transforms(source, target)
    for rotation in hypotheses:
        row_rotation, translation, score = symmetric_icp(source, target, rotation)
        if best is None or score < best[2]:
            best = (row_rotation, translation, score)
    assert best is not None
    row_rotation, translation, score = best
    transform = np.eye(4)
    transform[:3, :3] = row_rotation.T
    transform[:3, 3] = translation
    return transform, float(score), len(hypotheses)


def rotation_angle_degrees(transform: np.ndarray) -> float:
    cosine = np.clip((np.trace(transform[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def keep_identity_if_better(record: dict[str, Any]) -> dict[str, Any]:
    """The identity is a valid SE(3) candidate and must remain available."""
    if record["se3_aligned"]["chamfer_l1"] <= record["native"]["chamfer_l1"]:
        record["alignment"].setdefault("selected_identity_on_surface_metric", False)
        return record
    attempted = dict(record["alignment"])
    record["se3_aligned"] = dict(record["native"])
    record["alignment"] = {
        "matrix_predicted_to_target": np.eye(4).tolist(),
        "rotation_degrees": 0.0,
        "translation_norm": 0.0,
        "registration_chamfer": attempted["registration_chamfer"],
        "initial_hypotheses": attempted["initial_hypotheses"],
        "scale_fitted": False,
        "reflection_allowed": False,
        "selected_identity_on_surface_metric": True,
        "rejected_registration": attempted,
    }
    return record


def evaluate_case(case: dict[str, Any], registration_points: int, surface_points: int) -> dict[str, Any]:
    mesh_path, target_path = Path(case["mesh"]), Path(case["target"])
    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    target = trimesh.load(target_path, process=False, force="mesh")
    native = surface_metrics(
        mesh,
        target,
        count=surface_points,
        seed=case["surface_seed"],
        thresholds=THRESHOLDS,
    )
    source_points, _ = sample_surface(mesh, registration_points, case["registration_seed"])
    target_points, _ = sample_surface(target, registration_points, case["registration_seed"] + 1)
    transform, registration_cd, starts = fit_se3(source_points, target_points)
    aligned_mesh = mesh.copy()
    aligned_mesh.apply_transform(transform)
    aligned = surface_metrics(
        aligned_mesh,
        target,
        count=surface_points,
        seed=case["surface_seed"],
        thresholds=THRESHOLDS,
    )
    expected = case.get("expected_native")
    if expected is not None:
        discrepancy = max(abs(native[key] - expected[key]) for key in native)
        if discrepancy > 2.0e-6:
            raise RuntimeError(f"native metric reproduction failed for {case['id']}: {discrepancy}")
    return keep_identity_if_better({
        **case,
        "format": VERSION,
        "mesh_sha256": sha256(mesh_path),
        "target_sha256": sha256(target_path),
        "native": native,
        "se3_aligned": aligned,
        "alignment": {
            "matrix_predicted_to_target": transform.tolist(),
            "rotation_degrees": rotation_angle_degrees(transform),
            "translation_norm": float(np.linalg.norm(transform[:3, 3])),
            "registration_chamfer": registration_cd,
            "initial_hypotheses": starts,
            "scale_fitted": False,
            "reflection_allowed": False,
        },
    })


def ycb_cases() -> list[dict[str, Any]]:
    target = YCB / "gt_drill_canonical.ply"
    cases = []
    for group_index, group in enumerate(("A_light_8", "B_occluded_2_of_8", "C_occluded_8_of_8")):
        for seed in (42, 43):
            parent_metrics = json.loads((YCB / group / f"seed_{seed}" / "metrics.json").read_text())
            candidate_metrics = json.loads(
                (YCB_ADAPTED / group / f"seed_{seed}" / "full_raw/metrics.json").read_text()
            )
            for mode, mesh, expected in (
                ("parent", YCB / group / f"seed_{seed}/mesh_canonical.ply", parent_metrics["surface"]),
                (
                    "candidate",
                    YCB_ADAPTED / group / f"seed_{seed}/full_raw/mesh_canonical.ply",
                    candidate_metrics["surface"],
                ),
            ):
                cases.append(
                    {
                        "scope": "ycbv_drill",
                        "id": f"{group}_{seed}_{mode}",
                        "group": group,
                        "seed": seed,
                        "mode": mode,
                        "mesh": str(mesh.resolve()),
                        "target": str(target.resolve()),
                        "surface_seed": seed,
                        "registration_seed": 20260911 + group_index * 1009 + seed * 17 + (mode == "candidate"),
                        "expected_native": expected,
                    }
                )
    return cases


def dev64_cases() -> list[dict[str, Any]]:
    cases = []
    for path in sorted((DEV64 / "records").glob("*.json")):
        old = json.loads(path.read_text())
        index, seed, mode = int(old["index"]), int(old["seed"]), str(old["mode"])
        cases.append(
            {
                "scope": "formal_dev64",
                "id": path.stem,
                "index": index,
                "uid": old["uid"],
                "seed": seed,
                "mode": mode,
                "mesh": old["mesh"]["path"],
                "target": old["target_mesh"]["path"],
                "surface_seed": seed * 1009 + index * 9173,
                "registration_seed": 20260911 + index * 1009 + seed * 17 + (mode == "candidate"),
                "expected_native": old["surface"],
            }
        )
    if len(cases) != 256:
        raise RuntimeError(f"expected 256 Dev64 cases, found {len(cases)}")
    return cases


def mean_metrics(records: list[dict[str, Any]], field: str, mode: str) -> dict[str, float]:
    selected = [record[field] for record in records if record["mode"] == mode]
    return {key: float(np.mean([row[key] for row in selected])) for key in selected[0]}


def aggregate(records: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    modes = ("parent", "candidate")
    result: dict[str, Any] = {
        "format": VERSION,
        "scope": scope,
        "record_count": len(records),
        "protocol": {
            "alignment": "proper SE(3), multi-start symmetric ICP",
            "scale": "fixed; no fitted scale",
            "reflection": False,
            "surface_samples": 20000,
            "thresholds": list(THRESHOLDS),
        },
        "summary": {
            field: {mode: mean_metrics(records, field, mode) for mode in modes}
            for field in ("native", "se3_aligned")
        },
    }
    result["alignment"] = {
        mode: {
            "rotation_degrees_mean": float(
                np.mean([r["alignment"]["rotation_degrees"] for r in records if r["mode"] == mode])
            ),
            "rotation_degrees_median": float(
                np.median([r["alignment"]["rotation_degrees"] for r in records if r["mode"] == mode])
            ),
            "translation_norm_mean": float(
                np.mean([r["alignment"]["translation_norm"] for r in records if r["mode"] == mode])
            ),
        }
        for mode in modes
    }
    grouping = "group" if scope == "ycbv_drill" else "index"
    paired = {}
    for field in ("native", "se3_aligned"):
        by_key = {(r[grouping], r["seed"], r["mode"]): r for r in records}
        units = sorted({r[grouping] for r in records})
        keys = tuple(records[0][field])
        paired[field] = {}
        for metric in keys:
            sign = -1.0 if metric in {"pred_to_gt_mean", "gt_to_pred_mean", "chamfer_l1", "chamfer_l2"} else 1.0
            gains = []
            for unit in units:
                values = []
                for seed in (42, 43):
                    parent = by_key[unit, seed, "parent"][field][metric]
                    candidate = by_key[unit, seed, "candidate"][field][metric]
                    values.append(sign * (candidate - parent))
                gains.append(float(np.mean(values)))
            paired[field][metric] = {
                "mean_improvement": float(np.mean(gains)),
                "win_units": int(np.sum(np.asarray(gains) > 1.0e-12)),
                "loss_units": int(np.sum(np.asarray(gains) < -1.0e-12)),
                "tie_units": int(np.sum(np.abs(gains) <= 1.0e-12)),
            }
    result["paired"] = paired
    if scope == "ycbv_drill":
        result["groups"] = {
            group: {
                field: {
                    mode: {
                        key: float(
                            np.mean(
                                [
                                    r[field][key]
                                    for r in records
                                    if r["group"] == group and r["mode"] == mode
                                ]
                            )
                        )
                        for key in records[0][field]
                    }
                    for mode in modes
                }
                for field in ("native", "se3_aligned")
            }
            for group in ("A_light_8", "B_occluded_2_of_8", "C_occluded_8_of_8")
        }
    return result


def run(scope: str, workers: int, registration_points: int, surface_points: int) -> None:
    if scope == "ycbv":
        cases, output = ycb_cases(), YCB_ADAPTED / "rigid_aligned_metrics_v1"
        scope_name = "ycbv_drill"
    elif scope == "dev64":
        cases, output = dev64_cases(), DEV64 / "rigid_aligned_metrics_v1"
        scope_name = "formal_dev64"
    else:
        raise ValueError(scope)
    record_dir = output / "records"
    record_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    pending = []
    for case in cases:
        path = record_dir / f"{case['id']}.json"
        if path.exists():
            value = json.loads(path.read_text())
            if value.get("format") != VERSION:
                raise RuntimeError(f"incompatible existing record: {path}")
            before = value["se3_aligned"]["chamfer_l1"]
            value = keep_identity_if_better(value)
            if value["se3_aligned"]["chamfer_l1"] != before:
                dump(path, value)
            completed.append(value)
        else:
            pending.append(case)
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_case = {
                pool.submit(evaluate_case, case, registration_points, surface_points): case
                for case in pending
            }
            for number, future in enumerate(as_completed(future_to_case), start=1):
                result = future.result()
                dump(record_dir / f"{result['id']}.json", result)
                completed.append(result)
                print(
                    json.dumps(
                        {
                            "scope": scope_name,
                            "completed": len(completed),
                            "total": len(cases),
                            "id": result["id"],
                            "native_cd": result["native"]["chamfer_l1"],
                            "aligned_cd": result["se3_aligned"]["chamfer_l1"],
                            "rotation_degrees": result["alignment"]["rotation_degrees"],
                        }
                    ),
                    flush=True,
                )
    if len(completed) != len(cases):
        raise RuntimeError("incomplete aligned-metric records")
    completed.sort(key=lambda row: row["id"])
    dump(output / "aggregate.json", aggregate(completed, scope_name))
    print(output / "aggregate.json", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", choices=("ycbv", "dev64", "all"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--registration_points", type=int, default=4096)
    parser.add_argument("--surface_points", type=int, default=20000)
    args = parser.parse_args()
    if min(args.workers, args.registration_points, args.surface_points) <= 0:
        raise ValueError("worker and point counts must be positive")
    scopes = ("ycbv", "dev64") if args.scope == "all" else (args.scope,)
    for scope in scopes:
        run(scope, args.workers, args.registration_points, args.surface_points)


if __name__ == "__main__":
    main()
