"""Audited 2 x 2 SS-support/SLat-condition contrasts and native-frame galleries.

Importing this module does not load a model or render anything.  ``main`` runs
only after every logical cell and the current replay smoke have been verified.
The pure statistics helpers also serve the CPU-only protocol self-check.
"""
from __future__ import annotations

import argparse
import math
from itertools import product
from pathlib import Path
from statistics import fmean

from .common import (BASE_OUT, OUT, SHARED, binding, check_smoke, event,
                     freeze_runtime, get_result, logical_cases, now, plan,
                     read, verify, write)


LEVELS = ("518", "2072")
SEEDS = (42, 43, 44)
CELLS = tuple(product(LEVELS, repeat=2))
METRICS = ("chamfer_l1", "fscore_0p02", "ss_iou", "visible_recall",
           "occluded_recall", "free_space_violation")
PARTITIONS = ("visible", "externally_occluded", "other_unobserved")
EXPECTED_PARTITIONS = {
    "hssd_102344049_i075": dict(zip(PARTITIONS, (5247, 9477, 5276))),
    "hssd_102344049_i065": dict(zip(PARTITIONS, (5691, 14309, 0))),
}
DIRECTIONS = {
    "chamfer_l1": "lower", "fscore_0p02": "higher", "ss_iou": "higher",
    "visible_recall": "higher", "occluded_recall": "higher",
    "free_space_violation": "lower",
}
CONTRASTS = ("condition_at_support518", "condition_at_support2072",
             "support_at_condition518", "support_at_condition2072",
             "interaction")
SCOPE = (
    "事后诊断：appearance_control 的两个旧训练对象（table i075、stool i065），"
    "同一旧场景，三个 sampling seed（42/43/44）。种子不是独立物体或场景重复。"
    "这是 SS 支持与 SLat 条件的受控交叉干预，但两轴改变的是 representation；"
    "SS 支持同时改变数量、几何和 attention 拓扑，SLat 条件同时替换全局与投影 DINO，"
    "不能将效果单独归因于其中一路。"
    "518/2072 输入还涉及重渲染、抗锯齿和采样差异，不是纯分辨率因果实验。"
    "本结果不是训练效果、未见对象/场景泛化或 LaRI 贡献的证据。"
)


def cell_name(support, condition):
    return f"ss{support}_cond{condition}"


def metric_values(mesh, ss):
    part = mesh["partitioned"]
    values = {
        "chamfer_l1": mesh["surface"]["chamfer_l1"],
        "fscore_0p02": mesh["surface"]["fscore_0p02"],
        "ss_iou": ss["occupancy"]["iou"],
        "visible_recall": part["visible"]["recall_0p02"],
        "occluded_recall": part["externally_occluded"]["recall_0p02"],
        "free_space_violation": part["predicted_surface_free_space_violation_fraction"],
    }
    for key, value in values.items():
        if value is None and key in ("visible_recall", "occluded_recall"):
            continue
        if value is None or not math.isfinite(value):
            raise ValueError(f"Missing or nonfinite metric: {key}")
    return values


def averages(rows):
    """Do not silently change denominators when a GT stratum is empty."""
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot average an empty comparison")
    return {key: (None if any(row[key] is None for row in rows)
                  else fmean(row[key] for row in rows)) for key in METRICS}


def factorial_contrasts(cells):
    """Raw high-minus-low differences; L=518, H=2072, first axis=support.

    ``cells`` maps cell_name(support, condition) to a metric dictionary.  The
    interaction is HH - HL - LH + LL, with no sign reversal for CD.
    """
    expected = {cell_name(*cell) for cell in CELLS}
    if set(cells) != expected:
        raise ValueError("A complete, unique 2 x 2 table is required")
    ll, lh, hl, hh = (cells[cell_name(*cell)] for cell in CELLS)

    def subtract(a, b):
        return {key: None if a[key] is None or b[key] is None
                else a[key] - b[key] for key in METRICS}

    return {
        "condition_at_support518": subtract(lh, ll),
        "condition_at_support2072": subtract(hh, hl),
        "support_at_condition518": subtract(hl, ll),
        "support_at_condition2072": subtract(hh, lh),
        "interaction": {
            key: (None if any(cell[key] is None for cell in (ll, lh, hl, hh))
                  else hh[key] - hl[key] - lh[key] + ll[key])
            for key in METRICS
        },
    }


def summarize_statistics(rows, object_ids=None):
    """Paired seed effects, object means, and equal-weight object macro means."""
    rows = list(rows)
    object_ids = list(object_ids) if object_ids is not None else sorted({r["uid"] for r in rows})
    if len(object_ids) != 2 or len(set(object_ids)) != 2:
        raise ValueError("This diagnostic requires exactly two distinct objects")
    indexed = {}
    for row in rows:
        key = (row["uid"], row["seed"], row["support_level"], row["condition_level"])
        if key in indexed:
            raise ValueError(f"Duplicate logical cell: {key}")
        indexed[key] = row
    expected = set(product(object_ids, SEEDS, LEVELS, LEVELS))
    if set(indexed) != expected:
        raise ValueError("Expected exactly 2 objects x 3 seeds x 4 cells")

    per_seed, per_object = {}, {}
    for uid in object_ids:
        per_seed[uid] = {}
        for seed in SEEDS:
            cells = {cell_name(s, c): indexed[uid, seed, s, c]["metrics"] for s, c in CELLS}
            for support in LEVELS:
                if cells[cell_name(support, "518")]["ss_iou"] != cells[cell_name(support, "2072")]["ss_iou"]:
                    raise ValueError("SS IoU changed with condition while support was fixed")
            per_seed[uid][str(seed)] = {"cells": cells, "contrasts": factorial_contrasts(cells)}
        records = list(per_seed[uid].values())
        per_object[uid] = {
            "sampling_seeds": list(SEEDS),
            "cells": {name: averages(r["cells"][name] for r in records)
                      for name in (cell_name(*cell) for cell in CELLS)},
            "contrasts": {name: averages(r["contrasts"][name] for r in records)
                          for name in CONTRASTS},
        }
    macro = {
        "objects": 2, "sampling_seeds_per_object": 3, "scenes": 1,
        "weighting": "mean of the two object means; each object has three paired sampling seeds",
        "cells": {name: averages(r["cells"][name] for r in per_object.values())
                  for name in (cell_name(*cell) for cell in CELLS)},
        "contrasts": {name: averages(r["contrasts"][name] for r in per_object.values())
                      for name in CONTRASTS},
    }
    return {"per_seed": per_seed, "per_object": per_object, "macro": macro,
            "metric_directions": DIRECTIONS,
            "contrast_convention": "raw high(2072)-low(518); interaction=HH-HL-LH+LL; first axis=SS support",
            "ss_iou_role": "support diagnostic only; no SLat-condition gain; identical for each fixed-support pair",
            "null_policy": "any unavailable constituent propagates null; no available-case denominator changes"}


def self_check_statistics():
    """CPU-only algebra and pairing checks; no filesystem or model access."""
    rows = []
    for obj_index, uid in enumerate(("table", "stool")):
        for seed in SEEDS:
            for support, condition in CELLS:
                s, c = int(support == "2072"), int(condition == "2072")
                value = obj_index * 10 + seed + 2 * s + 3 * c + 5 * s * c
                metrics = {key: float(value) for key in METRICS}
                metrics["ss_iou"] = float(s)
                rows.append({"uid": uid, "seed": seed, "support_level": support,
                             "condition_level": condition, "metrics": metrics})
    result = summarize_statistics(rows)
    expected = dict(zip(CONTRASTS, (3., 8., 2., 7., 5.)))
    for name, value in expected.items():
        if result["macro"]["contrasts"][name]["chamfer_l1"] != value:
            raise AssertionError(f"Wrong 2 x 2 contrast: {name}")
    for invalid in (rows[:-1], rows + [rows[0]]):
        try:
            summarize_statistics(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete or duplicate cells accepted")
    changed = [dict(row, metrics=dict(row["metrics"])) for row in rows]
    changed[1]["metrics"]["ss_iou"] += .1
    try:
        summarize_statistics(changed)
    except ValueError:
        pass
    else:
        raise AssertionError("Condition-dependent SS IoU accepted")
    return {"passed": True, "interaction": 5., "missing_and_duplicate_rejected": True,
            "fixed_support_ss_iou_checked": True}


def _same_binding(left, right, label):
    if (Path(left["path"]).resolve(), left["sha256"]) != (Path(right["path"]).resolve(), right["sha256"]):
        raise ValueError(f"Binding mismatch: {label}")


def collect_rows(p, identity):
    """Fail closed on mixed identities, source/seed mismatches or partial runs."""
    objects = {obj["uid"]: obj for obj in p["objects"]}
    cases = logical_cases()
    if cases != p["cases"]:
        raise ValueError("Logical cases differ from the frozen plan")
    rows, counts, supports = [], {}, {}
    for case in cases:
        uid, seed = case["uid"], case["seed"]
        obj = objects[uid]
        support, condition = case["support_level"], case["condition_level"]
        if case["is_reference"] != (support == condition):
            raise ValueError("Only diagonal cells may reuse old mesh records")
        mp, mesh, sp, ss = get_result(case)
        mesh_binding, ss_binding = binding(mp), binding(sp)
        verify(case["support_record"])
        _same_binding(ss_binding, case["support_record"], "planned support record")
        _same_binding(mesh["ss"], ss_binding, "mesh uses the selected SS record")
        for record in (mesh, ss):
            if record["success"] is not True:
                raise ValueError(f"Incomplete result: {uid}/{seed}")
            if record["object_uid"] != uid or record["seed"] != seed or record["arm"] != "appearance_control":
                raise ValueError("Result object, seed or arm differs from planned case")
        if ss["variant"] != f"crop{support}_visible":
            raise ValueError("SS support level differs from its source variant")
        if ss["identity"] != p["base_identity"]:
            raise ValueError("Reused SS does not match the frozen baseline identity")
        condition_source = read(verify(case["condition_record"]))
        if (condition_source["identity"] != p["base_identity"]
                or condition_source["success"] is not True
                or condition_source["object_uid"] != uid
                or condition_source["seed"] != seed
                or condition_source["arm"] != "appearance_control"
                or condition_source["variant"] != f"crop{condition}_visible"):
            raise ValueError("SLat condition source differs from planned object/seed/level")
        verify(condition_source["inputs"]["sample"])
        if mesh["inputs"] != condition_source["inputs"]:
            raise ValueError("Mesh inputs do not match the selected SLat condition")
        if mesh["fitted_transform"] is not False:
            raise ValueError("Evaluation must use the untouched native frame")
        if mesh["calls"] != {"positive_calls": 25, "negative_calls": 19}:
            raise ValueError("Incomplete SLat CFG sampling")
        if mesh["slat_binding"] != p["slat_binding"]:
            raise ValueError("Complete frozen SLat binding differs from the plan")
        if case["is_reference"]:
            verify(case["diagonal_mesh"])
            _same_binding(mesh_binding, case["diagonal_mesh"], "archived diagonal")
            if mesh["identity"] != ss["identity"] or mesh["variant"] != f"crop{support}_visible":
                raise ValueError("Archived diagonal identity/variant differs from SS source")
        else:
            if mesh["identity"] != identity:
                raise ValueError("Cross result is not bound to the current runtime")
            if mesh["case"] != case:
                raise ValueError("Cross result case differs from the full frozen case")
            if (mesh["support_level"], mesh["condition_level"]) != (support, condition):
                raise ValueError("Cross mesh support/condition labels are inconsistent")
            if mesh["condition_inputs"] != condition_source["inputs"]:
                raise ValueError("Cross SLat condition binding differs from the selected source")
            expected_support = {"record": case["support_record"], "coords": ss["coords"],
                                "inputs": ss["inputs"]}
            expected_condition = {"record": case["condition_record"],
                                  "inputs": condition_source["inputs"]}
            if mesh["support_source"] != expected_support or mesh["condition_source"] != expected_condition:
                raise ValueError("Cross mesh support/condition provenance differs from the frozen sources")
            verify(ss["inputs"]["sample"])
            receipt = read(verify(mesh["sampled_slat_receipt"]))
            verify(receipt["latent"])
            if receipt["calls"] != {"positive_calls": 25, "negative_calls": 19}:
                raise ValueError("Sampled latent receipt has incomplete SLat CFG calls")
            if case["original_index"] != obj["original_index"]:
                raise ValueError("Case original object index differs from the frozen object")
            contract = receipt["contract"]
            expected_contract = {
                "identity": identity, "case": case, "support_source": expected_support,
                "condition_source": expected_condition, "slat_binding": p["slat_binding"],
                "original_index": obj["original_index"],
                "master_seed": seed * 2000003 + obj["original_index"] * 2017 + 7919,
            }
            for key, value in expected_contract.items():
                if contract[key] != value:
                    raise ValueError(f"Sampled latent receipt contract differs at {key}")
            voxel_count = ss["occupancy"]["predicted_voxels"]
            if not isinstance(voxel_count, int) or voxel_count <= 0:
                raise ValueError("Invalid SS support voxel count")
            for key, width, dtype in (("initial_coords", 4, "torch.int32"),
                                      ("initial_noise", 8, "torch.float32")):
                tensor = contract[key]
                if tensor["shape"] != [voxel_count, width] or tensor["dtype"] != dtype:
                    raise ValueError(f"Sampled latent receipt {key} does not match the SS support")
                digest = tensor["sha256"]
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError(f"Invalid tensor binding in sampled latent receipt: {key}")
        _same_binding(mesh["gt"], obj["mesh"], "fixed GT mesh")
        if mesh["canonical_from_world"] != obj["canonical_from_world"]:
            raise ValueError("Known canonical/world placement changed")
        for value in (mesh["mesh"], mesh["world_mesh"], mesh["gt"], mesh["ss"], ss["coords"]):
            verify(value)
        partition_counts = {name: mesh["partitioned"][name]["gt_samples"] for name in PARTITIONS}
        if partition_counts != EXPECTED_PARTITIONS[uid]:
            raise ValueError("GT partition differs from the original object-specific counts")
        if sum(partition_counts.values()) != 20000 or any(n < 0 for n in partition_counts.values()):
            raise ValueError("Expected the original 20,000-sample GT partition")
        partition_signature = {"counts": partition_counts,
                               "tolerance": mesh["partitioned"]["visibility_depth_tolerance_canonical"],
                               "scope": mesh["gt_partition_scope"]}
        if counts.setdefault(uid, partition_signature) != partition_signature:
            raise ValueError("GT partition changed across support, condition or seed")
        support_key = (uid, seed, support)
        support_signature = {"record": ss_binding, "coords": ss["coords"],
                             "occupancy": ss["occupancy"], "identity": ss["identity"]}
        if supports.setdefault(support_key, support_signature) != support_signature:
            raise ValueError("Fixed-support pair does not reuse identical SS data")
        rows.append({"uid": uid, "seed": seed, "support_level": support,
                     "condition_level": condition, "cell": cell_name(support, condition),
                     "is_reference": case["is_reference"], "mesh_record": mesh_binding,
                     "ss_record": ss_binding, "condition_record": case["condition_record"],
                     "mesh": mesh, "metrics": metric_values(mesh, ss)})
    if len(rows) != 24 or sum(r["is_reference"] for r in rows) != 12:
        raise ValueError("Expected 24 logical cells: 12 archived diagonals and 12 new crosses")
    return rows, {"passed": True, "logical_cases": 24, "archived_diagonals": 12,
                  "new_cross_meshes": 12, "unique_reused_ss_records": len(supports),
                  "fixed_gt_partitions": counts, "full_slat_calls": {"positive_calls": 25, "negative_calls": 19},
                  "complete_frozen_slat_binding": True, "cross_latent_receipts_verified": 12,
                  "native_frame": True, "fixed_support_ss_identical": True}


def _verify_render(record, expected_mesh):
    _same_binding(record["mesh"], expected_mesh, "render mesh")
    verify(record["mesh"])
    identity_matrix = [[float(i == j) for j in range(4)] for i in range(4)]
    if record["matrix_original_to_rendered"] != identity_matrix:
        raise ValueError("Unexpected fitted or transformed render")
    if [view["view"] for view in record["views"]] != [0, 1, 2]:
        raise ValueError("All three original fixed camera views are required")
    for view in record["views"]:
        verify({"path": view["png"], "sha256": view["png_sha256"]})


def figures(rows, p, identity):
    """Reuse all 12 archived diagonal renders; render only the 12 cross meshes."""
    from PIL import Image, ImageDraw
    from ..round2_v2.render_mesh_audit import render_file

    records, render_sources, clipping = {}, [], []
    for obj in p["objects"]:
        uid = obj["uid"]
        path = SHARED / "visuals/records" / f"{uid}_gt.json"
        record = read(path)
        _verify_render(record, obj["mesh"])
        records[uid, "gt", None] = record
        render_sources.append({"uid": uid, "cell": "gt", "reused": True, "record": binding(path)})
    for row in rows:
        uid, seed, cell = row["uid"], row["seed"], row["cell"]
        if row["is_reference"]:
            prefix = f"{uid}_crop{row['support_level']}_visible_appearance_control_{seed}"
            path = BASE_OUT / "visuals/records" / f"{prefix}.json"
            record = read(path)
        else:
            prefix = f"{uid}_{cell}_appearance_control_{seed}"
            event("render_cross_mesh", uid=uid, seed=seed, cell=cell)
            record = render_file(verify(row["mesh"]["mesh"]), prefix, output=OUT / "visuals",
                                 expected_sha256=row["mesh"]["mesh"]["sha256"],
                                 identity={"evaluation_identity": identity, "uid": uid, "seed": seed, "cell": cell})
            path = OUT / "visuals/records" / f"{prefix}.json"
        _verify_render(record, row["mesh"]["mesh"])
        records[uid, cell, seed] = record
        render_sources.append({"uid": uid, "seed": seed, "cell": cell,
                               "reused": row["is_reference"], "record": binding(path)})
    for (uid, cell, seed), record in records.items():
        for view in record["views"]:
            if view["foreground_touches_image_border"] or view["vertices_outside_xy"] or view["vertices_outside_depth"]:
                clipping.append({"uid": uid, "seed": seed, "cell": cell, "view": view["view"]})
    destination = OUT / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    links = []
    for obj in p["objects"]:
        uid = obj["uid"]
        for seed in SEEDS:
            side, header = 320, 32
            sheet = Image.new("RGB", (side * 3, header + 5 * (side + header)), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text((8, 8), f"{uid} / seed {seed} / native fixed cameras / no fitting", fill="black")
            for row_index, cell in enumerate(["gt"] + [cell_name(*cell) for cell in CELLS]):
                record = records[uid, "gt", None] if cell == "gt" else records[uid, cell, seed]
                diagonal = cell in ("ss518_cond518", "ss2072_cond2072")
                label = "GT (archived)" if cell == "gt" else cell + (" (archived)" if diagonal else " (cross)")
                for column, view in enumerate(record["views"]):
                    x, y = column * side, header + row_index * (side + header)
                    draw.text((x + 4, y + 8), f"{label} / view {view['view']}", fill="black")
                    with Image.open(view["png"]) as picture:
                        sheet.paste(picture.convert("RGB").resize((side, side), Image.Resampling.LANCZOS), (x, y + header))
            path = destination / f"{uid}_seed{seed}_ss_slat_cross.jpg"
            sheet.save(path, quality=95)
            links.append({"uid": uid, "seed": seed, **binding(path)})
    return {"identity": identity, "figures": links, "render_records": render_sources,
            "clipping": clipping, "review_required": True, "human_review_completed": False,
            "camera_policy": "three unchanged native orthographic views, identity render transform, same material/scale; no fitted alignment",
            "archived_diagonal_render_count": 12, "new_cross_render_count": 12,
            "scope": SCOPE}


def export_worlds(rows, identity):
    """Assemble the two already-placed world meshes without another transform."""
    import trimesh

    destination = OUT / "world"
    destination.mkdir(parents=True, exist_ok=True)
    worlds = []
    for support, condition in CELLS:
        for seed in SEEDS:
            selected = [r for r in rows if (r["support_level"], r["condition_level"], r["seed"]) == (support, condition, seed)]
            if len(selected) != 2 or len({r["uid"] for r in selected}) != 2:
                raise ValueError("Each world export must contain exactly the two planned objects")
            scene = trimesh.Scene()
            sources = []
            for row in selected:
                mesh = trimesh.load(verify(row["mesh"]["world_mesh"]), process=False, force="mesh")
                scene.add_geometry(mesh, node_name=row["uid"], geom_name=row["uid"])
                sources.append({"uid": row["uid"], "world_mesh": row["mesh"]["world_mesh"],
                                "canonical_from_world": row["mesh"]["canonical_from_world"]})
            cell = cell_name(support, condition)
            scene.metadata.update({"identity": identity, "cell": cell, "seed": seed,
                                   "layout": "known archived placements; not predicted layout"})
            path = destination / f"{cell}_seed{seed}_known_placements.glb"
            scene.export(path)
            worlds.append({"cell": cell, "seed": seed, "objects": 2, **binding(path),
                           "sources": sources, "placement_policy": "known archived placements; not predicted layout; no additional transforms"})
    return worlds


def report_text(result):
    stats, audit = result["statistics"], result["acceptance"]
    fmt = lambda value: "null" if value is None else f"{value:.7f}"
    lines = ["# SS 支持 × SLat 条件交叉诊断", "",
             "24 个逻辑单元格 = 12 个旧对角 mesh 引用 + 12 个新增交叉 mesh；复用 12 份已有 SS，不重新生成 SS。", "",
             SCOPE, "", result["plan_scope"], "", "## 统计约定", "",
             "L=518、H=2072；Y 的第一轴为 SS 支持，第二轴为 SLat 条件。CD 越低越好，F@0.02 越高越好。所有差值保持原始指标方向：高水平减低水平；CD 负值、F 正值表示改善。交互项为 Y(H,H)−Y(H,L)−Y(L,H)+Y(L,L)，其符号不单独代表整体收益。", "",
             "对象内对三个 sampling seed 成对汇总，再对两个对象等权宏平均。不把 24 个单元格当作独立样本，不报告泛化显著性。SS IoU 只诊断支持输入；固定支持时两个条件必须复用同一 SS，其 IoU 完全相同。", "",
             "## 每对象、每种子的完整 2×2", "",
             "| 对象 | seed | SS 支持 | SLat 条件 | 来源 | CD ↓ | F@0.02 ↑ | SS IoU | 可见 recall ↑ | 外遮挡 recall ↑ | free-space violation ↓ |",
             "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(result["records"], key=lambda r: (r["uid"], r["seed"],
                                                       LEVELS.index(r["support_level"]),
                                                       LEVELS.index(r["condition_level"]))):
        lines.append(f"| {row['uid']} | {row['seed']} | {row['support_level']} | {row['condition_level']} | "
                     + ("旧对角引用" if row["is_reference"] else "新增交叉") + " | "
                     + " | ".join(fmt(row["metrics"][key]) for key in METRICS) + " |")
    lines += ["", "## 对象三种子平均与两对象宏平均", "",
              "| 对象/汇总 | 单元格 | CD ↓ | F@0.02 ↑ | SS IoU | 可见 recall ↑ | 外遮挡 recall ↑ | free-space violation ↓ |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    aggregates = list(stats["per_object"].items()) + [("两对象宏平均", stats["macro"])]
    for label, item in aggregates:
        for cell, metrics in item["cells"].items():
            lines.append(f"| {label} | {cell} | " + " | ".join(fmt(metrics[key]) for key in METRICS) + " |")
    lines += ["", "## 固定支持、固定条件及交互差值", "",
              "condition_at_supportX：固定 SS=X，条件 2072−518；support_at_conditionX：固定条件 X，SS 2072−518。下表 ΔF 使用原始 [0,1] 分值，不是百分点。", "",
              "| 对象/seed/汇总 | 对照 | ΔCD | ΔF@0.02 | ΔSS IoU（仅审计） | Δ可见 recall | Δ外遮挡 recall | Δfree-space violation |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    paired = [(f"{uid} / {seed}", item) for uid, seeds in stats["per_seed"].items() for seed, item in seeds.items()]
    for label, item in paired + aggregates:
        for name, metrics in item["contrasts"].items():
            lines.append(f"| {label} | {name} | " + " | ".join(fmt(metrics[key]) for key in METRICS) + " |")
    lines += ["", "## 完整性和可视化", "",
              "全部记录通过 identity、对象/seed/SS 绑定、成功标志、25 次 positive / 19 次 negative 调用及 fitted_transform=False 检查。GT 表面分区沿用旧四帧，每物体 20,000 个固定 GT 样本，跨 seed 和四个单元格数量一致。原 native 坐标评估，不作 ICP、旋转、平移或缩放拟合。", "",
              "| 对象 | visible GT 样本 | externally occluded GT 样本 | other unobserved GT 样本 |",
              "|---|---:|---:|---:|"]
    for uid, item in audit["fixed_gt_partitions"].items():
        lines.append(f"| {uid} | " + " | ".join(str(item["counts"][key]) for key in PARTITIONS) + " |")
    lines += ["", "[完整记录与统计](summary.json)；[可视化审计与裁剪清单](visual_audit.json)。", "",
              "六张图各含 GT + 四个单元格，均用同一组三个固定 native 视角；GT 和旧对角图直接复用旧记录，新交叉 mesh 使用原渲染器。图像需人工检查重复结构、错误支撑、额外平面和裁剪；自动哈希/裁剪检查不代表视觉质量已验收。", ""]
    for figure in result["figures"]:
        lines.append(f"- [{figure['uid']} / seed {figure['seed']}]({Path(figure['path']).relative_to(OUT)})")
    lines += ["", "[12 个两物体组合 GLB](world/)覆盖四个单元格 × 三个种子。位置由旧 canonical_from_world 的已知 placement 给出，不是预测布局；此处没有新增布局估计或世界坐标拟合。", "",
              f"检测到 {len(result['visual_clipping'])} 个视图裁剪/越界记录，详见 visual_audit.json；人工审阅尚未完成。", ""]
    return "\n".join(lines)


def main():
    p = plan()["plan"]
    identity = freeze_runtime()
    smoke = check_smoke()
    rows, acceptance = collect_rows(p, identity)
    stats = summarize_statistics(rows, [obj["uid"] for obj in p["objects"]])
    event("summary_inputs_verified", logical_cases=len(rows), new_cross_meshes=12)
    worlds = export_worlds(rows, identity)
    visuals = figures(rows, p, identity)
    visuals["worlds"] = worlds
    visuals["world_placement_policy"] = "known archived placements, not predicted layout"
    write(OUT / "visual_audit.json", visuals)
    result = {"time": now(), "identity": identity, "plan": binding(OUT / "plan.json"), "smoke": smoke,
              "logical_cases": len(rows), "new_meshes": 12, "original_references": 12,
              "new_ss_samples": 0, "objects": 2, "scenes": 1, "sampling_seeds": list(SEEDS),
              "statistics": stats, "acceptance": acceptance, "records": rows,
              "figures": visuals["figures"], "worlds": worlds,
              "visual_clipping": visuals["clipping"], "visual_audit": binding(OUT / "visual_audit.json"),
              "scope": SCOPE, "plan_scope": p["scope"],
              "review_required": True, "human_review_completed": False}
    write(OUT / "summary.json", result)
    (OUT / "REPORT.md").write_text(report_text(result), encoding="utf-8")
    event("summary_complete", logical_cases=24, new_cross_meshes=12, figures=6, worlds=12)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="Run pure CPU-only statistics checks")
    args = parser.parse_args()
    if args.self_check:
        print(self_check_statistics())
    else:
        main()
