"""Paired predicted-support / GT-support diagnostic in the native frame.

GT support is privileged diagnostic input, never a deployable-method result.
All four archived objects and all three sampling seeds must be present before
the report is written. Importing this file does not load Torch or any model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import product
from pathlib import Path
from statistics import fmean

from .common import (BASE_OUT, OUT, SHARED, binding, check_smoke, event,
                     freeze_runtime, get_result, logical_cases, now, plan,
                     read, validate_saved_result, verify, write)

SUPPORTS = ("predicted", "gt")
SEEDS = (42, 43, 44)
METRICS = ("chamfer_l1", "fscore_0p02", "ss_iou", "visible_recall",
           "occluded_recall", "free_space_violation")
DIRECTIONS = {name: ("lower" if name in ("chamfer_l1", "free_space_violation") else "higher")
              for name in METRICS}
PARTITIONS = ("visible", "externally_occluded", "other_unobserved")
EXPECTED_PARTITIONS = {
    "hssd_102344049_i076": dict(zip(PARTITIONS, (9225, 9349, 1426))),
    "hssd_102344049_i075": dict(zip(PARTITIONS, (5247, 9477, 5276))),
    "hssd_102344049_i080": dict(zip(PARTITIONS, (862, 19138, 0))),
    "hssd_102344049_i065": dict(zip(PARTITIONS, (5691, 14309, 0))),
}
SCOPE = (
    "四个旧训练对象、一个 HSSD 场景、三个对象内采样种子。固定同一 crop2072_visible "
    "观测条件及冻结 SLat30K，仅将预测 SS 支持替换为完整目标mesh的64³表面体素支持（非填充实体）。GT 支持含不可见表面的"
    "特权信息，是诊断性上限对照，不是可部署方法成绩；也不保证每个指标一定更好。"
    "支持替换同时改变点数、形状与稀疏注意力拓扑，不能拆分这些因素。"
    "输入仍使用 GT 辅助相机、mask 和 canonical frame。没有新训练、未见场景泛化"
    "或 LaRI 贡献结论。三个 seeds 不是三个独立场景，24 个逻辑位置不是24个物体。"
)


def metric_values(mesh, ss):
    partition = mesh["partitioned"]
    values = {
        "chamfer_l1": mesh["surface"]["chamfer_l1"],
        "fscore_0p02": mesh["surface"]["fscore_0p02"],
        "ss_iou": ss["occupancy"]["iou"],
        "visible_recall": partition["visible"]["recall_0p02"],
        "occluded_recall": partition["externally_occluded"]["recall_0p02"],
        "free_space_violation": partition["predicted_surface_free_space_violation_fraction"],
    }
    for key, value in values.items():
        if value is None and key in ("visible_recall", "occluded_recall"):
            continue
        if value is None or not math.isfinite(value):
            raise ValueError(f"Missing/nonfinite metric: {key}")
    return values


def averages(rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot average an empty comparison")
    return {key: None if any(row[key] is None for row in rows)
            else fmean(row[key] for row in rows) for key in METRICS}


def delta(after, before):
    return {key: None if after[key] is None or before[key] is None
            else after[key] - before[key] for key in METRICS}


def ranges(rows):
    rows = list(rows)
    return {key: None if any(row[key] is None for row in rows)
            else [min(row[key] for row in rows), max(row[key] for row in rows)]
            for key in METRICS}


def win_counts(deltas):
    deltas = list(deltas)
    output = {}
    for metric in METRICS:
        values = [row[metric] for row in deltas]
        if any(value is None for value in values):
            output[metric] = None
            continue
        oriented = [value if DIRECTIONS[metric] == "higher" else -value for value in values]
        output[metric] = {"better": sum(v > 1e-12 for v in oriented),
                          "equal": sum(abs(v) <= 1e-12 for v in oriented),
                          "worse": sum(v < -1e-12 for v in oriented), "pairs": len(values)}
    return output


def summarize_statistics(rows, object_ids=None):
    """Retain every paired seed and use equal object weights, not best seeds."""
    rows = list(rows)
    object_ids = list(object_ids) if object_ids is not None else sorted({r["uid"] for r in rows})
    if len(object_ids) != 4 or len(set(object_ids)) != 4:
        raise ValueError("Exactly four distinct planned objects are required")
    indexed = {}
    for row in rows:
        key = row["uid"], row["seed"], row["support_kind"]
        if key in indexed:
            raise ValueError(f"Duplicate paired result: {key}")
        indexed[key] = row
    if set(indexed) != set(product(object_ids, SEEDS, SUPPORTS)):
        raise ValueError("Expected exactly four objects x three seeds x two support types")
    per_seed, per_object = {}, {}
    for uid in object_ids:
        per_seed[uid] = {}
        for seed in SEEDS:
            cells = {support: indexed[uid, seed, support]["metrics"] for support in SUPPORTS}
            per_seed[uid][str(seed)] = {"cells": cells,
                "gt_minus_predicted": delta(cells["gt"], cells["predicted"])}
        paired = list(per_seed[uid].values())
        differences = [r["gt_minus_predicted"] for r in paired]
        per_object[uid] = {
            "sampling_seeds": list(SEEDS),
            "cells": {support: averages(r["cells"][support] for r in paired) for support in SUPPORTS},
            "gt_minus_predicted": averages(differences),
            "delta_seed_range": ranges(differences),
            "seed_win_counts": win_counts(differences),
        }
    macro = {
        "objects": 4, "scenes": 1, "sampling_seeds_per_object": 3,
        "weighting": "equal mean of four object means, each from three paired sampling seeds",
        "cells": {support: averages(r["cells"][support] for r in per_object.values()) for support in SUPPORTS},
        "gt_minus_predicted": averages(r["gt_minus_predicted"] for r in per_object.values()),
        "object_win_counts": win_counts(r["gt_minus_predicted"] for r in per_object.values()),
    }
    return {"per_seed": per_seed, "per_object": per_object, "macro": macro,
            "metric_directions": DIRECTIONS, "contrast_convention": "GT support minus predicted support",
            "range_convention": "observed three-seed min/max, not confidence intervals",
            "null_policy": "any missing constituent propagates null, without changing the denominator",
            "ss_iou_role": "support-input diagnostic; GT support IoU=1 by construction, never a learned gain"}


def self_check_statistics():
    rows = []
    for i, uid in enumerate(("chair", "table", "bench", "stool")):
        for seed in SEEDS:
            for support in SUPPORTS:
                value = float(i + seed / 10 + (2 * i - 1 if support == "gt" else 0))
                rows.append({"uid": uid, "seed": seed, "support_kind": support,
                             "metrics": {name: value for name in METRICS}})
    stats = summarize_statistics(rows)
    if stats["macro"]["gt_minus_predicted"]["chamfer_l1"] != 2.0:
        raise AssertionError("Object-weighted pairing is incorrect")
    counts = stats["macro"]["object_win_counts"]
    if counts["chamfer_l1"]["better"] != 1 or counts["fscore_0p02"]["better"] != 3:
        raise AssertionError("CD/F direction handling is incorrect")
    for bad in (rows[:-1], rows + [rows[0]]):
        try:
            summarize_statistics(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete or duplicate case set was accepted")
    missing = [dict(row, metrics=dict(row["metrics"])) for row in rows]
    missing[0]["metrics"]["visible_recall"] = None
    if summarize_statistics(missing)["macro"]["gt_minus_predicted"]["visible_recall"] is not None:
        raise AssertionError("Unavailable stratum silently changed the denominator")
    return {"passed": True, "missing_and_duplicate_rejected": True,
            "raw_directions_and_object_weighting": True, "null_denominator_preserved": True}


def _same_binding(left, right, label):
    if (Path(left["path"]).resolve(), left["sha256"]) != (Path(right["path"]).resolve(), right["sha256"]):
        raise ValueError(f"Binding mismatch: {label}")


def collect_rows(p, identity):
    objects = {obj["uid"]: obj for obj in p["objects"]}
    cases = logical_cases()
    if cases != p["cases"] or set(objects) != set(EXPECTED_PARTITIONS):
        raise ValueError("Objects or cases differ from the complete frozen four-object plan")
    rows, partitions, condition_signatures = [], {}, {}
    for case in cases:
        uid, seed, support = case["uid"], case["seed"], case["support_kind"]
        obj = objects[uid]
        if support not in SUPPORTS or case["is_reference"] != (support == "predicted"):
            raise ValueError("Only predicted-support cells may reference old mesh outputs")
        mp, mesh, sp, ss = get_result(case)
        mesh_binding, ss_binding = binding(mp), binding(sp)
        condition = read(verify(case["condition_record"]))
        if (condition["identity"] != p["base_identity"] or not condition["success"]
                or condition["object_uid"] != uid or condition["seed"] != seed
                or condition["arm"] != "appearance_control" or condition["variant"] != "crop2072_visible"):
            raise ValueError("Condition source differs from the frozen common observation")
        verify(condition["inputs"]["sample"])
        if mesh["inputs"] != condition["inputs"]:
            raise ValueError("SLat observation changed between support types")
        signature = {"record": case["condition_record"], "inputs": condition["inputs"]}
        if condition_signatures.setdefault((uid, seed), signature) != signature:
            raise ValueError("A paired comparison uses different condition records")
        if (not mesh["success"] or mesh["object_uid"] != uid or mesh["seed"] != seed
                or mesh["arm"] != "appearance_control" or mesh["fitted_transform"] is not False
                or mesh["slat_binding"] != p["slat_binding"]
                or mesh["calls"] != {"positive_calls": 25, "negative_calls": 19}):
            raise ValueError("Incomplete, mismatched or transformed mesh result")
        _same_binding(mesh["ss"], ss_binding, "actual support record")
        if case["is_reference"]:
            _same_binding(mesh_binding, case["reference_mesh"], "predicted-support reference")
            if (mesh["identity"] != p["base_identity"] or ss["identity"] != p["base_identity"]
                    or mesh["variant"] != "crop2072_visible" or ss["variant"] != "crop2072_visible"
                    or ss["object_uid"] != uid or ss["seed"] != seed or not ss["success"]):
                raise ValueError("Predicted-support baseline differs from its frozen origin")
        else:
            validate_saved_result(mesh, case, identity, condition["inputs"])
            if ss["occupancy"]["iou"] != 1.0:
                raise ValueError("GT support differs from the GT occupancy target")
        _same_binding(mesh["gt"], obj["mesh"], "fixed GT mesh")
        if mesh["canonical_from_world"] != obj["canonical_from_world"]:
            raise ValueError("Known canonical/world transform changed")
        for value in (mesh["mesh"], mesh["world_mesh"], mesh["gt"], mesh["ss"], ss["coords"]):
            verify(value)
        counts = {name: mesh["partitioned"][name]["gt_samples"] for name in PARTITIONS}
        if counts != EXPECTED_PARTITIONS[uid] or sum(counts.values()) != 20000:
            raise ValueError("Original 20,000-sample GT partition changed")
        signature = {"counts": counts,
                     "tolerance": mesh["partitioned"]["visibility_depth_tolerance_canonical"],
                     "scope": mesh["gt_partition_scope"]}
        if partitions.setdefault(uid, signature) != signature:
            raise ValueError("GT partition differs across support types or seeds")
        rows.append({"uid": uid, "seed": seed, "support_kind": support,
                     "is_reference": case["is_reference"], "mesh_record": mesh_binding,
                     "ss_record": ss_binding, "condition_record": case["condition_record"],
                     "mesh": mesh, "metrics": metric_values(mesh, ss),
                     "support_voxels": ss["occupancy"]["predicted_voxels"]})
    if len(rows) != 24 or sum(row["is_reference"] for row in rows) != 12:
        raise ValueError("Require 24 logical results: 12 references plus 12 new GT-support meshes")
    summarize_statistics(rows, list(objects))
    return rows, {"passed": True, "logical_cases": 24, "original_references": 12,
                  "new_gt_support_meshes": 12, "fixed_gt_partitions": partitions,
                  "complete_frozen_slat_binding": True, "same_condition_in_every_pair": True,
                  "native_frame": True, "all_four_original_objects_retained": True}


def _verify_render(record, expected_mesh):
    from ..round2_v2.render_mesh_audit import parameters

    _same_binding(record["mesh"], expected_mesh, "render source mesh")
    verify(record["mesh"])
    if record["matrix_original_to_rendered"] != [[float(i == j) for j in range(4)] for i in range(4)]:
        raise ValueError("Unexpected per-result render transform")
    expected_binding = {"mesh_sha256": expected_mesh["sha256"], "parameters": parameters(),
                        "transform": record["matrix_original_to_rendered"]}
    expected_hash = hashlib.sha256(json.dumps(expected_binding, sort_keys=True).encode()).hexdigest()
    if record["binding_hash"] != expected_hash:
        raise ValueError("Render cameras, material, scale, versions or renderer source differ")
    if [view["view"] for view in record["views"]] != [0, 1, 2]:
        raise ValueError("Three original fixed cameras are required")
    for view in record["views"]:
        verify({"path": view["png"], "sha256": view["png_sha256"]})


def figures(rows, p, identity):
    from PIL import Image, ImageDraw
    from ..round2_v2.render_mesh_audit import render_file

    records, render_sources, clipping = {}, [], []
    for obj in p["objects"]:
        uid = obj["uid"]
        path = SHARED / "visuals/records" / f"{uid}_gt.json"
        record = read(path)
        _verify_render(record, obj["mesh"])
        records[uid, "mesh_gt", None] = record
        render_sources.append({"uid": uid, "support_kind": "mesh_gt", "reused": True, "record": binding(path)})
    for row in rows:
        uid, seed, support = row["uid"], row["seed"], row["support_kind"]
        if row["is_reference"]:
            prefix = f"{uid}_crop2072_visible_appearance_control_{seed}"
            path = BASE_OUT / "visuals/records" / f"{prefix}.json"
            record = read(path)
        else:
            prefix = f"{uid}_gt_support_{seed}"
            event("render_gt_support_gallery", uid=uid, seed=seed)
            record = render_file(verify(row["mesh"]["mesh"]), prefix, output=OUT / "visuals",
                                 expected_sha256=row["mesh"]["mesh"]["sha256"],
                                 identity={"evaluation_identity": identity, "uid": uid,
                                           "seed": seed, "support_kind": support})
            path = OUT / "visuals/records" / f"{prefix}.json"
        _verify_render(record, row["mesh"]["mesh"])
        records[uid, support, seed] = record
        render_sources.append({"uid": uid, "seed": seed, "support_kind": support,
                               "reused": row["is_reference"], "record": binding(path)})
    for (uid, support, seed), record in records.items():
        for view in record["views"]:
            if view["foreground_touches_image_border"] or view["vertices_outside_xy"] or view["vertices_outside_depth"]:
                clipping.append({"uid": uid, "seed": seed, "support_kind": support, "view": view["view"]})
    destination = OUT / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    links = []
    labels = {"mesh_gt": "GT mesh (archived)", "predicted": "Predicted SS (archived)",
              "gt": "GT support -> frozen SLat (oracle)"}
    for obj in p["objects"]:
        uid = obj["uid"]
        for seed in SEEDS:
            side, header = 320, 32
            sheet = Image.new("RGB", (side * 3, header + 3 * (side + header)), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text((8, 8), f"{uid} / seed {seed} / crop2072 / native fixed cameras / no fitting", fill="black")
            for row_index, support in enumerate(("mesh_gt", *SUPPORTS)):
                record = records[uid, support, None if support == "mesh_gt" else seed]
                for column, view in enumerate(record["views"]):
                    x, y = column * side, header + row_index * (side + header)
                    draw.text((x + 4, y + 8), f"{labels[support]} / view {view['view']}", fill="black")
                    with Image.open(view["png"]) as picture:
                        sheet.paste(picture.convert("RGB").resize((side, side), Image.Resampling.LANCZOS), (x, y + header))
            path = destination / f"{uid}_seed{seed}_gt_support.jpg"
            sheet.save(path, quality=95)
            links.append({"uid": uid, "seed": seed, **binding(path)})
    return {"identity": identity, "figures": links, "render_records": render_sources,
            "clipping": clipping, "review_required": True, "human_review_completed": False,
            "camera_policy": "same three native orthographic cameras/material/scale; no fitted transform",
            "archived_gt_render_count": 4, "archived_predicted_render_count": 12,
            "new_gt_support_render_count": 12, "scope": SCOPE}


def export_worlds(rows, identity):
    import trimesh

    destination = OUT / "world"
    destination.mkdir(parents=True, exist_ok=True)
    worlds = []
    for support, seed in product(SUPPORTS, SEEDS):
        selected = [r for r in rows if (r["support_kind"], r["seed"]) == (support, seed)]
        if len(selected) != 4 or {r["uid"] for r in selected} != set(EXPECTED_PARTITIONS):
            raise ValueError("Every world export must preserve all four objects")
        scene, sources = trimesh.Scene(), []
        for row in selected:
            mesh = trimesh.load(verify(row["mesh"]["world_mesh"]), process=False, force="mesh")
            scene.add_geometry(mesh, node_name=row["uid"], geom_name=row["uid"])
            sources.append({"uid": row["uid"], "world_mesh": row["mesh"]["world_mesh"],
                            "canonical_from_world": row["mesh"]["canonical_from_world"]})
        scene.metadata.update({"identity": identity, "support_kind": support, "seed": seed,
                               "layout": "known archived placements; not predicted layout"})
        path = destination / f"{support}_support_seed{seed}_known_placements.glb"
        scene.export(path)
        worlds.append({"support_kind": support, "seed": seed, "objects": 4,
                       **binding(path), "sources": sources,
                       "placement_policy": "known archived placements, not predicted layout; no additional transform"})
    return worlds


def report_text(result):
    stats = result["statistics"]
    fmt = lambda v: "null" if v is None else f"{v:.7f}"
    lines = ["# GT 支持 → 冻结 SLat：共享照片诊断", "",
             "已完成 12 个新增 GT-support mesh；另引用相同条件下的 12 个旧预测 SS mesh。共 24 个逻辑结果、4 个对象、1 个场景。", "",
             SCOPE, "", result["plan_scope"], "", "## 比较约定", "",
             "所有对象统一使用 crop2072_visible 条件（最终网络图像仍为 518），不按物体选择更优输入。固定 SLat30K、采样/解码器及原生坐标。GT 支持是完整目标 mesh 的 64³ 表面体素支持，并非填充实体；不能把其 SS IoU=1 视为模型学到的提升。", "",
             "差值统一为 GT-support 减 predicted-support：CD 和自由空间违规负值、F/recall 正值表示改善。先对三个种子配对平均，再对四对象等权平均；种子范围是观察范围，不是置信区间。", "",
             "## 完整逐对象、逐种子结果", "",
             "| 对象 | seed | 支持来源 | 支持点数 | CD ↓ | F@0.02 ↑ | SS IoU（诊断） | 可见 recall ↑ | 外遮挡 recall ↑ | 自由空间违规 ↓ |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["records"]:
        label = "预测支持（旧引用）" if row["is_reference"] else "GT支持（oracle）"
        lines.append(f"| {row['uid']} | {row['seed']} | {label} | {row['support_voxels']} | "
                     + " | ".join(fmt(row["metrics"][key]) for key in METRICS) + " |")
    lines += ["", "## 对象均值与宏平均", "",
              "| 对象/汇总 | 支持来源 | CD ↓ | F@0.02 ↑ | SS IoU（诊断） | 可见 recall ↑ | 外遮挡 recall ↑ | 自由空间违规 ↓ |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    aggregates = list(stats["per_object"].items()) + [("四对象宏平均", stats["macro"])]
    for label, item in aggregates:
        for support in SUPPORTS:
            lines.append(f"| {label} | {support} | " + " | ".join(fmt(item["cells"][support][key]) for key in METRICS) + " |")
    lines += ["", "## 配对差值（GT支持−预测支持）", "",
              "下表 ΔF、Δrecall 和 Δ违规率是原始 [0,1] 分值，不是百分点。", "",
              "| 对象/seed/汇总 | ΔCD | ΔF | ΔSS IoU（输入诊断） | Δ可见 recall | Δ外遮挡 recall | Δ自由空间违规 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    individual = [(f"{uid} / {seed}", item) for uid, seeds in stats["per_seed"].items() for seed, item in seeds.items()]
    for label, item in individual + aggregates:
        lines.append(f"| {label} | " + " | ".join(fmt(item["gt_minus_predicted"][key]) for key in METRICS) + " |")
    lines += ["", "## 验收与解释边界", "",
              "所有记录检查对象/seed/condition/support 绑定、完整 SLat 权重绑定、25 次 positive / 19 次 negative 调用、无拟合变换及固定 20,000 点 GT 分区。保留全部四个对象和三个种子，长凳失败不能删除或改成缺失。", "",
              "GT 支持改善结果，说明固定下游在更准确的支持上有恢复空间；不能单独证明 SS 可见/自由空间约束方案有效。GT 支持仍有错误时，需区分 SLat 生成、观测条件、离散支持与 decoder 能力，不能把全部残余误差都归因于 SLat 参数。已有 mesh→latent 检查可另作参考，但本轮不重新编码或直接解码 GT latent。", "",
              "[完整统计和来源绑定](summary.json)；[渲染审计](visual_audit.json)。12 张图均为 GT mesh / 旧预测支持 / 新 GT-support，使用原来三个固定 native 视角、同材质与尺度：", ""]
    for figure in result["figures"]:
        lines.append(f"- [{figure['uid']} / seed {figure['seed']}]({Path(figure['path']).relative_to(OUT)})")
    lines += ["", "[六个场景 GLB](world/)为两类支持 × 三种子，每个保留四物体；使用已有已知摆放位置，不是预测场景布局。", "",
              f"检测到 {len(result['visual_clipping'])} 个视图裁剪/越界记录；详见 visual_audit.json。自动完成不等于人工视觉验收，尚待审阅。", ""]
    return "\n".join(lines)


def main():
    p = plan()["plan"]
    identity = freeze_runtime()
    smoke = check_smoke()
    rows, acceptance = collect_rows(p, identity)
    stats = summarize_statistics(rows, [obj["uid"] for obj in p["objects"]])
    event("summary_inputs_verified", logical_cases=24, new_gt_support_meshes=12)
    worlds = export_worlds(rows, identity)
    visuals = figures(rows, p, identity)
    visuals["worlds"] = worlds
    write(OUT / "visual_audit.json", visuals)
    result = {"time": now(), "identity": identity, "plan": binding(OUT / "plan.json"), "smoke": smoke,
              "logical_cases": 24, "new_meshes": 12, "original_references": 12,
              "new_ss_samples": 0, "objects": 4, "scenes": 1, "sampling_seeds": list(SEEDS),
              "statistics": stats, "acceptance": acceptance, "records": rows,
              "figures": visuals["figures"], "worlds": worlds, "visual_clipping": visuals["clipping"],
              "visual_audit": binding(OUT / "visual_audit.json"), "scope": SCOPE,
              "plan_scope": p["scope"], "review_required": True, "human_review_completed": False}
    write(OUT / "summary.json", result)
    (OUT / "REPORT.md").write_text(report_text(result), encoding="utf-8")
    event("summary_complete", logical_cases=24, new_gt_support_meshes=12, figures=12, worlds=6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="CPU-only pairing/statistics checks")
    args = parser.parse_args()
    if args.self_check:
        print(self_check_statistics())
    else:
        main()
