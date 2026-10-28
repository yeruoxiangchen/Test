"""Fixed-camera mesh evidence for the Dev64 / strong16 final-mesh evaluation.

The source PLY files are never modified. Native images use identity transforms;
representative fitted images use exactly the independent SE(3) audit matrices.
Selection ranks are declared before any new cohort image is inspected.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import render_mesh_audit as renderer

PACKAGE = Path(__file__).resolve().parents[2]
OUT = PACKAGE / "outputs/round2_v2_heldout_mesh_20261018"
VIS = OUT / "visuals"
MODES = ("parent", "raw")
LABELS = {"parent": "Parent30K", "raw": "V2 step2000"}
DOMAINS = ("clean", "strong_occluded")
DOMAIN_LABELS = {"clean": "Clean Dev64", "strong_occluded": "Strong occlusion16"}
SELECTION_RANKS = {"clean": [1, 17, 33, 49, 64], "strong_occluded": [1, 9, 16]}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    renderer.save_json(Path(path), value)


def sha(path):
    return renderer.sha256(path)


def relative(path):
    return os.path.relpath(path, OUT)


def utc():
    return datetime.now(timezone.utc).isoformat()


def protocol():
    value = {
        "version": 1,
        "cohorts": {"clean": 64, "strong_occluded": 16},
        "modes": list(MODES), "seeds": [42, 43],
        "selection": {
            "score": "mean over inference seeds of native CD(parent) minus native CD(V2)",
            "order": "ascending score, then lexicographic object_uid",
            "one_based_ranks_per_independent_cohort": SELECTION_RANKS,
            "reason": "Include low, median and high effects without visual selection; complete cohort atlases also exported.",
            "declared_before_new_cohort_image_inspection": True,
        },
        "renderer": renderer.parameters(),
        "geometry": "Actual triangle PLY surfaces; native identity; no recentering, normalization, smoothing, filtering, or component removal.",
        "aligned_geometry": "Only preselected rows. Exact independently scored proper SE(3) matrix; GT-assisted diagnostic, not inference.",
        "native_expected_meshes_including_gt": 384, "native_expected_images": 1152,
        "aligned_expected_meshes": 32, "aligned_expected_images": 96,
        "atlas": {"seed": 42, "view": 0, "objects_per_page": 4},
        "gallery": "Offline, all 80 input rows, two seeds, three fixed views; no canvas pixel reading.",
        "units": "Normalized model coordinates, not physical meters.",
        "gt": "Official SLat VAE proxy, not original high-resolution asset.",
    }
    path = OUT / "visual_protocol.json"
    if path.exists():
        existing = read(path)
        if {k: v for k, v in existing.items() if k != "created_utc"} != value:
            raise ValueError("Frozen visual protocol changed")
        return existing
    value["created_utc"] = utc()
    save(path, value)
    return value


def load_plan():
    value = read(OUT / "plan.json")
    return value, value.get("plan", value)["rows"]


def load_records(require_complete=False):
    plan, rows = load_plan()
    by_index = {row["index"]: row for row in rows}
    records = {}
    for path in sorted((OUT / "records").glob("*.json")):
        r = read(path)
        if r.get("success") is False:
            continue
        key = (r["index"], r["seed"], r["mode"])
        row = by_index[r["index"]]
        if "identity_hash" in plan and r.get("identity_hash") != plan["identity_hash"]:
            raise ValueError(f"Different evaluation plan identity: {path}")
        if r["object_uid"] != row["object_uid"] or r["domain"] != row["domain"]:
            raise ValueError(f"Wrong row identity: {path}")
        if r["mode"] not in MODES or r["seed"] not in (42, 43):
            raise ValueError(f"Unexpected case: {path}")
        if r["target_mesh"] != row["target_mesh"]:
            raise ValueError(f"GT binding mismatch: {path}")
        if key in records:
            raise ValueError(f"Duplicate record: {path}")
        records[key] = r
    if require_complete and (len(rows) != 80 or len(records) != 320):
        raise ValueError(f"Incomplete native cohort: {len(rows)} rows, {len(records)} records")
    return rows, records


def native_jobs(rows, records):
    jobs, seen = [], {}
    for row in rows:
        obj, target = row["object_index"], row["target_mesh"]
        if obj in seen:
            if seen[obj] != target:
                raise ValueError("GT differs between input conditions")
            continue
        seen[obj] = target
        jobs.append({"mesh_path": target["path"], "prefix": f"gt_{obj:03d}",
                     "expected_sha256": target["sha256"],
                     "identity": {"mode": "gt", "object_index": obj, "object_uid": row["object_uid"]}})
    for key, r in sorted(records.items()):
        jobs.append({"mesh_path": r["mesh"]["path"], "prefix": f"{key[0]:03d}_{key[1]}_{key[2]}",
                     "expected_sha256": r["mesh"]["sha256"],
                     "identity": {k: r[k] for k in ("index", "object_index", "object_uid", "domain", "seed", "mode")}})
    return jobs


def run_jobs(jobs, workers):
    completed = []
    # Compile once before threaded execution; rasterizer releases the GIL.
    if jobs:
        completed.append(renderer.render_file(**jobs[0], output=VIS))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(renderer.render_file, **job, output=VIS): job["prefix"] for job in jobs[1:]}
        for future in as_completed(pending):
            completed.append(future.result())
            if len(completed) % 20 == 0:
                print(json.dumps({"utc": utc(), "rendered_or_cached_meshes": len(completed), "jobs": len(jobs)}), flush=True)
    return sorted(completed, key=lambda r: r["prefix"])


def manifest(records, complete, fitted=False):
    views = [v for r in records for v in r["views"]]
    return {"complete": complete, "updated_utc": utc(), "parameters": renderer.parameters(),
            "protocol_sha256": sha(OUT / "visual_protocol.json"),
            "frame": "GT-assisted proper SE(3) diagnostic" if fitted else "native identity",
            "mesh_count": len(records), "image_count": len(views),
            "empty_images": sum(v["foreground_pixels_supersampled"] == 0 for v in views),
            "clipped_mesh_views": sum(bool(v["vertices_outside_xy"] or v["vertices_outside_depth"] or v["foreground_touches_image_border"]) for v in views),
            "records": records}


def render_native(workers, watch):
    protocol()
    done = {}
    while True:
        rows, native = load_records()
        jobs = native_jobs(rows, native)
        new_records = run_jobs([j for j in jobs if j["prefix"] not in done], workers)
        done.update({r["prefix"]: r for r in new_records})
        records = [done[j["prefix"]] for j in jobs]
        complete = len(native) == 320
        value = manifest(records, complete)
        if complete and (len(records) != 384 or value["image_count"] != 1152):
            raise ValueError("Native render cohort size mismatch")
        save(VIS / "manifest.json", value)
        print(json.dumps({"utc": utc(), "native_cases_available": len(native), "native_images": value["image_count"], "complete": complete}), flush=True)
        if complete or not watch:
            return
        time.sleep(20)


def select(rows, records):
    protocol()
    value = {"protocol_sha256": sha(OUT / "visual_protocol.json"),
             "plan_sha256": sha(OUT / "plan.json"), "cohorts": {}}
    for domain in DOMAINS:
        ranked = []
        for row in rows:
            if row["domain"] != domain:
                continue
            means = {mode: float(np.mean([records[(row["index"], s, mode)]["surface"]["chamfer_l1"] for s in (42, 43)])) for mode in MODES}
            ranked.append({k: row[k] for k in ("index", "object_index", "object_uid", "domain")} |
                          {"mean_native_cd_gain_v2_over_parent": means["parent"] - means["raw"]})
        ranked.sort(key=lambda r: (r["mean_native_cd_gain_v2_over_parent"], r["object_uid"]))
        for rank, r in enumerate(ranked, 1):
            r["rank"] = rank
        chosen = [ranked[i - 1] for i in SELECTION_RANKS[domain]]
        value["cohorts"][domain] = {"ranked_rows": ranked, "representative_rows": chosen}
    path = OUT / "visual_selection.json"
    if path.exists() and read(path) != value:
        raise ValueError("Frozen selection changed")
    save(path, value)
    return value


def aligned_records(records, require_complete=True):
    result = {}
    for key, native in records.items():
        path = OUT / "alignment/records" / f"{key[0]:03d}_{key[1]}_{key[2]}.json"
        if not path.exists() and not require_complete:
            continue
        r = read(path)
        if r["mesh"] != native["mesh"]:
            raise ValueError(f"Alignment has different mesh: {path}")
        if r["target_mesh"] != native["target_mesh"]:
            raise ValueError(f"Alignment has different GT: {path}")
        if any(r[k] != native[k] for k in ("index", "object_index", "object_uid", "domain", "seed", "mode")):
            raise ValueError(f"Alignment case identity differs: {path}")
        if any(abs(r["native"][k] - value) > 2e-12 for k, value in native["surface"].items()):
            raise ValueError(f"Native metrics differ from alignment audit: {path}")
        matrix = np.asarray(r["alignment"]["matrix_predicted_to_target"])
        rotation = matrix[:3, :3]
        if matrix.shape != (4, 4) or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-12):
            raise ValueError("Invalid homogeneous transform")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10) or abs(np.linalg.det(rotation) - 1) > 1e-10:
            raise ValueError("Non-rigid or reflected alignment")
        if r["alignment"].get("scale_fitted", False) or r["alignment"].get("reflection_allowed", False):
            raise ValueError("Scale/reflection fitted")
        result[key] = r
    return result


def selected_indices(selection):
    return {r["index"] for c in selection["cohorts"].values() for r in c["representative_rows"]}


def render_aligned(workers):
    rows, native = load_records(True)
    selection = select(rows, native)
    selected = selected_indices(selection)
    fitted = aligned_records({k: r for k, r in native.items() if k[0] in selected})
    jobs = []
    for key, r in sorted(fitted.items()):
        jobs.append({"mesh_path": r["mesh"]["path"], "prefix": f"aligned_{key[0]:03d}_{key[1]}_{key[2]}",
                     "expected_sha256": r["mesh"]["sha256"],
                     "matrix": np.asarray(r["alignment"]["matrix_predicted_to_target"]),
                     "identity": {k: native[key][k] for k in ("index", "object_index", "object_uid", "domain", "seed", "mode")}})
    records = run_jobs(jobs, workers)
    value = manifest(records, len(records) == 32, fitted=True)
    if not value["complete"] or value["image_count"] != 96:
        raise ValueError("Incomplete representative aligned renders")
    save(VIS / "aligned_manifest.json", value)
    print(json.dumps({"aligned_images": value["image_count"], "selected_indices": sorted(selected)}), flush=True)


def imgpath(row, seed, mode, view, fitted=False):
    prefix = f"gt_{row['object_index']:03d}" if mode == "gt" else f"{'aligned_' if fitted else ''}{row['index']:03d}_{seed}_{mode}"
    return VIS / "renders" / f"{prefix}_v{view}.png"


def caption(draw, xy, value, size=16, bold=False, color="#263648"):
    draw.text(xy, value, font=ImageFont.truetype(BOLD if bold else FONT, size), fill=color)


def paste(sheet, path, x, y, size):
    with Image.open(path) as im:
        sheet.paste(im.convert("RGB").resize((size, size), Image.Resampling.LANCZOS), (x, y))


def figure(sheet, name):
    path = OUT / "figures" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, dpi=(180, 180))
    sheet.save(path.with_suffix(".pdf"), resolution=180)
    return path


def native_sheet(rows, records, name, title, view=0, ranks=None):
    left, cell, top, row_height, footer = 190, 350, 135, 405, 105
    sheet = Image.new("RGB", (left + 3 * cell, top + len(rows) * row_height + footer), "white")
    draw = ImageDraw.Draw(sheet)
    caption(draw, (20, 16), title, 25, True)
    caption(draw, (20, 57), f"Native frame | seed 42 | fixed view {view+1} | same camera / material / scale", 18)
    for j, mode in enumerate(("gt", *MODES)):
        caption(draw, (left + j * cell + 17, 100), "GT (VAE proxy)" if mode == "gt" else LABELS[mode], 21, True)
    for i, row in enumerate(rows):
        y = top + i * row_height
        draw.line((15, y, sheet.width - 15, y), fill="#d6dde6")
        caption(draw, (18, y + 28), f"Dev object {row['object_index']:02d}", 20, True)
        caption(draw, (18, y + 66), row["object_uid"][:14], 13)
        if ranks:
            r = ranks[row["index"]]
            caption(draw, (18, y + 105), f"Rank {r['rank']}/{64 if row['domain']=='clean' else 16}", 18)
            caption(draw, (18, y + 138), "Mean CD gain", 16)
            caption(draw, (18, y + 166), f"{r['mean_native_cd_gain_v2_over_parent']:+.6f}", 17)
        for j, mode in enumerate(("gt", *MODES)):
            x = left + j * cell
            paste(sheet, imgpath(row, 42, mode, view), x, y + 4, cell)
            if mode != "gt":
                m = records[(row["index"], 42, mode)]["surface"]
                caption(draw, (x + 13, y + cell + 11), f"CD {m['chamfer_l1']:.5f}   F@.02 {m['fscore_0p02']:.3f}", 17)
    caption(draw, (20, sheet.height - 82), "Development objects; official SLat VAE-proxy GT. Distances use normalized model units.", 17)
    caption(draw, (20, sheet.height - 53), "CD = mean bidirectional Euclidean surface distance. No per-mesh display normalization.", 16)
    caption(draw, (20, sheet.height - 25), "Representative ranks use both seeds and are fixed by metrics; full cohort atlases are also available.", 15)
    return figure(sheet, name)


def overlay(gt_path, pred_path):
    with Image.open(gt_path) as im:
        gt = np.asarray(im.convert("RGB"), dtype=np.int16)
    with Image.open(pred_path) as im:
        pred = np.asarray(im.convert("RGB"), dtype=np.int16)
    background = np.array([249, 250, 251])
    g = np.max(np.abs(gt - background), axis=2) > 20
    p = np.max(np.abs(pred - background), axis=2) > 20
    rgb = np.full(gt.shape, 250, dtype=np.uint8)
    rgb[g & p] = [127, 143, 158]
    rgb[g & ~p] = [30, 164, 195]
    rgb[p & ~g] = [231, 131, 60]
    return Image.fromarray(rgb)


def make_overlays(rows, native, selection):
    by_index = {r["index"]: r for r in rows}
    selected = selected_indices(selection)
    destination = VIS / "overlays"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for key in sorted(native):
        index, seed, mode = key
        row = by_index[index]
        for fitted in ([False, True] if index in selected else [False]):
            for view in range(3):
                gt = imgpath(row, seed, "gt", view)
                pred = imgpath(row, seed, mode, view, fitted)
                output = destination / pred.name
                overlay(gt, pred).save(output)
                records.append({"index": index, "seed": seed, "mode": mode, "view": view,
                                "aligned": fitted, "gt_png": relative(gt), "prediction_png": relative(pred),
                                "path": relative(output), "sha256": sha(output)})
    if len(records) != 1056:
        raise ValueError("Overlay cohort incomplete")
    save(VIS / "overlay_manifest.json", {"complete": True, "image_count": len(records),
         "method": "RGB silhouette difference >20 from [249,250,251]; cyan GT-only, orange prediction-only, gray overlap; no depth comparison or scored metric.",
         "records": records})


def pose_sheet(row, native, fitted):
    left, cell, top, row_height = 175, 290, 140, 345
    sheet = Image.new("RGB", (left + 5 * cell, top + 6 * row_height + 110), "white")
    draw = ImageDraw.Draw(sheet)
    caption(draw, (20, 17), f"{DOMAIN_LABELS[row['domain']]} | object {row['object_index']:02d}: native / GT-assisted rigid fit", 25, True)
    caption(draw, (20, 57), "Seed 42 | proper rotation + translation only | identical fixed cameras | no scale or reflection", 17)
    for j, label in enumerate(("GT (VAE proxy)", "Parent30K", "V2 step2000", "Parent vs GT", "V2 vs GT")):
        caption(draw, (left + j * cell + 12, 103), label, 19, True)
    for view in range(3):
        for a in range(2):
            y = top + (2 * view + a) * row_height
            draw.line((15, y, sheet.width - 15, y), fill="#d6dde6")
            caption(draw, (19, y + 38), f"View {view+1}", 20, True)
            caption(draw, (19, y + 75), "SE(3) fit" if a else "Native", 19)
            paste(sheet, imgpath(row, 42, "gt", view), left, y + 4, cell)
            for j, mode in enumerate(MODES, 1):
                path = imgpath(row, 42, mode, view, bool(a))
                paste(sheet, path, left + j * cell, y + 4, cell)
                paste(sheet, VIS / "overlays" / path.name, left + (j + 2) * cell, y + 4, cell)
                r = fitted[(row["index"], 42, mode)]
                m = r["se3_aligned" if a else "native"]
                caption(draw, (left + j * cell + 12, y + cell + 10), f"CD {m['chamfer_l1']:.5f}   F {m['fscore_0p02']:.3f}", 16)
                if a:
                    p = r["alignment"]
                    caption(draw, (left + (j + 2) * cell + 8, y + cell + 10), f"fit {p['rotation_degrees']:.1f} deg; |t| {p['translation_norm']:.4f}", 15)
    caption(draw, (20, sheet.height - 84), "Cyan: GT only. Orange: prediction only. Gray: 2D silhouette overlap; depth is not compared here.", 17)
    caption(draw, (20, sheet.height - 55), "SE(3) uses GT and is a diagnostic, not an inference result. No mesh-specific display fitting is applied.", 17)
    caption(draw, (20, sheet.height - 26), "Fitted angles are not absolute pose errors: symmetry and shape differences affect the best rigid correction.", 16)
    return figure(sheet, f"pose_{row['domain']}_{row['object_index']:02d}")


def verify_renders(native, fitted, selection):
    selected = selected_indices(selection)
    expected = {f"{k[0]:03d}_{k[1]}_{k[2]}" for k in native} | {f"gt_{r['object_index']:03d}" for r in native.values()}
    expected_fitted = {f"aligned_{k[0]:03d}_{k[1]}_{k[2]}" for k in native if k[0] in selected}
    checked = 0
    for path, wanted, aligned in ((VIS / "manifest.json", expected, False), (VIS / "aligned_manifest.json", expected_fitted, True)):
        m = read(path)
        if not m["complete"] or {r["prefix"] for r in m["records"]} != wanted:
            raise ValueError("Render cohort incomplete")
        if m["parameters"] != renderer.parameters() or m["protocol_sha256"] != sha(OUT / "visual_protocol.json"):
            raise ValueError("Inconsistent renderer/protocol")
        for r in m["records"]:
            if sha(r["mesh"]["path"]) != r["mesh"]["sha256"]:
                raise ValueError("Source mesh changed")
            matrix = np.asarray(r["matrix_original_to_rendered"])
            if aligned:
                i = r["identity"]
                source = fitted[(i["index"], i["seed"], i["mode"])]
                if r["mesh"] != source["mesh"] or not np.array_equal(matrix, np.asarray(source["alignment"]["matrix_predicted_to_target"])):
                    raise ValueError("Displayed alignment differs from scored transform")
            elif not np.array_equal(matrix, np.eye(4)) or r["source_bounds"] != r["rendered_bounds"]:
                raise ValueError("Native mesh was transformed")
            if {v["view"] for v in r["views"]} != {0, 1, 2}:
                raise ValueError("Camera missing")
            for v in r["views"]:
                if not v["foreground_pixels_supersampled"] or v["vertices_outside_xy"] or v["vertices_outside_depth"] or v["foreground_touches_image_border"]:
                    raise ValueError("Empty/clipped/border touching render")
                if sha(v["png"]) != v["png_sha256"]:
                    raise ValueError("Render hash changed")
                with Image.open(v["png"]) as im:
                    if im.size != (512, 512) or im.mode != "RGB":
                        raise ValueError("Invalid image format")
                checked += 1
    if checked != 1248:
        raise ValueError("Image count mismatch")
    save(OUT / "visual_integrity.json", {"passed": True, "render_images_checked": checked,
         "native_identity_meshes": 384, "aligned_matrices_exact_equal_scored": 32,
         "same_cameras_material_lights_framing": True, "source_mesh_hashes_unchanged": True,
         "empty_images": 0, "clipped_images": 0, "border_touching_images": 0,
         "all_png_hashes_match": True, "all_pngs_512_rgb": True})


HTML = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dev64 / 强遮挡16 · 最终 mesh 验证</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f2f5f8;color:#213147;font:15px/1.6 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:26px}h1{font-size:28px}h2{font-size:20px}a{color:#096889}select,button{font:inherit;padding:8px;border:1px solid #bccad5;border-radius:6px;background:white}label{display:inline-flex;gap:8px;align-items:center;margin:5px 14px 5px 0}.controls,.box{background:white;border:1px solid #dce4eb;border-radius:10px;padding:16px;margin:18px 0}.controls{position:sticky;top:0;z-index:2}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px}.overgrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:13px}.card{background:white;border:1px solid #dce4eb;border-radius:10px;padding:13px}.card img{width:100%;display:block}.metric{font-variant-numeric:tabular-nums;font-size:14px}.subtle{color:#57677b;font-size:13px}.links{display:flex;flex-wrap:wrap;gap:14px}code{overflow-wrap:anywhere}@media(max-width:750px){main{padding:12px}.controls{position:static}.grid{grid-template-columns:1fr}.overgrid{grid-template-columns:1fr}}</style>
<main><h1>未见训练对象的最终 mesh：实际效果与位姿诊断</h1>
<p>干净 Dev64 与固定强遮挡16子集 · 推理种子42/43 · 父模型与当前 V2 · 共320个预测mesh。三组固定显示相机、材质、灯光和尺度完全一致，原生视图展示实际输出。Dev64 为开发集，不能替代独立论文测试集。</p>
<p class="subtle">参考表面为官方 SLat 经同一VAE解码的GT代理，并非原始高精度资产。CD为双向欧氏最近邻距离均值的平均；距离使用规范化模型坐标，0.02不表示2cm。</p>
<div class="controls"><label>输入条件 <select id="domain"><option value="clean">干净 Dev64</option><option value="strong_occluded">强遮挡16</option></select></label><label>对象 <select id="object"></select></label><label>种子 <select id="seed"><option>42</option><option>43</option></select></label><label>显示视角 <select id="view"><option value="0">1 · -45°</option><option value="1">2 · 75°</option><option value="2">3 · 195°</option></select></label><label>坐标 <select id="frame"><option value="native">原生坐标</option><option value="aligned">SE(3)拟合诊断</option></select></label><button id="prev">上一个</button> <button id="next">下一个</button><div id="frameNote" class="subtle"></div></div>
<div class="box"><strong id="identity"></strong><p id="rank" class="subtle"></p><p class="subtle">显示相机用于检查几何，并非输入图像相机。各输入条件的对象配对由固定计划文件绑定。</p></div>
<div id="cards" class="grid"></div><div class="box"><h2>同相机GT轮廓叠加</h2><p class="subtle">青色=仅GT；橙色=仅预测；灰色=二维轮廓重叠。此图不比较深度，不作为新增评测指标。</p><div id="overlays" class="overgrid"></div></div>
<div class="box"><h2>协议与完整数据</h2><p>SE(3)仅允许旋转和平移，禁止缩放、镜像和局部形变。拟合使用GT，是诊断结果；原生坐标仍是位姿约束任务的主要结果。拟合角度受对称性与形状差异影响，不能直接视为真实姿态误差。</p><p>干净输入的父模型结果复用已审计历史mesh；跨运行可有小幅数值波动，详情见<a href="runtime_repeatability/compare.json">重复性核查</a>。正式320项结果及可视化选择协议保持固定。</p><div class="links"><a href="plan.json">数据与模型绑定</a><a href="alignment/protocol.json">对齐协议</a><a href="alignment/aggregate.json">全量对齐指标</a><a href="visual_protocol.json">显示与选择协议</a><a href="visual_selection.json">代表样例完整排名</a><a href="visual_integrity.json">图片与变换核查</a><a href="final_report.json">最终报告</a></div></div>
<div class="box"><h2>全量图册</h2><p>按对象编号完整收录两个条件；每页4对象，固定种子42、视角1。浏览器覆盖两种子、三视角。</p><div id="atlasLinks" class="links"></div></div>
<div class="box"><h2>固定规则代表样例</h2><p>各条件独立按对象平均原生CD改善从小到大排序；干净选第1/17/33/49/64名，强遮挡选第1/9/16名。规则在检查图片之前声明，同时包含较差和较好结果。</p><div id="representativeLinks" class="links"></div></div>
</main><script>
const D=__DATA__;
const $=id=>document.getElementById(id);
const modes=['parent','raw'];
const labels={parent:'Parent30K',raw:'V2 step2000'};
function activeRows(){return D.rows.filter(r=>r.domain===$('domain').value)}
function populate(){const prior=Number($('object').value);$('object').replaceChildren();for(const row of activeRows()){const option=document.createElement('option');option.value=String(row.object_index);option.textContent=`${String(row.object_index).padStart(2,'0')} · ${row.object_uid.slice(0,10)}${row.selected?' · 代表样例':''}`;$('object').appendChild(option)}if(activeRows().some(r=>r.object_index===prior))$('object').value=String(prior);else $('object').value=String(activeRows()[0].object_index)}
function imgPath(row,mode,view,aligned){const prefix=mode==='gt'?`gt_${String(row.object_index).padStart(3,'0')}`:`${aligned?'aligned_':''}${String(row.index).padStart(3,'0')}_${$('seed').value}_${mode}`;return `visuals/renders/${prefix}_v${view}.png`}
function update(){const row=activeRows().find(r=>r.object_index===Number($('object').value));const view=Number($('view').value);$('frame').options[1].disabled=!row.selected;if(!row.selected)$('frame').value='native';const isAligned=$('frame').value==='aligned';$('frameNote').textContent=isAligned?'当前为GT辅助刚体拟合诊断；不是推理输出。显示相机、尺度保持不变。':`当前为原生坐标，无逐物体居中、旋转或尺度归一化。${row.selected?'可切换SE(3)图。':'拟合图仅导出预声明代表样例；全量拟合指标均已计算。'}`;$('identity').textContent=`Dev对象 ${row.object_index} · ${row.object_uid} · ${row.domain}`;$('rank').textContent=`该条件内原生CD改善排名 ${row.rank}/${activeRows().length}（从小到大），两种子平均改善 ${row.gain.toFixed(6)}`;$('cards').replaceChildren();$('overlays').replaceChildren();for(const mode of ['gt',...modes]){const card=document.createElement('div');card.className='card';const title=document.createElement('h2');title.textContent=mode==='gt'?'GT · VAE proxy':labels[mode];const image=document.createElement('img');image.src=imgPath(row,mode,view,isAligned);image.alt=title.textContent;card.append(title,image);const metric=document.createElement('div');metric.className='metric';if(mode==='gt'){metric.textContent='相同参考表面，原生坐标。';const link=document.createElement('a');link.href=row.target;link.textContent='参考PLY';card.append(metric,link)}else{const r=D.records[`${row.index}_${$('seed').value}_${mode}`];const m=r[isAligned?'aligned':'native'];metric.innerHTML=`CD ${m.chamfer_l1.toFixed(5)} ↓<br>F@0.02 ${m.fscore_0p02.toFixed(4)} ↑<br><span class="subtle">原生CD ${r.native.chamfer_l1.toFixed(5)} → 拟合后 ${r.aligned.chamfer_l1.toFixed(5)}<br>拟合旋转 ${r.angle.toFixed(2)}°；平移范数 ${r.translation.toFixed(5)}</span>`;const links=document.createElement('div');for(const [href,label] of [[r.mesh,'原始PLY'],[r.record,'指标/变换']]){const a=document.createElement('a');a.href=href;a.textContent=label;links.append(a,' ')}card.append(metric,links);const over=document.createElement('div');over.className='card';const name=document.createElement('strong');name.textContent=labels[mode];const overlay=document.createElement('img');overlay.src=imgPath(row,mode,view,isAligned).replace('/renders/','/overlays/');overlay.alt=`${labels[mode]}与GT轮廓`;over.append(name,overlay);$('overlays').appendChild(over)}$('cards').appendChild(card)}}
$('domain').addEventListener('change',()=>{populate();update()});for(const id of ['object','seed','view','frame'])$(id).addEventListener('change',update);for(const [id,delta] of [['prev',-1],['next',1]])$(id).onclick=()=>{const rows=activeRows();const i=rows.findIndex(r=>r.object_index===Number($('object').value));$('object').value=String(rows[(i+delta+rows.length)%rows.length].object_index);update()};
for(const f of D.atlases){for(const ext of ['png','pdf']){const a=document.createElement('a');a.href=`${f.path}.${ext}`;a.textContent=`${f.label} ${ext.toUpperCase()}`;$('atlasLinks').appendChild(a)}}for(const f of D.representatives){for(const ext of ['png','pdf']){const a=document.createElement('a');a.href=`${f.path}.${ext}`;a.textContent=`${f.label} ${ext.toUpperCase()}`;$('representativeLinks').appendChild(a)}}populate();update();
</script></html>'''


JS_HARNESS = r'''
const fs=require('fs'),vm=require('vm');
class Element {
  constructor(tag){this.tag=tag;this.children=[];this._value='';this.events={}}
  appendChild(x){this.children.push(x);return x}
  append(...xs){this.children.push(...xs)}
  replaceChildren(...xs){this.children=xs;if(this.tag==='select')this._value=''}
  addEventListener(name,f){this.events[name]=f}
  get options(){return this.children}
  get value(){return this._value||((this.tag==='select'&&this.children.length)?this.children[0].value:'')}
  set value(v){this._value=String(v)}
}
const elements={};
for(const id of ['domain','object','seed','view','frame'])elements[id]=new Element('select');
for(const id of ['cards','overlays','frameNote','identity','rank','prev','next','atlasLinks','representativeLinks'])elements[id]=new Element('div');
for(const [id,values] of [['domain',['clean','strong_occluded']],['seed',['42','43']],['view',['0','1','2']],['frame',['native','aligned']]])for(const v of values){const e=new Element('option');e.value=v;elements[id].appendChild(e)}
const context=vm.createContext({document:{getElementById:id=>elements[id],createElement:tag=>new Element(tag)},console});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
vm.runInContext(`
let checked=0;
for(const domain of ['clean','strong_occluded']){
  $('domain').value=domain;populate();
  const rows=activeRows();if(rows.length!==(domain==='clean'?64:16))throw Error('wrong domain size');
  for(const row of rows){$('object').value=row.object_index;for(const seed of [42,43]){$('seed').value=seed;for(const view of [0,1,2]){$('view').value=view;for(const frame of ['native',...(row.selected?['aligned']:[])]){$('frame').value=frame;update();if($('cards').children.length!==3||$('overlays').children.length!==2)throw Error('missing cards');if($('frame').options[1].disabled===row.selected)throw Error('wrong aligned availability');if(!$('identity').textContent.includes(row.object_uid))throw Error('identity mismatch');checked++}}}}
  $('object').value=rows[rows.length-1].object_index;$('next').onclick();if(Number($('object').value)!==rows[0].object_index)throw Error('next wrap');$('prev').onclick();if(Number($('object').value)!==rows[rows.length-1].object_index)throw Error('previous wrap');
}
$('domain').value='clean';populate();$('object').value='63';update();$('domain').value='strong_occluded';$('domain').events.change();if(Number($('object').value)!==0)throw Error('invalid object survived domain switch');
if(checked!==528)throw Error('wrong selectable combinations');
console.log(JSON.stringify({executed_combinations:checked,domain_switch:true,next_previous_wrap:true,aligned_availability:true}));
`,context);
'''


def make_gallery(rows, native, fitted, selection, atlas, representatives):
    rankings = {r["index"]: r for c in selection["cohorts"].values() for r in c["ranked_rows"]}
    selected = selected_indices(selection)
    data = {"rows": [], "records": {}, "atlases": atlas, "representatives": representatives}
    for row in sorted(rows, key=lambda r: r["index"]):
        rank = rankings[row["index"]]
        data["rows"].append({k: row[k] for k in ("index", "object_index", "object_uid", "domain")} |
                            {"rank": rank["rank"], "gain": rank["mean_native_cd_gain_v2_over_parent"],
                             "selected": row["index"] in selected, "target": relative(row["target_mesh"]["path"])})
    for key, r in sorted(native.items()):
        a = fitted[key]
        data["records"][f"{key[0]}_{key[1]}_{key[2]}"] = {
            "native": r["surface"], "aligned": a["se3_aligned"],
            "angle": a["alignment"]["rotation_degrees"], "translation": a["alignment"]["translation_norm"],
            "mesh": relative(r["mesh"]["path"]), "record": f"alignment/records/{key[0]:03d}_{key[1]}_{key[2]}.json"}
    save(OUT / "gallery_data.json", data)
    script = HTML.split("<script>", 1)[1].split("</script>", 1)[0].replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    (OUT / "index.html").write_text(HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))))
    script_path = OUT / "gallery_script.js"
    script_path.write_text(script)
    subprocess.run(["node", "--check", str(script_path)], check=True, capture_output=True, text=True)
    execution = subprocess.run(["node", "-e", JS_HARNESS, str(script_path)], check=True, capture_output=True, text=True)
    js_result = json.loads(execution.stdout)
    # Enumerate every selectable combination and its actual disk resources.
    checked, available = 0, []
    for row in data["rows"]:
        for seed in (42, 43):
            for view in range(3):
                for aligned in ([False, True] if row["selected"] else [False]):
                    for mode in ("gt", *MODES):
                        p = imgpath(row, seed, mode, view, aligned)
                        if not p.exists():
                            raise ValueError(f"Gallery image missing: {p}")
                        if mode != "gt":
                            if not (VIS / "overlays" / p.name).exists():
                                raise ValueError("Gallery overlay missing")
                            r = data["records"][f"{row['index']}_{seed}_{mode}"]
                            for field in ("mesh", "record"):
                                if not (OUT / r[field]).exists():
                                    raise ValueError("Gallery artifact missing")
                    checked += 1
        if not (OUT / row["target"]).exists():
            raise ValueError("Gallery GT link missing")
        available.append({"index": row["index"], "domain": row["domain"], "object_index": row["object_index"], "aligned_display": row["selected"]})
    for f in atlas + representatives:
        for ext in ("png", "pdf"):
            if not (OUT / f"{f['path']}.{ext}").exists():
                raise ValueError("Gallery figure missing")
    repeatability = OUT / "runtime_repeatability/compare.json"
    if not repeatability.exists():
        raise ValueError("Historical mesh runtime-repeatability link target is missing")
    save(OUT / "gallery_validation.json", {"passed": True, "node_syntax_check": True,
         "actual_gallery_js_execution_with_dom_stub": js_result,
         "selection_combinations_checked": checked, "condition_aware_rows": available,
         "all_images_overlays_meshes_metrics_and_figures_exist": True,
         "getImageData_used": False, "external_network_resources": False,
         "historical_parent_reuse_runtime_note_and_link": True,
         "runtime_repeatability_sha256": sha(repeatability),
         "html_sha256": sha(OUT / "index.html"), "script_sha256": sha(script_path)})


def build_report():
    rows, native = load_records(True)
    selection = select(rows, native)
    fitted = aligned_records(native)
    verify_renders(native, fitted, selection)
    make_overlays(rows, native, selection)
    figures, atlas, representatives = [], [], []
    by_index = {r["index"]: r for r in rows}
    for domain in DOMAINS:
        cohort = sorted([r for r in rows if r["domain"] == domain], key=lambda r: r["object_index"])
        for start in range(0, len(cohort), 4):
            name = f"native_atlas_{domain}_{start//4+1:02d}"
            title = f"{DOMAIN_LABELS[domain]} | full mesh atlas {start//4+1}/{len(cohort)//4}"
            figures.append(native_sheet(cohort[start:start+4], native, name, title))
            atlas.append({"path": f"figures/{name}", "label": f"{domain} 对象 {cohort[start]['object_index']:02d}–{cohort[start+3]['object_index']:02d}"})
        chosen = selection["cohorts"][domain]["representative_rows"]
        ranks = {r["index"]: r for r in chosen}
        for view in range(3):
            name = f"representative_native_{domain}_v{view}"
            figures.append(native_sheet([by_index[r["index"]] for r in chosen], native, name,
                                        f"{DOMAIN_LABELS[domain]} | fixed-rank representative meshes", view, ranks))
            representatives.append({"path": f"figures/{name}", "label": f"{domain} 代表样例 视角{view+1}"})
        for chosen_row in chosen:
            row = by_index[chosen_row["index"]]
            path = pose_sheet(row, native, fitted)
            figures.append(path)
            representatives.append({"path": relative(path.with_suffix("")), "label": f"{domain} 对象{row['object_index']:02d} 原生/SE(3)"})
    make_gallery(rows, native, fitted, selection, atlas, representatives)
    files = [OUT / x for x in ("index.html", "gallery_data.json", "gallery_script.js", "gallery_validation.json", "visual_integrity.json", "visual_protocol.json", "visual_selection.json", "visuals/manifest.json", "visuals/aligned_manifest.json", "visuals/overlay_manifest.json")]
    for f in figures:
        files.extend([f, f.with_suffix(".pdf")])
    save(OUT / "visual_report_manifest.json", {"complete": True, "completed_utc": utc(),
         "native_cases": 320, "gt_meshes": 64, "native_render_images": 1152,
         "aligned_render_images": 96, "overlay_images": 1056,
         "atlas_pages": 20, "figure_pairs_png_pdf": len(figures),
         "script": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
         "representative_indices": sorted(selected_indices(selection)),
         "artifacts": [{"path": relative(p), "sha256": sha(p)} for p in files]})
    print(json.dumps({"complete": True, "gallery": str(OUT / "index.html"), "figure_pairs": len(figures)}), flush=True)


def refresh_gallery():
    """Update method disclosure without regenerating or modifying any figure."""
    manifest_path = OUT / "visual_report_manifest.json"
    value = read(manifest_path)
    mutable = {"index.html", "gallery_data.json", "gallery_script.js", "gallery_validation.json"}
    if not value["complete"]:
        raise ValueError("The original visual report must finish first")
    for binding in value["artifacts"]:
        if sha(OUT / binding["path"]) != binding["sha256"]:
            raise ValueError(f"Existing visual artifact changed: {binding['path']}")
    data = read(OUT / "gallery_data.json")
    rows, native = load_records(True)
    selection = select(rows, native)
    fitted = aligned_records(native)
    make_gallery(rows, native, fitted, selection, data["atlases"], data["representatives"])
    for binding in value["artifacts"]:
        current = sha(OUT / binding["path"])
        if binding["path"] in mutable:
            binding["sha256"] = current
        elif current != binding["sha256"]:
            raise ValueError("A figure or render manifest changed during the gallery update")
    value["script"]["sha256"] = sha(__file__)
    value["html_method_note_updated_utc"] = utc()
    value["figures_unchanged_during_method_note_update"] = True
    save(manifest_path, value)
    print(json.dumps({"gallery_method_note_updated": True, "figures_unchanged": True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["protocol", "native", "selection", "aligned", "report", "finish", "gallery"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 3:
        parser.error("Use 1–3 rendering workers")
    if args.action == "protocol":
        protocol()
    elif args.action == "native":
        render_native(args.workers, args.watch)
    elif args.action == "selection":
        rows, records = load_records(True)
        select(rows, records)
    elif args.action == "aligned":
        render_aligned(args.workers)
    elif args.action == "gallery":
        refresh_gallery()
    elif args.action == "finish":
        previous = None
        while True:
            native_path = VIS / "manifest.json"
            count = (len(list((OUT / "records").glob("*.json"))),
                     len(list((OUT / "alignment/records").glob("*.json"))),
                     read(native_path)["complete"] if native_path.exists() else False)
            if count != previous:
                print(json.dumps({"utc": utc(), "waiting_for_final_artifacts": {
                    "native_cases": count[0], "aligned_cases": count[1], "native_renders_complete": count[2]}}), flush=True)
                previous = count
            if count == (320, 320, True):
                break
            if not args.watch:
                raise ValueError("Final native/aligned prerequisites incomplete; --watch waits for them")
            time.sleep(20)
        render_aligned(args.workers)
        build_report()
    else:
        build_report()


if __name__ == "__main__":
    main()
