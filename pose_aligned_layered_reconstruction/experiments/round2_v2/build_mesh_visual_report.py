"""Build an offline mesh gallery and scientific figures from audited renders.

This file never changes or aligns a mesh. It composes the images made by
render_mesh_audit.py and reads the independently recorded SE(3) diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PACKAGE = Path(__file__).resolve().parents[2]
SOURCE = PACKAGE / "outputs/round2_v2_diagnostics_20261018/mesh"
OUT = PACKAGE / "outputs/round2_v2_mesh_visual_audit_20261018"
MODES = ("parent", "v1", "raw")
LABELS = {"parent": "Parent30K", "v1": "V1 step2000", "raw": "V2 step2000"}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def relative(path):
    return os.path.relpath(path, OUT)


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def render_path(obj, index, seed, mode, view, aligned=False):
    stem = f"gt_{obj:03d}" if mode == "gt" else f"{'aligned_' if aligned else ''}{index:03d}_{seed}_{mode}"
    return OUT / "visuals/renders" / f"{stem}_v{view}.png"


def caption(draw, xy, value, size=16, color="#263648", bold=False):
    draw.text(xy, value, font=font(size, bold), fill=color)


def paste_image(sheet, path, box):
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail(box[2:4], Image.Resampling.LANCZOS)
        sheet.paste(image, (box[0] + (box[2] - image.width) // 2,
                            box[1] + (box[3] - image.height) // 2))


def write_figure(image, name, pdf=True):
    path = OUT / "figures" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(180, 180))
    if pdf:
        image.save(path.with_suffix(".pdf"), resolution=180.0)
    return path


def load_data():
    plan = read(SOURCE / "plan.json")
    selection = read(OUT / "visual_selection.json")
    if sha(SOURCE / "plan.json") != selection["source_plan_sha256"]:
        raise ValueError("selection/source plan changed")
    rows = plan["plan"]["rows"]
    records = {}
    for row in rows:
        for case in row["cases"]:
            key = (row["index"], case["seed"], case["mode"])
            record = read(SOURCE / "records" / f"{key[0]:03d}_{key[1]}_{key[2]}.json")
            if record["identity_hash"] != plan["identity_hash"] or record["object_uid"] != row["object_uid"]:
                raise ValueError("source identity mismatch")
            records[key] = record
    if len(records) != 384:
        raise ValueError("incomplete native data")
    aligned = {}
    for path in sorted((OUT / "alignment/records").glob("*.json")):
        record = read(path)
        key = (record["index"], record["seed"], record["mode"])
        if key not in records or record["mesh"] != records[key]["mesh"]:
            raise ValueError("alignment identity mismatch")
        aligned[key] = record
    return rows, records, aligned, selection


def make_inputs(rows):
    paths = {}
    bindings = []
    recovery = read(OUT / "clean_input_recovery.json")
    recovered = {r["index"]: r for r in recovery["records"]}
    if not recovery["complete"] or len(recovered) != 32:
        raise ValueError("clean input recovery incomplete")
    destination = OUT / "visuals/inputs"
    destination.mkdir(parents=True, exist_ok=True)
    for row in rows:
        source = Path(row["sample"]["path"]).parent / ("raw_view0.png" if row["domain"] == "legacy_clean" else "raw_00.png")
        path = destination / f"{row['index']:03d}.png"
        if row["domain"] == "legacy_clean":
            if not path.exists() or not (OUT / "clean_input_recovery.json").exists():
                raise FileNotFoundError("Run recover_mesh_input_previews.py first")
            binding = recovered[row["index"]]
            if binding["object_uid"] != row["object_uid"] or binding["sample"] != row["sample"]:
                raise ValueError("recovered input belongs to a different sample")
            if Path(binding["output"]["path"]).resolve() != path.resolve() or sha(path) != binding["output"]["sha256"]:
                raise ValueError("recovered input image changed")
            source = path
        else:
            if not source.exists():
                raise FileNotFoundError(source)
            with Image.open(source) as image:
                image = image.convert("RGB")
                image.thumbnail((512, 512), Image.Resampling.LANCZOS)
                image.save(path)
        paths[row["index"]] = relative(path)
        bindings.append({"index": row["index"], "object_uid": row["object_uid"],
                         "source": str(source), "source_sha256": sha(source),
                         "output": relative(path), "output_sha256": sha(path)})
    save(OUT / "visuals/input_manifest.json", {
        "description": "Input view 0: clean rebuilt with original preprocessing, occluded from cached PNG. Not the fixed visualization camera. Models receive eight views; this shows one only.",
        "clean_recovery": {"path": "clean_input_recovery.json", "sha256": sha(OUT / "clean_input_recovery.json")},
        "records": bindings})
    return paths


def native_sheet(objects, rows, records, view, name, title, seed=42, cell=300):
    left, top, gap, footer = 180, 132, 50, 90
    row_height = cell + gap
    sheet = Image.new("RGB", (left + 4 * cell, top + len(objects) * 2 * row_height + footer), "white")
    draw = ImageDraw.Draw(sheet)
    caption(draw, (22, 17), title, 26, bold=True)
    caption(draw, (22, 55), f"Native frame | seed {seed} | fixed view {view + 1} | same camera / material / scale", 18)
    for j, mode in enumerate(("gt", *MODES)):
        caption(draw, (left + j * cell + 18, 95), "GT (VAE proxy)" if mode == "gt" else LABELS[mode], 20, bold=True)
    for n, obj in enumerate(objects):
        object_rows = sorted([r for r in rows if r["object_index"] == obj], key=lambda r: r["index"])
        for d, row in enumerate(object_rows):
            y = top + (2 * n + d) * row_height
            draw.line((15, y, sheet.width - 15, y), fill="#d6dde6")
            caption(draw, (18, y + 26), f"Object {obj:02d}", 20, bold=True)
            caption(draw, (18, y + 57), "Clean" if d == 0 else "Occluded", 18)
            caption(draw, (18, y + 86), row["object_uid"][:10], 13)
            for j, mode in enumerate(("gt", *MODES)):
                x = left + j * cell
                paste_image(sheet, render_path(obj, row["index"], seed, mode, view), (x, y + 4, cell, cell))
                if mode != "gt":
                    metric = records[(row["index"], seed, mode)]["surface"]
                    caption(draw, (x + 14, y + cell + 8), f"CD {metric['chamfer_l1']:.5f}   F@.02 {metric['fscore_0p02']:.3f}", 16)
    caption(draw, (20, sheet.height - 67), "32 training objects; GT is an official SLat VAE proxy. CD = mean bidirectional Euclidean distance.", 15)
    caption(draw, (20, sheet.height - 40), "Coordinates are normalized model units. No per-mesh recentering, scaling, smoothing or component filtering.", 15)
    return write_figure(sheet, name)


def silhouette_overlay(gt_path, pred_path):
    # The fixed renderer background is [249,250,251]; material is much darker.
    # This is a derived 2D silhouette illustration, not a new scored metric.
    with Image.open(gt_path) as image:
        gt = np.asarray(image.convert("RGB"), dtype=np.int16)
    with Image.open(pred_path) as image:
        pred = np.asarray(image.convert("RGB"), dtype=np.int16)
    background = np.array([249, 250, 251])
    g = np.max(np.abs(gt - background), axis=2) > 20
    p = np.max(np.abs(pred - background), axis=2) > 20
    result = np.full(gt.shape, 250, np.uint8)
    result[g & p] = [127, 143, 158]
    result[g & ~p] = [30, 164, 195]
    result[p & ~g] = [231, 131, 60]
    return Image.fromarray(result)


def make_overlays(rows, records, selected):
    destination = OUT / "visuals/overlays"
    destination.mkdir(parents=True, exist_ok=True)
    by_index = {r["index"]: r for r in rows}
    manifest = []
    for index, seed, mode in sorted(records):
        obj = by_index[index]["object_index"]
        for aligned in ([False, True] if obj in selected else [False]):
            for view in range(3):
                gt = render_path(obj, index, seed, "gt", view)
                pred = render_path(obj, index, seed, mode, view, aligned)
                target = destination / pred.name
                silhouette_overlay(gt, pred).save(target)
                manifest.append({"index": index, "object_index": obj, "seed": seed,
                                 "mode": mode, "aligned": aligned, "view": view,
                                 "gt_png": relative(gt), "prediction_png": relative(pred),
                                 "path": relative(target), "sha256": sha(target)})
    save(OUT / "visuals/overlay_manifest.json", {
        "complete": True, "image_count": len(manifest),
        "method": "RGB silhouette threshold; cyan GT only, orange prediction only, gray overlap; no depth comparison",
        "records": manifest})


def main_comparison(rows, records, inputs):
    # The worst, median and best objects from the predeclared ranks, not hand-picked views.
    objects = [26, 27, 0]
    cell, left, top, row_height = 350, 180, 145, 405
    sheet = Image.new("RGB", (left + 4 * cell, top + 3 * row_height + 105), "white")
    draw = ImageDraw.Draw(sheet)
    caption(draw, (22, 17), "Actual mesh outputs: objects ranked by mean improvement over Parent", 27, bold=True)
    caption(draw, (22, 57), "Occluded input | seed 42 | fixed view 1 | unchanged native mesh coordinates", 20)
    for j, name in enumerate(("Input view 0", "GT (VAE proxy)", "Parent30K", "V2 step2000")):
        caption(draw, (left + j * cell + 22, 106), name, 23, bold=True)
    ranked = {r["object_index"]: (i + 1, r["mean_native_cd_gain_v2_over_parent"])
              for i, r in enumerate(read(OUT / "visual_selection.json")["ranked_objects"])}
    for i, obj in enumerate(objects):
        row = next(r for r in rows if r["object_index"] == obj and r["domain"] != "legacy_clean")
        y = top + i * row_height
        draw.line((15, y, sheet.width - 15, y), fill="#d6dde6")
        caption(draw, (20, y + 70), f"Object {obj:02d}", 22, bold=True)
        rank, gain = ranked[obj]
        caption(draw, (20, y + 108), f"Rank {rank}/32", 20)
        caption(draw, (20, y + 146), "Mean CD gain", 16)
        caption(draw, (20, y + 173), f"{gain:+.5f}", 18)
        paste_image(sheet, OUT / inputs[row["index"]], (left + 20, y + 25, cell - 40, cell - 40))
        for j, mode in enumerate(("gt", "parent", "raw"), 1):
            paste_image(sheet, render_path(obj, row["index"], 42, mode, 0), (left + j * cell, y + 4, cell, cell))
            if mode != "gt":
                m = records[(row["index"], 42, mode)]["surface"]
                caption(draw, (left + j * cell + 20, y + cell + 8), f"CD {m['chamfer_l1']:.5f}   F@.02 {m['fscore_0p02']:.3f}", 18)
    caption(draw, (22, sheet.height - 85), "Selection uses each object's mean CD gain across BOTH input conditions and BOTH seeds, before viewing renders.", 18)
    caption(draw, (22, sheet.height - 55), "Input shows view 0 of 8; it may itself be unobstructed. GT / Parent / V2 share the same display camera and scale.", 18)
    caption(draw, (22, sheet.height - 25), "Training objects and VAE-proxy GT; this is a diagnostic, not evidence of held-out final-mesh generalization.", 17)
    return write_figure(sheet, "mesh_comparison")


def pose_sheet(obj, rows, records, aligned):
    # One object, all three complementary views, occluded input, seed 42.
    row = next(r for r in rows if r["object_index"] == obj and r["domain"] != "legacy_clean")
    cell, left, top, row_height = 300, 180, 140, 355
    sheet = Image.new("RGB", (left + 5 * cell, top + 6 * row_height + 105), "white")
    draw = ImageDraw.Draw(sheet)
    caption(draw, (22, 17), f"Object {obj:02d}: native output and GT-assisted rigid diagnostic", 26, bold=True)
    caption(draw, (22, 58), "Occluded input | seed 42 | proper rotation + translation only | no scale or reflection", 18)
    headings = ["GT (VAE proxy)", "Parent30K", "V2 step2000", "Parent vs GT", "V2 vs GT"]
    for j, name in enumerate(headings):
        caption(draw, (left + j * cell + 18, 104), name, 19, bold=True)
    for view in range(3):
        for a in range(2):
            y = top + (view * 2 + a) * row_height
            draw.line((15, y, sheet.width - 15, y), fill="#d6dde6")
            caption(draw, (20, y + 35), f"View {view + 1}", 20, bold=True)
            caption(draw, (20, y + 70), "Native" if not a else "SE(3) fit", 19)
            gt = render_path(obj, row["index"], 42, "gt", view)
            paste_image(sheet, gt, (left, y + 3, cell, cell))
            for j, mode in enumerate(("parent", "raw"), 1):
                pred = render_path(obj, row["index"], 42, mode, view, bool(a))
                paste_image(sheet, pred, (left + j * cell, y + 3, cell, cell))
                overlay = silhouette_overlay(gt, pred).resize((cell, cell), Image.Resampling.LANCZOS)
                sheet.paste(overlay, (left + (j + 2) * cell, y + 3))
                record = aligned[(row["index"], 42, mode)]
                metric = record["se3_aligned" if a else "native"]
                caption(draw, (left + j * cell + 12, y + cell + 7), f"CD {metric['chamfer_l1']:.5f}   F {metric['fscore_0p02']:.3f}", 16)
                if a:
                    pose = record["alignment"]
                    caption(draw, (left + (j + 2) * cell + 12, y + cell + 7), f"fit {pose['rotation_degrees']:.1f} deg; |t| {pose['translation_norm']:.4f}", 15)
    caption(draw, (22, sheet.height - 85), "Overlay: cyan = GT only; orange = prediction only; gray = overlap (2D silhouette, no depth comparison).", 17)
    caption(draw, (22, sheet.height - 56), "Fit rows use GT for SE(3); they are diagnostics, not inference outputs. Identical fixed cameras in every row.", 17)
    caption(draw, (22, sheet.height - 29), "Fitted rotation is not absolute pose error: symmetries and shape differences can change the best rigid correction.", 16)
    return write_figure(sheet, f"pose_object_{obj:02d}")


def make_plots(aligned, selection):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {"parent": "#738499", "v1": "#b39042", "raw": "#147eab"}
    for mode in MODES:
        records = [r for r in aligned.values() if r["mode"] == mode]
        x = [r["native"]["chamfer_l1"] for r in records]
        y = [r["se3_aligned"]["chamfer_l1"] for r in records]
        axes[0, 0].scatter(x, y, s=17, alpha=.65, color=colors[mode], label=LABELS[mode])
    maximum = max(r["native"]["chamfer_l1"] for r in aligned.values()) * 1.03
    axes[0, 0].plot([0, maximum], [0, maximum], "--", color="#b0b5bd", lw=1)
    axes[0, 0].set(xlabel="Native CD (model units)", ylabel="SE(3)-aligned CD", title="All 384 reconstructions; below line = alignment helps")
    axes[0, 0].legend(frameon=False)
    ordered = [r["object_index"] for r in selection["ranked_objects"]]
    for field, label, color in (("native", "Native", "#96a4b4"), ("se3_aligned", "After SE(3)", "#147eab")):
        gains = []
        for obj in ordered:
            subset = [r for r in aligned.values() if r["object_index"] == obj]
            means = {mode: np.mean([r[field]["chamfer_l1"] for r in subset if r["mode"] == mode]) for mode in MODES}
            gains.append(means["parent"] - means["raw"])
        shift = -.18 if field == "native" else .18
        axes[0, 1].bar(np.arange(32) + shift, gains, width=.36, label=label, color=color)
    axes[0, 1].axhline(0, lw=.8, color="#626c7a")
    axes[0, 1].set_xticks(range(32), [f"{x:02d}" for x in ordered], rotation=90, fontsize=8)
    axes[0, 1].set(ylabel="CD(Parent) - CD(V2); positive is better", xlabel="Object ID (ordered by native improvement)", title="Does V2's improvement survive pose removal?")
    axes[0, 1].legend(frameon=False)
    for ax, key, ylabel in ((axes[1, 0], "rotation_degrees", "Best-fit rotation (degrees)"),
                            (axes[1, 1], "translation_norm", "Best-fit translation norm (model units)")):
        values = [[r["alignment"][key] for r in aligned.values() if r["mode"] == mode] for mode in MODES]
        ax.boxplot(values, tick_labels=[LABELS[m] for m in MODES], showfliers=False)
        rng = np.random.default_rng(42)
        for j, (mode, data) in enumerate(zip(MODES, values), 1):
            ax.scatter(j + rng.uniform(-.13, .13, len(data)), data, s=12, alpha=.5, color=colors[mode])
        ax.set(ylabel=ylabel, title="GT-assisted rigid correction; 128 cases per method")
    fig.suptitle("Pose and shape audit | 32 training objects, clean + occluded, seeds 42/43", fontsize=17)
    fig.text(.5, .012, "No scale/reflection. GT is a VAE proxy. Large fitted angles can be symmetry-equivalent; they are not ground-truth pose errors.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .96))
    path = OUT / "figures/pose_shape_audit.png"
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


HTML = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mesh 可视化与坐标审计</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f2f5f8;color:#213147;font:15px/1.6 system-ui,sans-serif}main{max-width:1600px;margin:auto;padding:28px}h1{font-size:29px;margin:0 0 12px}h2{font-size:19px}p{max-width:1200px}a{color:#096889}select,button{font:inherit;padding:8px 12px;border:1px solid #bccad5;border-radius:6px;background:white}label{display:inline-flex;gap:9px;align-items:center;margin:5px 18px 5px 0}.controls,.box{background:white;border:1px solid #dce4eb;border-radius:10px;padding:16px;margin:18px 0}.controls{position:sticky;top:0;z-index:2}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px}.card{background:white;border-radius:10px;padding:13px;border:1px solid #dce4eb}.card h2{margin:0}.card img{width:100%;display:block}.metric{font-variant-numeric:tabular-nums;font-size:14px}.subtle{color:#57677b;font-size:13px}.warning{color:#8b4b09}.input{display:flex;gap:24px;align-items:center}.input img{width:180px;max-height:180px;object-fit:contain}.links{display:flex;flex-wrap:wrap;gap:16px}.figure{max-width:100%;border:1px solid #dce4eb}#contact{max-width:100%;max-height:1000px;object-fit:contain}canvas{width:100%;display:block}code{overflow-wrap:anywhere}@media(max-width:850px){.grid{grid-template-columns:repeat(2,1fr)}main{padding:12px}.controls{position:static}}@media(max-width:450px){.grid{grid-template-columns:1fr}.input{display:block}}
</style><main>
<h1>Mesh 实际效果与旋转、平移检查</h1>
<p>32 个训练对象 · 干净 / 首个遮挡输入 · 推理种子 42 / 43 · Parent、V1、V2 共 384 个重建。所有 mesh 使用相同的三组固定相机、材质、灯光和显示尺度。<strong>原生坐标视图展示模型实际输出。</strong></p>
<p class="subtle">参考表面为官方 SLat 经同一 VAE 解码得到的 GT 代理，并非原始高精度资产。所有对象均见于 SLat30K 训练。距离单位为规范化模型坐标，0.02 不能解读为 2 cm。V1 是内部历史版本。</p>
<div class="controls">
<label>对象 <select id="object"></select></label><label>输入 <select id="domain"><option value="legacy_clean">干净</option><option value="legacy_composite" selected>遮挡</option></select></label>
<label>种子 <select id="seed"><option>42</option><option>43</option></select></label><label>视角 <select id="view"><option value="0">1 · -45°</option><option value="1">2 · 75°</option><option value="2">3 · 195°</option></select></label>
<label>显示 <select id="frame"><option value="native">原生坐标</option><option value="aligned">SE(3) 对齐诊断</option></select></label><button id="prev">上一个</button> <button id="next">下一个</button>
<div id="frameNote" class="subtle"></div></div>
<div class="box input"><img id="input" alt="缓存输入视图 0"><div><strong id="identity"></strong><p class="subtle">左侧是输入视图 0 的缩略图；模型实际接收 8 个视图。该输入相机与下方用于检查形状的固定显示相机不同。</p><p id="rank" class="subtle"></p></div></div>
<div id="cards" class="grid"></div>
<div class="box"><h2>相同显示相机下的 GT 轮廓叠加</h2><p class="subtle">青色 = 仅 GT；橙色 = 仅预测；灰色 = 轮廓重叠。此图只比较二维轮廓，不代表深度一致。</p><div id="overlays" class="grid"></div></div>
<div class="box"><h2>完整证据</h2><p>对齐仅允许旋转和平移，禁止缩放、镜像、局部形变和删除连通块。它使用 GT 寻找刚体修正，属于诊断；不能用它替代带位姿约束任务的原生坐标结果。拟合角度受形状对称性影响，不能直接称为真实姿态误差。</p><div class="links"><a href="coordinate_audit.json">坐标链审计</a><a href="alignment/aggregate.json">384 项原生 / 对齐指标</a><a href="alignment/protocol.json">对齐协议</a><a href="visual_selection.json">代表样例选择规则</a><a href="final_report.json">最终汇总</a></div></div>
<div class="box"><h2>姿态与形状诊断</h2><a href="figures/pose_shape_audit.pdf">下载 PDF</a><img class="figure" src="figures/pose_shape_audit.png" alt="原生和刚体对齐指标图"></div>
<div class="box"><h2>全量图册</h2><p>每页 4 个对象，均展示干净与遮挡输入。图册固定种子 42、视角 1；上方浏览器覆盖两颗种子和全部三个视角。</p><div id="atlasLinks" class="links"></div></div>
<div class="box"><h2>代表对象</h2><p>在查看图片前，按 V2 相对 Parent 的对象平均原生 CD 改善排序，选第 1、9、17、25、32 名，包含退步样例；不据视觉效果挑图。各图均有 PNG / PDF。</p><div id="representativeLinks" class="links"></div></div>
</main><script>
const D=__DATA__;
const $=id=>document.getElementById(id);
const labels={parent:'Parent30K',v1:'V1 step2000',raw:'V2 step2000'};
const modes=['parent','v1','raw'];
for(const o of D.objects){const option=document.createElement('option');option.value=o.id;option.textContent=`${String(o.id).padStart(2,'0')} · ${o.uid.slice(0,10)}${o.selected?' · 代表样例':''}`;$('object').appendChild(option)}
$('object').value='27';
function imgPath(row,mode,view,aligned){let stem=mode==='gt'?`gt_${String(row.object_index).padStart(3,'0')}`:`${aligned?'aligned_':''}${String(row.index).padStart(3,'0')}_${$('seed').value}_${mode}`;return `visuals/renders/${stem}_v${view}.png`}
function overlay(a,b,target){target.src=b.replace('/renders/','/overlays/');target.alt='GT / prediction silhouette overlay'}
function update(){const obj=D.objects.find(o=>o.id===Number($('object').value));const row=D.rows.find(r=>r.object_index===obj.id&&r.domain===$('domain').value);const view=Number($('view').value);$('frame').options[1].disabled=!obj.selected;if(!obj.selected)$('frame').value='native';const isAligned=$('frame').value==='aligned';$('frameNote').textContent=isAligned?'当前为 GT 辅助对齐诊断：不是推理输出。相机与原生视图完全相同。':`当前为原生坐标，无逐物体居中、旋转或尺度归一化。${obj.selected?'可切换对齐图。':'对齐图仅导出了预先选定的五个代表对象；完整对齐指标已计算。'}`;$('input').src=row.input;$('identity').textContent=`对象 ${obj.id} · ${obj.uid}`;$('rank').textContent=`V2 原生 CD 改善排名 ${obj.rank}/32（从小到大）；对象平均改善 ${obj.gain.toFixed(6)}`;$('cards').replaceChildren();$('overlays').replaceChildren();for(const mode of ['gt',...modes]){const card=document.createElement('div');card.className='card';const title=document.createElement('h2');title.textContent=mode==='gt'?'GT · VAE proxy':labels[mode];card.appendChild(title);const image=document.createElement('img');image.src=imgPath(row,mode,view,isAligned);image.alt=title.textContent;card.appendChild(image);const text=document.createElement('div');text.className='metric';if(mode==='gt'){text.textContent='同一参考表面；保持原生坐标。';const link=document.createElement('a');link.href=row.target;link.textContent='下载参考 PLY';card.append(text,link)}else{const r=D.records[`${row.index}_${$('seed').value}_${mode}`];const m=r[isAligned?'aligned':'native'];text.innerHTML=`CD ${m.chamfer_l1.toFixed(5)} ↓<br>F@0.02 ${m.fscore_0p02.toFixed(4)} ↑<br><span class="subtle">原生 CD ${r.native.chamfer_l1.toFixed(5)} → 对齐 ${r.aligned.chamfer_l1.toFixed(5)}<br>拟合旋转 ${r.angle.toFixed(2)}°；平移范数 ${r.translation.toFixed(5)}</span>`;const links=document.createElement('div');const ply=document.createElement('a');ply.href=r.mesh;ply.textContent='原始 PLY';const data=document.createElement('a');data.href=r.record;data.textContent='指标 / 变换';links.append(ply,' · ',data);card.append(text,links);const over=document.createElement('div');over.className='card';const name=document.createElement('strong');name.textContent=labels[mode];const canvas=document.createElement('img');over.append(name,canvas);$('overlays').append(over);overlay(imgPath(row,'gt',view,false),image.src,canvas)}$('cards').appendChild(card)}}
for(const id of ['object','domain','seed','view','frame'])$(id).addEventListener('change',update);
for(const [id,delta] of [['prev',-1],['next',1]])$(id).onclick=()=>{$('object').value=String((Number($('object').value)+delta+32)%32);update()};
for(let i=1;i<=8;i++){const a=document.createElement('a');a.href=`figures/native_atlas_${String(i).padStart(2,'0')}.pdf`;a.textContent=`对象 ${(i-1)*4}–${i*4-1} PDF`;$('atlasLinks').appendChild(a)}
for(const o of D.objects.filter(o=>o.selected)){for(const type of ['png','pdf']){const a=document.createElement('a');a.href=`figures/pose_object_${String(o.id).padStart(2,'0')}.${type}`;a.textContent=`对象 ${o.id} 原生/对齐 ${type.toUpperCase()}`;$('representativeLinks').appendChild(a)}}
update();
</script></html>'''


def make_gallery(rows, records, aligned, selection, inputs):
    selected = {r["object_index"] for r in selection["representative_objects"]}
    ranked = {r["object_index"]: (i + 1, r) for i, r in enumerate(selection["ranked_objects"])}
    data = {"objects": [], "rows": [], "records": {}}
    for obj in range(32):
        rank, r = ranked[obj]
        data["objects"].append({"id": obj, "uid": r["object_uid"], "rank": rank,
                                "gain": r["mean_native_cd_gain_v2_over_parent"], "selected": obj in selected})
    for row in rows:
        data["rows"].append({k: row[k] for k in ("index", "object_index", "object_uid", "domain")} |
                            {"input": inputs[row["index"]], "target": relative(SOURCE / "targets" / f"{row['object_index']:03d}.ply")})
    for key, native in records.items():
        r = aligned[key]
        data["records"][f"{key[0]}_{key[1]}_{key[2]}"] = {
            "native": r["native"], "aligned": r["se3_aligned"],
            "angle": r["alignment"]["rotation_degrees"], "translation": r["alignment"]["translation_norm"],
            "mesh": relative(native["mesh"]["path"]), "record": f"alignment/records/{key[0]:03d}_{key[1]}_{key[2]}.json"}
    (OUT / "index.html").write_text(HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))))
    save(OUT / "gallery_data.json", data)


def verify_renders(aligned, selection):
    native = read(OUT / "visuals/manifest.json")
    fitted = read(OUT / "visuals/aligned_manifest.json")
    representatives = {r["object_index"] for r in selection["representative_objects"]}
    expected_fitted = {f"aligned_{k[0]:03d}_{k[1]}_{k[2]}" for k, r in aligned.items()
                       if r["object_index"] in representatives}
    if {k: v for k, v in native["parameters"].items() if k != "mesh_transform"} != {
            k: v for k, v in fitted["parameters"].items() if k != "mesh_transform"}:
        raise ValueError("native/aligned cameras or material differ")
    actual_fitted = {r["prefix"] for r in fitted["records"]}
    if actual_fitted != expected_fitted or len(actual_fitted) != 60:
        raise ValueError("incomplete or unexpected aligned render cohort")
    image_count, transform_count = 0, 0
    for manifest, expected_count, is_aligned in ((native, 1248, False), (fitted, 180, True)):
        if not manifest["complete"] or manifest["image_count"] != expected_count:
            raise ValueError("incomplete render manifest")
        for record in manifest["records"]:
            matrix = np.asarray(record["matrix_original_to_rendered"])
            identity = record["identity"]
            if is_aligned:
                key = (identity["index"], identity["seed"], identity["mode"])
                expected = np.asarray(aligned[key]["alignment"]["matrix_predicted_to_target"])
                if record["mesh"] != aligned[key]["mesh"] or not np.array_equal(matrix, expected):
                    raise ValueError("rendered rigid transform differs from scored transform")
                transform_count += 1
            elif not np.array_equal(matrix, np.eye(4)) or record["source_bounds"] != record["rendered_bounds"]:
                raise ValueError("native geometry changed")
            if {v["view"] for v in record["views"]} != {0, 1, 2}:
                raise ValueError("missing fixed camera")
            for view in record["views"]:
                if not view["foreground_pixels_supersampled"] or view["vertices_outside_xy"] or view["vertices_outside_depth"] or view["foreground_touches_image_border"]:
                    raise ValueError("empty / clipped / border touching render")
                path = Path(view["png"])
                if sha(path) != view["png_sha256"]:
                    raise ValueError("render image changed")
                with Image.open(path) as image:
                    if image.size != (512, 512) or image.mode != "RGB":
                        raise ValueError("inconsistent image format")
                image_count += 1
    save(OUT / "visual_integrity.json", {
        "passed": True, "render_images_checked": image_count,
        "native_identity_transforms": len(native["records"]),
        "aligned_matrices_exact_equal_scored": transform_count,
        "same_cameras_material_lights_framing": True,
        "empty_images": 0, "clipped_images": 0, "border_touching_images": 0,
        "all_png_hashes_match": True, "all_pngs_512_rgb": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="Compose object 0 only while full renders are running.")
    args = parser.parse_args()
    rows, records, aligned, selection = load_data()
    if args.preview:
        # One clean row is sufficient to verify layout before the occluded mesh finishes.
        row = next(r for r in rows if r["index"] == 0)
        path = native_sheet([0], [row], records, 0, "layout_preview", "Mesh comparison layout preview")
        print(path)
        return
    native_manifest = read(OUT / "visuals/manifest.json")
    if not native_manifest["complete"] or native_manifest["image_count"] != 1248 or len(aligned) != 384:
        raise ValueError("native renders or SE(3) audit incomplete")
    if native_manifest["empty_images"] or native_manifest["clipped_mesh_views"]:
        raise ValueError("empty or clipped native views")
    verify_renders(aligned, selection)
    figures = []
    for i in range(8):
        figures.append(native_sheet(list(range(4 * i, 4 * i + 4)), rows, records, 0,
                                    f"native_atlas_{i+1:02d}", f"All-object mesh atlas {i+1}/8"))
    representative = [r["object_index"] for r in selection["representative_objects"]]
    for view in range(3):
        figures.append(native_sheet(representative, rows, records, view,
                                    f"representative_native_v{view}", "Preselected examples: native mesh comparison"))
    for obj in representative:
        figures.append(pose_sheet(obj, rows, records, aligned))
    figures.append(make_plots(aligned, selection))
    inputs = make_inputs(rows)
    figures.append(main_comparison(rows, records, inputs))
    make_overlays(rows, records, set(representative))
    make_gallery(rows, records, aligned, selection, inputs)
    files = [OUT / "index.html", OUT / "gallery_data.json", OUT / "visuals/input_manifest.json", OUT / "visuals/overlay_manifest.json"]
    for figure in figures:
        files.extend([figure, figure.with_suffix(".pdf")])
    save(OUT / "visual_report_manifest.json", {
        "complete": True, "native_records": len(records), "aligned_records": len(aligned),
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
        "atlas_objects": list(range(32)), "atlas_seed": 42, "atlas_view": 0,
        "gallery_seeds": [42, 43], "gallery_views": [0, 1, 2],
        "selected_objects": representative, "selection_sha256": sha(OUT / "visual_selection.json"),
        "silhouette_overlay": "Threshold >20 RGB difference from fixed [249,250,251] background; illustrative, not scored. No depth comparison.",
        "artifacts": [{"path": relative(path), "sha256": sha(path)} for path in files]})
    print(json.dumps({"complete": True, "figures_png_and_pdf": len(figures), "gallery": str(OUT / "index.html")}))


if __name__ == "__main__":
    main()
