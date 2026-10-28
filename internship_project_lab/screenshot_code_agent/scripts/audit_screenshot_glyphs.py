#!/usr/bin/env python3
"""Reject invisible reference characters, repair unused inputs, and rebuild CodeOCR.

Only construction-time image assets are changed. Task IDs, source text, tests,
patches, groups and budgets stay fixed. Old images are retained for audit.
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from screenshot_agent.codeocr_benchmark import upstream
from screenshot_agent.control import atomic_json

MONO = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
SYMBOL = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def charset(font):
    values = subprocess.check_output(["fc-query", "--format=%{charset}", str(font)], text=True)
    ranges = []
    for value in values.split():
        first, _, last = value.partition("-")
        ranges.append((int(first, 16), int(last or first, 16)))
    return ranges


def missing_glyphs(text, ranges):
    # Variation selectors do not require standalone glyphs in a monochrome font.
    return sorted({c for c in text if not c.isspace() and not 0xFE00 <= ord(c) <= 0xFE0F
                   and not any(a <= ord(c) <= b for a, b in ranges)})


def render_text(text, path, font_path, theme):
    font = ImageFont.truetype(str(font_path), 21)
    lines = text.expandtabs(4).splitlines()
    width = max(640, math.ceil(max((font.getlength(s) for s in lines), default=0))+62)
    height = max(150, 48+31*len(lines))
    bg, fg = ("#fafafa", "#18222e") if theme == "light" else ("#17202a", "#e8edf2")
    picture = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(picture)
    layout = []
    for n, line in enumerate(lines):
        y = 24+31*n
        draw.text((26, y), line, font=font, fill=fg, anchor="lt")
        layout.append([{"text": c, "bbox": [26+int(font.getlength(line[:i])), y,
            26+math.ceil(font.getlength(line[:i+1])), y+25]} for i,c in enumerate(line)])
    temporary = path.with_name("new_" + path.name)
    picture.save(temporary)
    temporary.replace(path)
    atomic_json(path.with_suffix(".layout.json"), layout)


def repair_audit(dataset, runs, backup):
    dataset, backup = dataset.resolve(), backup.resolve()
    previous_path = dataset / "glyph_audit.json"
    previous = json.loads(previous_path.read_text()).get("tasks", {}) if previous_path.exists() else {}
    tasks = {}
    for reference in sorted(dataset.glob("*/screenshot_light.txt")):
        instance = reference.parent.name
        prior = previous.get(instance, {})
        font = Path(prior.get("font", str(MONO)))
        text = reference.read_text()
        missing = missing_glyphs(text, charset(font))
        item = {**prior, "font": str(font), "font_sha256": sha(font), "missing_visible_glyphs": missing}
        if missing:
            if runs and any(runs.glob(instance + "_*")):
                raise ValueError(f"Refuse to change an input already exposed to a model: {instance}")
            if missing_glyphs(text, charset(SYMBOL)):
                raise ValueError(f"No installed complete font for {instance}")
            saved = backup / instance
            saved.mkdir(parents=True, exist_ok=True)
            old_images = {}
            for theme in ["light", "dark"]:
                image = reference.parent / f"screenshot_{theme}.png"
                old_images[theme] = sha(image)
                shutil.copy2(image, saved / image.name)
                shutil.copy2(image.with_suffix(".layout.json"), saved / image.with_suffix(".layout.json").name)
                render_text(text, image, SYMBOL, theme)
            item = {"font": str(SYMBOL), "font_sha256": sha(SYMBOL), "missing_visible_glyphs": [],
                    "original_missing_characters": missing, "old_image_sha256": old_images,
                    "new_image_sha256": {theme: sha(reference.parent/f"screenshot_{theme}.png") for theme in old_images},
                    "correction_before_model_exposure": True, "unchanged_reference_sha256": sha(reference)}
        tasks[instance] = item
    report = {"tasks": tasks, "all_visible_glyphs_supported": all(not x["missing_visible_glyphs"] for x in tasks.values()),
              "correction_scope": "Only missing-glyph inputs not yet exposed to any model; source text and task selection unchanged."}
    if not tasks:
        raise ValueError("No screenshot references were found in the dataset")
    atomic_json(previous_path, report)
    return report


def rebuild_codeocr(source, output, font):
    source, output, font = source.resolve(), output.resolve(), font.resolve()
    if output.exists():
        raise ValueError("Use a new CodeOCR output dataset directory")
    old_rows = [json.loads(s) for s in (source/"images.jsonl").read_text().splitlines() if s.strip()]
    build = json.loads((source/"build_report.json").read_text())
    render, corrupt, _ = upstream(Path(build["checkout"]))
    coverage = charset(font)
    face = ImageFont.truetype(str(font), 20)
    groups = {}
    for row in old_rows:
        groups.setdefault(row["reference"], []).append(row)
    output.mkdir(parents=True)
    audit = []
    for reference, variants in groups.items():
        src = source/reference
        text = src.read_text()
        missing = missing_glyphs(text, coverage)
        if missing:
            raise ValueError(f"Reference has unrenderable characters: {variants[0]['source_id']}: {missing}")
        folder = (output/reference).parent
        folder.mkdir(parents=True)
        shutil.copy2(src, output/reference)
        shutil.copy2(src.parent/"source.json", folder/"source.json")
        lines = text.splitlines()
        width = max(800, math.ceil(max((face.getlength(s) for s in lines),default=0))+64)
        height = max(200,(len(lines)+3)*28+48)
        pages = render(text,width=width,height=height,font_size=20,line_height=1.4,margin_px=24,
                       preserve_newlines=True,enable_syntax_highlight=True,
                       filename=variants[0]["source_id"],font_path=str(font),theme="light")
        if len(pages)!=1:
            raise ValueError("Unexpected image pagination")
        baseline = folder/"original.png"
        pages[0].save(baseline)
        for row in variants:
            image = Path(corrupt(str(baseline),row["ratio"]))
            assert image == output/row["image"]
            assert sha(output/reference)==sha(src)
        audit.append({"source_id": variants[0]["source_id"], "missing_visible_glyphs": [],
                      "reference_sha256": sha(src)})
    (output/"images.jsonl").write_text("".join(json.dumps(r)+"\n" for r in old_rows))
    build.update(font_file=str(font),font_sha256=sha(font),font_glyph_audit=audit,
                 original_dataset=str(source),same_source_ids_and_references=True,
                 old_font_data_status="Diagnostic only: four source files had unsupported CJK glyphs.")
    atomic_json(output/"build_report.json",build)
    return build


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command",required=True)
    repair = sub.add_parser("repair")
    repair.add_argument("--dataset",type=Path,required=True)
    repair.add_argument("--runs",type=Path)
    repair.add_argument("--backup",type=Path,default=Path("outputs/official_v3_setup/font_audit"))
    codeocr = sub.add_parser("codeocr")
    codeocr.add_argument("--source",type=Path,default=Path("datasets/generated/codeocr_v3"))
    codeocr.add_argument("--output",type=Path,default=Path("datasets/generated/codeocr_v3_unicode"))
    codeocr.add_argument("--font",type=Path,default=Path(".cache/NotoSansMonoCJKsc-Regular.otf"))
    args=p.parse_args()
    result=repair_audit(args.dataset,args.runs,args.backup) if args.command=="repair" else rebuild_codeocr(args.source,args.output,args.font)
    print(json.dumps(result,ensure_ascii=False,indent=2))
