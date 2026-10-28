"""CodeOCR author-released source subset, upstream rendering and reconstruction metrics."""

from __future__ import annotations

import ast
import difflib
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from Levenshtein import distance
from PIL import Image, ImageDraw, ImageFont

from .config import ImageInput, RunConfig
from .control import Budget, Events, atomic_json
from .images import ImageStore, OCRReader, VisionReader
from .models import APIModel


def upstream(checkout: Path):
    sys.path.insert(0, str(checkout.resolve()))
    render = importlib.import_module("CodeOCR.core.rendering").text_to_image
    # Upstream's unused API utility imports a missing AIHUBMIX_BASE_URL constant.
    # Load the two pure, UNMODIFIED function ASTs without those unrelated imports.
    def pure_function(relative, name):
        path = checkout / relative
        module = ast.parse(path.read_text())
        node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == name)
        namespace = {"os": os, "difflib": difflib}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        return namespace[name]

    corrupt = pure_function("reconstruction/pipeline/data_pipeline.py", "apply_visual_corruption")
    metrics = pure_function("reconstruction/pipeline/evaluator.py", "_compute_codediff_metrics_no_blank")
    return render, corrupt, metrics


def build_codeocr(checkout: Path, destination: Path, limit: int = 10) -> dict:
    checkout, destination = checkout.resolve(), destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Use a new empty CodeOCR dataset directory")
    render, corrupt, _ = upstream(checkout)
    dataset_path = checkout / "reconstruction/dataset/dataset.json"
    original = json.loads(dataset_path.read_text())
    # Complete original files only; bound text size BEFORE querying any reader.
    selected = sorted((r for r in original if len(r["code"].splitlines()) <= 80
                       and len(r["code"]) <= 2200), key=lambda r: r["id"])[:limit]
    if len(selected) != limit:
        raise ValueError("Not enough complete source files satisfying the frozen size limits")
    destination.mkdir(parents=True, exist_ok=True)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    font = ImageFont.truetype(font_path, 20)
    rows = []
    for index, record in enumerate(selected):
        folder = destination / f"sample_{index:03d}"
        folder.mkdir()
        source = record["code"].expandtabs(4)
        (folder / "reference.py").write_text(source)
        group = record["repo"]
        split = "test" if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 3 == 0 else "dev"
        lines = source.splitlines()
        width = max(800, math.ceil(max((font.getlength(line) for line in lines), default=0)) + 64)
        height = max(200, (len(lines) + 3) * 28 + 48)
        pages = render(source, width=width, height=height, font_size=20, line_height=1.4,
                       margin_px=24, preserve_newlines=True, enable_syntax_highlight=True,
                       filename=record["id"], font_path=font_path, theme="light")
        if len(pages) != 1:
            raise RuntimeError("Expected a complete single-page source rendering")
        baseline = folder / "original.png"
        pages[0].save(baseline)
        for ratio in (1, 2):
            image = Path(corrupt(str(baseline), ratio))
            rows.append({"sample_id": f"codeocr_{index:03d}_ratio{ratio}", "source_id": record["id"],
                         "group_id": group, "split": split, "ratio": ratio,
                         "image": str(image.relative_to(destination)),
                         "reference": str((folder / "reference.py").relative_to(destination)),
                         "source_url": record["url"], "source_sha256": hashlib.sha256(source.encode()).hexdigest()})
        atomic_json(folder / "source.json", record)
    (destination / "images.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = {"source": "https://github.com/YerbaPage/CodeOCR", "checkout": str(checkout),
               "original_files": len(original), "selected_files": len(selected), "images": len(rows),
               "groups": len({r["group_id"] for r in rows}), "synthetic_screenshots": True,
               "selection_rule": "Sort original IDs; first 10 COMPLETE files with <=80 lines and <=2200 characters. No crops of source, no model-based selection.",
               "grouping": "Repository hash assignment before ratios; both image variants share group/split.",
               "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
               "font_sha256": hashlib.sha256(Path(font_path).read_bytes()).hexdigest(),
               "metrics": "Upstream normalized CER/WER/line similarity; supplementary exact CER and Python syntax check.",
               "scope": "Small reconstruction subset; no repository repair tasks or full CodeOCR paper reproduction."}
    summary["upstream_import_adaptation"] = "Two unmodified pure function ASTs loaded without unused API imports: upstream utils references a missing AIHUBMIX_BASE_URL config constant. Renderer imported normally."
    atomic_json(destination / "build_report.json", summary)
    return summary


def font_characters(font: Path) -> set[int]:
    values = subprocess.check_output(["fc-query", "--format=%{charset}", str(font)], text=True)
    characters = set()
    for value in values.split():
        first, _, last = value.partition("-")
        characters.update(range(int(first, 16), int(last or first, 16)+1))
    return characters


@contextmanager
def fallback_characters(characters: set[str], font: Path):
    """Supply missing single glyphs to the unmodified upstream renderer.

    Upstream draws each code point separately. Noto CJK's missing-glyph advance
    is 20 pixels at size 20; the emoji font is fitted into that same cell.
    """
    original = ImageDraw.ImageDraw.text
    face = ImageFont.truetype(str(font), 16)

    def draw_text(draw, xy, text, *args, **kwargs):
        if text in characters:
            kwargs["font"] = face
        return original(draw, xy, text, *args, **kwargs)

    with patch.object(ImageDraw.ImageDraw, "text", draw_text):
        yield


def build_full_codeocr(checkout: Path, destination: Path, font: Path, fallback: Path) -> dict:
    checkout, destination, font, fallback = [p.resolve() for p in (checkout,destination,font,fallback)]
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Use a new empty CodeOCR dataset directory")
    render, corrupt, _ = upstream(checkout)
    source_path = checkout / "reconstruction/dataset/dataset.json"
    original = json.loads(source_path.read_text())
    selected = sorted(original, key=lambda r:r["id"])
    if len({r["id"] for r in selected}) != len(selected):
        raise ValueError("Duplicate upstream source IDs")
    main_chars, extra_chars = font_characters(font), font_characters(fallback)
    needed = {c for r in selected for c in r["code"] if not c.isspace()
              and not 0xFE00 <= ord(c) <= 0xFE0F and ord(c) not in main_chars}
    unsupported = sorted(c for c in needed if ord(c) not in extra_chars)
    if unsupported:
        raise ValueError(f"Unrenderable characters: {unsupported}")
    face = ImageFont.truetype(str(font),20)
    extra_face = ImageFont.truetype(str(fallback),16)
    if any(extra_face.getlength(c)>face.getlength(c)+.5 for c in needed):
        raise ValueError("Fallback glyph is wider than the upstream character cell")
    destination.mkdir(parents=True,exist_ok=True)
    rows, audit = [], []
    with fallback_characters(needed,fallback):
        for index, record in enumerate(selected):
            folder = destination/f"sample_{index:03d}"
            folder.mkdir()
            source = record["code"].expandtabs(4)
            (folder/"reference.py").write_text(source)
            atomic_json(folder/"source.json",record)
            lines = source.splitlines()
            width = max(800,math.ceil(max((face.getlength(s) for s in lines),default=0))+64)
            height = max(200,(len(lines)+3)*28+48)
            pages = render(source,width=width,height=height,font_size=20,line_height=1.4,
                           margin_px=24,preserve_newlines=True,enable_syntax_highlight=True,
                           filename=record["id"],font_path=str(font),theme="light")
            if len(pages)!=1:
                raise ValueError(f"Unexpected pagination: {record['id']}; no source will be truncated")
            if pages[0].width*pages[0].height>16_000_000:
                raise ValueError(f"Image exceeds model input limit: {record['id']}")
            baseline = folder/"original.png"
            pages[0].save(baseline)
            group = record["repo"]
            split = "test" if int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%3==0 else "dev"
            for ratio in (1,2):
                image = Path(corrupt(str(baseline),ratio))
                with Image.open(image) as picture:
                    dimensions = list(picture.size)
                rows.append({"sample_id":f"codeocr_{index:03d}_ratio{ratio}","source_id":record["id"],
                    "group_id":group,"split":split,"ratio":ratio,"image":str(image.relative_to(destination)),
                    "reference":str((folder/"reference.py").relative_to(destination)),
                    "source_url":record["url"],"source_sha256":hashlib.sha256(source.encode()).hexdigest(),
                    "image_sha256":hashlib.sha256(image.read_bytes()).hexdigest(),"dimensions":dimensions})
            audit.append({"source_id":record["id"],"lines":len(lines),"characters":len(source),
                          "fallback_characters":sorted(needed & set(source)),"missing_visible_glyphs":[],
                          "reference_sha256":hashlib.sha256(source.encode()).hexdigest()})
    (destination/"images.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    summary = {"source":"https://github.com/YerbaPage/CodeOCR","checkout":str(checkout),
        "original_files":len(original),"selected_files":len(selected),"groups":len({r['group_id'] for r in rows}),
        "images":len(rows),"ratios":[1,2],"synthetic_screenshots":True,"full_reconstruction_source_coverage":True,
        "selection_rule":"ALL author-released reconstruction records, sorted by source ID. No length, syntax or model-quality filtering.",
        "dataset_sha256":hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "font_file":str(font),"font_sha256":hashlib.sha256(font.read_bytes()).hexdigest(),
        "fallback_font":str(fallback),"fallback_sha256":hashlib.sha256(fallback.read_bytes()).hexdigest(),
        "font_glyph_audit":audit,"all_visible_glyphs_supported":True,
        "grouping":"Repository hash assignment before ratio variants; same source stays in one split.",
        "metrics":"Upstream normalized CER/WER/line similarity; raw CER, exact match and Python syntax checks.",
        "rendering_adaptation":"Unmodified upstream renderer plus a draw-time font fallback for missing individual symbols; no source replacements. Complete single-page files.",
        "scope":"Full coverage of published reconstruction source data at ratios 1 and 2; not the complete CodeOCR task suite or all upstream ratios. No code execution or repair claims."}
    atomic_json(destination/"build_report.json",summary)
    return summary


def evaluate_codeocr(manifest: Path, config: RunConfig, output: Path, readers=("ocr", "vlm"),
                     *, shard_index=0, shard_count=1, resume=False) -> dict:
    if not 0<=shard_index<shard_count or not readers or set(readers)-{"ocr","vlm"} or len(set(readers))!=len(readers):
        raise ValueError("Invalid readers or shard selection")
    if output.exists() and not resume:
        raise ValueError("CodeOCR output directory already exists")
    output.mkdir(parents=True,exist_ok=True)
    build = json.loads((manifest.parent / "build_report.json").read_text())
    _, _, metrics = upstream(Path(build["checkout"]))
    rows = [json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    ids = sorted({r["source_id"] for r in rows})
    selected_ids = set(ids[shard_index::shard_count])
    rows = [r for r in rows if r["source_id"] in selected_ids]
    if not rows or len({r["sample_id"] for r in rows})!=len(rows):
        raise ValueError("Empty or duplicate image selection")
    plan = {"manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),"config":config.model_dump(),
            "readers":list(readers),"shard_index":shard_index,"shard_count":shard_count,
            "sample_ids":[r["sample_id"] for r in rows]}
    if (output/"plan.json").exists() and json.loads((output/"plan.json").read_text())!=plan:
        raise ValueError("Cannot resume with a different dataset, configuration or shard plan")
    atomic_json(output/"plan.json",plan)
    records = []
    for row in rows:
        for name in readers:
            directory = output / row["sample_id"] / name
            if resume and (directory/"result.json").exists():
                saved = json.loads((directory/"result.json").read_text())
                if any(saved.get(k)!=v for k,v in {**row,"reader":name}.items()):
                    raise ValueError("Stored result belongs to a different reading input")
                records.append(saved)
                continue  # Retain failed attempts too; do not retry for a better score.
            if directory.exists():
                raise ValueError(f"Interrupted case needs archival before resuming: {directory}")
            directory.mkdir(parents=True)
            events = Events(directory / "events.jsonl")
            budget = Budget(config.limits, events)
            store = ImageStore([ImageInput(image_id="shot", path=str(manifest.parent / row["image"]))],
                               directory / "screenshots", events)
            reader = OCRReader(store) if name == "ocr" else VisionReader(store,
                APIModel(config.vision, budget, events, kind="vision"))
            started = time.monotonic()
            record = {**row, "reader": name, "status": "error"}
            try:
                reading = reader.read("shot")
                source = (manifest.parent / row["reference"]).read_text()
                hypothesis = reading.raw_text
                (directory / "prediction.txt").write_text(hypothesis)
                score = metrics(source, hypothesis)

                def parses(text):
                    try:
                        ast.parse(text)
                        return True
                    except (SyntaxError, ValueError):
                        return False

                record.update(status="complete", metrics=score, raw_cer=distance(source, hypothesis)/max(1,len(source)),
                              exact_match=source == hypothesis, reference_parses=parses(source),
                              hypothesis_parses=parses(hypothesis), reading=reading.model_dump(),
                              reading_schema_valid=reading.format_status in {"native","strict_json"},
                              reading_format_status=reading.format_status)
            except Exception as error:
                record.update(error=str(error), error_type=type(error).__name__)
            record.update(elapsed_seconds=round(time.monotonic()-started,3), usage=budget.snapshot())
            record['output_cap_reached'] = name=='vlm' and budget.completion_tokens>=config.vision.max_output_tokens
            atomic_json(directory / "result.json", record)
            records.append(record)
            atomic_json(output / "status.json", {"status":"running","completed": len(records), "total": len(rows)*len(readers),
                                                  "current": row["sample_id"], "reader": name,
                                                  "errors":sum(r['status']=='error' for r in records)})
            print(f"CodeOCR [{len(records)}/{len(rows)*len(readers)}] {row['sample_id']} / {name}: {record['status']}", flush=True)
    return codeocr_report(build,records,readers,output)


def codeocr_report(build, records, readers, output):
    summaries = []
    for name in readers:
        for ratio in sorted({r["ratio"] for r in records}):
            selected = [r for r in records if r["reader"] == name and r["ratio"] == ratio]
            done = [r for r in selected if r["status"] == "complete"]
            summaries.append({"reader": name, "ratio": ratio, "attempted": len(selected), "completed": len(done),
                "normalized_cer_mean": sum(r["metrics"]["cer"] for r in done)/len(done) if done else None,
                "cer_including_errors_mean": sum(r.get('metrics',{}).get('cer',1.0) for r in selected)/len(selected),
                "normalized_wer_mean": sum(r["metrics"]["wer"] for r in done)/len(done) if done else None,
                "line_similarity_mean": sum(r["metrics"]["line_sim"] for r in done)/len(done) if done else None,
                "raw_cer_mean": sum(r["raw_cer"] for r in done)/len(done) if done else None,
                "syntax_valid": sum(r["hypothesis_parses"] for r in done),
                "reference_syntax_valid":sum(r["reference_parses"] for r in done),
                "schema_valid":sum(r.get('reading_schema_valid',True) for r in done),
                "exact_matches":sum(r['exact_match'] for r in done),
                "errors":len(selected)-len(done),
                "mean_seconds": sum(r["elapsed_seconds"] for r in done)/len(done) if done else None})
    report = {"scope": build, "records": records, "summary": summaries, "complete": True}
    atomic_json(output / "report.json", report)
    atomic_json(output/"status.json",{"status":"complete","completed":len(records),"total":len(records),
                                      "errors":sum(r['status']=='error' for r in records)})
    lines = ["# CodeOCR reconstruction evaluation", "", "Upstream renderer, source files and normalized metrics. No repair-success claim.", "",
             "| Reader | Ratio | Complete / attempts | CER | Python syntax valid |",
             "|---|---:|---:|---:|---:|"]
    for s in summaries:
        cer = f"{s['normalized_cer_mean']:.2%}" if s["normalized_cer_mean"] is not None else "N/A"
        lines.append(f"| {s['reader']} | {s['ratio']} | {s['completed']}/{s['attempted']} | {cer} | {s['syntax_valid']} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return report
