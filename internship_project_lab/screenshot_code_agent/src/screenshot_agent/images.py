"""Immutable screenshot registry, deterministic crops and three interchangeable readers."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import statistics
import threading
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, Field

from .config import ImageInput
from .control import Events
from .models import APIModel


class Reading(BaseModel):
    raw_text: str
    kind: str = "unknown"
    file_hints: list[str] = Field(default_factory=list)
    line_hints: list[int] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    regions: list[dict] = Field(default_factory=list)
    format_status: str = "native"


def text_hints(text: str) -> dict:
    files = sorted(set(re.findall(r"[\w./-]+\.(?:py|js|ts|tsx|java|go|rs|cpp|c|h)", text)))
    lines = sorted(set(int(x) for x in re.findall(r"\bline\s+(\d+)\b", text)))
    return {"file_hints": files, "line_hints": lines}


class ImageStore:
    def __init__(self, inputs: list[ImageInput], directory: Path, events: Events):
        self.directory, self.events = directory, events
        self.directory.mkdir(parents=True, exist_ok=True)
        self.entries: dict[str, dict] = {}
        self.references: dict[str, dict] = {}
        self.lock = threading.RLock()
        for item in inputs:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", item.image_id) or item.image_id in self.entries:
                raise ValueError("Screenshot IDs must be unique and contain only letters, digits, '_', '-' or '.'")
            with Image.open(item.path) as source:
                if source.width * source.height > 32_000_000:
                    raise ValueError("Screenshot exceeds 32 megapixels")
                image = source.convert("RGB")
                path = directory / f"{item.image_id}.png"
                image.save(path)
            self.entries[item.image_id] = {
                "image_id": item.image_id, "path": str(path), "width": image.width, "height": image.height,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "parent": None,
            }
            if item.reference:
                self.references[item.image_id] = {"text": Path(item.reference).read_text(encoding="utf-8"),
                    "layout": json.loads(Path(item.layout).read_text()) if item.layout else None}

    def get(self, image_id: str) -> dict:
        with self.lock:
            if image_id not in self.entries:
                raise ValueError(f"Unknown screenshot ID: {image_id}")
            return dict(self.entries[image_id])

    def public(self) -> list[dict]:
        with self.lock:
            return [{k: v for k, v in row.items() if k != "path"} for row in self.entries.values()]

    def crop(self, image_id: str, bbox: list[int]) -> dict:
        with self.lock:
            parent = self.get(image_id)
            if len(bbox) != 4 or any(isinstance(v, bool) or not isinstance(v, int) for v in bbox):
                raise ValueError("bbox must be [left, top, right, bottom] in integer pixel coordinates")
            x0, y0, x1, y1 = bbox
            if not (0 <= x0 < x1 <= parent["width"] and 0 <= y0 < y1 <= parent["height"]):
                raise ValueError("Crop rectangle must be nonempty and entirely inside the image")
            suffix = hashlib.sha256(f"{image_id}:{bbox}".encode()).hexdigest()[:12]
            new_id = f"crop_{suffix}"
            if new_id in self.entries:
                return {k: v for k, v in self.entries[new_id].items() if k != "path"}
            if len(self.entries) >= 32:
                raise ValueError("Maximum of 32 registered screenshots/crops reached")
            path = self.directory / f"{new_id}.png"
            with Image.open(parent["path"]) as image:
                image.crop(bbox).save(path)
            self.entries[new_id] = {"image_id": new_id, "path": str(path), "width": x1-x0,
                "height": y1-y0, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "parent": image_id, "bbox": bbox}
            reference = self.references.get(image_id)
            if reference and reference["layout"] is not None:
                visible = []
                rows = []
                for row in reference["layout"]:
                    chars = []
                    for char in row:
                        bx0, by0, bx1, by1 = char["bbox"]
                        if bx0 >= x0 and by0 >= y0 and bx1 <= x1 and by1 <= y1:
                            chars.append({"text": char["text"], "bbox": [bx0-x0, by0-y0, bx1-x0, by1-y0]})
                    if chars:
                        visible.append("".join(c["text"] for c in chars))
                        rows.append(chars)
                self.references[new_id] = {"text": "\n".join(visible), "layout": rows}
            self.events.emit("screenshot_crop", **self.get(new_id))
            return {k: v for k, v in self.entries[new_id].items() if k != "path"}


class OracleReader:
    name = "oracle"

    def __init__(self, store: ImageStore):
        self.store = store

    def read(self, image_id: str) -> Reading:
        self.store.get(image_id)
        if image_id not in self.store.references:
            raise ValueError("No exact visible transcription is available for this screenshot/crop")
        text = self.store.references[image_id]["text"]
        return Reading(raw_text=text, **text_hints(text))


class OCRReader:
    name = "ocr"

    def __init__(self, store: ImageStore):
        self.store, self.engine = store, None
        self.lock = threading.Lock()

    def read(self, image_id: str) -> Reading:
        with self.lock:
            if self.engine is None:
                from rapidocr_onnxruntime import RapidOCR
                self.engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
            result, _ = self.engine(self.store.get(image_id)["path"])
        words = []
        for box, text, score in result or []:
            xs, ys = zip(*box)
            words.append({"x": min(xs), "y": (min(ys)+max(ys))/2, "w": max(xs)-min(xs),
                          "h": max(ys)-min(ys), "text": text, "score": float(score)})
        if not words:
            return Reading(raw_text="", uncertainties=["OCR detected no text"])
        char_width = statistics.median(w["w"] / max(1, len(w["text"])) for w in words)
        origin = min(w["x"] for w in words)
        rows: list[list[dict]] = []
        for word in sorted(words, key=lambda w: (w["y"], w["x"])):
            if rows and abs(word["y"]-statistics.mean(w["y"] for w in rows[-1])) < word["h"] * .55:
                rows[-1].append(word)
            else:
                rows.append([word])
        lines, uncertainties = [], []
        for row in rows:
            line = ""
            for word in sorted(row, key=lambda w: w["x"]):
                column = max(0, round((word["x"]-origin)/max(1, char_width)))
                line += " " * max(0, column-len(line)) + word["text"]
                if word["score"] < .9:
                    uncertainties.append(f"Low OCR confidence {word['score']:.2f}: {word['text']}")
            lines.append(line)
        text = "\n".join(lines)
        return Reading(raw_text=text, uncertainties=uncertainties, **text_hints(text))


VISION_PROMPT = """Transcribe only what is visible in this screenshot of source code or a terminal.
Preserve indentation, punctuation, operators, identifiers and error messages. Do not repair code or invent hidden lines.
Return a JSON object with: raw_text (string), kind (code/error/mixed/unknown), file_hints (string array),
line_hints (integer array), uncertainties (string array), regions (array of {description,bbox:[left,top,right,bottom]}).
All coordinates refer to the provided image's pixel dimensions. Distinguish ambiguous characters explicitly.
Do not infer a repository path or line number that the screenshot does not show."""


class VisionReader:
    name = "vlm"

    def __init__(self, store: ImageStore, model: APIModel):
        self.store, self.model = store, model

    def read(self, image_id: str) -> Reading:
        entry = self.store.get(image_id)
        payload = base64.b64encode(Path(entry["path"]).read_bytes()).decode()
        response = self.model.complete([
            {"role": "system", "content": VISION_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"Screenshot {image_id}, {entry['width']}x{entry['height']} pixels."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,"+payload}},
            ]},
        ])
        text = response.choices[0].message.content or ""
        return parse_vision_response(text)


def parse_vision_response(text: str) -> Reading:
    """Recover literal control characters without inventing or repairing code."""
    try:
        start, end = text.index("{"), text.rindex("}")+1
        payload = text[start:end]
        try:
            reading = Reading.model_validate_json(payload)
            reading.format_status = "strict_json"
            return reading
        except (ValueError, TypeError):
            # strict=False only permits literal newline/tab characters in JSON
            # strings. It does not repair quotes, add delimiters or decode code
            # a second time, so ambiguous/truncated responses remain explicit.
            reading = Reading.model_validate(json.loads(payload,strict=False))
            reading.format_status = "relaxed_json"
            reading.uncertainties.append("Vision JSON contained literal control characters; source text was preserved by relaxed parsing")
            return reading
    except (ValueError, TypeError):
        return Reading(raw_text=text,format_status="unparsed_text",
                       uncertainties=["Vision response did not satisfy the reading JSON schema"])
