"""Self-authored synthetic micro-repositories, visible transcriptions and real captured test failures.

These are engineering fixtures, not SWE-bench instances or organically collected screenshots.
Split assignment happens at the original task level BEFORE any style rendering.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .control import Events, atomic_json
from .sandbox import BubblewrapSandbox

# id, problem, buggy body, corrected body, independent test assertions.
CASES = [
    ("add", "Adding two integers gives the wrong result.", "return a - b", "return a + b",
     "def add(a, b):", ["self.assertEqual(m.add(7, 2), 9)", "self.assertEqual(m.add(-2, 2), 0)"]),
    ("mean", "The arithmetic mean is too small.", "return sum(values) / (len(values) + 1)",
     "return sum(values) / len(values)", "def mean(values):",
     ["self.assertEqual(m.mean([2, 4, 6]), 4)", "self.assertEqual(m.mean([9]), 9)"]),
    ("clamp", "Clamping should keep a value inside the inclusive bounds.", "return max(low, max(value, high))",
     "return max(low, min(value, high))", "def clamp(value, low, high):",
     ["self.assertEqual(m.clamp(5, 0, 10), 5)", "self.assertEqual(m.clamp(20, 0, 10), 10)"]),
    ("discount", "A percentage discount produces a negative price.", "return price * (1 - percent)",
     "return price * (1 - percent / 100)", "def discount(price, percent):",
     ["self.assertEqual(m.discount(100, 20), 80)", "self.assertEqual(m.discount(100, 100), 0)"]),
    ("inclusive_range", "The final endpoint is missing from an inclusive range.", "return list(range(start, end))",
     "return list(range(start, end + 1))", "def inclusive_range(start, end):",
     ["self.assertEqual(m.inclusive_range(2, 4), [2, 3, 4])", "self.assertEqual(m.inclusive_range(1, 1), [1])"]),
    ("normalize", "Usernames should be lowercase with surrounding whitespace removed.", "return name.lower()",
     "return name.strip().lower()", "def normalize(name):",
     ["self.assertEqual(m.normalize(' Alice '), 'alice')", "self.assertEqual(m.normalize('BOB'), 'bob')"]),
    ("csv_fields", "Comma-separated fields are not parsed or trimmed correctly.", "return text.split(' ')",
     "return [part.strip() for part in text.split(',')] if text.strip() else []", "def csv_fields(text):",
     ["self.assertEqual(m.csv_fields('a, b,c'), ['a', 'b', 'c'])", "self.assertEqual(m.csv_fields(''), [])"]),
    ("merge", "Merging overrides must not mutate the defaults dictionary.",
     "defaults.update(overrides)\n    return defaults", "return {**defaults, **overrides}",
     "def merge(defaults, overrides):",
     ["base = {'a': 1}", "self.assertEqual(m.merge(base, {'b': 2}), {'a': 1, 'b': 2})",
      "self.assertEqual(base, {'a': 1})"]),
    ("first_unique", "Return the first non-repeating item, or None if there is none.",
     "return next((x for x in values if values.count(x) > 0), None)",
     "return next((x for x in values if values.count(x) == 1), None)", "def first_unique(values):",
     ["self.assertEqual(m.first_unique([1, 1, 2, 3]), 2)", "self.assertIsNone(m.first_unique([1, 1]))"]),
    ("leap", "Century years have incorrect leap-year classification.", "return year % 4 == 0",
     "return year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)", "def leap(year):",
     ["self.assertFalse(m.leap(1900))", "self.assertTrue(m.leap(2000))", "self.assertTrue(m.leap(2024))"]),
    ("factorial", "factorial(0) should return 1 instead of recursing indefinitely.",
     "if n == 1:\n        return 1\n    return n * factorial(n - 1)",
     "if n <= 1:\n        return 1\n    return n * factorial(n - 1)", "def factorial(n):",
     ["self.assertEqual(m.factorial(0), 1)", "self.assertEqual(m.factorial(5), 120)"]),
    ("find_index", "Searching for a missing item should return -1.", "return values.index(item)",
     "return values.index(item) if item in values else -1", "def find_index(values, item):",
     ["self.assertEqual(m.find_index([1, 2], 3), -1)", "self.assertEqual(m.find_index([1, 2], 2), 1)"]),
    ("safe_divide", "Dividing by zero should return the supplied default.", "return a / b",
     "return a / b if b != 0 else default", "def safe_divide(a, b, default=0):",
     ["self.assertEqual(m.safe_divide(3, 0), 0)", "self.assertEqual(m.safe_divide(6, 2), 3)"]),
    ("dedup", "Deduplication must preserve the first occurrence order.", "return list(set(values))",
     "return list(dict.fromkeys(values))", "def dedup(values):",
     ["self.assertEqual(m.dedup([3, 1, 3, 2]), [3, 1, 2])", "self.assertEqual(m.dedup([]), [])"]),
    ("extension", "Return the last filename extension, or an empty string for extensionless names.",
     "return filename.split('.')[1]", "return filename.rsplit('.', 1)[1] if '.' in filename else ''",
     "def extension(filename):",
     ["self.assertEqual(m.extension('a.test.py'), 'py')", "self.assertEqual(m.extension('README'), '')"]),
    ("chunks", "Splitting a list into fixed-size chunks loses some items.",
     "return [values[i:i + size] for i in range(0, len(values), size + 1)]",
     "return [values[i:i + size] for i in range(0, len(values), size)]", "def chunks(values, size):",
     ["self.assertEqual(m.chunks([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])",
      "self.assertEqual(m.chunks([], 2), [])"]),
    ("transpose", "Transposing a rectangular matrix returns incorrectly shaped rows.",
     "return list(zip(rows))", "return [list(row) for row in zip(*rows)]", "def transpose(rows):",
     ["self.assertEqual(m.transpose([[1, 2], [3, 4]]), [[1, 3], [2, 4]])",
      "self.assertEqual(m.transpose([]), [])"]),
    ("sorted_keys", "The function should return sorted dictionary keys.", "return sorted(mapping.values())",
     "return sorted(mapping.keys())", "def sorted_keys(mapping):",
     ["self.assertEqual(m.sorted_keys({'b': 1, 'a': 2}), ['a', 'b'])", "self.assertEqual(m.sorted_keys({}), [])"]),
    ("count_words", "Repeated whitespace should not create extra words.", "return len(text.split(' '))",
     "return len(text.split())", "def count_words(text):",
     ["self.assertEqual(m.count_words('a  b\tc'), 3)", "self.assertEqual(m.count_words(''), 0)"]),
    ("ceil_div", "Integer division must round up for nonnegative dividends.", "return a // b",
     "return (a + b - 1) // b", "def ceil_div(a, b):",
     ["self.assertEqual(m.ceil_div(7, 3), 3)", "self.assertEqual(m.ceil_div(0, 3), 0)"]),
]


def render_screenshot(text: str, path: Path, theme: str = "dark") -> dict:
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
    if not font_path.exists():
        raise RuntimeError("Install DejaVu Sans Mono or adjust the renderer's explicit font path")
    font = ImageFont.truetype(str(font_path), 21)
    cw = float(font.getlength("M"))
    line_height, left, top = 31, 26, 24
    lines = text.expandtabs(4).splitlines()
    width = max(640, int(max(map(len, lines), default=1)*cw + left*2 + 10))
    height = max(150, top*2 + line_height*len(lines))
    bg, fg = (("#17202a", "#e8edf2") if theme == "dark" else ("#fafafa", "#18222e"))
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    layout = []
    for index, line in enumerate(lines):
        y = top+index*line_height
        draw.text((left, y), line, font=font, fill=fg, anchor="lt")
        layout.append([{"text": char, "bbox": [int(left+i*cw), y, int(left+(i+1)*cw), y+25]}
                       for i, char in enumerate(line)])
    image.save(path)
    path.with_suffix(".txt").write_text(text.expandtabs(4), encoding="utf-8")
    atomic_json(path.with_suffix(".layout.json"), layout)
    return {"width": width, "height": height, "theme": theme,
            "font_sha256": hashlib.sha256(font_path.read_bytes()).hexdigest()}


def build_dataset(destination: Path, limit: int = 20) -> dict:
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Dataset destination must be empty; use a new versioned output directory")
    destination.mkdir(parents=True, exist_ok=True)
    manifest, audit = [], []
    events = Events(destination / "build_events.jsonl")
    for index, (name, problem, buggy_body, fixed_body, signature, assertions) in enumerate(CASES[:limit]):
        # Original task assignment is frozen before either screenshot style is rendered.
        split = "dev" if index < 12 else "test"
        task_id = f"synthetic_{name}"
        group_id = task_id
        directory = destination / task_id
        repo = directory / "repo"
        (repo / "tests").mkdir(parents=True)
        buggy = signature+"\n    "+buggy_body+"\n"
        fixed = signature+"\n    "+fixed_body+"\n"
        (repo / "miniapp.py").write_text(buggy)
        test = ("import unittest\nimport miniapp as m\n\nclass TestRepair(unittest.TestCase):\n"
                "    def test_target(self):\n" + "".join("        "+line+"\n" for line in assertions)
                + "\n    def test_regression(self):\n        self.assertTrue(callable(m."
                + name + "))\n")
        (repo / "tests/test_repair.py").write_text(test)
        (repo / "README.md").write_text("Self-authored synthetic repair fixture. Run python3 -m unittest discover -s tests -v.\n")
        sandbox = BubblewrapSandbox(repo, events, ["tests"], timeout=15)
        sandbox.probe()
        before = sandbox.run("python3 -m unittest discover -s tests -v")
        if before["returncode"] == 0 or before["timed_out"]:
            raise RuntimeError(f"Invalid fixture: {task_id} does not reproduce a bounded test failure")
        (directory / "baseline_test.log").write_text(before["output"])
        (repo / "miniapp.py").write_text(fixed)
        after = sandbox.run("python3 -m unittest discover -s tests -v")
        (repo / "miniapp.py").write_text(buggy)
        if after["returncode"] != 0:
            raise RuntimeError(f"Reference patch does not pass: {task_id}: {after['output']}")
        (directory / "reference_test.log").write_text(after["output"])
        patch = "diff --git a/miniapp.py b/miniapp.py\n" + "".join(difflib.unified_diff(
            buggy.splitlines(True), fixed.splitlines(True), fromfile="a/miniapp.py", tofile="b/miniapp.py"))
        (directory / "reference.patch").write_text(patch)
        kind = "code" if index % 2 == 0 else "error"
        if kind == "code":
            visible = "miniapp.py\n\n" + buggy
        else:
            # A window on REAL captured failing-test output; the exact visible window is the reference.
            visible = "$ python3 -m unittest discover -s tests -v\n" + "\n".join(
                line[:115] for line in before["output"].splitlines()[-18:])
        for theme in ("light", "dark"):
            image = directory / f"{kind}_{theme}.png"
            render = render_screenshot(visible, image, theme)
            bbox = [0, 55, render["width"], render["height"]]
            crop_id = "crop_"+hashlib.sha256(f"shot:{bbox}".encode()).hexdigest()[:12]
            steps = [
                {"tool": "read_file", "arguments": {"file_path": "miniapp.py"}},
                {"tool": "run_tests", "arguments": {}},
                {"tool": "crop_screenshot", "arguments": {"image_id": "shot", "bbox": bbox}},
                {"tool": "read_screenshot", "arguments": {"image_id": crop_id}},
                {"tool": "edit_file", "arguments": {"file_path": "miniapp.py", "old_string": buggy,
                                                      "new_string": fixed}},
                {"tool": "run_tests", "arguments": {}},
                {"content": "Scripted plumbing check: applied the known reference edit and ran tests. No model inference."},
            ]
            atomic_json(directory / f"replay_{theme}.json", steps)
            task = {"task_id": task_id, "group_id": group_id, "split": split,
                "provenance": "self_authored_synthetic_micro_repository", "repo": "repo", "problem": problem,
                "test_command": "python3 -m unittest discover -s tests -v", "protected_paths": ["tests"],
                "reference_patch": "reference.patch", "replay": f"replay_{theme}.json",
                "images": [{"image_id": "shot", "path": image.name,
                            "reference": image.with_suffix(".txt").name,
                            "layout": image.with_suffix(".layout.json").name}]}
            atomic_json(directory / f"task_{theme}.json", task)
            if theme == "light":
                manifest.append({"task_file": f"{task_id}/task_{theme}.json", "task_id": task_id,
                                 "group_id": group_id, "split": split, "screenshot_kind": kind})
        audit.append({"task_id": task_id, "split": split, "baseline_returncode": before["returncode"],
                      "reference_returncode": after["returncode"], "screenshot_kind": kind,
                      "buggy_source_sha256": hashlib.sha256(buggy.encode()).hexdigest()})
    with (destination / "tasks.jsonl").open("w") as f:
        for item in manifest:
            f.write(json.dumps(item)+"\n")
    report = {"task_count": len(manifest), "style_variants_per_task": 2,
              "independent_task_count": len(manifest), "synthetic": True,
              "split_policy": "original task/group split before rendering; all style variants stay in that split",
              "real_user_screenshots": 0, "audit": audit}
    atomic_json(destination / "build_report.json", report)
    return report
