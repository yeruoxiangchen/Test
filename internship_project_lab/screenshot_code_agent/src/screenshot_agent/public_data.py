"""Pinned public bug histories with synthetic screenshots and sandbox-validated tests.

The three tasks share one repository group and stay in the demo split. They are
development cases, not an independently held-out benchmark or real user captures.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import textwrap
from pathlib import Path

from .control import Events, atomic_json
from .data import render_screenshot
from .repository import git, validate_tree
from .sandbox import BubblewrapSandbox

UPSTREAM = "https://github.com/mahmoud/boltons"
PROVENANCE = "public_historical_fix_with_synthetic_screenshot"
TEST_COMMAND = "python3 -m unittest discover -s repair_tests -v"

# Test adapters are authored here using the upstream regression scenarios and
# additional invariants. The complete archived repository retains its BSD license.
CASES = [
    {
        "name": "copy_function_kwdefaults", "kind": "code", "file": "boltons/funcutils.py",
        "base": "0341ab3d79e6430c0e6a4149df9c92679988a534",
        "fix": "c2b04b4cb16410d7bf9ff781c78dee3ca908dc32",
        "problem": "The function-copying behavior in the screenshot loses part of the original callable's behavior. "
                   "Restore compatible calls while preserving ordinary function-copy behavior.",
        "anchor": "    ret = FunctionType(",
        "old": "    if copy_dict:\n        ret.__dict__.update(orig.__dict__)",
        "new": "    if hasattr(orig, \"__kwdefaults__\"):\n        ret.__kwdefaults__ = orig.__kwdefaults__\n"
               "    if copy_dict:\n        ret.__dict__.update(orig.__dict__)",
        "test": '''
            import unittest
            from boltons.funcutils import copy_function

            class CopyFunctionRegression(unittest.TestCase):
                def test_keyword_defaults(self):
                    def target(a=2, *, b=5):
                        return a + b
                    copied = copy_function(target)
                    self.assertEqual(copied(), 7)
                    self.assertEqual(copied(3, b=8), 11)
                    self.assertEqual(copied.__kwdefaults__, target.__kwdefaults__)

                def test_required_keyword_remains_required(self):
                    def target(*, value):
                        return value
                    copied = copy_function(target)
                    with self.assertRaises(TypeError):
                        copied()
                    self.assertEqual(copied(value=9), 9)

                def test_closure_and_custom_attributes(self):
                    factor = 3
                    def target(a=4):
                        return factor * a
                    target.tag = 'kept'
                    copied = copy_function(target)
                    self.assertEqual(copied(), 12)
                    self.assertEqual(copied.tag, 'kept')
                    self.assertIsNot(copied, target)
                    self.assertFalse(hasattr(copy_function(target, copy_dict=False), 'tag'))
        ''',
    },
    {
        "name": "url_chained_navigation", "kind": "error", "file": "boltons/urlutils.py",
        "base": "270e974975984f662f998c8f6eb0ebebd964de82",
        "fix": "395f690f4a24331c4554e2169ac18a15955a4eab",
        "problem": "A URL operation fails as shown in the attached terminal screenshot. "
                   "Fix the root cause and preserve existing URL resolution behavior.",
        "anchor": "                new_path_parts = self.path_parts[:-1] + dest.path_parts",
        "old": "                new_path_parts = self.path_parts[:-1] + dest.path_parts",
        "new": "                new_path_parts = list(self.path_parts[:-1]) \\\n                               + list(dest.path_parts)",
        "test": '''
            import unittest
            from boltons.urlutils import URL

            class NavigationRegression(unittest.TestCase):
                def test_chained_relative_paths(self):
                    scenarios = [
                        ('https://host/b', 'https://host', ('a', '/b')),
                        ('https://host/b', 'https://host', ('a', 'b')),
                        ('https://host/a/b', 'https://host', ('a/', 'b')),
                        ('https://host/b', 'https://host', ('/a', 'b')),
                        ('https://host/a/b', 'https://host/a/', (None, 'b')),
                        ('https://host/b', 'https://host/a', (None, 'b')),
                    ]
                    for expected, base, paths in scenarios:
                        with self.subTest(base=base, paths=paths):
                            current = URL(base)
                            for path in paths:
                                current = current.navigate(path)
                            self.assertEqual(current.to_text(), expected)

                def test_single_step_and_parent(self):
                    base = URL('https://host/dir/page')
                    self.assertEqual(base.navigate('next').to_text(), 'https://host/dir/next')
                    self.assertEqual(base.navigate('../x').to_text(), 'https://host/x')
                    self.assertEqual(base.to_text(), 'https://host/dir/page')
        ''',
    },
    {
        "name": "one_to_one_empty_update", "kind": "error", "file": "boltons/dictutils.py",
        "base": "985df86bbf511b244ce6acf7a9c3b8a0d09d0fb3",
        "fix": "6cac49ce4eba506a3582e8089bcc347c6b2fa70f",
        "problem": "A mapping update fails as shown in the attached terminal screenshot. "
                   "Make the operation consistent with normal mapping behavior and keep the inverse mapping valid.",
        "anchor": "    def update(self, dict_or_iterable, **kw):",
        "old": "    def update(self, dict_or_iterable, **kw):\n        if isinstance(dict_or_iterable, dict):",
        "new": "    def update(self, dict_or_iterable, **kw):\n        keys_vals = []\n"
               "        if isinstance(dict_or_iterable, dict):",
        "test": '''
            import unittest
            from boltons.dictutils import OneToOne

            class MappingRegression(unittest.TestCase):
                def check_inverse(self, mapping):
                    self.assertEqual(dict(mapping.inv), {v: k for k, v in mapping.items()})

                def test_empty_updates(self):
                    for empty in ({}, [], iter(())):
                        with self.subTest(kind=type(empty).__name__):
                            mapping = OneToOne({'a': 1})
                            mapping.update(empty)
                            self.assertEqual(dict(mapping), {'a': 1})
                            self.check_inverse(mapping)

                def test_empty_with_keywords_and_nonempty(self):
                    mapping = OneToOne()
                    mapping.update({}, cat='dog')
                    mapping.update([('x', 'y')])
                    self.assertEqual(dict(mapping), {'cat': 'dog', 'x': 'y'})
                    self.check_inverse(mapping)

                def test_collision_keeps_bijection(self):
                    mapping = OneToOne({'a': 1, 'b': 2})
                    mapping.update({'c': 2})
                    self.assertEqual(dict(mapping), {'a': 1, 'c': 2})
                    self.check_inverse(mapping)
        ''',
    },
]


def extract_snapshot(source: Path, revision: str, target: Path) -> None:
    """Export tracked files only; history and fix objects never enter the task repo."""
    payload = git(source, "archive", "--format=tar", revision).stdout
    target.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        for member in archive.getmembers():
            rel = Path(member.name)
            if rel.is_absolute() or ".." in rel.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported archive member: {member.name}")
            path = target / rel
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.extractfile(member).read())
    validate_tree(target)


def build_public_dataset(source: Path, destination: Path) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Use an empty, versioned dataset destination")
    destination.mkdir(parents=True, exist_ok=True)
    events = Events(destination / "build_events.jsonl")
    audit, manifest = [], []
    for case in CASES:
        # Repository-level group assignment is made before rendering variants.
        task_id = f"boltons_{case['name']}"
        directory = destination / task_id
        repo = directory / "repo"
        parent = git(source, "rev-parse", case["fix"]+"^").stdout.decode().strip()
        if parent != case["base"]:
            raise ValueError(f"Unexpected parent for pinned fix {case['fix']}")
        extract_snapshot(source, case["base"], repo)
        (repo / "repair_tests").mkdir()
        test = "# Regression adapter; upstream BSD license: ../LICENSE\n" + textwrap.dedent(case["test"]).lstrip()
        (repo / "repair_tests/test_regression.py").write_text(test)
        filename = repo / case["file"]
        buggy = filename.read_text()
        fixed = git(source, "show", f"{case['fix']}:{case['file']}").stdout.decode()
        if buggy.count(case["old"]) != 1 or buggy.replace(case["old"], case["new"], 1) != fixed:
            raise ValueError(f"Recipe does not reproduce upstream source patch: {task_id}")
        protected = ["tests", "repair_tests"]
        sandbox = BubblewrapSandbox(repo, events, protected, timeout=30)
        sandbox.probe()
        before = sandbox.run(TEST_COMMAND)
        if before["returncode"] == 0 or before["timed_out"]:
            raise ValueError(f"Public bug did not reproduce: {task_id}: {before['output']}")
        try:
            filename.write_text(fixed)
            after = sandbox.run(TEST_COMMAND)
        finally:
            filename.write_text(buggy)
        if after["returncode"] != 0 or after["timed_out"]:
            raise ValueError(f"Upstream fix did not pass adapters: {task_id}: {after['output']}")
        (directory / "baseline_test.log").write_text(before["output"])
        (directory / "reference_test.log").write_text(after["output"])
        patch = git(source, "diff", case["base"], case["fix"], "--", case["file"]).stdout
        (directory / "reference.patch").write_bytes(patch)
        if case["kind"] == "code":
            lines = buggy.splitlines()
            anchor = next(i for i, line in enumerate(lines) if case["anchor"] in line)
            start, end = max(0, anchor-5), min(len(lines), anchor+18)
            visible = f"{case['file']} (lines {start+1}-{end})\n\n" + "\n".join(
                f"{i+1:4d}  {lines[i]}" for i in range(start, end))
        else:
            visible = "$ " + TEST_COMMAND + "\n" + "\n".join(before["output"].splitlines()[-25:])
        for theme in ("light", "dark"):
            image = directory / f"{case['kind']}_{theme}.png"
            render_screenshot(visible, image, theme)
            atomic_json(directory / f"replay_{theme}.json", [
                {"tool": "read_file", "arguments": {"file_path": case["file"]}},
                {"tool": "edit_file", "arguments": {"file_path": case["file"],
                                                       "old_string": case["old"], "new_string": case["new"]}},
                {"tool": "run_tests", "arguments": {}},
                {"content": "Known upstream patch replay; this is a plumbing check, not model repair."},
            ])
            atomic_json(directory / f"task_{theme}.json", {
                "task_id": task_id, "group_id": "boltons", "split": "demo", "provenance": PROVENANCE,
                "repo": "repo", "problem": case["problem"], "test_command": TEST_COMMAND,
                "protected_paths": protected, "reference_patch": "reference.patch", "replay": f"replay_{theme}.json",
                "images": [{"image_id": "shot", "path": image.name, "reference": image.with_suffix(".txt").name,
                            "layout": image.with_suffix(".layout.json").name}],
            })
            if theme == "light":
                manifest.append({"task_file": f"{task_id}/task_{theme}.json", "task_id": task_id,
                                 "group_id": "boltons", "split": "demo", "screenshot_kind": case["kind"],
                                 "provenance": PROVENANCE})
        files = validate_tree(repo)
        item = {"task_id": task_id, "upstream": UPSTREAM, "base_commit": case["base"], "fix_commit": case["fix"],
                "fix_url": f"{UPSTREAM}/commit/{case['fix']}", "source_file": case["file"],
                "baseline_returncode": before["returncode"], "reference_returncode": after["returncode"],
                "buggy_source_sha256": hashlib.sha256(buggy.encode()).hexdigest(),
                "patch_sha256": hashlib.sha256(patch).hexdigest(),
                "license_sha256": hashlib.sha256((repo / "LICENSE").read_bytes()).hexdigest(),
                "repository_files": len(files), "repository_bytes": sum(p.stat().st_size for p in files),
                "screenshot_kind": case["kind"], "screenshot_provenance": "synthetic_rendering",
                "test_provenance": "project-authored stdlib unittest adapters of upstream regressions plus extra checks"}
        atomic_json(directory / "provenance.json", item)
        audit.append(item)
    (destination / "tasks.jsonl").write_text("".join(json.dumps(row)+"\n" for row in manifest))
    for row in manifest:
        (destination / f"{row['task_id']}.jsonl").write_text(json.dumps(row)+"\n")
    report = {"task_count": len(manifest), "repository_groups": 1, "split": "demo", "synthetic_screenshots": True,
              "real_user_screenshots": 0, "style_variants_per_task": 2,
              "split_policy": "all revisions and styles of the same repository stay in demo",
              "audit": audit}
    atomic_json(destination / "build_report.json", report)
    return report


def combine_manifests(sources: list[Path], destination: Path) -> dict:
    """Reference existing tasks; never copy tasks or split their style variants."""
    if destination.exists():
        raise ValueError("Combined manifest already exists; select a new versioned path")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows, ids, groups = [], set(), {}
    for source in sources:
        for line in source.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["task_id"] in ids:
                raise ValueError(f"Duplicate task ID: {row['task_id']}")
            group = row.get("group_id", row["task_id"])
            if group in groups and groups[group] != row["split"]:
                raise ValueError(f"Group crosses splits: {group}")
            groups[group] = row["split"]
            ids.add(row["task_id"])
            row["task_file"] = os.path.relpath((source.parent / row["task_file"]).resolve(), destination.parent)
            rows.append(row)
    destination.write_text("".join(json.dumps(row)+"\n" for row in rows))
    return {"manifest": str(destination), "task_count": len(rows), "repository_or_task_groups": len(groups)}
