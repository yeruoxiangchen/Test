"""Trusted external scoring; reference metadata is never mounted into Agent tools."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def canonicalize_expected_ids(spec, parser) -> dict:
    """Use the selected official parser's naming rules for expected test IDs too.

    The caller limits this adaptation to the official pytest-options parser and
    its aliases. No expected test may disappear or merge with another test.
    Synthetic PASSED lines are used only to obtain names, never to score a run.
    """
    raw = {group: list(getattr(spec, group)) for group in ("FAIL_TO_PASS", "PASS_TO_PASS")}
    canonical, changes, seen = {}, [], {}
    for group, names in raw.items():
        canonical[group] = []
        for name in names:
            if not isinstance(name, str) or not name or "\n" in name or "\r" in name:
                raise ValueError("Expected test IDs must be nonempty single-line strings")
            parsed = parser(f"PASSED {name}\n", spec)
            if len(parsed) != 1 or next(iter(parsed.values())) != "PASSED":
                raise ValueError(f"Official parser cannot canonicalize expected test ID: {name!r}")
            normalized = next(iter(parsed))
            if normalized in seen:
                prior_group, prior_name = seen[normalized]
                raise ValueError("Expected test ID collision after official parser normalization: "
                                 f"{prior_group}:{prior_name!r} and {group}:{name!r} -> {normalized!r}")
            seen[normalized] = (group, name)
            canonical[group].append(normalized)
            if normalized != name:
                changes.append({"group": group, "raw": name, "canonical": normalized})
    return {"protocol": "pinned_pytest_options_expected_ids_v1", "parser": spec.log_parser,
            "raw": raw, "canonical": canonical, "changes": changes,
            "counts_preserved": all(len(raw[g]) == len(canonical[g]) for g in raw)}


def official_grade(metadata: Path, log: Path) -> dict:
    helper = Path(__file__).resolve().parents[2] / "scripts/official_grade.py"
    result = subprocess.run([sys.executable, str(helper), "--metadata", str(metadata), "--log", str(log)],
                            capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError("Official grading failed: " + result.stderr[-4000:])
    return json.loads(result.stdout)
