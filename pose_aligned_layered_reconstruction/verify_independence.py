#!/usr/bin/env python3
"""Fail if production modules import either legacy project package."""

from __future__ import annotations

import ast
import json
from pathlib import Path


FORBIDDEN = ("pose_aligned_reconstruction", "pose_point_depth_mv")


def main() -> None:
    package = Path(__file__).resolve().parent
    violations: list[dict[str, object]] = []
    checked = 0
    for path in sorted(package.rglob("*.py")):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        checked += 1
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name == FORBIDDEN[0] or name.startswith(FORBIDDEN[0] + ".") or name == FORBIDDEN[1] or name.startswith(FORBIDDEN[1] + "."):
                    violations.append(
                        {
                            "path": str(path.relative_to(package)),
                            "line": int(node.lineno),
                            "import": name,
                        }
                    )
    report = {
        "passed": not violations,
        "checked_python_files": checked,
        "forbidden_packages": list(FORBIDDEN),
        "violations": violations,
    }
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

