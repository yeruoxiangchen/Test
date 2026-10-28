"""Post-hoc edge-case audit; never changes the registered evaluation verdicts."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
from pathlib import Path

from screenshot_agent.control import Events, atomic_json
from screenshot_agent.repository import git, prepare_repository
from screenshot_agent.sandbox import BubblewrapSandbox


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tests = [("a,,b", ["a", "", "b"]), (",a", ["", "a"]), ("a,", ["a", ""])]
    events = Events(args.output / "events.jsonl")
    task = json.loads((args.runs / "runs/synthetic_csv_fields_oracle/task.json").read_text())
    reference_dir = args.output / "reference"
    reference_dir.mkdir()
    _, workspace = prepare_repository(Path(task["repo"]), reference_dir)
    git(workspace, "apply", "-", stdin=Path(task["reference_patch"]).read_bytes())

    def check(repo):
        sandbox = BubblewrapSandbox(repo, events, ["tests"], timeout=15)
        sandbox.probe()
        checks = []
        try:
            for text, expected in tests:
                program = (f"import miniapp; actual=miniapp.csv_fields({text!r}); print(repr(actual)); "
                           f"assert actual == {expected!r}, (actual, {expected!r})")
                result = sandbox.run("python3 -c " + shlex.quote(program))
                checks.append({"input": text, "expected": expected,
                               "passed": result["returncode"] == 0 and not result["timed_out"], **result})
        finally:
            sandbox.cancel()
        return checks

    reference = check(workspace)
    if not all(row["passed"] for row in reference):
        raise ValueError("Supplementary checks do not match the known reference behavior")
    records = []
    for reader in ("oracle", "ocr", "vlm"):
        source = args.runs / f"runs/synthetic_csv_fields_{reader}"
        original = json.loads((source / "result.json").read_text())
        target = args.output / reader
        shutil.copytree(source / "verification", target, ignore=shutil.ignore_patterns("__pycache__"))
        checks = check(target)
        records.append({"reader": reader, "primary_resolved": original["resolved"],
                        "supplementary_passed": all(row["passed"] for row in checks),
                        "primary_and_supplementary_passed": original["resolved"] and all(row["passed"] for row in checks),
                        "checks": checks})
    report = {"source_run": str(args.runs.resolve()), "task_id": "synthetic_csv_fields",
              "posthoc": True, "changes_registered_verdicts": False,
              "purpose": "Audit whether fixed-test success preserves empty fields as the reference patch does",
              "reference_checks": reference, "records": records}
    atomic_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
