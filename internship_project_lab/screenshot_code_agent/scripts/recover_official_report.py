#!/usr/bin/env python3
"""Rebuild a finished batch report with a separately verified reporting source.

The original model runtime, raw journals, patches and verdicts remain unchanged.
This is postprocessing only; it never starts inference or reruns a repair.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recover(run: Path, source: Path, model_released: bool):
    evaluation = run / "evaluation"
    state = json.loads((evaluation / "status.json").read_text())
    results = sorted(evaluation.glob("runs/*/result.json"))
    if state["status"] != "complete" or len(results) != state["completed"] or len(results) != state["total"]:
        raise ValueError("Only fully completed batches can be recovered")
    protected = [*results, *evaluation.glob("runs/*/events.jsonl"), *evaluation.glob("runs/*/patch.diff")]
    before = {str(p.relative_to(run)): digest(p) for p in protected}
    environment = dict(os.environ, PYTHONPATH=str(source))
    command = [sys.executable, "-m", "screenshot_agent", "report", "--runs", str(evaluation)]
    subprocess.run(command, env=environment, check=True, capture_output=True, text=True)
    if before != {str(p.relative_to(run)): digest(p) for p in protected}:
        raise ValueError("Recovery unexpectedly changed an original journal, patch or result")
    report = json.loads((evaluation / "report.json").read_text())
    if report["finished_runs"] != len(results):
        raise ValueError("Recovered report differs from the registered attempt count")
    recovery = {
        "report_complete": True, "record_count_verified": True, "completed_runs": len(results),
        "model_released": model_released, "stats_available": (run / "final_model_stats.json").exists(),
        "raw_results_modified": False, "model_requests": 0, "original_sha256": before,
        "reporting_source": str(source / "screenshot_agent/report.py"),
        "reporting_source_sha256": digest(source / "screenshot_agent/report.py"),
        "launcher_exit_code": 2,
        "reason": "Original report split valid JSONL strings at Unicode NEL. Recovery reads physical JSONL records; no model rerun or verdict change. Launcher ended before final CUDA statistics.",
    }
    (run / "completion_recovery.json").write_text(json.dumps(recovery, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in recovery.items() if k != "original_sha256"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model-released", action="store_true", help="Assert separately checked launcher/GPU cleanup")
    options = parser.parse_args()
    recover(options.run.resolve(), options.source.resolve(), options.model_released)
