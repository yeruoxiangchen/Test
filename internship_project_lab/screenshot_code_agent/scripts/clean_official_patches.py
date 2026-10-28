#!/usr/bin/env python3
"""Export cache-free model patches and regrade them without changing raw verdicts."""

import argparse
import json
from pathlib import Path

from screenshot_agent.benchmark import official_grade
from screenshot_agent.config import RunConfig, Task
from screenshot_agent.control import Events, atomic_json
from screenshot_agent.repository import git, replay_patch
from screenshot_agent.runtime import create_sandbox

EXCLUDES = ["__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"]


def export_clean(workspace, destination):
    specifications = [".", *[f":(exclude,glob)**/{name}/**" for name in EXCLUDES],
                      ":(exclude,glob)**/*.pyc", ":(exclude,glob)**/*.pyo"]
    payload = git(workspace,"diff","--cached","--binary","--no-ext-diff","--no-textconv",
                  "--no-renames","HEAD","--",*specifications).stdout
    destination.write_bytes(payload)
    return len(payload)


def check_batch(run):
    state = json.loads((run/"evaluation/status.json").read_text())
    if state["status"] != "complete" or state["completed"] != state["total"]:
        raise ValueError("Wait until the repair comparison is complete")
    root = run/"clean_patch_checks"
    root.mkdir(exist_ok=True)
    results = []
    for folder in sorted((run/"evaluation/runs").iterdir()):
        if not (folder/"result.json").exists():
            continue
        original = json.loads((folder/"result.json").read_text())
        output = root/folder.name
        if (output/"check.json").exists():
            results.append(json.loads((output/"check.json").read_text()))
            continue
        output.mkdir(exist_ok=True)
        item = {"task_id": original["task_id"],"reader":original["reader"],
                "original_resolved": original["resolved"],"clean_resolved":False,"model_requests":0,
                "uses_reference_patch":False,"raw_result_modified":False}
        try:
            patch = folder/"patch.clean.diff"
            item["patch_bytes"] = export_clean(folder/"workspace",patch)
            item["original_patch_bytes"] = original.get("patch_bytes",0)
            item["replay"] = replay_patch(folder/"baseline",output/"verification",patch)
            if item["replay"]["applied"]:
                task = Task.from_file(folder/"task.json")
                config = RunConfig.model_validate_json((folder/"config.json").read_text())
                sb = create_sandbox(output/"verification",Events(output/"events.jsonl"),task.protected_paths,
                                    config.limits.command_timeout_seconds,folder/"baseline",task.runtime)
                try:
                    execution = sb.run(task.test_command,log_path=output/"verification_test.log")
                    item["official"] = official_grade(Path(task.benchmark),output/"verification_test.log")
                    item["clean_resolved"] = item["official"]["resolved"] and not execution["timed_out"]
                finally:
                    sb.cancel()
            item["matches_original"] = item["clean_resolved"] == item["original_resolved"]
        except Exception as error:
            item.update(error=str(error),matches_original=False)
        atomic_json(output/"check.json",item)
        results.append(item)
        print(f"Clean patch {folder.name}: {item['clean_resolved']} (raw {item['original_resolved']})",flush=True)
    report = {"complete":True,"attempts":len(results),"original_passed":sum(r["original_resolved"] for r in results),
              "clean_passed":sum(r["clean_resolved"] for r in results),
              "all_verdicts_preserved":all(r["matches_original"] for r in results),"records":results,
              "scope":"Only generic Python/pytest cache paths removed. No model calls, semantic edits, new tests or reference-patch use. Raw registered results unchanged."}
    atomic_json(run/"clean_patch_audit.json",report)
    return report


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,required=True)
    args=p.parse_args()
    report=check_batch(args.run.resolve())
    print(json.dumps({k:v for k,v in report.items() if k!="records"},indent=2))
