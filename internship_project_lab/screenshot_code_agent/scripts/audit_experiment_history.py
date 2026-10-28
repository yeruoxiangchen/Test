#!/usr/bin/env python3
"""Audit saved experiment timestamps and v5 failure traces without model calls."""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def utc(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="seconds")


def audit():
    hashes = {}

    def read_bytes(relative):
        data = (ROOT / relative).read_bytes()
        hashes[str(relative)] = hashlib.sha256(data).hexdigest()
        return data

    def read_json(relative):
        return json.loads(read_bytes(relative))

    def read_events(relative):
        # Split on physical JSONL lines, preserving Unicode NEL inside strings.
        return [json.loads(line) for line in read_bytes(relative).split(b"\n") if line.strip()]

    timeline = []
    batches = {
        "v1_known_patch_replay": "outputs/validation_20261021T134255Z/status.json",
        "v1_protocol_failure_1": "outputs/local_smoke_20261021T135939Z/evaluation/status.json",
        "v1_protocol_failure_2": "outputs/local_smoke_20261021T140508Z/evaluation/status.json",
        "v1_model_smoke": "outputs/local_smoke_20261021T140810Z/evaluation/status.json",
        "v2_model_smoke_repeat": "outputs/local_smoke_20261022T021829Z/evaluation/status.json",
        "v2_main": "docs/artifacts/v2/batch_status.json",
        "v2_public_known_patch_replay": "outputs/validation_20261022T055100Z/status.json",
        "v3_pilot": "outputs/official_v3_pilot/evaluation/status.json",
        "v3_main": "docs/artifacts/v3/batch_status.json",
        "post_v3_development_case": "docs/artifacts/post_v3/result.json",
        "v5_old_nine_regression": "outputs/v5_regression_gpu6/evaluation/status.json",
        "v5_ablation_service": "outputs/v5_supplementary/ablation/status.json",
        "v5_ablation_pilot": "outputs/v5_supplementary/ablation/pilot/status.json",
        "v5_extended_service": "outputs/v5_supplementary/extended/status.json",
        "v5_extended_model_batch": "outputs/v5_supplementary/extended/three_readers/evaluation/status.json",
    }
    for phase in ("clean_available", "clean_initial_only", "degraded_available", "degraded_initial_only"):
        batches[f"v5_ablation_{phase}"] = f"outputs/v5_supplementary/ablation/{phase}/evaluation/status.json"
    for name, path in batches.items():
        data = read_json(path)
        timeline.append({
            "name": name, "source": path, "basis": "recorded started_at/finished_at",
            "started_at": data["started_at"], "finished_at": data["finished_at"],
            "started_utc": utc(data["started_at"]), "finished_utc": utc(data["finished_at"]),
            "status": data.get("status"), "completed": data.get("completed"), "total": data.get("total"),
        })
    state = read_json("outputs/codeocr_full_v4/source_state.json")
    status = read_json("outputs/codeocr_full_v4/status.json")
    end = state["started_at"] + status["elapsed_seconds"]
    timeline.append({
        "name": "v4_codeocr_full", "started_at": state["started_at"], "finished_at": end,
        "started_utc": utc(state["started_at"]), "finished_utc": utc(end),
        "basis": "end derived from source_state.started_at + status.elapsed_seconds",
        "sources": ["outputs/codeocr_full_v4/source_state.json", "outputs/codeocr_full_v4/status.json"],
        "completed": status["completed"], "total": status["total"],
    })
    pattern = "outputs/official_v3_codeocr_unicode/codeocr/*/vlm/events.jsonl"
    reading_times = [event["time"] for path in sorted(ROOT.glob(pattern))
                     for event in read_events(path.relative_to(ROOT)) if "time" in event]
    reading_window = {
        "pattern": pattern, "first_event_utc": utc(min(reading_times)),
        "last_event_utc": utc(max(reading_times)),
        "basis": "VLM event window only; excludes preceding OCR and service startup/shutdown",
    }

    counts = Counter()
    rows = []
    run_root = ROOT / "outputs/v5_supplementary/extended/three_readers/evaluation/runs"
    for folder in sorted(path for path in run_root.iterdir() if path.is_dir()):
        relative = folder.relative_to(ROOT)
        result = read_json(relative / "result.json")
        events = read_events(relative / "events.jsonl")
        c = Counter()
        previous_edit = None
        for event in events:
            if event["event"] == "tool_start" and event.get("name") == "edit_file":
                c["edit_attempts"] += 1
                arguments = event["arguments"]
                if previous_edit == arguments:
                    c["identical_to_previous_edit"] += 1
                previous_edit = arguments
            if event["event"] == "tool_result":
                c["tool_results"] += 1
                output = event.get("output", "")
                if len(output) > 1500:
                    c["long_tool_results"] += 1
                    if len(output.splitlines()) <= 6:
                        c["long_results_skipped_by_line_snip"] += 1
                        if event.get("name") == "read_file":
                            c["long_read_file_results_skipped"] += 1
                if event.get("name") == "edit_file":
                    data = json.loads(output)
                    if data.get("returncode", 0) != 0:
                        c["edit_nonzero_returncode"] += 1
                    if "old_string must be nonempty and match exactly once" in output:
                        c["edit_exact_match_failures"] += 1
            if event["event"] == "model_request" and event.get("kind") == "code":
                c["code_requests"] += 1
                if not event.get("tools"):
                    c["code_requests_without_tools"] += 1
                    system = str(event.get("messages", [{}])[0].get("content", ""))
                    if "summary" in system.lower() or "summarize" in system.lower():
                        c["context_summary_requests"] += 1
        c["runs"] = 1
        c["resolved"] = int(bool(result.get("resolved")))
        c["no_changed_files"] = int(not result.get("changed_files"))
        c["budget_exceeded_runs"] = int(result.get("exit_reason") == "budget_exceeded")
        c["model_call_limit_runs"] = int(
            result.get("agent_error") == "Model step limit reached (includes context summaries)"
        )
        for key in ("edit_exact_match_failures", "identical_to_previous_edit", "context_summary_requests"):
            c[f"runs_with_{key}"] = int(c[key] > 0)
        counts.update(c)
        rows.append({"run": str(relative), "reader": result["reader"],
                     "exit_reason": result.get("exit_reason"), "counts": dict(c)})
    if len(rows) != 150:
        raise ValueError(f"Expected all 150 extended runs, found {len(rows)}")
    before = read_json("docs/artifacts/v5_regression/source_state.json")["sha256"]
    after = read_json("outputs/v5_eval_source/source_state.json")["sha256"]
    changed = sorted(key for key in before.keys() & after.keys() if before[key] != after[key])
    for path in ("vendor/CoreCoder/corecoder/context.py", "src/screenshot_agent/sandbox.py",
                 "scripts/local_model_api.py", "vendor/CoreCoder/corecoder/agent.py"):
        read_bytes(path)
    snapshot = ROOT / "outputs/v5_eval_source"
    frozen_mismatches = [name for name, digest in after.items()
                         if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != digest]
    if frozen_mismatches:
        raise ValueError(f"Frozen runtime changed: {frozen_mismatches}")
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "Offline retrospective audit; no model/test execution and no original result changes.",
        "timestamp_policy": "UTC; no filesystem mtime used as historical execution or edit time.",
        "timeline": sorted(timeline, key=lambda row: row["started_at"]),
        "v3_codeocr_vlm_event_window": reading_window,
        "v5_extended_diagnostics": dict(counts), "v5_extended_per_run": rows,
        "count_definitions": {
            "identical_to_previous_edit": "Same arguments as previous edit_file in the same run; other tools may intervene.",
            "context_summary_requests": "Code requests without tools whose system prompt requests a summary.",
            "long_results_skipped_by_line_snip": "Stored output >1500 chars and <=6 splitlines; structural skip condition, not an observed compression invocation.",
            "no_changed_files": "Empty or absent changed_files; does not by itself prove a localization failure.",
        },
        "runtime_manifest_comparison": {
            "before_entries": len(before), "after_entries": len(after), "changed_common_paths": changed,
            "added_manifest_entries": sorted(after.keys() - before.keys()),
            "note": "A newly inventoried path is not proof the file was first created in this phase.",
            "frozen_files_checked": len(after), "frozen_mismatches": frozen_mismatches,
        },
        "source_sha256": hashes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; existing files are not overwritten")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a new audit path")
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "counts": report["v5_extended_diagnostics"]},
                     ensure_ascii=False, indent=2))
