#!/usr/bin/env python3
"""Offline audit of immutable, completed workflow runs; no inference or test execution.

Gold-file overlap is a diagnostic, not a localization correctness label. Public
checks and final hidden grading are kept separate. Intermediate patches are not
graded, so the audit cannot claim that a retry changed hidden-test correctness.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

SOURCE_PREFIXES = {
    "marshmallow-code/marshmallow": ("src/marshmallow/",),
    "pvlib/pvlib-python": ("pvlib/",),
    "pydicom/pydicom": ("pydicom/",),
    "pylint-dev/astroid": ("astroid/",),
    "pyvista/pyvista": ("pyvista/",),
    "sqlfluff/sqlfluff": ("src/sqlfluff/",),
}
TERMINAL = {"verified", "unresolved", "error"}


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def gold_files(patch):
    return sorted(set(re.findall(r"^diff --git a/(.*?) b/.*$", patch, re.M)))


def identifier(key):
    # Diagnostic grouping only: never changes the official score or feedback.
    return key.split("[", 1)[0]


def summarize_check(check, baseline):
    reg = check.get("regression", {})
    old, new = baseline.get("cases") or {}, check.get("cases") or {}
    lost = reg.get("lost_passing_tests", [])
    new_passing = [k for k, v in new.items() if v == "passed" and k not in old]
    potential_id_drift = [k for k in lost if k not in new and any(identifier(k) == identifier(n) for n in new_passing)]
    return {
        "label": check.get("label"),
        "patch_sha256": check.get("patch_sha256"),
        "passed": check.get("passed"),
        "retry_needed": check.get("retry_needed"),
        "regression_status": reg.get("status"),
        "regression_reason": reg.get("reason"),
        "lost_passing_tests": lost,
        "additional_failures": reg.get("additional_failures", []),
        "potential_parameter_id_drift": potential_id_drift,
        "drift_new_passing_examples": [
            k for k in new_passing if any(identifier(k) == identifier(n) for n in potential_id_drift)
        ][:12],
        "reproduction_status": check.get("reproduction", {}).get("status"),
        "reproduction_baseline_verified": check.get("reproduction", {}).get("baseline_reproduced"),
        "reproduction_candidate_returncode": check.get("reproduction", {}).get("candidate", {}).get("returncode"),
        "timed_out": check.get("timed_out"),
        "returncode": check.get("returncode"),
        "changed_files": check.get("changed_files", []),
        "candidate_output_tail": check.get("output", "")[-1400:],
    }


def audit_task(folder):
    result, flow, task = [read(folder / f) for f in ("result.json", "workflow.json", "task.json")]
    benchmark = read(Path(task["benchmark"]))
    record = benchmark["record"]
    expected = gold_files(record.get("patch", ""))
    changed = result.get("changed_files", [])
    prefixes = SOURCE_PREFIXES.get(record["repo"], ())
    source_changed = [
        f for f in changed if f.startswith(prefixes) and "/tests/" not in f and not Path(f).name.startswith("test_")
    ]
    retrieved = read(folder / "retrieval.json") if (folder / "retrieval.json").exists() else []
    # retrieval.json is refreshed after phases, not necessarily the initial retrieval.
    retrieved_paths = sorted({v["path"] for v in retrieved})
    hidden = result.get("validation", {}).get("hidden", {})
    official = hidden.get("official", {})
    statuses = official.get("tests_status", {})
    baseline = flow.get("public_baseline", {})
    checks = [summarize_check(c, baseline) for c in flow.get("checks", [])]
    repros = []
    for rep in flow.get("reproduction_attempts", []):
        b = rep.get("baseline", {})
        text = b.get("output", "")
        if rep.get("baseline_reproduced"):
            reason = "verified_assertion"
        elif b.get("timed_out"):
            reason = "timeout"
        elif b.get("returncode") == 0:
            reason = "baseline_exited_zero"
        elif "ModuleNotFoundError" in text or "ImportError" in text:
            reason = "import_error"
        elif "AssertionError" in text:
            reason = "assertion_contract_mismatch"
        else:
            reason = "non_assertion_failure"
        repros.append(
            {
                "script_path": rep.get("script_path"),
                "baseline_reproduced": rep.get("baseline_reproduced"),
                "expected_failure": rep.get("expected_failure"),
                "returncode": b.get("returncode"),
                "reason": reason,
                "output_tail": text[-1000:],
            }
        )

    stage, pending_request = "setup", None
    stage_calls, stage_tools, event_types, tool_counts = [collections.Counter() for _ in range(4)]
    stage_seconds = collections.Counter()
    stage_started = None
    reads, edits, observations, phase_ends = [], [], [], []
    compactions, model_seconds = [], 0.0
    requests, responses = 0, 0
    final_model_notes = []
    for line_no, line in enumerate((folder / "events.jsonl").open(), 1):
        event = json.loads(line)
        typ = event.get("event")
        event_types[typ] += 1
        if typ == "workflow_stage":
            now = event["time"]
            if stage_started is not None:
                stage_seconds[stage] += now - stage_started
            stage, stage_started = event["stage"], now
        elif typ == "model_request":
            stage_calls[stage] += 1
            requests += 1
            pending_request = event["time"]
        elif typ in ("model_response", "model_error"):
            if pending_request is not None:
                model_seconds += event["time"] - pending_request
                pending_request = None
            if typ == "model_response":
                responses += 1
                choices = event.get("response", {}).get("choices", [])
                msg = choices[0].get("message", {}) if choices else {}
                if not msg.get("tool_calls") and msg.get("content"):
                    final_model_notes.append({"line": line_no, "stage": stage, "text": msg["content"][:1800]})
            else:
                observations.append(
                    {
                        "line": line_no,
                        "stage": stage,
                        "kind": "model_error",
                        "detail": str(event.get("error", ""))[:500],
                    }
                )
        elif typ == "tool_start":
            name, args = event.get("name"), event.get("arguments", {})
            tool_counts[name] += 1
            stage_tools[stage] += 1
            path = args.get("file_path", args.get("path"))
            if name == "read_file" and path:
                reads.append({"line": line_no, "stage": stage, "path": str(path)})
            if name in ("edit_file", "write_file", "apply_patch"):
                edits.append({"line": line_no, "stage": stage, "name": name, "path": path})
        elif typ in ("tool_result", "tool_error"):
            text = str(event.get("output", event.get("error", "")))
            patterns = {
                "anchor_mismatch": r"not found|no match|must match|ambiguous|multiple matches",
                "reproduction_limit": r"Reproduction registration limit reached",
                "protected_path": r"protected|read.only",
                "missing_file": r"No such file|FileNotFoundError",
            }
            for kind, pattern in patterns.items():
                if re.search(pattern, text, re.I):
                    # These are text signals, not proof of tool errors (output may quote code).
                    observations.append(
                        {"line": line_no, "stage": stage, "name": event.get("name"), "kind": kind, "detail": text[:350]}
                    )
        elif typ == "phase_finished":
            phase_ends.append({"line": line_no, "stage": event.get("stage"), "reason": event.get("reason")})
        elif typ == "context_compacted":
            compactions.append(
                {
                    "line": line_no,
                    **{k: event.get(k) for k in ("before_tokens", "after_tokens", "limit", "task_anchor_preserved")},
                }
            )

    read_paths = {x["path"].removeprefix("/testbed/").removeprefix("./") for x in reads}
    return {
        "task_id": task["task_id"],
        "repo": record["repo"],
        "resolved": result.get("resolved"),
        "status": result["status"],
        "exit_reason": result.get("exit_reason"),
        "stop_reason": flow.get("stop_reason"),
        "runtime_failure": bool(result.get("runtime_failure")),
        "usage": result.get("usage", {}),
        "gold_files": expected,
        "changed_files": changed,
        "source_changed": source_changed,
        "gold_changed_overlap": sorted(set(expected) & set(changed)),
        "read_file_gold_overlap": sorted(set(expected) & read_paths),
        "last_retrieval_gold_overlap": sorted(set(expected) & set(retrieved_paths)),
        "retry_count": result.get("retry_count"),
        "plan": flow.get("plan"),
        "public_passed": flow.get("public_test_passed"),
        "checks": checks,
        "reproductions": repros,
        "public_baseline": {
            "returncode": baseline.get("returncode"),
            "timed_out": baseline.get("timed_out"),
            "case_counts": dict(collections.Counter((baseline.get("cases") or {}).values())),
            "output_tail": baseline.get("output", "")[-1800:],
        },
        "hidden": {
            "parsed": official.get("parsed"),
            "grader_started": hidden.get("grader_started"),
            "timed_out": hidden.get("test", {}).get("timed_out"),
            "test_counts": {k: {s: len(v) for s, v in group.items()} for k, group in statuses.items()},
            "failed_tests": {k: v.get("failure", []) for k, v in statuses.items()},
            "output_tail": hidden.get("test", {}).get("output", "")[-2500:],
        },
        "events": {
            "requests": requests,
            "responses": responses,
            "event_types": dict(event_types),
            "stage_calls": dict(stage_calls),
            "stage_tools": dict(stage_tools),
            "stage_seconds": dict(stage_seconds),
            "model_wait_seconds": model_seconds,
            "tool_counts": dict(tool_counts),
            "phase_ends": phase_ends,
            "compaction_count": len(compactions),
            "compactions": compactions,
            "reads": reads,
            "edits": edits,
            "text_signals": observations,
            "last_model_notes": final_model_notes[-4:],
        },
        "reviews": [{"issues": v.get("issues"), "next_action": v.get("next_action")} for v in flow.get("reviews", [])],
    }


def aggregate(rows):
    counts = collections.Counter()
    repos = {}
    phases = {}
    tools = collections.Counter()
    calls = collections.Counter()
    repro = collections.Counter()
    reg = collections.Counter()
    pairs = collections.Counter()
    failures = collections.Counter()
    stage_seconds = collections.Counter()
    for row in rows:
        counts["completed"] += 1
        counts["resolved"] += bool(row["resolved"])
        counts["runtime_failures"] += row["runtime_failure"]
        counts["no_source_change"] += not row["source_changed"]
        counts["gold_file_changed"] += bool(row["gold_changed_overlap"])
        counts["gold_file_read_via_read_file"] += bool(row["read_file_gold_overlap"])
        counts["failed_with_gold_file_changed"] += not row["resolved"] and bool(row["gold_changed_overlap"])
        counts["failed_with_no_source_change"] += not row["resolved"] and not row["source_changed"]
        counts["hidden_parsed"] += bool(row["hidden"]["parsed"])
        counts["grader_started"] += bool(row["hidden"]["grader_started"])
        counts["retry_twice"] += row["retry_count"] == 2
        counts["any_verified_reproduction_baseline"] += any(x["baseline_reproduced"] for x in row["reproductions"])
        counts["no_reproduction_registered"] += not row["reproductions"]
        counts["possible_public_id_drift_tasks"] += any(x["potential_parameter_id_drift"] for x in row["checks"])
        counts["public_regression_ever_inconclusive"] += any(
            x["regression_status"] == "inconclusive" for x in row["checks"]
        )
        counts["final_reproduction_verified"] += (
            bool(row["checks"]) and row["checks"][-1]["reproduction_status"] == "verified"
        )
        counts["final_public_pass"] += row["public_passed"] is True
        if row["checks"]:
            counts["reproduction_ever_verified"] += any(x["reproduction_status"] == "verified" for x in row["checks"])
            counts["public_pass_ever_then_final_not_passed"] += (
                any(x["passed"] for x in row["checks"]) and row["public_passed"] is not True
            )
        pairs[f"public={row['public_passed']},hidden={row['resolved']}"] += 1
        failures[str(row["stop_reason"] or row["exit_reason"] or "normal_workflow_end")] += 1
        repos.setdefault(row["repo"], collections.Counter()).update(total=1, resolved=int(bool(row["resolved"])))
        for x in row["events"]["phase_ends"]:
            phases.setdefault(x["stage"], collections.Counter())[x["reason"]] += 1
        calls.update(row["events"]["stage_calls"])
        tools.update(row["events"]["tool_counts"])
        stage_seconds.update(row["events"]["stage_seconds"])
        repro.update(x["reason"] for x in row["reproductions"])
        reg.update(str(x["regression_status"]) + ":" + str(x["regression_reason"]) for x in row["checks"])
    return {
        "counts": dict(counts),
        "by_repo": {k: dict(v) for k, v in repos.items()},
        "public_hidden_pairs": dict(pairs),
        "stop_reasons": dict(failures),
        "phase_exits": {k: dict(v) for k, v in phases.items()},
        "stage_calls": dict(calls),
        "stage_seconds": dict(stage_seconds),
        "tool_counts": dict(tools),
        "reproduction_attempt_reasons": dict(repro),
        "public_check_statuses": dict(reg),
        "total_model_wait_hours": sum(x["events"]["model_wait_seconds"] for x in rows) / 3600,
        "total_task_hours": sum(x["usage"].get("elapsed_seconds", 0) for x in rows) / 3600,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not output.is_relative_to(Path("/data/zjr/screenshot_code_agent")) or output == source:
        raise ValueError("Use a separate audit directory under /data/zjr/screenshot_code_agent")
    prepared = read(source / "prepared.json")
    selected = []
    for item in prepared["tasks"]:
        f = source / "runs" / item["task_id"] / "result.json"
        if f.is_file() and read(f).get("status") in TERMINAL:
            selected.append(item["task_id"])
    rows = []
    for i, task_id in enumerate(selected, 1):
        folder = source / "runs" / task_id
        fingerprint = {
            name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in ("result.json", "workflow.json")
        }
        cached = output / "tasks" / (task_id + ".json")
        old = read(cached) if cached.exists() else {}
        if old.get("fingerprint") == fingerprint:
            row = old
        else:
            row = audit_task(folder)
            row["fingerprint"] = fingerprint
            write(cached, row)
        rows.append(row)
        if i % 20 == 0:
            print(json.dumps({"audited": i, "total": len(selected)}), flush=True)
    result = {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_total": len(prepared["tasks"]),
        "selected_tasks": selected,
        "model_requests_started": 0,
        "tests_executed": 0,
        "limitations": [
            "Gold-file overlap does not prove localization correctness.",
            "Parameter-ID drift is a candidate signal requiring case inspection.",
            "Only final patches have hidden grades; retry benefit is not inferred.",
            "Read-file coverage omits reads via bash or search tool output.",
        ],
        **aggregate(rows),
    }
    assert result["counts"]["completed"] == len(selected)
    assert result["counts"]["resolved"] == sum(
        bool(read(source / "runs" / t / "result.json")["resolved"]) for t in selected
    )
    write(output / "analysis.json", result)
    with (output / "tasks.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "task_id",
                "resolved",
                "repo",
                "source_changed_count",
                "gold_file_overlap",
                "public_passed",
                "repro_attempts",
                "verified_baseline_repro",
                "retries",
                "model_calls",
                "elapsed_minutes",
                "stop_reason",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["task_id"],
                    row["resolved"],
                    row["repo"],
                    len(row["source_changed"]),
                    len(row["gold_changed_overlap"]),
                    row["public_passed"],
                    len(row["reproductions"]),
                    any(x["baseline_reproduced"] for x in row["reproductions"]),
                    row["retry_count"],
                    row["usage"].get("code_model_calls"),
                    round(row["usage"].get("elapsed_seconds", 0) / 60, 2),
                    row["stop_reason"],
                ]
            )
    print(json.dumps({k: v for k, v in result.items() if k != "selected_tasks"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
