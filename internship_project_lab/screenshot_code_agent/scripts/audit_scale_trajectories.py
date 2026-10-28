#!/usr/bin/env python3
"""Offline trace accounting: no model calls, task execution, or input mutations."""

import argparse
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path

FILESYSTEM_TOOLS = {"glob", "grep", "read_file", "edit_file", "write_file"}
MUTATION_TOOLS = {"edit_file", "write_file"}


def read_json(path, sources):
    data = path.read_bytes()
    sources[str(path)] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    return json.loads(data)


def metadata(output):
    """Parse only tool metadata, never classify source text as an error."""
    try:
        value = json.loads(output)
        if isinstance(value, dict):
            return value
    except (ValueError, TypeError):
        pass
    result = {}
    for line in output.split("\n--- output ---\n", 1)[0].splitlines():
        key, sep, value = line.partition(": ")
        if sep:
            try:
                result[key] = json.loads(value)
            except ValueError:
                pass
    return result


def labels_for(name, output, meta):
    labels = []
    error = str(meta.get("error", ""))
    if name in FILESYSTEM_TOOLS:
        if (meta.get("returncode", 0) != 0 and "/sca_worker.py" in output
                and "SyntaxError: future feature annotations is not defined" in output):
            labels.append("worker_python36_syntax")
        if meta.get("error_type") == "AttributeError" and "has no attribute 'is_relative_to'" in error:
            labels.append("worker_pathlib_api")
        if meta.get("error") in {"edit_match_failed", "repeated_failed_edit"}:
            labels.append(meta["error"])
        if "Read-only file system" in error:
            labels.append("protected_path_readonly")
        if "Path must stay inside /workspace" in error:
            labels.append("path_rejected")
        if meta.get("error_type") == "FileNotFoundError":
            labels.append("path_not_found")
    if isinstance(meta.get("python_syntax"), dict) and meta["python_syntax"].get("valid") is False:
        labels.append("edit_python_syntax_invalid")
    return labels


def analyze_task(folder, arm, expected, sources):
    result = read_json(folder / "result.json", sources)
    config = read_json(folder / "config.json", sources)
    if result["task_id"] != expected or not result.get("model_inference"):
        raise ValueError(f"Unexpected result identity: {folder}")
    if result.get("status") == "cancelled" or (
        result.get("exit_reason") == "model_error" and result.get("agent_error_type") == "APIConnectionError"
    ):
        raise ValueError(f"Interrupted task cannot be treated as complete: {folder}")
    patch = (folder / "patch.diff").read_bytes() if (folder / "patch.diff").exists() else None
    if patch is not None:
        sources[str(folder / "patch.diff")] = {
            "sha256": hashlib.sha256(patch).hexdigest(), "bytes": len(patch)}
    row = {
        "arm": arm, "task_id": expected, "repository": result["group_id"],
        "resolved": result["resolved"], "status": result["status"],
        "exit_reason": result.get("exit_reason"), "agent_error": result.get("agent_error"),
        "failure_stage": result.get("failure_stage"), "error": result.get("error"),
        "elapsed_seconds": result.get("elapsed_seconds"), "usage": result.get("usage", {}),
        "patch_bytes": None if patch is None else len(patch),
        "changed_files": result.get("changed_files", []),
        "zero_patch_failure": not result["resolved"] and patch == b"" and not result.get("changed_files"),
        "config": {"limits": config["limits"], "agent": config["agent"],
                   "max_context_tokens": config["max_context_tokens"]},
        "calls": [], "summary_evidence": [], "event_counts": {}, "label_counts": {},
        "response_finish_reasons": {}, "unmatched_tool_events": [],
    }
    events_path = folder / "events.jsonl"
    digest, size = hashlib.sha256(), 0
    counts, finishes, labels = collections.Counter(), collections.Counter(), collections.Counter()
    pending = collections.defaultdict(collections.deque)
    seen, previous = collections.Counter(), None
    decision_index = 0
    with events_path.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            digest.update(raw)
            size += len(raw)
            if not raw.strip():
                continue
            event = json.loads(raw)
            kind = event["event"]
            counts[kind] += 1
            if kind == "model_request" and event.get("kind") == "code":
                decision_index += 1
            elif kind == "model_response":
                choice = event.get("response", {}).get("choices", [{}])[0]
                finishes[f"{event.get('kind')}:{choice.get('finish_reason')}"] += 1
                if event.get("kind") == "summary":
                    row["summary_evidence"].append({
                        "line": number, "content": choice.get("message", {}).get("content")})
            elif kind == "tool_start":
                name, args = event["name"], event.get("arguments", {})
                signature = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
                call = {
                    "tool_call": event["tool_call"], "decision_index": decision_index,
                    "name": name, "arguments": args, "start_line": number,
                    "repeated_exact_request": seen[signature] > 0,
                    "consecutive_exact_repeat": previous == signature,
                }
                seen[signature] += 1
                previous = signature
                row["calls"].append(call)
                pending[name].append(call)
            elif kind in {"tool_result", "tool_error"}:
                name = event["name"]
                if not pending[name]:
                    row["unmatched_tool_events"].append({"line": number, "event": kind, "name": name})
                    continue
                call = pending[name].popleft()
                output = event.get("output", "")
                meta = metadata(output) if kind == "tool_result" else event
                call_labels = labels_for(name, output, meta)
                labels.update(call_labels)
                call.update(result_line=number, labels=call_labels,
                            returncode=meta.get("returncode"), error=meta.get("error"),
                            error_type=meta.get("error_type"), timed_out=meta.get("timed_out", False),
                            elapsed_seconds=meta.get("elapsed_seconds"),
                            match_count=meta.get("match_count"),
                            successful_mutation=name in MUTATION_TOOLS and meta.get("ok") is True
                            and meta.get("changed") is True and meta.get("returncode", 0) == 0,
                            output_sha256=hashlib.sha256(output.encode()).hexdigest())
                if call_labels or kind == "tool_error":
                    call["evidence_excerpt"] = output[:900] if output else str(meta.get("error", ""))[:900]
    sources[str(events_path)] = {"sha256": digest.hexdigest(), "bytes": size}
    row["event_counts"], row["label_counts"] = dict(counts), dict(labels)
    row["response_finish_reasons"] = dict(finishes)
    row["tool_counts"] = dict(collections.Counter(c["name"] for c in row["calls"]))
    row["unclosed_tool_calls"] = [c["tool_call"] for queue in pending.values() for c in queue]
    if row["unmatched_tool_events"] or row["unclosed_tool_calls"]:
        raise ValueError(f"Ambiguous/incomplete tool event pairing: {folder}")
    if len(row["calls"]) != row["usage"].get("tool_calls", 0):
        raise ValueError(f"Tool count differs from recorded usage: {folder}")
    row["explicit_mutation_attempts"] = sum(c["name"] in MUTATION_TOOLS for c in row["calls"])
    row["successful_explicit_mutations"] = sum(c.get("successful_mutation", False) for c in row["calls"])
    row["repeat_counts_by_tool"] = dict(collections.Counter(
        c["name"] for c in row["calls"] if c["repeated_exact_request"]))
    row["first_explicit_mutation_call"] = next(
        (c["tool_call"] for c in row["calls"] if c["name"] in MUTATION_TOOLS), None)
    if row["zero_patch_failure"]:
        # Observed trace partitions, not claims that a single factor caused failure.
        if labels["worker_python36_syntax"] or labels["worker_pathlib_api"]:
            row["zero_patch_observed_group"] = "worker_incompatible"
        elif not row["explicit_mutation_attempts"]:
            row["zero_patch_observed_group"] = "no_explicit_mutation_request"
        elif labels["protected_path_readonly"]:
            row["zero_patch_observed_group"] = "protected_path_edit_rejected"
        elif labels["edit_match_failed"] or labels["repeated_failed_edit"]:
            row["zero_patch_observed_group"] = "edit_match_failure"
        else:
            row["zero_patch_observed_group"] = "other_requires_manual_review"
    return row


def aggregate(rows):
    def summed(mapping):
        counts = collections.Counter()
        for row in rows:
            counts.update(row[mapping])
        return dict(counts)
    labels = sorted({label for row in rows for label in row["label_counts"]})
    return {
        "attempted": len(rows), "resolved": sum(r["resolved"] for r in rows),
        "statuses": dict(collections.Counter(r["status"] for r in rows)),
        "exit_reasons": dict(collections.Counter(r["exit_reason"] for r in rows)),
        "budget_errors": dict(collections.Counter(
            r["agent_error"] for r in rows if r["exit_reason"] == "budget_exceeded")),
        "budget_exceeded_but_resolved": sum(r["exit_reason"] == "budget_exceeded" and r["resolved"] for r in rows),
        "tool_counts": summed("tool_counts"), "label_call_counts": summed("label_counts"),
        "label_task_counts": {label: sum(r["label_counts"].get(label, 0) > 0 for r in rows) for label in labels},
        "repeat_counts_by_tool": summed("repeat_counts_by_tool"),
        "event_counts": summed("event_counts"), "response_finish_reasons": summed("response_finish_reasons"),
        "zero_patch_failures": sum(r["zero_patch_failure"] for r in rows),
        "zero_patch_observed_groups": dict(collections.Counter(
            r["zero_patch_observed_group"] for r in rows if r["zero_patch_failure"])),
        "tasks_without_explicit_mutation": sum(r["explicit_mutation_attempts"] == 0 for r in rows),
        "successful_explicit_mutations": sum(r["successful_explicit_mutations"] for r in rows),
        "summary_calls": sum(r["usage"].get("summary_model_calls", 0) for r in rows),
        "decision_calls": sum(r["usage"].get("decision_model_calls", 0) for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must not exist; preserve prior analysis artifacts")
    args.run = args.run.resolve()
    sources = {}
    plan = read_json(args.run / args.split / "plan.json", sources)
    comparison = read_json(args.run / args.split / "comparison.json", sources)
    ids = [r["task_id"] for r in plan["tasks"]]
    if len(ids) != len(set(ids)) or not comparison["complete"]:
        raise ValueError("Need a complete comparison and a distinct fixed task manifest")
    rows = []
    for arm in ["baseline", "coder"]:
        for task_id in ids:
            rows.append(analyze_task(args.run / args.split / arm / "runs" / task_id, arm, task_id, sources))
    arms = {arm: aggregate([r for r in rows if r["arm"] == arm]) for arm in ["baseline", "coder"]}
    for arm, summary in arms.items():
        for key in ["attempted", "resolved"]:
            if summary[key] != comparison["arms"][arm][key]:
                raise ValueError(f"Summary differs from original comparison: {arm}:{key}")
    report = {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run": str(args.run), "split": args.split, "arms": arms, "rows": rows,
        "sources": sources,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation": [
            "Observed requests and tool feedback, not counterfactual causal attribution.",
            "Repeated exact requests can be justified, especially tests after edits.",
            "No explicit edit/write tool request does not by itself exclude shell mutations; inspect bash commands.",
            "Worker errors are counted only in filesystem tool outputs, not quoted model prompts/source text.",
            "Original scores, run files, frozen runtime, and task selection are unchanged.",
        ],
    }
    args.output.mkdir(parents=True)
    (args.output / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    fields = ["arm", "task_id", "repository", "resolved", "status", "exit_reason", "patch_bytes",
              "zero_patch_failure", "zero_patch_observed_group", "explicit_mutation_attempts",
              "successful_explicit_mutations", "first_explicit_mutation_call", "elapsed_seconds"]
    with (args.output / "tasks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(args.output), "arms": arms}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
