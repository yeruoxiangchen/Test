"""Paired batch summaries; scripted plumbing runs never become model-accuracy claims."""

from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path

from .control import atomic_json


def read_events(path: Path) -> list[dict]:
    # JSON strings may contain U+0085/U+2028. str.splitlines() treats those
    # characters as record separators, but JSONL is separated by actual LF.
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    row = list(range(len(b)+1))
    for i, aa in enumerate(a, 1):
        new = [i]
        for j, bb in enumerate(b, 1):
            new.append(min(new[-1]+1, row[j]+1, row[j-1]+(aa != bb)))
        row = new
    return row[-1]


def reading_metrics(run_dir: Path) -> dict:
    task_file, events_file = run_dir / "task.json", run_dir / "events.jsonl"
    if not task_file.exists() or not events_file.exists():
        return {}
    task = json.loads(task_file.read_text())
    refs = {i["image_id"]: i.get("reference") for i in task.get("images", [])}
    measured, seen = [], set()
    for event in read_events(events_file):
        if event["event"] != "screenshot_read":
            continue
        image_id = event["image_id"]
        if image_id in seen or not refs.get(image_id):
            continue
        seen.add(image_id)
        reference = Path(refs[image_id]).read_text()
        actual = event["reading"]["raw_text"]
        if max(len(reference), len(actual)) > 10000:
            measured.append({"image_id": image_id, "exact": reference == actual, "cer": None,
                             "reason": "Reference/output exceeds 10k-character CER computation limit"})
        else:
            measured.append({"image_id": image_id, "exact": reference == actual,
                             "cer": edit_distance(reference, actual) / max(1, len(reference)),
                             "reference_characters": len(reference), "output_characters": len(actual)})
    return {"readings": measured}


def summarize_rows(records: list[dict], fields: tuple[str, ...]) -> list[dict]:
    groups = collections.defaultdict(list)
    for row in records:
        groups[tuple(row.get(field, "unknown") for field in fields)].append(row)
    summary = []
    for key, rows in sorted(groups.items()):
        usage = [r.get("usage", {}) for r in rows]
        cers = [v["cer"] for r in rows for v in r["reading_metrics"].get("readings", []) if v["cer"] is not None]
        checks = [r.get("validation", {}) for r in rows]
        summary.append({**dict(zip(fields, key)), "attempted": len(rows),
            "resolved": sum(r["resolved"] for r in rows),
            "resolved_fraction": sum(r["resolved"] for r in rows)/len(rows),
            "independent_task_groups": len({r["group_id"] for r in rows}),
            "elapsed_seconds_mean": statistics.mean(r.get("elapsed_seconds", 0) for r in rows),
            "tool_calls_mean": statistics.mean(u.get("tool_calls", 0) for u in usage),
            "code_model_calls_mean": statistics.mean(u.get("code_model_calls", 0) for u in usage),
            "vision_model_calls_mean": statistics.mean(u.get("vision_model_calls", 0) for u in usage),
            "prompt_tokens_total": sum(u.get("prompt_tokens", 0) for u in usage),
            "completion_tokens_total": sum(u.get("completion_tokens", 0) for u in usage),
            "cost_usd_total": sum(u.get("cost_usd", 0) for u in usage),
            "reading_cer_mean": statistics.mean(cers) if cers else None,
            "readings_with_differences": sum(cer > 0 for cer in cers),
            "readings_measured": len(cers),
            "baseline_failures_reproduced": sum(c.get("baseline", {}).get("returncode") not in (None, 0)
                                                and not c.get("baseline", {}).get("timed_out") for c in checks),
            "patch_replays_passed": sum(bool(c.get("patch_replay", {}).get("applied")) for c in checks),
            "final_tests_passed": sum(c.get("final", {}).get("returncode") == 0
                                      and not c.get("final", {}).get("timed_out") for c in checks),
            "reference_file_overlap_count": sum(bool(r.get("localization_diagnostic", {}).get("reference_file_overlap"))
                                                for r in rows),
            "status_counts": dict(collections.Counter(r["status"] for r in rows)),
            "failure_stages": dict(collections.Counter(r.get("failure_stage") for r in rows if not r["resolved"])),
            "interpretation": "model repair results" if rows[0]["code_backend"] == "api"
                              else "scripted plumbing checks; reference patch known"})
    return summary


def make_report(root: Path) -> dict:
    records = []
    for path in sorted(root.rglob("result.json")):
        if any(x in path.relative_to(root).parts for x in ["baseline", "baseline_check", "workspace", "verification"]):
            continue
        row = json.loads(path.read_text())
        events_path = path.parent / "events.jsonl"
        if "screenshot_read_calls" not in row.get("usage", {}) and events_path.exists():
            # Early V1 traces used the vision counter for all screenshot readers.
            # Reconstruct exact request counts without changing the original journals or verdicts.
            journal = read_events(events_path)
            legacy = row["usage"].get("vision_model_calls", 0)
            row["usage"]["screenshot_read_calls"] = sum(
                e["event"] == "tool_start" and e.get("name") == "read_screenshot" for e in journal)
            row["usage"]["vision_model_calls"] = sum(
                e["event"] == "model_request" and e.get("kind") == "vision" for e in journal)
            row["usage_accounting_note"] = (
                f"Reconstructed from immutable events; the initial vision counter ({legacy}) counted screenshot reads")
        row["run_dir"] = str(path.parent)
        row["reading_metrics"] = reading_metrics(path.parent)
        task_path = path.parent / "task.json"
        if task_path.exists():
            task = json.loads(task_path.read_text())
            reference = task.get("reference_patch")
            if reference and Path(reference).exists():
                reference_files = set()
                for line in Path(reference).read_text().splitlines():
                    if line.startswith(("+++ b/", "--- a/")):
                        reference_files.add(line[6:])
                changed = set(row.get("changed_files", []))
                row["localization_diagnostic"] = {
                    "reference_files": sorted(reference_files),
                    "touched_reference_files": sorted(reference_files & changed),
                    "reference_file_overlap": bool(reference_files & changed),
                    "interpretation": "Reference-file overlap is a diagnostic cue, not proof of correct/incorrect localization",
                }
        records.append(row)
    summary = summarize_rows(records, ("code_backend", "reader"))
    cohorts = summarize_rows(records, ("provenance", "split", "code_backend", "reader"))
    # Pair only the same task ID and backend. No mixing of different agents/budgets.
    pairs = collections.defaultdict(dict)
    for row in records:
        pairs[(row["code_backend"], row["task_id"])][row["reader"]] = row["resolved"]
    paired = [{"code_backend": backend, "task_id": task, "outcomes": outcomes}
              for (backend, task), outcomes in sorted(pairs.items())]
    job_file = root / "batch_job.json"
    job = json.loads(job_file.read_text()) if job_file.exists() else None
    scheduled = len(job["tasks"])*len(job["readers"]) if job else len(records)
    result = {"summary": summary, "summary_by_cohort": cohorts,
              "scheduled_runs": scheduled, "finished_runs": len(records),
              "unfinished_or_not_attempted": max(0, scheduled-len(records)),
              "paired_outcomes": paired, "records": records,
              "limitations": ["Self-authored synthetic micro-repositories are engineering fixtures, not SWE-bench scores.",
                  "Replay uses a known reference patch and does not measure a model's reasoning or repair ability.",
                  "CER is whitespace-sensitive and measured on the first full-image reading.",
                  "Style variants share a group ID and must not be counted as independent tasks.",
                  "No organically collected user screenshots have been evaluated in the bundled dataset.",
                  "Public historical fixes use synthetic screenshots and project-authored regression adapters; "
                  "they share one demo repository group and do not run the complete upstream test suite.",
                  "Reference-file overlap and transcription differences are diagnostics, not causal failure labels.",
                  "A zero local API price excludes GPU, power and operator costs."]}
    atomic_json(root / "report.json", result)
    lines = ["# Screenshot Code Agent evaluation", "", "All attempted runs, including failures, remain in the denominator.", "",
             "| Code backend | Reader | Passed / attempted | Mean seconds | Mean tools | Cost USD | Reading CER |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        cer = "n/a" if row["reading_cer_mean"] is None else f"{row['reading_cer_mean']:.4f}"
        lines.append(f"| {row['code_backend']} | {row['reader']} | {row['resolved']}/{row['attempted']} | "
                     f"{row['elapsed_seconds_mean']:.2f} | {row['tool_calls_mean']:.2f} | {row['cost_usd_total']:.6f} | {cer} |")
    lines += ["", f"Finished: {len(records)} / scheduled: {scheduled}. "
              f"Unfinished or not attempted: {result['unfinished_or_not_attempted']}.",
              "", "## Results by source and split", "",
              "| Provenance | Split | Backend | Reader | Passed / attempted | Task groups | Mean seconds | Reading CER |",
              "|---|---|---|---|---:|---:|---:|---:|"]
    for row in cohorts:
        cer = "n/a" if row["reading_cer_mean"] is None else f"{row['reading_cer_mean']:.4f}"
        lines.append(f"| {row['provenance']} | {row['split']} | {row['code_backend']} | {row['reader']} | "
                     f"{row['resolved']}/{row['attempted']} | {row['independent_task_groups']} | "
                     f"{row['elapsed_seconds_mean']:.2f} | {cer} |")
    lines += ["", "## Interpretation", "", *[f"- {v}" for v in result["limitations"]]]
    lines += ["", "## Failures", ""]
    for row in records:
        if not row["resolved"]:
            lines.append(f"- {row['task_id']} / {row['reader']}: {row['status']}, stage={row.get('failure_stage')}, "
                         f"error={row.get('error', row.get('agent_error', 'see test log'))}")
    (root / "report.md").write_text("\n".join(lines)+"\n")
    return result
