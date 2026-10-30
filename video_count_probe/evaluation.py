"""Blinded human annotation and paired descriptive analysis.

Counts come from complete cycles, never from prompt text or an unvalidated VLM.
"""
from __future__ import annotations

import collections
import json
import math
import random
import statistics
from pathlib import Path

from .core import METHODS, atomic_json, checked_plan, file_sha, read_json


def completed_records(run_dir, plan):
    run_dir = Path(run_dir)
    records, errors = {}, []
    identity = read_json(run_dir / "identity.json") if (run_dir / "identity.json").exists() else None
    for job in plan["jobs"]:
        path = run_dir / "records" / (job["id"] + ".json")
        if not path.exists():
            if (run_dir / "errors" / (job["id"] + ".json")).exists():
                errors.append(job["id"])
            continue
        record = read_json(path)
        if record.get("status") != "ok" or record.get("job") != job or record.get("identity") != identity:
            raise ValueError(f"Mixed or invalid result: {path}")
        if not identity or identity["plan_hash"] != plan["plan_hash"]:
            raise ValueError("Result/plan mismatch")
        video = run_dir / record["video"]
        if not video.is_file() or file_sha(video) != record["video_sha256"]:
            raise ValueError(f"Missing or corrupt video: {video}")
        records[job["id"]] = record
    return records, errors


def validate_annotation(annotation, record, case, plan):
    required = ("job_id", "video_sha256", "complete", "visually_usable", "action_present", "ambiguous", "completion_times", "rater")
    if any(k not in annotation for k in required):
        raise ValueError("Missing annotation fields")
    if annotation["complete"] is not True or not str(annotation["rater"]).strip():
        raise ValueError("Annotation must be marked complete and identify its rater")
    if annotation["job_id"] != record["job"]["id"] or annotation["video_sha256"] != record["video_sha256"]:
        raise ValueError("Annotation refers to a different video")
    for key in ("visually_usable", "action_present", "ambiguous"):
        if not isinstance(annotation[key], bool):
            raise ValueError(f"{key} must be a boolean")
    times = annotation["completion_times"]
    duration = plan["config"]["frames"] / plan["config"]["fps"]
    if not isinstance(times, list) or any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or not 0 <= t <= duration for t in times):
        raise ValueError("Completion times must be finite seconds within the video")
    if times != sorted(set(times)):
        raise ValueError("Completion times must be unique and ascending")
    if times and not annotation["action_present"]:
        raise ValueError("Cannot mark cycles complete when the action is absent")


def score_annotation(annotation, case, plan):
    times = annotation["completion_times"]
    judgeable = annotation["visually_usable"] and not annotation["ambiguous"]
    count_ok = len(times) == case["target_count"]
    exact = judgeable and annotation["action_present"] and count_ok
    # Timing score: match the ith completed cycle to the ith requested action window.
    # A 0.25s tolerance reflects one Wan latent frame. Also report raw timing error.
    windows = [s for s in case["segments"] if s["kind"] == "action"]
    timing_errors = []
    if exact:
        for t, s in zip(times, windows):
            timing_errors.append(max(s["start_seconds"] - t, 0.0, t - s["end_seconds"]))
    return {"exact_usable": int(exact), "judgeable": int(judgeable),
            "visually_usable": int(annotation["visually_usable"]),
            "action_present": int(annotation["action_present"]), "ambiguous": int(annotation["ambiguous"]),
            "absolute_count_error": abs(len(times) - case["target_count"]) if judgeable else None,
            "timing_success": int(exact and all(e <= 0.25 for e in timing_errors)),
            "mean_outside_window_seconds": statistics.mean(timing_errors) if timing_errors else None,
            "pause_success": int(exact and annotation.get("pause_respected") is True) if case["schedule"] == "pause_after_two" else None}


def collect_annotations(paths, plan, records):
    cases = {c["id"]: c for c in plan["cases"]}
    annotations = {}
    for path in paths:
        data = read_json(path)
        if data.get("plan_hash") != plan["plan_hash"]:
            raise ValueError(f"Annotation/plan mismatch: {path}")
        for annotation in data["annotations"]:
            if annotation.get("complete") is not True:
                continue
            job_id = annotation["job_id"]
            if job_id in annotations:
                raise ValueError(f"Duplicate annotation {job_id}; use an adjudicated single annotation per video")
            if job_id not in records:
                raise ValueError(f"Annotation has no verified video: {job_id}")
            case = cases[records[job_id]["job"]["case_id"]]
            validate_annotation(annotation, records[job_id], case, plan)
            if case["schedule"] == "pause_after_two" and not isinstance(annotation.get("pause_respected"), bool):
                raise ValueError("Pause cases require pause_respected=true/false")
            annotations[job_id] = annotation
    return annotations


def report(run_dir, annotation_paths=()):
    run_dir = Path(run_dir)
    plan = checked_plan(run_dir / "plan.json")
    records, errors = completed_records(run_dir, plan)
    annotations = collect_annotations(annotation_paths, plan, records)
    cases = {c["id"]: c for c in plan["cases"]}
    rows = []
    for job in plan["jobs"]:
        if job["id"] not in annotations:
            continue
        case = cases[job["case_id"]]
        row = {**job, "action": case["action"], "target_count": case["target_count"], "schedule": case["schedule"]}
        row.update(score_annotation(annotations[job["id"]], case, plan))
        rows.append(row)
    groups = {}
    for method in METHODS:
        part = [r for r in rows if r["method"] == method]
        if part:
            groups[method] = {"annotated": len(part), "exact_usable_rate": statistics.mean(r["exact_usable"] for r in part),
                              "visually_usable_rate": statistics.mean(r["visually_usable"] for r in part),
                              "action_present_rate": statistics.mean(r["action_present"] for r in part),
                              "ambiguous_rate": statistics.mean(r["ambiguous"] for r in part),
                              "timing_success_rate": statistics.mean(r["timing_success"] for r in part),
                              "count_mae_judgeable": statistics.mean([r["absolute_count_error"] for r in part if r["absolute_count_error"] is not None])
                              if any(r["absolute_count_error"] is not None for r in part) else None}
    subgroups = {}
    for method in METHODS:
        subgroups[method] = {}
        for field in ("action", "target_count", "schedule"):
            buckets = collections.defaultdict(list)
            for row in rows:
                if row["method"] == method:
                    buckets[str(row[field])].append(row)
            subgroups[method][field] = {value: {"n": len(part), "exact_usable_rate": statistics.mean(r["exact_usable"] for r in part),
                                               "timing_success_rate": statistics.mean(r["timing_success"] for r in part),
                                               "pause_success_rate": statistics.mean(r["pause_success"] for r in part if r["pause_success"] is not None)
                                               if any(r["pause_success"] is not None for r in part) else None}
                                        for value, part in buckets.items()}
    paired = {}
    for control in ("count_prompt", "timed_prompt", "relay_unrouted"):
        left = {(r["case_id"], r["seed"]): r for r in rows if r["method"] == control}
        right = {(r["case_id"], r["seed"]): r for r in rows if r["method"] == "relay"}
        keys = sorted(left.keys() & right.keys())
        deltas = [right[k]["exact_usable"] - left[k]["exact_usable"] for k in keys]
        paired["relay_vs_" + control] = {"pairs": len(keys),
            "exact_usable_delta": statistics.mean(deltas) if deltas else None,
            "wins": deltas.count(1), "losses": deltas.count(-1), "ties": deltas.count(0)}
        for key in keys:
            rid = right[key]["id"]
            lid = left[key]["id"]
            if records[rid]["initial_latent_sha256"] != records[lid]["initial_latent_sha256"]:
                raise ValueError("Paired initial noise mismatch")
    ready = len(records) == len(plan["jobs"]) == len(annotations) and plan["scientific_evaluation"]
    result = {"plan_hash": plan["plan_hash"], "expected": len(plan["jobs"]), "generated": len(records),
              "missing": len(plan["jobs"]) - len(records), "execution_errors": len(errors),
              "annotated": len(annotations), "metrics_ready": ready, "methods": groups, "subgroups": subgroups, "paired": paired,
              "per_video": rows, "limitations": plan["limitations"],
              "interpretation": "Descriptive pilot only; inspect action/count subgroups and clips. No automatic publish/abandon decision.",
              "timing_tolerance_seconds": 0.25}
    atomic_json(run_dir / "summary.json", result)
    print(f"generated={len(records)}/{len(plan['jobs'])} annotated={len(annotations)} errors={len(errors)} metrics_ready={ready}; {run_dir / 'summary.json'}", flush=True)
    for name, values in paired.items():
        if values["pairs"]:
            print(name, values, flush=True)
    return result


def make_review(run_dir):
    run_dir = Path(run_dir)
    plan = checked_plan(run_dir / "plan.json")
    records, _ = completed_records(run_dir, plan)
    if not records:
        raise RuntimeError("No completed videos to review")
    cases = {c["id"]: c for c in plan["cases"]}
    items = []
    for record in records.values():
        case = cases[record["job"]["case_id"]]
        items.append({"id": record["job"]["id"], "video": record["video"], "sha256": record["video_sha256"],
                      "action": case["action"], "criterion": case["criterion"]})
    random.Random(82415).shuffle(items)
    data = json.dumps({"plan_hash": plan["plan_hash"], "items": items}, ensure_ascii=False).replace("<", "\\u003c")
    template = (Path(__file__).parent / "review_template.html").read_text()
    (run_dir / "review.html").write_text(template.replace("__PROBE_DATA__", data))
    print(f"Blind review: {run_dir / 'review.html'}; copy this directory with videos/ to a computer with a browser.")

