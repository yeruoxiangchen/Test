"""Fail-closed gate: complete 32-object learned-layer + SS occupancy overfit."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from .evaluate_ss_occupancy import aggregate_records
from .prepare_official_experiment import atomic_json, read_json


def collect(directory, expected_mode):
    directory = Path(directory)
    identities = [read_json(p) for p in sorted(directory.glob("identity_*.json"))]
    if not identities:
        raise ValueError("missing evaluation identities")
    shared = ("manifest_sha256", "parent", "candidate", "lari", "decoder", "steps", "cfg", "seeds", "mode")
    for identity in identities:
        if any(identity[key] != identities[0][key] for key in shared):
            raise ValueError("evaluation shards have different model/data/sampling identities")
    records = [r for path in sorted(directory.glob("records/*.json")) for r in read_json(path)["records"]]
    if any(r["mode"] != expected_mode for r in records):
        raise ValueError("unexpected evaluation mode")
    return records, identities[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--lari_initial_report", required=True)
    parser.add_argument("--lari_final_report", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_report", required=True)
    args = parser.parse_args()
    uids = {r["uid"] for r in read_json(args.manifest)["samples"]}
    parent, parent_identity = collect(args.baseline_dir, "parent")
    candidate, candidate_identity = collect(args.candidate_dir, "candidate")
    for key in ("manifest_sha256", "parent", "decoder", "steps", "cfg", "seeds"):
        if parent_identity[key] != candidate_identity[key]:
            raise ValueError(f"parent/candidate comparison mismatch: {key}")
    if candidate_identity["layer_source"] != "predicted":
        raise ValueError("oracle layers cannot pass the learned-model overfit gate")
    seeds = [int(x) for x in parent_identity["seeds"].split(",")]
    expected = {(uid, seed) for uid in uids for seed in seeds}
    for records in (parent, candidate):
        actual = [(r["uid"], r["seed"]) for r in records]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("incomplete or duplicated occupancy evaluation")
        if not all(np.isfinite(r[k]) for r in records for k in ("iou", "precision", "recall", "f1")):
            raise ValueError("non-finite evaluation")
    initial, final = read_json(args.lari_initial_report), read_json(args.lari_final_report)
    layer_metrics = {}
    for label, report in (("initial", initial), ("final", final)):
        rows = report["records"]
        if len(rows) != len(uids) or {r["uid"] for r in rows} != uids:
            raise ValueError("layer evaluation object mismatch")
        layer_metrics[label] = {key: float(np.mean([r[key] for r in rows])) for key in
                                ("valid_depth_mae", "stop_accuracy", "target_ownership_iou_on_layer_slots")}
    if final["checkpoint"]["sha256"] != candidate_identity["lari"]["sha256"]:
        raise ValueError("layer evaluation and SS input checkpoint mismatch")
    summary = aggregate_records(parent + candidate)
    checks = {
        "exactly_32_objects": len(uids) == 32,
        "complete_two_seed_sampling": len(seeds) >= 2,
        "depth_mae_halved": layer_metrics["final"]["valid_depth_mae"] < .5 * layer_metrics["initial"]["valid_depth_mae"],
        "stop_accuracy_at_least_90pct": layer_metrics["final"]["stop_accuracy"] >= .9,
        "mean_iou_gain_positive": summary["paired"]["iou_gain"] > 0,
        "object_iou_win_rate_at_least_half": summary["paired"]["object_iou_win_rate"] >= .5,
    }
    report = {"passed": all(checks.values()), "checks": checks, "layer_metrics": layer_metrics,
              "occupancy": summary, "parent_identity": parent_identity,
              "candidate_identity": candidate_identity,
              "scope": "training-object overfit gate; not generalization",
              "formal_training_allowed": all(checks.values())}
    atomic_json(args.output_report, report)
    import json
    print(json.dumps({k: report[k] for k in ("passed", "checks", "layer_metrics", "occupancy")}), flush=True)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
