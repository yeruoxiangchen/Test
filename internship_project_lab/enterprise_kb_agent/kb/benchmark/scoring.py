"""Transparent automatic diagnostics; keyword matches are NOT semantic judges."""
from collections import defaultdict
import math
import random
import re
import statistics


def covered(evidence, hits):
    spans = sorted((max(evidence["start"], h["start"]), min(evidence["end"], h["end"]))
                   for h in hits if h["document_id"] == evidence["document_id"]
                   and h["start"] < evidence["end"] and h["end"] > evidence["start"])
    end, total = evidence["start"], 0
    for a, b in spans:
        total += max(0, b - max(a, end))
        end = max(end, b)
    return total / max(1, evidence["end"] - evidence["start"])


def retrieval_metrics(case, hits):
    gold = case["evidence"]
    if not gold:
        return {"evaluable": False, "returned": len(hits)}
    result = {"evaluable": True, "returned": len(hits)}
    for k in (1, 3, 5, 10):
        found = [covered(e, hits[:k]) >= .95 for e in gold]
        result[f"evidence_recall_at_{k}"] = sum(found) / len(gold)
        result[f"all_evidence_at_{k}"] = int(all(found))
        result[f"any_evidence_at_{k}"] = int(any(found))
    first = next((rank for rank in range(1, min(10, len(hits)) + 1)
                  if any(covered(e, hits[:rank]) >= .95 for e in gold)), None)
    result["mrr_at_10"] = 1 / first if first else 0
    # Unique evidence units prevent overlapping chunks from earning repeated credit.
    previous, gain = set(), 0.
    for rank in range(1, min(10, len(hits)) + 1):
        now = {i for i, e in enumerate(gold) if covered(e, hits[:rank]) >= .95}
        gain += len(now - previous) / math.log2(rank + 1)
        previous = now
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(gold))))
    result["evidence_ndcg_at_10_diagnostic"] = min(1., gain / ideal) if ideal else 0
    return result


def answer_metrics(case, prediction):
    answer = prediction.get("answer") or ""
    normalized = re.sub(r"\s+", "", answer).lower()
    facts = case["required_facts"]
    matched = [any(re.sub(r"\s+", "", alias).lower() in normalized for alias in alternatives) for alternatives in facts]
    citations = prediction.get("citations", [])
    labels = {c.get("label") for c in citations}
    inline = set(re.findall(r"\[(S\d+)\]", answer))
    gold = case["evidence"]
    coverage = sum(covered(e, citations) >= .95 for e in gold) / len(gold) if gold else None
    supported = sum(any(covered(e, [c]) >= .95 for e in gold) for c in citations)
    return {"action_correct": prediction["status"] == case["expected_action"],
        "failed": prediction["status"] not in {"answered", "insufficient", "clarify"},
        "false_refusal": prediction["status"] in {"insufficient", "clarify"} if case["expected_action"] == "answered" else None,
        "unsupported_answer": prediction["status"] == "answered" if case["expected_action"] != "answered" else None,
        "correct_abstention": prediction["status"] == "insufficient" if case["expected_action"] == "insufficient" else None,
        "correct_clarification": prediction["status"] == "clarify" if case["expected_action"] == "clarify" else None,
        "required_fact_match_diagnostic": sum(matched) / len(matched) if matched else None,
        "all_required_facts_diagnostic": all(matched) if matched else None,
        "citation_format_valid": prediction["status"] in {"answered", "insufficient", "clarify"} and inline == labels and (prediction["status"] != "answered" or bool(labels)),
        "citation_gold_evidence_recall": coverage,
        "citation_gold_evidence_precision": supported / len(citations) if citations and gold else None,
        "forbidden_output": any(word.lower() in answer.lower() for word in case["forbidden_strings"]) if case["forbidden_strings"] else None,
        "unauthorized_citation": any(c.get("visibility") == "admin" for c in citations) if case.get("user_role", "member") != "admin" else None,
        "semantic_correctness": None, "faithfulness": None,
        "note": "Fact string and gold-span overlap are diagnostics, not semantic entailment or answer accuracy."}


def cluster_interval(rows, key, n=1000):
    groups = defaultdict(list)
    for row in rows:
        value = row["metrics"].get(key)
        if value is not None:
            groups[row["family_id"]].append(float(value))
    if not groups:
        return None
    values = [statistics.mean(v) for v in groups.values()]
    rng = random.Random(42)
    samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(n))
    return {"mean": statistics.mean(values), "ci95_family_bootstrap": [samples[int(n*.025)], samples[min(n-1, int(n*.975))]],
            "families": len(values)}


def paired_difference(left, right, key, n=2000):
    a, b = {r["id"]: r for r in left}, {r["id"]: r for r in right}
    if len(a) != len(left) or len(b) != len(right):
        raise ValueError("Duplicate case ids in paired comparison")
    if a.keys() != b.keys():
        raise ValueError("Paired comparison requires the same completed case ids; failures must remain in results")
    rows = []
    for task_id, x in a.items():
        y = b[task_id]
        if x["family_id"] != y["family_id"]:
            raise ValueError("Paired case family mismatch: " + task_id)
        v, w = x["metrics"].get(key), y["metrics"].get(key)
        if (v is None) != (w is None):
            raise ValueError(f"Unequal metric availability: {key}, case {task_id}")
        if v is not None:
            rows.append({"family_id": x["family_id"], "metrics": {key: float(w) - float(v)}})
    return cluster_interval(rows, key, n)


def metric_keys(rows):
    """Use the same score/metadata boundary in summaries and paired reports."""
    return sorted({k for r in rows for k, v in r["metrics"].items()
                   if isinstance(v, (float, int)) and k != "evaluable"})


def summary(rows, total):
    keys = metric_keys(rows)
    latencies = sorted(r["latency_seconds"] for r in rows)
    return {"expected": total, "records": len(rows), "complete": len(rows) == total,
        "metrics": {k: cluster_interval(rows, k) for k in keys},
        "by_category": {c: {k: cluster_interval([r for r in rows if r["category"] == c], k, 200) for k in keys}
                        for c in sorted({r["category"] for r in rows})},
        "latency_seconds": {"p50": statistics.median(latencies), "p95": latencies[min(len(latencies)-1, math.ceil(len(latencies)*.95)-1)]} if rows else {},
        "model_calls": sum(r.get("model_calls", 0) for r in rows),
        "tool_calls": sum(r.get("tool_calls", 0) for r in rows),
        "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in rows),
        "completion_tokens": sum(r.get("completion_tokens", 0) for r in rows),
        "semantic_scores": "not_scored_pending_independent_judge_or_human_review"}
