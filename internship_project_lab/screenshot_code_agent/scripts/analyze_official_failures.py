#!/usr/bin/env python3
"""Count observable trace events; do not infer a causal localization error rate."""

import argparse
import collections
import json
from pathlib import Path


def analyze(run):
    rows = []
    for path in sorted((run/"evaluation/runs").glob("*/result.json")):
        result = json.loads(path.read_text())
        with (path.parent/"events.jsonl").open(encoding="utf-8") as stream:
            events = [json.loads(line) for line in stream if line.strip()]
        tools = [e for e in events if e["event"]=="tool_start"]
        responses = [e for e in events if e["event"]=="model_response" and e.get("kind")=="code"]
        limit = json.loads((path.parent/"config.json").read_text())["code"]["max_output_tokens"]
        row = {"task_id": result["task_id"], "reader": result["reader"], "resolved": result["resolved"],
               "exit_reason": result.get("exit_reason"), "tools": dict(collections.Counter(e["name"] for e in tools)),
               "usage": result.get("usage"),
               "read_paths": [e.get("arguments",{}).get("file_path") for e in tools if e["name"]=="read_file"],
               "code_responses_at_output_cap": sum(e.get("response",{}).get("usage",{}).get("completion_tokens",0)>=limit for e in responses),
               "tool_errors": [str(e)[:500] for e in events if e["event"]=="tool_error"]}
        rows.append(row)
    report = {"records": rows, "scope": "Observed tool requests and output lengths. A request need not succeed; file overlap does not establish correct localization. Reaching the output cap is a truncation cue, not a provider finish_reason."}
    (run/"trace_diagnostics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps([{k:v for k,v in r.items() if k not in ["usage","tool_errors"]} for r in rows],ensure_ascii=False,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    args = parser.parse_args()
    analyze(args.run.resolve())
