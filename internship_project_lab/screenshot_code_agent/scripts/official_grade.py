#!/usr/bin/env python3
"""Run the pinned, UNMODIFIED SWE-bench log parser and grader in a small process.

Only the harness package initializer is bypassed: it eagerly imports optional
Docker/cloud collectors and auto-loads dotenv files. Every grading function,
constant and log parser comes directly from the recorded official checkout.
Expected pytest-options IDs are adapted with that same unmodified parser;
the report retains raw IDs and rejects any normalization collisions.
"""

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

from screenshot_agent.benchmark import canonicalize_expected_ids


def grade(checkout: Path, metadata: dict, log: Path) -> dict:
    checkout = checkout.resolve()
    sys.path.insert(0, str(checkout))
    harness = types.ModuleType("swebench.harness")
    harness.__path__ = [str(checkout / "swebench/harness")]
    sys.modules["swebench.harness"] = harness
    grading = importlib.import_module("swebench.harness.grading")
    spec_type = importlib.import_module("swebench.types").TestSpec
    record = metadata["record"]

    def sequence(key):
        value = record[key]
        return json.loads(value) if isinstance(value, str) else value

    spec = spec_type(instance_id=record["instance_id"], image=record["image"],
                     eval_script_list=[], repo=record["repo"], version=record["version"],
                     FAIL_TO_PASS=sequence("FAIL_TO_PASS"), PASS_TO_PASS=sequence("PASS_TO_PASS"),
                     log_parser=record["log_parser"], eval_type=record["eval_type"])
    parser = grading.PARSER_REGISTRY[spec.log_parser]
    normalization = None
    if parser is grading.PARSER_REGISTRY["parse_log_pytest_options"]:
        normalization = canonicalize_expected_ids(spec, parser)
        spec.FAIL_TO_PASS = normalization["canonical"]["FAIL_TO_PASS"]
        spec.PASS_TO_PASS = normalization["canonical"]["PASS_TO_PASS"]
    status_map, parsed = grading.get_logs_eval(spec, str(log))
    report = grading.get_eval_report(spec, {"instance_id": spec.instance_id,
        "model_patch": "externally applied and verified patch", "model_name_or_path": "screenshot-code-agent"},
        str(log), True)[spec.instance_id]
    return {**report, "parsed": parsed, "status_map": status_map,
            "grading_source": str(Path(grading.__file__).resolve()),
            "expected_id_normalization": normalization,
            "expected_fail_to_pass": spec.FAIL_TO_PASS, "expected_pass_to_pass": spec.PASS_TO_PASS}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    args = p.parse_args()
    metadata = json.loads(args.metadata.read_text())
    print(json.dumps(grade(Path(metadata["harness_checkout"]), metadata, args.log)))
