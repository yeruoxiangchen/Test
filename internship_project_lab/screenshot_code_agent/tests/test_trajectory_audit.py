import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "trajectory_audit", Path(__file__).resolve().parents[1] / "scripts/audit_scale_trajectories.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_quoted_worker_errors_in_successful_source_reads_are_not_counted():
    text = ('returncode: 0\n--- output ---\n# /sca_worker.py\n'
            '# SyntaxError: future feature annotations is not defined\n'
            'error_type: "AttributeError"\nerror: "has no attribute \'is_relative_to\'"')
    assert audit.labels_for("read_file", text, audit.metadata(text)) == []


@pytest.mark.parametrize("output,label", [
    ('returncode: 1\n--- output ---\nFile "/sca_worker.py", line 3\n'
     'SyntaxError: future feature annotations is not defined', "worker_python36_syntax"),
    ('error_type: "AttributeError"\nerror: "\'PosixPath\' object has no attribute \'is_relative_to\'"\n'
     'returncode: 1', "worker_pathlib_api"),
])
def test_real_worker_compatibility_errors_are_detected(output, label):
    assert label in audit.labels_for("read_file", output, audit.metadata(output))
    assert label not in audit.labels_for("run_tests", output, audit.metadata(output))


def task_fixture(tmp_path, *, omit_result_event=False):
    result = {"task_id": "sample", "group_id": "org/repo", "model_inference": True,
              "resolved": True, "status": "verified", "exit_reason": "budget_exceeded",
              "agent_error": "Shared model call limit reached (decisions + summaries)",
              "usage": {"tool_calls": 1}, "changed_files": ["source.py"]}
    (tmp_path / "result.json").write_text(json.dumps(result))
    (tmp_path / "config.json").write_text(json.dumps({"limits": {}, "agent": {}, "max_context_tokens": 8192}))
    (tmp_path / "patch.diff").write_text("a genuine source patch")
    events = [{"event": "model_request", "kind": "code"},
              {"event": "tool_start", "name": "edit_file", "arguments": {}, "tool_call": 1}]
    if not omit_result_event:
        events.append({"event": "tool_result", "name": "edit_file",
                       "output": 'ok: true\nchanged: true\nreturncode: 0'})
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return tmp_path


def test_verified_patch_remains_successful_when_agent_exhausts_budget(tmp_path):
    row = audit.analyze_task(task_fixture(tmp_path), "coder", "sample", {})
    assert row["resolved"] and not row["zero_patch_failure"]
    assert row["successful_explicit_mutations"] == 1
    summary = audit.aggregate([row])
    assert summary["resolved"] == summary["budget_exceeded_but_resolved"] == 1


def test_incomplete_tool_trace_cannot_silently_be_counted_as_successful(tmp_path):
    with pytest.raises(ValueError, match="Incomplete|incomplete"):
        audit.analyze_task(task_fixture(tmp_path, omit_result_event=True), "coder", "sample", {})
