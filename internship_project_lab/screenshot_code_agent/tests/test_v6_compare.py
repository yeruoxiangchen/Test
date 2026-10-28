import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/run_v6_compare.py"
    spec = importlib.util.spec_from_file_location("v6_compare", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_selection_is_deterministic_balanced_and_retains_controls():
    rows = [{"task_id": f"task{i}", "group_id": f"repo{i%3}"} for i in range(15)]
    rows.append({"task_id": "sphinx-doc__sphinx-8713", "group_id": "control"})
    selected = module().choose_tasks(rows, 7)
    assert selected == module().choose_tasks(rows, 7)
    assert len({r["task_id"] for r in selected}) == 7
    assert {r["group_id"] for r in selected} == {"repo0", "repo1", "repo2", "control"}
    assert module().choose_tasks(rows, 0) == rows
    with pytest.raises(ValueError):
        module().choose_tasks(rows, 100)


def test_status_does_not_infer_process_liveness_from_timestamps(tmp_path):
    (tmp_path / "plan.json").write_text(json.dumps({"runs_per_arm": 12, "arms": [{"name": "v5"}, {"name": "v6"}]}))
    (tmp_path / "status.json").write_text(json.dumps({"stage": "evaluating", "pid": 999999999, "start_ticks": "bad"}))
    result = module().status(tmp_path)
    assert not result["controller_alive"]
    assert result["progress"]["v6"]["completed"] == 0


def test_diagnostics_counts_legacy_and_v6_failures_consistently(tmp_path):
    rows = [
        {"event": "model_request", "kind": "code", "tools": [], "messages": [
            {"role": "system", "content": "Compress this conversation into a brief summary."}]},
        {"event": "model_request", "kind": "summary"},
        {"event": "tool_start", "name": "edit_file"},
        {"event": "tool_result", "name": "edit_file", "output": json.dumps({"returncode": 1,
            "output": json.dumps({"error": "old_string must be nonempty and match exactly once"})})},
        {"event": "tool_start", "name": "edit_file"},
        {"event": "edit_failure", "error": "repeated_failed_edit", "repeated": True},
        {"event": "tool_result", "name": "edit_file", "output": "error: repeated_failed_edit\nactual source"},
    ]
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    result = module().diagnostics(tmp_path)
    assert result["code_requests"] == 2 and result["summary_requests"] == 2
    assert result["failed_edit_requests"] == 2 and result["blocked_repeats"] == 1


def test_report_keeps_failures_and_computes_paired_changes(tmp_path):
    plan = {"task_ids": ["a", "b"], "readers": ["oracle"], "runs_per_arm": 2,
            "interpretation": "development only", "arms": [{"name": "v5"}, {"name": "v6"}]}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    for arm, outcomes in [("v5", [False, True]), ("v6", [True, False])]:
        directory = tmp_path / arm / "evaluation"
        records = []
        for task, solved in zip(["a", "b"], outcomes):
            folder = directory / "runs" / (task + "_oracle")
            folder.mkdir(parents=True)
            (folder / "events.jsonl").write_text("")
            records.append({"task_id": task, "reader": "oracle", "model_inference": True, "resolved": solved,
                            "status": "verified" if solved else "unresolved", "changed_files": ["app.py"] if solved else [],
                            "elapsed_seconds": 1, "usage": {"prompt_tokens": 10, "completion_tokens": 2}})
        (directory / "report.json").write_text(json.dumps({"records": records}))
    result = module().report(tmp_path)
    assert result["paired"] == {"repaired": 1, "regressed": 1}
    assert result["arms"]["v6"]["attempted"] == 2
    assert result["complete"]
    bad = tmp_path / "v6/evaluation/report.json"
    records = json.loads(bad.read_text())["records"]
    records.pop()
    bad.write_text(json.dumps({"records": records}))
    with pytest.raises(ValueError, match="Incomplete"):
        module().report(tmp_path)


def test_runtime_fingerprint_rejects_modified_source(tmp_path):
    code = tmp_path / "runtime.py"
    code.write_text("original")
    (tmp_path / "source_state.json").write_text(json.dumps({"sha256": {"runtime.py": module().digest(code)}}))
    module().verify_runtime(tmp_path)
    code.write_text("modified")
    with pytest.raises(ValueError, match="changed"):
        module().verify_runtime(tmp_path)


def test_budget_comparison_rejects_extra_model_calls(tmp_path):
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "configs/v6-local.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps(data))
    arms = [{"name": "test", "runtime": str(tmp_path), "config": "config.json"}]
    module().check_budgets(arms)
    data["limits"]["max_model_calls"] = 24
    (tmp_path / "config.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Matched-budget"):
        module().check_budgets(arms)
