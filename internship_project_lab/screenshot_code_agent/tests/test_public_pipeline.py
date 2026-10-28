import importlib.util
import json
import os
from pathlib import Path

import pytest

from screenshot_agent.public_data import combine_manifests
from screenshot_agent.report import make_report


def manifest(directory, name, rows):
    path = directory / name / "tasks.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    return path


def row(task, group="boltons", split="demo"):
    return {"task_id": task, "group_id": group, "split": split, "task_file": f"{task}/task_light.json"}


def test_combined_manifest_preserves_task_paths_and_repository_groups(tmp_path):
    source = manifest(tmp_path, "public", [row("first"), row("second")])
    output = tmp_path / "suite/tasks.jsonl"
    result = combine_manifests([source], output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert result["task_count"] == 2
    assert result["repository_or_task_groups"] == 1
    assert (output.parent / rows[0]["task_file"]).resolve() == source.parent / "first/task_light.json"


@pytest.mark.parametrize("second", [row("first"), row("second", split="test")])
def test_combined_manifest_rejects_duplicates_and_group_leakage(tmp_path, second):
    a = manifest(tmp_path, "a", [row("first")])
    b = manifest(tmp_path, "b", [second])
    output = tmp_path / "suite/tasks.jsonl"
    with pytest.raises(ValueError):
        combine_manifests([a, b], output)
    assert not output.exists()


def test_reports_keep_public_demo_and_synthetic_test_denominators_separate(tmp_path):
    definitions = [
        ("public_a", "boltons", "demo", "public_historical_fix_with_synthetic_screenshot", True),
        ("public_b", "boltons", "demo", "public_historical_fix_with_synthetic_screenshot", False),
        ("synthetic", "synthetic", "test", "self_authored_synthetic_micro_repository", True),
    ]
    for task, group, split, provenance, resolved in definitions:
        directory = tmp_path / "runs" / task
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(json.dumps({
            "task_id": task, "group_id": group, "split": split, "provenance": provenance,
            "reader": "vlm", "code_backend": "api", "status": "verified" if resolved else "unresolved",
            "resolved": resolved, "failure_stage": None if resolved else "repair_or_verification",
            "validation": {"baseline": {"returncode": 1}, "patch_replay": {"applied": True},
                           "final": {"returncode": 0 if resolved else 1}},
        }))
    (tmp_path / "batch_job.json").write_text(json.dumps({"tasks": ["a", "b", "c", "not_started"], "readers": ["vlm"]}))
    result = make_report(tmp_path)
    cohorts = {r["split"]: r for r in result["summary_by_cohort"]}
    assert cohorts["demo"]["attempted"] == 2
    assert cohorts["demo"]["resolved"] == 1
    assert cohorts["demo"]["independent_task_groups"] == 1
    assert cohorts["test"]["resolved_fraction"] == 1
    assert result["unfinished_or_not_attempted"] == 1
    assert cohorts["demo"]["baseline_failures_reproduced"] == 2
    assert cohorts["demo"]["final_tests_passed"] == 1
    assert "public_b / vlm: unresolved" in Path(tmp_path, "report.md").read_text()


def analysis_module():
    path = Path(__file__).resolve().parents[1] / "scripts/archive_evaluation.py"
    spec = importlib.util.spec_from_file_location("archive_evaluation_test", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_final_archive_refuses_unfinished_batch(tmp_path):
    run = tmp_path / "evaluation"
    run.mkdir()
    (run / "status.json").write_text(json.dumps({"status": "running", "completed": 1, "total": 2}))
    with pytest.raises(ValueError, match="fully completed"):
        analysis_module().archive(run, tmp_path / "artifacts", tmp_path / "final_report.md")
    assert not (tmp_path / "final_report.md").exists()


def test_model_analysis_cannot_credit_known_patch_replays(tmp_path):
    run = tmp_path / "evaluation"
    task = run / "runs/replay"
    task.mkdir(parents=True)
    (run / "status.json").write_text(json.dumps({"status": "complete", "completed": 1, "total": 1}))
    (task / "result.json").write_text(json.dumps({"task_id": "known", "group_id": "known", "split": "demo",
        "provenance": "synthetic", "reader": "oracle", "code_backend": "replay", "model_inference": False,
        "status": "verified", "resolved": True}))
    with pytest.raises(ValueError, match="known-patch replay"):
        analysis_module().archive(run, tmp_path / "artifacts", tmp_path / "final_report.md")
    assert not (tmp_path / "final_report.md").exists()


def test_archive_waits_for_launcher_statistics_after_batch_counters_complete(tmp_path, monkeypatch):
    module = analysis_module()
    run = tmp_path / "evaluation"
    run.mkdir()
    state = {"status": "complete", "completed": 3, "total": 3}
    (run / "status.json").write_text(json.dumps(state))
    (tmp_path / "model_server.pid").write_text(str(os.getpid()))
    sleeps = []

    def collect_stats(seconds):
        sleeps.append(seconds)
        (tmp_path / "final_model_stats.json").write_text('{}')

    monkeypatch.setattr(module.time, "sleep", collect_stats)
    assert module.wait_for_completion(run, 5) == state
    assert sleeps == [2]
