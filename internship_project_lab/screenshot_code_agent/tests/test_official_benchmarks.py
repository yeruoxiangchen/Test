import json
import shutil
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from screenshot_agent.benchmark import official_grade
from screenshot_agent.config import RuntimeConfig, Task
from screenshot_agent.control import Events
from screenshot_agent.official_data import install_test_patch, select_records, source_archive
from screenshot_agent.official_data import test_command as extract_test_command
from screenshot_agent.repository import git
from screenshot_agent.runtime import create_sandbox

PROJECT = Path(__file__).resolve().parents[1]


def metadata(tmp_path):
    if not (PROJECT / "vendor/SWE-bench/swebench/harness/grading.py").is_file():
        pytest.skip("Run bash scripts/setup_official.sh for optional official-source grading checks")
    p = tmp_path / "benchmark.json"
    p.write_text(json.dumps({"harness_checkout": str(PROJECT / "vendor/SWE-bench"), "record": {
        "instance_id": "fixture", "image": "unused", "repo": "pytest-dev/pytest", "version": "4.4",
        "FAIL_TO_PASS": ["test_a.py::test_bug"], "PASS_TO_PASS": ["test_a.py::test_keep"],
        "log_parser": "parse_log_pytest", "eval_type": "pass_and_fail"}}))
    return p


@pytest.mark.parametrize("bug,keep,resolved", [("PASSED", "PASSED", True),
    ("SKIPPED", "PASSED", False), ("PASSED", "FAILED", False), ("FAILED", "PASSED", False)])
def test_upstream_grader_requires_fix_and_regression_preservation(tmp_path, bug, keep, resolved):
    log = tmp_path / "test.log"
    log.write_text(f">>>>> Start Test Output\n{bug} test_a.py::test_bug\n{keep} test_a.py::test_keep\n"
                   ">>>>> End Test Output\n>>>>> Test Exit Code: 0\n")
    report = official_grade(metadata(tmp_path), log)
    assert report["parsed"]
    assert report["resolved"] is resolved
    assert report["grading_source"].startswith(str(PROJECT / "vendor/SWE-bench"))


def test_upstream_grader_rejects_a_suite_that_never_ran(tmp_path):
    log = tmp_path / "test.log"
    log.write_text(">>>>> Start Test Output\nModuleNotFoundError: absent\n>>>>> End Test Output\n>>>>> Test Exit Code: 2\n")
    result = official_grade(metadata(tmp_path), log)
    assert not result["parsed"] and not result["resolved"]


def test_task_selection_is_frozen_and_groups_precede_rendering(tmp_path):
    rows = []
    for repo in ["pytest-dev/pytest", "sympy/sympy"]:
        for number in reversed(range(8)):
            rows.append({"repo": repo, "instance_id": repo.replace("/", "__") + f"-{number}",
                         "problem_statement": "Problem\n```python\na = 1\nb = 2\n```"})
    parquet = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    records = select_records(parquet, tmp_path / "selection")
    assert len(records) == 10
    assert all(r["instance_id"].endswith(tuple(str(i) for i in range(5))) for r in records)
    assert len({(r["repo"], r["sca_split"]) for r in records}) == 2
    assert records == select_records(parquet, tmp_path / "selection")


def test_source_archive_rejects_escape_members(tmp_path):
    record = {"repo": "unit/repo", "base_commit": "abc"}
    cache = tmp_path / "cache"
    cache.mkdir()
    with tarfile.open(cache / "unit_repo_abc.tar.gz", "w:gz") as archive:
        info = tarfile.TarInfo("root/../../escape")
        archive.addfile(info)
    with pytest.raises(ValueError, match="unsafe"):
        source_archive(record, cache, tmp_path / "repo")
    assert not (tmp_path / "escape").exists()


def test_test_patch_applies_inside_export_not_the_surrounding_checkout(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "target.py").write_text("old\n")
    exported = tmp_path / "nested/export"
    exported.mkdir(parents=True)
    (exported / "target.py").write_text("old\n")
    patch = "diff --git a/target.py b/target.py\n--- a/target.py\n+++ b/target.py\n@@ -1 +1 @@\n-old\n+new\n"
    install_test_patch(exported, patch)
    assert (exported / "target.py").read_text() == "new\n"
    assert (tmp_path / "target.py").read_text() == "old\n"


def test_original_test_command_extraction_does_not_copy_gold_or_setup():
    script = "pip install private\ngit apply gold\n: '>>>>> Start Test Output'\npytest -rA testing/a.py\n: '>>>>> End Test Output'\ngit checkout base\n"
    assert extract_test_command(script) == "pytest -rA testing/a.py"
    with pytest.raises(ValueError):
        extract_test_command("pytest test.py")


def test_runtime_and_external_grading_paths_resolve_relative_to_task(tmp_path):
    p = tmp_path / "task.json"
    p.write_text(json.dumps({"task_id": "unit", "repo": "repo", "problem": "x", "benchmark": "private.json",
                            "runtime": {"backend": "venv", "prefix": "runtime"}}))
    task = Task.from_file(p)
    assert task.benchmark == str(tmp_path / "private.json")
    assert task.runtime.prefix == str(tmp_path / "runtime")


def test_dependency_prefix_is_read_only_and_uses_repository_pythonpath(tmp_path):
    prefix, repo = tmp_path / "prefix", tmp_path / "repo"
    (prefix / "bin").mkdir(parents=True)
    shutil.copyfile("/usr/bin/python3", prefix / "bin/python")
    (prefix / "bin/python").chmod(0o755)
    repo.mkdir()
    (repo / "value.py").write_text("answer = 7\n")
    sb = create_sandbox(repo, Events(tmp_path / "events.jsonl"), [], 10, repo,
                        RuntimeConfig(backend="venv", prefix=str(prefix)))
    try:
        sb.probe()
        value = sb.run("python -c 'import value; print(value.answer)'")
        assert value["returncode"] == 0 and "7" in value["output"]
        assert sb.run(f"touch {prefix}/forbidden")["returncode"] != 0
        assert not (prefix / "forbidden").exists()
        log = tmp_path / "full.log"
        result = sb.run("python -c 'print(\"x\" * 100000)'", log_path=log)
        assert len(log.read_bytes()) > 100000
        assert "output truncated" in result["output"]
    finally:
        sb.cancel()
