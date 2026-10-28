"""Pinned SWE-bench Lite tasks, original tests, screenshot derivatives and gold audits."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path

from .benchmark import official_grade
from .config import RuntimeConfig
from .control import Events, atomic_json
from .data import render_screenshot
from .repository import git, validate_tree
from .runtime import create_sandbox

DATASET_REVISION = "b0dde1093fe417d83b7184254edf8199c1f0dff5"


def install_test_patch(repo: Path, patch: str):
    """Keep git discovery inside the exported task, even in a parent checkout."""
    git(repo, "init", "-q")
    git(repo, "apply", "--check", "--binary", "-", stdin=patch.encode())
    git(repo, "apply", "--binary", "-", stdin=patch.encode())


def select_records(parquet: Path, output: Path) -> list[dict]:
    import pyarrow.parquet as pq
    rows = pq.read_table(parquet).to_pylist()
    selected = []
    for repo, split in [("pytest-dev/pytest", "dev"), ("sympy/sympy", "test")]:
        group = []
        for record in sorted((r for r in rows if r["repo"] == repo), key=lambda r: r["instance_id"]):
            usable = [b for b in re.finditer(r"```[^\n]*\n(.*?)```", record["problem_statement"], re.S)
                      if 2 <= len(b[1].splitlines()) <= 35 and len(b[1]) <= 2200]
            if not usable:
                continue
            record.update(sca_split=split, screenshot_block=usable[0][1], screenshot_span=list(usable[0].span()))
            group.append(record)
            if len(group) == 5:
                break
        if len(group) != 5:
            raise ValueError("Selection must contain five predeclared tasks per repository")
        selected.extend(group)
    output.mkdir(parents=True, exist_ok=True)
    selection = {"dataset": "SWE-bench/SWE-bench_Lite", "revision": DATASET_REVISION, "original_split": "test",
                 "source_parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
                 "selection_rule": "For each of pytest-dev/pytest and sympy/sympy, sort instance IDs and take the first 5 with a fenced block of 2..35 lines and <=2200 chars. Frozen before model calls.",
                 "grouping": "pytest=development, sympy=test, assigned before any screenshot variants.",
                 "selected_ids": [r["instance_id"] for r in selected],
                 "evaluation_scope": "screenshot-derived subset, not original leaderboard results"}
    if (output / "selected_records.json").exists():
        old = json.loads((output / "selected_records.json").read_text())
        if old != selected:
            raise ValueError("Frozen selection differs; create a new dataset version")
    atomic_json(output / "selected_records.json", selected)
    atomic_json(output / "selection.json", selection)
    return selected


def source_archive(record: dict, cache: Path, destination: Path):
    archive = cache / (record["repo"].replace("/", "_") + "_" + record["base_commit"] + ".tar.gz")
    cache.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        subprocess.run(["curl", "-fL", "--retry", "3", "--retry-all-errors", "--connect-timeout", "20",
                        "--max-time", "300", f"https://api.github.com/repos/{record['repo']}/tarball/{record['base_commit']}",
                        "-o", str(archive) + ".part"], check=True)
        Path(str(archive) + ".part").replace(archive)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as source:
        members = source.getmembers()
        roots = {Path(m.name).parts[0] for m in members}
        if len(roots) != 1:
            raise ValueError("GitHub source archive must have one top-level directory")
        for member in members:
            rel = Path(*Path(member.name).parts[1:])
            if rel == Path("."):
                continue
            if rel.is_absolute() or ".." in rel.parts or not (member.isfile() or member.isdir()):
                raise ValueError("Source archive contains an unsafe or unsupported member")
            member.name = rel.as_posix()
            source.extract(member, destination, filter="data")
    validate_tree(destination)
    return hashlib.sha256(archive.read_bytes()).hexdigest()


def test_command(eval_script: str) -> str:
    match = re.search(r": ['\"]>>>>> Start Test Output['\"]\s*\n(.*?)\n: ['\"]>>>>> End Test Output['\"]",
                      eval_script, re.S)
    if not match:
        raise ValueError("Cannot identify the original official test command")
    return match[1].strip()


def build_official(output: Path, task_checkout: Path, harness_checkout: Path, prefix: Path,
                   archive_cache: Path, timeout: int = 120) -> dict:
    output, task_checkout, harness_checkout = output.resolve(), task_checkout.resolve(), harness_checkout.resolve()
    records = json.loads((output / "selected_records.json").read_text())
    audit = []
    manifest = []
    all_tasks = []
    for number, record in enumerate(records, 1):
        instance = record["instance_id"]
        folder = output / instance
        print(f"Official task [{number}/{len(records)}]: {instance}", flush=True)
        if (folder / "audit.json").exists():
            saved = json.loads((folder / "audit.json").read_text())
            audit.append(saved)
            row = saved.get("manifest_row")
            if row:
                all_tasks.append(row)
                if saved["status"] == "validated":
                    manifest.append(row)
            continue
        folder.mkdir(exist_ok=True)
        item = {"instance_id": instance, "repo": record["repo"], "status": "error"}
        try:
            repo = folder / "repo"
            if repo.exists():
                raise ValueError("Unfinished task directory exists; inspect it and use a new version or remove only its incomplete repo")
            item["archive_sha256"] = source_archive(record, archive_cache, repo)
            # An explicit repository prevents git from discovering the surrounding
            # Tracker checkout and silently skipping repository-relative patches.
            upstream_task = task_checkout / "tasks" / instance
            # Require exact agreement with the pinned HF dataset, not merely matching IDs.
            for filename, key in [("test.patch", "test_patch"), ("gold.patch", "patch")]:
                if (upstream_task / filename).read_text().strip() != record[key].strip():
                    raise ValueError(f"HF and official task checkout differ for {filename}")
            script = (upstream_task / "eval.sh").read_text()
            command = test_command(script)
            # Dataset test patch is installed before any screenshot/model interaction.
            install_test_patch(repo, record["test_patch"])
            eval_body = ("#!/bin/bash\nset -uo pipefail\nprintf '\\n>>>>> Start Test Output\\n'\n" + command
                         + "\nSCA_TEST_EXIT=$?\nprintf '\\n>>>>> End Test Output\\n>>>>> Test Exit Code: %s\\n' \"$SCA_TEST_EXIT\"\nexit \"$SCA_TEST_EXIT\"\n")
            (repo / ".sca_eval.sh").write_text(eval_body)
            protected = sorted({".sca_eval.sh", *re.findall(r"^\+\+\+ b/(.+)$", record["test_patch"], re.M),
                                *[str(p.relative_to(repo)) for p in repo.rglob("conftest.py")]})
            runtime = RuntimeConfig(backend="venv", prefix=str(prefix.resolve()))
            if record["repo"] == "pytest-dev/pytest":
                version = record["version"] + ".0" if record["version"].count(".") == 1 else record["version"]
                generator = ("from setuptools_scm import get_version; "
                             "print(get_version(root='/workspace',write_to='src/_pytest/_version.py'))")
                setup = create_sandbox(repo, Events(folder / "setup_events.jsonl"), [], timeout, repo, runtime)
                setup_result = setup.run("SETUPTOOLS_SCM_PRETEND_VERSION=" + shlex.quote(version)
                    + " python -c " + shlex.quote(generator), log_path=folder / "runtime_setup.log")
                if setup_result["returncode"]:
                    raise RuntimeError("Pytest runtime version generation failed: " + setup_result["output"])
                protected.append("src/_pytest/_version.py")
            # Gold metadata never enters the repository or model's task prompt.
            metadata = {"record": {k: v for k, v in record.items() if not k.startswith("screenshot_")},
                        "harness_checkout": str(harness_checkout), "task_checkout": str(task_checkout),
                        "official_eval_script_sha256": hashlib.sha256(script.encode()).hexdigest(),
                        "test_command": command, "runtime_adaptation": "isolated Python prefix, original test command and test patch; not official Docker image"}
            atomic_json(folder / "benchmark.json", metadata)
            (folder / "reference.patch").write_text(record["patch"])
            begin, end = record["screenshot_span"]
            # Replace the selected original issue block. All groups see this same description.
            problem = record["problem_statement"][:begin] + "[See the attached screenshot for this code/output block.]" + record["problem_statement"][end:]
            visible = record["screenshot_block"].replace("\r\n", "\n").rstrip() + "\n"
            render_screenshot(visible, folder / "screenshot_light.png", "light")
            render_screenshot(visible, folder / "screenshot_dark.png", "dark")
            task = {"task_id": instance, "repo": "repo", "problem": problem, "test_command": "bash .sca_eval.sh",
                    "protected_paths": protected, "split": record["sca_split"], "group_id": record["repo"],
                    "provenance": "swebench_lite_screenshot_derived", "reference_patch": "reference.patch",
                    "benchmark": "benchmark.json", "runtime": runtime.model_dump(),
                    "images": [{"image_id": "shot", "path": "screenshot_light.png", "reference": "screenshot_light.txt",
                                "layout": "screenshot_light.layout.json"}]}
            atomic_json(folder / "task_light.json", task)
            row = {"task_file": instance + "/task_light.json", "task_id": instance, "group_id": record["repo"],
                   "split": record["sca_split"], "provenance": task["provenance"]}
            all_tasks.append(row)
            item["manifest_row"] = row
            events = Events(folder / "validation_events.jsonl")
            sandbox = create_sandbox(repo, events, protected, timeout, repo, runtime)
            sandbox.probe()
            before = sandbox.run("bash .sca_eval.sh", log_path=folder / "baseline_test.log")
            baseline = official_grade(folder / "benchmark.json", folder / "baseline_test.log")
            item["baseline"] = {"returncode": before["returncode"], "timed_out": before["timed_out"], "official": baseline}
            gold_repo = folder / "gold_check"
            shutil.copytree(repo, gold_repo)
            git(gold_repo, "apply", "--binary", "-", stdin=record["patch"].encode())
            gold = create_sandbox(gold_repo, events, protected, timeout, repo, runtime)
            after = gold.run("bash .sca_eval.sh", log_path=folder / "reference_test.log")
            graded = official_grade(folder / "benchmark.json", folder / "reference_test.log")
            item["gold"] = {"returncode": after["returncode"], "timed_out": after["timed_out"], "official": graded}
            # Infrastructure gate only; original ten-task selection is retained in the report.
            f2p = baseline.get("tests_status", {}).get("FAIL_TO_PASS", {})
            item["status"] = "validated" if (baseline["parsed"] and not baseline["resolved"]
                and f2p.get("failure") and graded["resolved"] and not before["timed_out"] and not after["timed_out"]) else "environment_failed"
            if item["status"] == "validated":
                manifest.append(row)
            shutil.rmtree(gold_repo)
        except Exception as error:
            item.update(error=str(error), error_type=type(error).__name__)
        audit.append(item)
        atomic_json(folder / "audit.json", item)
        print(f"  {instance}: {item['status']} {item.get('error','')}", flush=True)
        atomic_json(output / "build_status.json", {"completed": number, "total": len(records),
                    "validated": sum(a["status"] == "validated" for a in audit)})
    for name, rows in [("tasks.jsonl", manifest), ("all_tasks.jsonl", all_tasks)]:
        (output / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = {"dataset": "SWE-bench/SWE-bench_Lite", "dataset_revision": DATASET_REVISION,
              "original_selection": len(records), "validated": len(manifest), "audits": audit,
              "groups": len({r["repo"] for r in records}), "synthetic_screenshots": True,
              "official_container_used": False, "interpretation": "Screenshot-derived fixed subset using official tests and grader in an adapted runtime; no leaderboard comparability."}
    atomic_json(output / "build_report.json", report)
    return report
