"""Versioned clean inputs: reference tests are installed only by the final grader."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .benchmark import official_grade
from .config import Task
from .control import atomic_json
from .harness import PROTOCOL
from .repository import IGNORED, git, validate_tree
from .runtime import create_sandbox
from .scale_data import sha256, tree_hash

GRADER_STARTED = "__SCA_HIDDEN_GRADER_STARTED__"


def read(path):
    return json.loads(Path(path).read_text())


def public_protected_paths(repo):
    """Derive protection from public filenames, never from reference test locations."""
    result = set()
    for p in validate_tree(repo, max_files=50000, max_bytes=1_000_000_000,
                           allowed_symlinks={str(p.relative_to(repo)): p.readlink().as_posix()
                                             for p in repo.rglob("*") if p.is_symlink()}):
        rel = p.relative_to(repo)
        test_dir = next((i for i, part in enumerate(rel.parts[:-1]) if part in {"tests", "test", "testing"}), None)
        if test_dir is not None:
            result.add(str(Path(*rel.parts[:test_dir+1])))
        elif p.name.startswith("test_") or p.name in {"conftest.py", "setup.py", "setup.cfg", "pyproject.toml"}:
            result.add(str(rel))
    return sorted(result)


def prepare_hidden_task(old_task_file, destination, runtime=None):
    """Copy, reverse the installed test patch, and prove round-trip equivalence.

    No old repository or receipt is modified. The original failed/gold-passed
    environment check remains valid because reapplying tests recovers its tree.
    """
    old_task_file, destination = Path(old_task_file), Path(destination)
    old = Task.from_file(old_task_file)
    audit_path = old_task_file.parent / "audit.json"
    audit = read(audit_path)
    if audit.get("status") != "validated" or sha256(old_task_file) != audit["task_sha256"]:
        raise ValueError("Source task has no matching validated preflight")
    if sha256(old.benchmark) != audit["benchmark_sha256"]:
        raise ValueError("Source grading metadata changed")
    if destination.exists():
        receipt = read(destination / "prepared.json")
        if receipt["source_task_sha256"] != sha256(old_task_file):
            raise ValueError("Prepared task belongs to a different source")
        return receipt
    destination.mkdir(parents=True)
    repo = destination / "repo"
    shutil.copytree(old.repo, repo, symlinks=True, ignore=shutil.ignore_patterns(*IGNORED))
    metadata = read(old.benchmark)
    patch = metadata["record"]["test_patch"].encode()
    script = (repo / ".sca_eval.sh").read_bytes()
    (repo / ".sca_eval.sh").unlink()
    git(repo, "init", "-q")
    git(repo, "apply", "--reverse", "--check", "--binary", "-", stdin=patch)
    git(repo, "apply", "--reverse", "--binary", "-", stdin=patch)
    clean_hash = tree_hash(repo)
    # This checks all source and dependency-metadata files, not just patch text.
    git(repo, "apply", "--binary", "-", stdin=patch)
    (repo / ".sca_eval.sh").write_bytes(script)
    if tree_hash(repo) != audit["repo_sha256"]:
        raise ValueError("Clean source + reference tests do not reproduce validated source")
    git(repo, "apply", "--reverse", "--binary", "-", stdin=patch)
    (repo / ".sca_eval.sh").unlink()
    if tree_hash(repo) != clean_hash:
        raise ValueError("Reference test reversal did not restore clean source")
    # The task's .git is newly initialized (no gold/future history); the actual
    # per-attempt private base commit is created by prepare_repository.
    private = destination / "grader"
    private.mkdir()
    (private / "test.patch").write_bytes(patch)
    (private / "eval.sh").write_bytes(script)
    metadata.update(sca_protocol=PROTOCOL, test_patch_file=str(private / "test.patch"),
                    eval_script=str(private / "eval.sh"))
    atomic_json(private / "benchmark.json", metadata)
    task = old.model_copy(update={"repo": str(repo), "test_command": "python -m pytest",
        "protected_paths": public_protected_paths(repo), "reference_patch": None, "replay": None,
        "benchmark": str(private / "benchmark.json"), "provenance": "swebench_full_hidden_tests_oci_adapted",
        "runtime": runtime or old.runtime})
    atomic_json(destination / "task.json", task.model_dump())
    receipt = {"protocol": PROTOCOL, "task_id": task.task_id, "task_file": str(destination / "task.json"),
               "source_task_sha256": sha256(old_task_file), "source_audit_sha256": sha256(audit_path),
               "repo_sha256": clean_hash, "reference_roundtrip_verified": True,
               "files": {name: sha256(destination / name) for name in
                         ("task.json", "grader/test.patch", "grader/eval.sh", "grader/benchmark.json")}}
    atomic_json(destination / "prepared.json", receipt)
    return receipt


def validate_hidden_task(task):
    if not task.benchmark:
        raise ValueError("Hidden evaluation requires trusted grading metadata")
    meta = read(task.benchmark)
    if meta.get("sca_protocol") != PROTOCOL:
        raise ValueError("Refusing visible-reference-test input in hidden protocol")
    repo = Path(task.repo).resolve()
    for name in (task.benchmark, meta["test_patch_file"], meta["eval_script"]):
        if Path(name).resolve().is_relative_to(repo):
            raise ValueError("Reference metadata must be outside the Agent repository")
    if (repo / ".sca_eval.sh").exists():
        raise ValueError("Reference evaluation script leaked into Agent repository")
    return meta


def grade_hidden(task, repo, events, log_path, timeout=600):
    """Called only AFTER the Agent has terminated, in its independent replay copy."""
    meta = validate_hidden_task(task)
    patch = Path(meta["test_patch_file"]).read_bytes()
    check = git(repo, "apply", "--check", "--binary", "-", stdin=patch, check=False)
    if check.returncode:
        return {"resolved": False, "error": "reference_test_application_failed",
                "details": check.stderr.decode(errors="replace")}
    git(repo, "apply", "--binary", "-", stdin=patch)
    # Protected files come from this graded copy, not the pre-patch Agent base.
    sb = create_sandbox(repo, events, [], timeout, repo, task.runtime)
    try:
        # A bootstrap error is an evaluation failure, not an incorrect repair.
        # Emit this before any candidate code runs; retain it in the complete log.
        command = f"test -r /grader/eval.sh && printf '{GRADER_STARTED}\\n' && exec /bin/bash /grader/eval.sh"
        test = sb.run(command, mounts={"/grader/eval.sh": Path(meta["eval_script"])},
                      log_path=log_path)
    finally:
        sb.cancel()
    if GRADER_STARTED not in test["output"].splitlines():
        return {"test": test, "resolved": False, "grader_started": False,
                "runtime_failure": True, "error": "hidden_grader_start_failed"}
    score = official_grade(Path(task.benchmark), log_path)
    failure = bool(score.get("infra_failure"))
    return {"test": test, "official": score, "grader_started": True, "runtime_failure": failure,
            "resolved": bool(score["resolved"] and not test["timed_out"] and not failure)}


def validate_prepared(folder, verify_repo=True):
    folder = Path(folder)
    receipt = read(folder / "prepared.json")
    if receipt.get("protocol") != PROTOCOL or not receipt.get("reference_roundtrip_verified"):
        raise ValueError("Prepared input has an invalid protocol receipt")
    for name, expected in receipt["files"].items():
        if sha256(folder / name) != expected:
            raise ValueError(f"Prepared input changed: {folder / name}")
    task = Task.from_file(folder / "task.json")
    validate_hidden_task(task)
    if verify_repo and tree_hash(Path(task.repo)) != receipt["repo_sha256"]:
        raise ValueError("Prepared clean repository changed")
    return receipt
