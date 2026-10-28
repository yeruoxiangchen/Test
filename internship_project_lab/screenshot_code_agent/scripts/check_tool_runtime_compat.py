#!/usr/bin/env python3
"""Bounded CPU-only filesystem-tool checks in existing OCI Python runtimes.

Only runtime metadata is reused from prepared tasks. The original repositories,
tests, audits, and historical results are never used as writable workspaces.
"""

import argparse
import contextlib
import datetime
import hashlib
import json
import signal
import tempfile
import time
from pathlib import Path

from screenshot_agent.config import Task
from screenshot_agent.control import Events, atomic_json
from screenshot_agent.runtime import create_sandbox

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("python36", "pydicom__pydicom-816", "3.6"),
    ("python38", "pydicom__pydicom-1090", "3.8"),
    ("python39", "pvlib__pvlib-python-1616", "3.9"),
)
REQUIRED_CHECKS = {
    *(f"v{version}.{operation}" for version in (1, 2)
      for operation in ("write_file", "read_file", "glob", "grep", "edit_file")),
    "ambiguous_edit", "repeated_edit_guard", "protected_write", "path_escape",
    "symlink_escape", "git_access", "syntax_feedback", "v2.mutation_feedback",
    "v2.read_feedback",
}


class ValidationDeadline(RuntimeError):
    pass


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextlib.contextmanager
def deadline(seconds):
    """Interrupt a blocked sandbox command so its existing cleanup kills its child."""
    def expired(signum, frame):
        raise ValidationDeadline(f"Runtime validation exceeded {seconds} seconds")
    old_handler = signal.signal(signal.SIGALRM, expired)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
        signal.signal(signal.SIGALRM, old_handler)


def check_receipt(receipt, version, worker_sha):
    if receipt.get("protocol") != "filesystem_tools_v1":
        raise ValueError("Sandbox probe did not return the filesystem-tools protocol receipt")
    checks = receipt.get("checks", {})
    missing = sorted(name for name in REQUIRED_CHECKS if checks.get(name) is not True)
    if missing:
        raise ValueError(f"Incomplete filesystem contract checks: {missing}")
    if receipt.get("python_version") != version:
        raise ValueError("AST feedback did not use the actual task Python interpreter")
    if receipt.get("worker_sha256") != worker_sha:
        raise ValueError("Filesystem receipt belongs to different worker code")
    if receipt.get("cleanup_complete") is not True:
        raise ValueError("Disposable probe cleanup was not confirmed")


def run_case(data, output, case, environment_timeout):
    name, instance, expected_minor = case
    folder = output / name
    folder.mkdir()
    task_path = data / "tasks" / instance / "task.json"
    task_sha = sha256(task_path)
    task = Task.from_file(task_path)
    if task.runtime.backend != "oci":
        raise ValueError(f"Expected a real OCI task runtime: {task_path}")
    worker_sha = sha256(ROOT / "src/screenshot_agent/worker.py")
    started = time.monotonic()
    result = {
        "case": name, "task_runtime_source": str(task_path), "task_sha256": task_sha,
        "expected_python_minor": expected_minor, "runtime": task.runtime.model_dump(),
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "passed": False,
    }
    sandbox = None
    print(json.dumps({"stage": "checking_runtime", "case": name, "task": instance}), flush=True)
    try:
        with deadline(environment_timeout):
            # This tiny fresh directory, never the prepared task repository, is mounted writable.
            with tempfile.TemporaryDirectory(prefix="sca-tool-matrix-", dir=folder) as temporary:
                repo = Path(temporary) / "repo"
                repo.mkdir()
                events = Events(folder / "events.jsonl")
                sandbox = create_sandbox(repo, events, [], 20, repo, task.runtime)
                info = sandbox.run([
                    task.runtime.python, "-c",
                    "import json,sys; print(json.dumps({'version':sys.version.split()[0],"
                    "'executable':sys.executable}))",
                ], timeout=20)
                if info["returncode"] != 0 or info["timed_out"]:
                    raise ValueError(f"Task interpreter failed: {info['output'][:1500]}")
                python = json.loads(info["output"].strip())
                result["python"] = python
                if ".".join(python["version"].split(".")[:2]) != expected_minor:
                    raise ValueError(f"Unexpected Python version for {name}: {python['version']}")
                probe = sandbox.probe()
                result["probe"] = probe
                check_receipt(probe.get("filesystem_probe", {}), python["version"], worker_sha)
                # The probe owns its own disposable fixture and must not modify even this empty repo.
                if list(repo.iterdir()):
                    raise ValueError("Sandbox probe modified its caller's repository")
                result["caller_repository_unchanged"] = True
        if sha256(task_path) != task_sha:
            raise ValueError("Prepared task metadata changed during the check")
        result["task_metadata_unchanged"] = True
        result["passed"] = True
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error))
    finally:
        if sandbox is not None:
            sandbox.cancel()
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_json(folder / "result.json", result)
    print(json.dumps({"stage": "runtime_finished", "case": name, "passed": result["passed"],
                      "elapsed_seconds": result["elapsed_seconds"], "error": result.get("error")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/generated/swebench_full_20261024")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-timeout", type=int, default=120)
    args = parser.parse_args()
    if not 30 <= args.environment_timeout <= 180:
        parser.error("Each environment timeout must be between 30 and 180 seconds")
    args.output = args.output.resolve()
    args.data = args.data.resolve()
    if args.output.exists():
        parser.error("Output must not exist; preserve previous evidence")
    args.output.mkdir(parents=True)
    report = {
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Filesystem-tool contracts only; no models, GPUs, repository tests or prepared-source writes",
        "required_checks": sorted(REQUIRED_CHECKS), "cases": [],
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [
            Path(__file__).resolve(), ROOT / "src/screenshot_agent/worker.py",
            ROOT / "src/screenshot_agent/sandbox.py", ROOT / "src/screenshot_agent/runtime.py",
            ROOT / "src/screenshot_agent/tool_probe.py",
        ]},
    }
    for case in CASES:
        report["cases"].append(run_case(args.data, args.output, case, args.environment_timeout))
        atomic_json(args.output / "report.json", report)
    report["passed"] = all(case["passed"] for case in report["cases"])
    report["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    atomic_json(args.output / "report.json", report)
    print(json.dumps({"report": str(args.output / "report.json"), "passed": report["passed"]}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
