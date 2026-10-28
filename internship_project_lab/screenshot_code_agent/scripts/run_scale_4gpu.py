#!/usr/bin/env python3
"""Explicit stages for pinned resources, Full environments and four-GPU code evaluation.

No action launches the next expensive stage implicitly. status/prepare/report are CPU-only.
Long stages own their child process groups and can resume without replacing completed results.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import queue
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections import Counter
from pathlib import Path

from screenshot_agent.cli import start_ticks
from screenshot_agent.config import RunConfig, Task
from screenshot_agent.control import Events, atomic_json
from screenshot_agent.report import make_report
from screenshot_agent.runner import run_task
from screenshot_agent.runtime import create_sandbox
from screenshot_agent.scale_data import (
    audit_one,
    finalize,
    immutable_json,
    prepare,
    read,
    sha256,
    tree_hash,
)
from screenshot_agent.tool_probe import validate_receipt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/scale_4gpu_20261024"
DEFAULT_DATA = ROOT / "datasets/generated/swebench_full_20261024"
MODEL = ROOT / ".cache/models/Qwen3-Coder-30B-A3B-Instruct"
BASELINE = Path("/data/zjr/models/Qwen3-VL-8B-Instruct")


def jsonlines(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_gpus(value):
    values = value.split(",")
    if len(values) != 4 or len(set(values)) != 4 or any(not x.isdecimal() for x in values):
        raise ValueError("--gpus requires exactly four distinct physical GPU indices, e.g. 4,5,6,7")
    return values


def server_command(root, model, gpus, port, pin, gpu_memory_utilization=0.80,
                   context_tokens=16384, max_num_seqs=2):
    parse_gpus(gpus)
    if not 1024 <= port <= 65535:
        raise ValueError("Invalid server port")
    if model == "coder":
        return [str(root / ".coder-env/bin/python"), "-u", "-m", "vllm.entrypoints.openai.api_server",
                "--model", str(root / ".cache/models/Qwen3-Coder-30B-A3B-Instruct"),
                "--served-model-name", "sca-coder", "--host", "127.0.0.1", "--port", str(port),
                "--tensor-parallel-size", "4", "--dtype", "bfloat16", "--max-model-len", str(context_tokens),
                "--max-num-seqs", str(max_num_seqs), "--gpu-memory-utilization", str(gpu_memory_utilization), "--enforce-eager",
                "--disable-custom-all-reduce", "--enable-auto-tool-choice",
                "--tool-call-parser", pin["tool_call_parser"], "--seed", "0"]
    return [str(root / ".model-env/bin/python"), "-u", str(root / "scripts/local_model_api.py"),
            "--model-path", str(BASELINE), "--port", str(port), "--max-context", "16384", "--max-output", "4096"]


def signal_trace_command(command, log_path):
    """Trace only model-service termination signals, preserving its PID and exit status.

    strace daemonizes into a separate session; the model remains our direct child.
    The tracer exits when its tracees exit and cannot swallow our group cleanup.
    No syscall arguments, HTTP bodies or environment values are logged.
    """
    tracer = shutil.which("strace")
    if tracer is None:
        raise ValueError("--trace-server-signals requires strace; no model service was started")
    return [tracer, "-DDD", "-f", "-qq", "-ttt", "--seccomp-bpf", "-e", "trace=none",
            "-e", "signal=SIGTERM,SIGINT,SIGHUP,SIGKILL", "-o", str(log_path), "--", *command]


def signal_trace_summary(service):
    """Read bounded signal metadata only; a missing sender remains explicitly unknown."""
    path = service.get("signal_trace_log")
    summary = {"enabled": bool(path), "log": path, "status": "disabled" if not path else "unavailable",
               "last_sigterm": None}
    if not path:
        return summary
    try:
        with Path(path).open("rb") as trace:
            trace.seek(0, os.SEEK_END)
            size = trace.tell()
            trace.seek(max(0, size - 65536))
            lines = trace.read(65536).decode("utf-8", errors="replace").splitlines()
    except OSError:
        return summary
    summary["status"] = "no_sigterm_observed"
    chosen = None
    for line in reversed(lines):
        if not re.search(r"--- SIGTERM\s+\{.*\bsi_signo=SIGTERM\b", line):
            continue
        sender_pid = re.search(r"\bsi_pid=(\d+)\b", line)
        sender_uid = re.search(r"\bsi_uid=(\d+)\b", line)
        code = re.search(r"\bsi_code=([A-Z0-9_]+)\b", line)
        recipient = re.match(r"^(?:\[pid\s+(\d+)\]|(\d+))\s+", line)
        recipient_pid = int(recipient.group(1) or recipient.group(2)) if recipient else None
        direct = recipient_pid is not None and recipient_pid == service.get("pid")
        observed = {
            "line": line[:2048], "si_code": code.group(1) if code else None,
            "sender_pid": int(sender_pid.group(1)) if sender_pid else None,
            "sender_uid": int(sender_uid.group(1)) if sender_uid else None,
            "recipient_pid": recipient_pid,
            "scope": "model_process" if direct else "observed_tracee" if recipient else "unattributed",
        }
        if chosen is None or direct:
            chosen = observed
        if direct:
            break  # Prefer the model PID; a later worker signal is not its proven cause of exit.
    if chosen is not None:
        summary["last_sigterm"] = chosen
        summary["status"] = ("sender_recorded" if chosen["sender_pid"] is not None
                             and chosen["sender_uid"] is not None else "sender_unknown")
    return summary


def gpu_policy(args):
    return {"allow_shared_gpus": args.allow_shared_gpus,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "headroom_mib": args.gpu_headroom_mib}


def assess_gpu_memory(text, gpus, policy):
    """Check every device's free memory, not the sum across tensor-parallel ranks."""
    fraction, headroom = policy["gpu_memory_utilization"], policy["headroom_mib"]
    if not 0.5 <= fraction <= 0.95 or headroom < 0:
        raise ValueError("Invalid GPU memory budget")
    rows = []
    for line in text.splitlines():
        index, total, used, free = [int(x.strip()) for x in line.split(",")]
        if total <= 0 or min(used, free) < 0 or used + free > total:
            raise ValueError("Invalid GPU memory telemetry")
        required = math.ceil(total * fraction) + headroom
        permitted = free >= required and (policy["allow_shared_gpus"] or used <= 1000)
        rows.append({"gpu": str(index), "total_mib": total, "used_mib": used, "free_mib": free,
                     "required_free_mib": required, "ready": permitted})
    if len(rows) != len(gpus) or {r["gpu"] for r in rows} != set(gpus):
        raise ValueError("GPU telemetry does not match the requested devices")
    return {"ready": all(r["ready"] for r in rows), "devices": rows, "policy": policy}


def wait_for_gpu_capacity(c, args, gpus, arm):
    stable = 0
    while True:
        raw = subprocess.check_output([
            "nvidia-smi", "--id=" + ",".join(gpus),
            "--query-gpu=index,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"], text=True)
        sample = assess_gpu_memory(raw, gpus, gpu_policy(args))
        stable = stable + 1 if sample["ready"] else 0
        # Background runs wait for two consecutive samples before attempting model loading.
        ready = sample["ready"] and (not args.wait_for_gpus or stable >= 2)
        c.update(stage="gpu_memory_ready" if ready else "waiting_for_gpu_memory", model_arm=arm,
                 gpu_memory=sample, stable_gpu_samples=stable)
        if ready:
            with (c.output / "gpu_checks.jsonl").open("a") as output:
                output.write(json.dumps({"time": time.time(), "arm": arm, **sample}) + "\n")
            return
        if not args.wait_for_gpus:
            raise ValueError("Insufficient free GPU memory or sharing disabled; inspect gpu_memory in status.json")
        time.sleep(15)


def alive(pid, ticks):
    return bool(pid and ticks and start_ticks(pid) == ticks)


def interrupted_result(result):
    """Transport interruptions are retryable; completed model mistakes are not."""
    return result.get("status") == "cancelled" or (
        result.get("exit_reason") == "model_error"
        and result.get("agent_error_type") == "APIConnectionError"
    )


def terminate(proc):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=15)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=15)


class Controller:
    def __init__(self, output, action):
        self.output, self.action = output, action
        self.children = []
        self.resource_leases = []
        # Source snapshots share the original project's mutable model/environment
        # paths, so storage switching and every new controller must share a lock.
        anchor = ROOT
        if (ROOT / "source_state.json").is_file():
            anchor = Path(read(ROOT / "source_state.json").get("original_project", str(ROOT)))
        self.storage_lock = (anchor / ".storage-migration.lock").open("a")
        self.lock = None
        try:
            try:
                fcntl.flock(self.storage_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Storage migration is active; wait for it to finish before starting evaluation") from error
            output.mkdir(parents=True, exist_ok=True)
            (output / "logs").mkdir(exist_ok=True)
            self.lock = (output / "controller.lock").open("a")
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.state = {"action": action, "stage": "starting", "pid": os.getpid(),
                          "start_ticks": start_ticks(os.getpid()), "started_at": time.time()}
            self.last_beat = 0
            self.update()
        except BaseException:
            if self.lock is not None:
                self.lock.close()
            self.storage_lock.close()
            raise

    def update(self, **kwargs):
        self.state.update(kwargs, updated_at=time.time())
        visible = [item for item in self.children if item["proc"].poll() is None or item in self.children[-2:]]
        self.state["children"] = [{k: item[k] for k in ("label", "pid", "start_ticks", "log")}
                                  | {"alive": item["proc"].poll() is None,
                                     "returncode": item["proc"].returncode} for item in visible]
        self.state["children_completed"] = sum(item["proc"].poll() is not None for item in self.children)
        atomic_json(self.output / "status.json", self.state)
        if time.monotonic() - self.last_beat >= 15:
            print(json.dumps(self.state), flush=True)
            self.last_beat = time.monotonic()

    def launch(self, command, label, env=None, cwd=None):
        log = self.output / "logs" / f"{label}.log"
        # Append on retry: old diagnostics are not silently overwritten.
        with log.open("ab") as f:
            f.write(f"\n=== attempt {time.time()} ===\n".encode())
            proc = subprocess.Popen(command, cwd=cwd or ROOT, env=env, stdout=f, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        entry = {"proc": proc, "label": label, "pid": proc.pid, "start_ticks": start_ticks(proc.pid), "log": str(log)}
        self.children.append(entry)
        self.update()
        return entry

    def wait(self, entry, timeout=None):
        started = time.monotonic()
        while entry["proc"].poll() is None:
            if timeout and time.monotonic() - started > timeout:
                terminate(entry["proc"])
                raise RuntimeError(f"{entry['label']} exceeded {timeout}s; see {entry['log']}")
            self.update()
            time.sleep(1)
        if entry["proc"].returncode:
            raise RuntimeError(f"{entry['label']} exited {entry['proc'].returncode}; see {entry['log']}")

    def close(self):
        try:
            for entry in reversed(self.children):
                terminate(entry["proc"])
            self.update()
        finally:
            for lease in self.resource_leases:
                lease.close()
            self.lock.close()
            self.storage_lock.close()


def inspect_status(output, data):
    state = read(output / "status.json") if (output / "status.json").exists() else {"stage": "not_started"}
    state["controller_alive"] = alive(state.get("pid"), state.get("start_ticks"))
    for child in state.get("children", []):
        child["alive"] = alive(child["pid"], child["start_ticks"])
    if (data / "selection.json").exists():
        selection = read(data / "selection.json")
        splits = {row["instance_id"]: row["sca_split"] for row in read(data / "selected_records.json")}
        counts = Counter()
        by_split = {split: Counter() for split in ("dev", "test")}
        for name in selection["selected_ids"]:
            p = data / "tasks" / name / "audit.json"
            status = read(p)["status"] if p.exists() else "pending"
            counts[status] += 1
            by_split[splits[name]][status] += 1
        state["preflight"] = {"targets": selection["targets"], "candidates": len(selection["selected_ids"]),
                              "active_split": state.get("preflight_split", "all"), **counts,
                              "by_split": {split: dict(values) for split, values in by_split.items()}}
    state["model_download_complete"] = (MODEL / "download_complete.json").exists()
    state["coder_environment_ready"] = (ROOT / ".coder-env/setup_complete.json").exists()
    evaluations = {}
    for split in ("dev", "test"):
        for arm in ("baseline", "coder"):
            folder = output / split / arm
            if folder.exists():
                records = [read(p) for p in folder.glob("runs/*/result.json")]
                interrupted = [r for r in records if interrupted_result(r)]
                completed = [r for r in records if not interrupted_result(r)]
                evaluations[f"{split}/{arm}"] = {
                    "completed": len(completed),
                    "resolved": sum(r.get("resolved", False) for r in completed),
                    "interrupted": len(interrupted),
                    "interrupted_task_ids": sorted(r["task_id"] for r in interrupted),
                }
    state["evaluations"] = evaluations
    return state


def resources(c, args):
    for stage, python, action in [("metadata", ROOT / ".venv/bin/python", "metadata"),
                                  ("coder_environment", ROOT / ".venv/bin/python", "setup"),
                                  ("model_download", ROOT / ".model-env/bin/python", "model")]:
        c.update(stage=stage)
        c.wait(c.launch([str(python), "-u", str(ROOT / "scripts/scale_resources.py"), action], stage))
        if stage == "metadata":
            prepare(ROOT, args.data, dev_count=args.dev_count, test_count=args.test_count)
            c.update(stage="registry_probe")
            c.wait(c.launch([str(python), "-u", str(ROOT / "scripts/scale_resources.py"), "registry-probe",
                             "--selection", str(args.data / "selected_records.json")], "registry_probe"))


def verify_resource_receipts():
    """Resume without redownloading weights or re-installing a working environment."""
    pins = read(ROOT / "configs/scale_4gpu_resources.json")
    model = read(MODEL / "download_complete.json")
    setup = read(ROOT / ".coder-env/setup_complete.json")
    if model["revision"] != pins["coder"]["revision"] or setup["vllm_version"] != pins["vllm_version"]:
        raise ValueError("Resource receipts differ from pins; run resources before resuming")
    if not model.get("files") or not (ROOT / ".coder-env/bin/python").is_file():
        raise ValueError("Resource receipts are incomplete; run resources before resuming")
    for name, item in model["files"].items():
        path = MODEL / name
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Downloaded model file is missing or changed: {name}")
    return {"model_revision": model["revision"], "vllm_version": setup["vllm_version"],
            "model_receipt_sha256": sha256(MODEL / "download_complete.json"),
            "environment_receipt_sha256": sha256(ROOT / ".coder-env/setup_complete.json")}


def preflight_is_ready(report, split):
    if "ready_by_split" in report:
        if split == "all":
            return all(report["ready_by_split"].get(s, False) for s in ("dev", "test"))
        return report["ready_by_split"].get(split, False)
    return bool(report.get("ready"))  # Compatibility with completed pre-split reports.


def preflight(c, args):
    requested = ("dev", "test") if args.preflight_split == "all" else (args.preflight_split,)
    if (args.data / "evaluation_seal.json").exists() or any(
            (args.data / f"evaluation_seal_{split}.json").exists()
            or (args.output / split / "plan.json").exists() for split in requested):
        raise ValueError("An evaluation plan already uses this split; preserve it and create a new version")
    selected = read(args.data / "selected_records.json")
    selected = [row for row in selected if row["sca_split"] in requested]
    selection = read(args.data / "selection.json")
    for name, checksum in selection["upstream_files_sha256"].items():
        if sha256(name) != checksum:
            raise ValueError(f"Pinned upstream task definition changed: {name}")
    if args.limit:
        selected = selected[:args.limit]
    pending = []
    for row in selected:
        p = args.data / "tasks" / row["instance_id"] / "audit.json"
        if p.exists() and (read(p)["status"] == "validated" or not args.retry_failed):
            continue
        pending.append(row)
    active = []
    while pending or active:
        while pending and len(active) < args.workers:
            row = pending.pop(0)
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "audit-one", "--output", str(args.output),
                       "--data", str(args.data), "--instance", row["instance_id"], "--test-timeout", str(args.test_timeout)]
            if args.retry_failed:
                command.append("--retry-failed")
            entry = c.launch(command, "preflight_" + row["instance_id"])
            active.append((row, entry))
        for row, entry in list(active):
            if entry["proc"].poll() is not None:
                if entry["proc"].returncode or not (args.data / "tasks" / row["instance_id"] / "audit.json").exists():
                    raise RuntimeError(f"Preflight worker crashed; retry preserves old attempts: {entry['log']}")
                active.remove((row, entry))
        c.update(stage="preflight", preflight_split=args.preflight_split, pending=len(pending), running=len(active))
        if active:
            time.sleep(1)
    report = finalize(args.data, split=None if args.preflight_split == "all" else args.preflight_split)
    ready = preflight_is_ready(report, args.preflight_split)
    c.update(stage="preflight_ready" if ready else "preflight_incomplete", preflight_split=args.preflight_split,
             preflight={k: v for k, v in report.items() if k != "audits"})
    if not ready and not args.limit:
        raise RuntimeError(f"Not enough validated {args.preflight_split} tasks or incomplete audits; "
                           "inspect preflight_report.json. "
                           "No reduced evaluation is launched implicitly.")


def frozen_runtime(output):
    runtime = output / "runtime"
    if not runtime.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts/freeze_v5_source.py"), "--output", str(runtime)],
                       cwd=ROOT, check=True)
        (runtime / ".coder-env").symlink_to(ROOT / ".coder-env", target_is_directory=True)
    snapshot = read(runtime / "source_state.json")
    for name, checksum in snapshot["sha256"].items():
        if sha256(runtime / name) != checksum:
            raise ValueError(f"Frozen runtime changed: {name}")
    return runtime


def plan_evaluation(args, progress=None):
    report = read(args.data / "preflight_report.json")
    if not preflight_is_ready(report, args.split):
        raise ValueError(f"Preflight must satisfy the full declared {args.split} target before evaluation")
    runtime = frozen_runtime(args.output)
    manifest = args.data / f"{args.split}.jsonl"
    rows = jsonlines(manifest)
    selection = read(args.data / "selection.json")
    target = selection["targets"][args.split]
    ids = [row["task_id"] for row in rows]
    if (len(rows) != target or len(set(ids)) != target
            or any(row["split"] != args.split for row in rows)
            or not set(ids).issubset(selection["selected_ids"])):
        raise ValueError(f"Manifest must contain exactly {target} distinct selected {args.split} tasks")
    inputs = {str(manifest): sha256(manifest), str(args.data / "selection.json"): sha256(args.data / "selection.json")}
    validation_started = time.monotonic()
    for index, row in enumerate(rows):
        if progress:
            progress(stage="checking_frozen_inputs", validation_completed=index, validation_total=len(rows),
                     validation_task=row["task_id"], validation_seconds=round(time.monotonic() - validation_started, 1))
        folder = Path(row["task_file"]).parent
        audit = read(folder / "audit.json")
        if audit["status"] != "validated" or sha256(folder / "audit.json") != row["audit_sha256"]:
            raise ValueError("Task preflight changed")
        for name, key in (("task.json", "task_sha256"), ("benchmark.json", "benchmark_sha256"),
                          ("reference.patch", "reference_sha256")):
            if sha256(folder / name) != audit[key]:
                raise ValueError(f"Prepared input changed: {folder / name}")
            inputs[str(folder / name)] = audit[key]
        if tree_hash(folder / "repo") != audit["repo_sha256"]:
            raise ValueError(f"Prepared repository changed: {folder}")
    if progress:
        progress(validation_completed=len(rows), validation_task=None,
                 validation_seconds=round(time.monotonic() - validation_started, 1))
    common = read(runtime / "configs/scale-4gpu.json")
    common["agent"]["mode"] = getattr(args, "agent_mode", "v6")
    RunConfig.model_validate(common)
    model_receipt = read(MODEL / "download_complete.json")
    setup_receipt = read(ROOT / ".coder-env/setup_complete.json")
    pins = read(runtime / "configs/scale_4gpu_resources.json")
    if model_receipt["revision"] != pins["coder"]["revision"] or setup_receipt["vllm_version"] != pins["vllm_version"]:
        raise ValueError("Run resources first; model/runtime receipts differ from pins")
    plan = {"split": args.split, "tasks": rows, "inputs_sha256": inputs,
            "runtime": str(runtime), "runtime_sha256": sha256(runtime / "source_state.json"),
            "common_config": common, "service_context": 16384, "gpu_policy": gpu_policy(args),
            "model_receipt_sha256": sha256(MODEL / "download_complete.json"),
            "coder_setup_sha256": sha256(ROOT / ".coder-env/setup_complete.json"),
            "resources": read(runtime / "configs/scale_4gpu_resources.json"),
            "input_mode": "original_issue_text", "reader": "oracle (no images; no vision calls)",
            "tool_preflight_required": True,
            "interpretation": f"Paired model/service comparison on a fixed {common['agent']['mode']} harness; equal numeric budgets. "
                              "36 calls/4096 output differs from the old 12-task run. "
                              "Official-image Bubblewrap adaptation, not an official leaderboard submission."}
    folder = args.output / args.split
    folder.mkdir(exist_ok=True)
    immutable_json(folder / "plan.json", plan)
    immutable_json(args.data / f"evaluation_seal_{args.split}.json", {
        "split": args.split, "records": target,
        "selection_sha256": sha256(args.data / "selection.json"),
        "manifest_sha256": sha256(manifest)})
    return plan


def worker_python(runtime):
    return runtime.python if runtime.backend in {"oci", "docker"} else "/usr/bin/python3"


def validate_task_tool_receipt(checked, task, task_sha, worker_sha):
    """Bind a complete contract to the selected task, runtime and frozen worker."""
    if not isinstance(checked, dict):
        raise ValueError("Tool preflight receipt must be an object")
    if checked.get("task_id") != task.task_id or checked.get("task_sha256") != task_sha:
        raise ValueError("Tool preflight task identity differs from the evaluation plan")
    if checked.get("runtime") != task.runtime.model_dump():
        raise ValueError("Tool preflight used a different task runtime")
    if checked.get("caller_repository_unchanged") is not True or checked.get("probe_workspace_cleaned") is not True:
        raise ValueError("Tool preflight workspace preservation and cleanup were not confirmed")
    python = checked.get("python", {})
    if (not isinstance(python, dict) or python.get("executable") != worker_python(task.runtime)
            or not isinstance(python.get("version"), str) or not re.fullmatch(r"\d+\.\d+\.\d+", python["version"])):
        raise ValueError("Tool preflight did not record the actual task worker interpreter")
    probe = checked.get("probe", {})
    if not isinstance(probe, dict) or probe.get("returncode") != 0 or probe.get("timed_out") is not False:
        raise ValueError("Tool preflight sandbox probe did not succeed")
    validate_receipt(probe.get("filesystem_probe"), python_version=python["version"], worker_sha256=worker_sha)


def probe_task_tools(task_path, receipt_path):
    """Check tools in a tiny disposable repository with the task's actual runtime."""
    task = Task.from_file(task_path)
    task_sha = sha256(task_path)
    worker_sha = sha256(ROOT / "src/screenshot_agent/worker.py")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    if receipt_path.exists():
        raise ValueError("Refusing to overwrite a tool preflight receipt")
    with tempfile.TemporaryDirectory(prefix="sca-tools-") as temporary:
        base = Path(temporary)
        repo = base / "repo"
        repo.mkdir()
        events = Events(receipt_path.with_suffix(".events.jsonl"))
        sandbox = create_sandbox(repo, events, [], 30, repo, task.runtime)
        try:
            info = sandbox.run([worker_python(task.runtime), "-c",
                                "import json,sys; print(json.dumps({'version':sys.version.split()[0],"
                                "'executable':sys.executable}))"], timeout=20)
            if info["returncode"] != 0 or info["timed_out"]:
                raise RuntimeError(f"Task worker interpreter failed: {info['output'][:1500]}")
            python = json.loads(info["output"].strip())
            checked = sandbox.probe()
            if list(repo.iterdir()):
                raise RuntimeError("Tool preflight modified its caller repository")
        finally:
            sandbox.cancel()
    if sha256(task_path) != task_sha:
        raise ValueError("Task metadata changed during tool preflight")
    receipt = {"task_id": task.task_id, "task_sha256": task_sha, "runtime": task.runtime.model_dump(),
               "python": python, "probe": checked, "caller_repository_unchanged": True,
               "probe_workspace_cleaned": not base.exists(), "completed_at": time.time()}
    validate_task_tool_receipt(receipt, task, task_sha, worker_sha)
    atomic_json(receipt_path, receipt)


def preflight_tools(c, args, plan):
    """All distinct task runtimes must pass the frozen worker contract before GPU startup."""
    runtime = Path(plan["runtime"])
    if not (runtime / "src/screenshot_agent/tool_probe.py").is_file():
        raise ValueError("This output freezes the old tools. Use a new output directory for the repaired experiment")
    if sha256(runtime / "source_state.json") != plan["runtime_sha256"]:
        raise ValueError("Frozen runtime receipt changed after the evaluation plan")
    worker_sha = read(runtime / "source_state.json")["sha256"].get("src/screenshot_agent/worker.py")
    if not worker_sha or worker_sha != sha256(runtime / "src/screenshot_agent/worker.py"):
        raise ValueError("Frozen filesystem worker differs from its source receipt")
    if not plan["tasks"]:
        raise ValueError("Tool preflight requires a nonempty evaluation task list")
    groups = {}
    for row in plan["tasks"]:
        task_path = Path(row["task_file"])
        task = Task.from_file(task_path)
        if task.task_id != row["task_id"]:
            raise ValueError("Tool preflight task identity differs from the manifest")
        if sha256(task_path) != plan["inputs_sha256"].get(str(task_path)):
            raise ValueError("Task metadata differs from the evaluation plan")
        identity = json.dumps(task.runtime.model_dump(), sort_keys=True)
        groups.setdefault(identity, []).append(row)
    attempt = c.output / "tool_preflight" / str(time.time_ns())
    attempt.mkdir(parents=True)
    env = {**os.environ, "PYTHONPATH": str(runtime / "src")}
    receipts = []
    for index, rows in enumerate(groups.values()):
        representative = rows[0]
        receipt = attempt / f"{representative['task_id']}.json"
        c.update(stage="tool_preflight", tool_preflight_completed=index, tool_preflight_total=len(groups),
                 tool_preflight_task=representative["task_id"], tool_preflight_covered_tasks=sum(
                     len(item["task_ids"]) for item in receipts))
        command = [sys.executable, "-u", str(runtime / "scripts/run_scale_4gpu.py"), "probe-one",
                   "--task", representative["task_file"], "--receipt", str(receipt)]
        entry = c.launch(command, "tool_preflight_" + representative["task_id"], env=env, cwd=runtime)
        c.wait(entry, timeout=300)
        checked = read(receipt)
        task_path = Path(representative["task_file"])
        task = Task.from_file(task_path)
        task_sha = plan["inputs_sha256"][str(task_path)]
        if sha256(task_path) != task_sha or task.task_id != representative["task_id"]:
            raise ValueError("Task metadata changed during tool preflight")
        validate_task_tool_receipt(checked, task, task_sha, worker_sha)
        receipts.append({"task_ids": [row["task_id"] for row in rows], "receipt": str(receipt),
                         "receipt_sha256": sha256(receipt)})
    report = attempt / "report.json"
    atomic_json(report, {"complete": True, "runtime_sha256": plan["runtime_sha256"],
                        "task_count": len(plan["tasks"]), "receipts": receipts})
    c.update(stage="tools_verified", tool_preflight_completed=len(groups),
             tool_preflight_covered_tasks=len(plan["tasks"]), tool_preflight_report=str(report))


def startup_telemetry(pid):
    """Read bounded procfs fields without exposing command lines or environment keys."""
    sample = {"observed_at": time.time()}
    try:
        sample["host_memory_kib"] = {
            fields[0].rstrip(":"): int(fields[1])
            for line in Path("/proc/meminfo").read_text().splitlines()
            if (fields := line.split())[0] in {"MemAvailable:", "SwapFree:"}
        }
        sample["host_pressure"] = {name: Path(f"/proc/pressure/{name}").read_text().strip()
                                   for name in ("memory", "io")}
    except (OSError, ValueError):
        sample["host_telemetry_unavailable"] = True
    if pid:
        try:
            sample["server_status"] = {
                key: value.strip() for line in Path(f"/proc/{pid}/status").read_text().splitlines()
                if (key := line.partition(":")[0]) in {"State", "VmRSS", "VmSwap"}
                for value in [line.partition(":")[2]]
            }
            sample["server_io_bytes"] = {
                fields[0].rstrip(":"): int(fields[1])
                for line in Path(f"/proc/{pid}/io").read_text().splitlines()
                if (fields := line.split())[0] in {"read_bytes:", "rchar:"}
            }
        except (OSError, ValueError):
            sample["server_telemetry_unavailable"] = True
    return sample


def startup_chat(c, entry, opener, request, model, step, timeout=600):
    """Send one startup probe while the controller monitors its bounded wait.

    Do not retry a timed-out POST: the server could still be generating its reply.
    The daemon request thread is released when start_service cleans up the server.
    """
    replies = queue.Queue(maxsize=1)
    started = time.monotonic()
    last_sample = -15

    def send_once():
        try:
            with opener.open(request, timeout=timeout) as response:
                result = json.load(response)
            replies.put((True, result))
        except BaseException as error:
            replies.put((False, error))

    def timed_out():
        return TimeoutError(f"Startup protocol smoke {step} exceeded {timeout:g}s without a reply; "
                            f"no retry was queued; inspect {entry['log']}")

    c.update(stage="protocol_smoke", model=model, protocol_smoke_step=step,
             protocol_smoke_seconds=0, protocol_smoke_timeout_seconds=timeout)
    threading.Thread(target=send_once, name="startup-protocol-request", daemon=True).start()
    while True:
        returncode = entry["proc"].poll()
        if returncode is not None:
            raise RuntimeError(f"Model server exited during startup protocol smoke {step} "
                               f"(returncode={returncode}); inspect {entry['log']}")
        elapsed = time.monotonic() - started
        telemetry = {}
        if elapsed - last_sample >= 15:
            telemetry["startup_progress"] = startup_telemetry(entry.get("pid"))
            last_sample = elapsed
        c.update(stage="protocol_smoke", model=model, protocol_smoke_step=step,
                 protocol_smoke_seconds=round(elapsed, 1), protocol_smoke_timeout_seconds=timeout, **telemetry)
        remaining = timeout - elapsed
        if remaining <= 0:
            raise timed_out()
        try:
            succeeded, result = replies.get(timeout=min(1, remaining))
        except queue.Empty:
            continue
        if not succeeded:
            if isinstance(result, TimeoutError) or isinstance(getattr(result, "reason", None), TimeoutError):
                raise timed_out() from result
            raise result
        returncode = entry["proc"].poll()
        if returncode is not None:
            raise RuntimeError(f"Model server exited during startup protocol smoke {step} "
                               f"(returncode={returncode}); inspect {entry['log']}")
        return result


def ready_server(c, entry, port, key, model, probe_timeout=600):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.monotonic()
    while time.monotonic() - started < 1200:
        if entry["proc"].poll() is not None:
            raise RuntimeError(f"Model server exited; inspect {entry['log']}")
        try:
            request = urllib.request.Request(f"http://127.0.0.1:{port}/health",
                                             headers={"Authorization": "Bearer " + key})
            with opener.open(request, timeout=2) as response:
                if response.status == 200:
                    break
        except (OSError, ValueError):
            pass
        c.update(stage="model_loading", model=model, startup_seconds=round(time.monotonic() - started))
        time.sleep(1)
    else:
        raise RuntimeError("Model server did not become ready within 1200 seconds")
    # Real tool round trip before spending hundreds of task runs on a parser mismatch.
    tool = {"type": "function", "function": {"name": "echo_probe", "description": "Echo the supplied text.",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}
    payload = {"model": model, "messages": [{"role": "user", "content": "Call echo_probe with text hello."}],
               "tools": [tool], "temperature": 0, "max_tokens": 256}
    def chat(value, step):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(value).encode(),
                                     {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        return startup_chat(c, entry, opener, req, model, step, timeout=probe_timeout)
    first = chat(payload, "tool_call")
    message = first["choices"][0]["message"]
    calls = message.get("tool_calls", [])
    if len(calls) != 1 or calls[0]["function"]["name"] != "echo_probe":
        raise ValueError("Tool protocol smoke failed: expected echo_probe")
    if json.loads(calls[0]["function"]["arguments"]) != {"text": "hello"}:
        raise ValueError("Tool argument decoding failed")
    payload["messages"] += [message, {"role": "tool", "tool_call_id": calls[0]["id"], "content": "hello"},
                            {"role": "user", "content": "The tool succeeded. Reply with hello; do not call tools again."}]
    second = chat(payload, "tool_response")
    if (second["choices"][0]["finish_reason"] == "length"
            or second["choices"][0]["message"].get("tool_calls")
            or not second["choices"][0]["message"].get("content")):
        raise ValueError("Tool-response round trip was truncated")
    atomic_json(c.output / "logs" / f"{model}_protocol_smoke.json", {"first": first, "second": second})
    c.update(stage="protocol_smoke_passed", model=model)


def record_startup_failure(c, args, arm, service, error):
    """Distinguish an observed server failure from our subsequent cleanup signal."""
    returncode = service["proc"].poll()
    receipt = {
        "observed_at": time.time(), "split": args.split, "arm": arm,
        "pid": service["pid"], "start_ticks": service["start_ticks"], "log": service["log"],
        "phase": c.state.get("stage"), "protocol_smoke_step": c.state.get("protocol_smoke_step"),
        "protocol_smoke_seconds": c.state.get("protocol_smoke_seconds"),
        "protocol_smoke_timeout_seconds": c.state.get("protocol_smoke_timeout_seconds"),
        "error_type": type(error).__name__, "error": str(error),
        "server_returncode_before_cleanup": returncode, "server_alive_before_cleanup": returncode is None,
        "controller_cleanup_requested": returncode is None,
    }
    receipt["startup_progress"] = startup_telemetry(service["pid"])
    receipt["signal_trace"] = signal_trace_summary(service)
    path = c.output / "logs" / f"{args.split}_{arm}_startup_failure_{time.time_ns()}.json"
    atomic_json(path, receipt)
    c.update(model_startup_failure={**receipt, "receipt": str(path)})
    return receipt


def start_service(c, args, runtime, arm, pins):
    automatic = args.gpus == "auto"
    if automatic and arm != "coder":
        raise ValueError("Automatic four-GPU selection is supported for the Coder service")
    gpus = None if automatic else parse_gpus(args.gpus)
    if not 1024 <= args.port <= 65535:
        raise ValueError("Invalid server port")
    model = "sca-coder" if arm == "coder" else "local-qwen3-vl-8b"
    trace_path = None
    if getattr(args, "trace_server_signals", False):
        trace_path = c.output / "logs" / f"{args.split}_{arm}_signals_{time.time_ns()}.log"
        if shutil.which("strace") is None:
            raise ValueError("--trace-server-signals requires strace; no model service was started")
    if arm == "coder":
        receipt = read(MODEL / "download_complete.json")
        if receipt["revision"] != pins["coder"]["revision"]:
            raise ValueError("Coder weights differ from the pinned revision")
        for name, item in receipt["files"].items():
            if (MODEL / name).stat().st_size != item["bytes"]:
                raise ValueError("Coder weight file size changed")
        setup = read(ROOT / ".coder-env/setup_complete.json")
        if setup["vllm_version"] != pins["vllm_version"]:
            raise ValueError("Unexpected vLLM version")
    lease = service = None
    try:
        if automatic:
            from screenshot_agent.gpu_queue import QueuePolicy, wait_for_idle
            policy = QueuePolicy(candidates=tuple(args.gpu_candidates.split(",")),
                poll_seconds=args.gpu_poll_seconds,stable_samples=args.gpu_stable_samples,
                memory_fraction=args.gpu_memory_utilization,headroom_mib=args.gpu_headroom_mib,
                allow_shared_gpus=getattr(args,'allow_shared_gpus',False))
            lease = wait_for_idle(c,policy)
            c.resource_leases.append(lease)
            gpus = lease.gpus
        used = gpus if arm == "coder" else gpus[:1]
        if not automatic:
            wait_for_gpu_capacity(c, args, used, arm)
        command = server_command(runtime, arm, ",".join(gpus), args.port, pins, args.gpu_memory_utilization,
                                 getattr(args, "service_context_tokens", 16384), getattr(args, "max_num_seqs", 2))
        if trace_path:
            command = signal_trace_command(command, trace_path)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", args.port))
        key = secrets.token_urlsafe(32)
        env = os.environ.copy()
        env.update(PYTHONPATH=str(runtime / "src"), CUDA_VISIBLE_DEVICES=",".join(used),
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_API_KEY=key, SCA_LOCAL_KEY=key,
                   SCA_CODE_API_KEY=key, NO_PROXY="127.0.0.1,localhost,::1", no_proxy="127.0.0.1,localhost,::1",
                   OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false", PYTHONFAULTHANDLER="1",
                   XDG_CACHE_HOME=os.environ.get("XDG_CACHE_HOME", str(ROOT / ".cache")),
                   VLLM_CACHE_ROOT=os.environ.get("VLLM_CACHE_ROOT", str(ROOT / ".cache/vllm")))
        service = c.launch(command, f"{args.split}_{arm}_server", env=env)
        if trace_path:
            service["signal_trace_log"] = str(trace_path)
        c.update(model_signal_trace={"enabled": bool(trace_path), "log": str(trace_path) if trace_path else None,
                                     "model_pid": service["pid"], "signals_only": True})
        try:
            ready_server(c, service, args.port, key, model)
        except BaseException as error:
            record_startup_failure(c, args, arm, service, error)
            raise
    except BaseException:
        if service is not None:
            terminate(service["proc"])
        if lease is not None:
            lease.close()
        raise
    return service, env


def smoke(c, args):
    runtime = frozen_runtime(args.output)
    service, _ = start_service(c, args, runtime, "coder", read(runtime / "configs/scale_4gpu_resources.json"))
    terminate(service["proc"])
    c.update(stage="coder_protocol_verified")


def record_server_exit(c, args, arm, service, active, returncode):
    """Persist the observed server exit before terminating any remaining workers."""
    receipt = {
        "observed_at": time.time(), "split": args.split, "arm": arm,
        "pid": service["pid"], "start_ticks": service["start_ticks"],
        "returncode": returncode, "signal": -returncode if returncode < 0 else None,
        "log": service["log"], "active_task_ids": [row["task_id"] for row, _, _ in active],
        "signal_trace": signal_trace_summary(service),
    }
    try:
        receipt["host_memory_kib"] = {
            fields[0].rstrip(":"): int(fields[1])
            for line in Path("/proc/meminfo").read_text().splitlines()
            if (fields := line.split())[0] in {"MemAvailable:", "SwapFree:"}
        }
    except (OSError, ValueError):
        receipt["host_memory_kib"] = None
    path = c.output / "logs" / f"{args.split}_{arm}_server_exit_{time.time_ns()}.json"
    atomic_json(path, receipt)
    c.update(model_server_exit={**receipt, "receipt": str(path)})
    return receipt


def run_arm(c, args, plan, arm):
    folder = args.output / args.split / arm
    folder.mkdir(exist_ok=True)
    runtime = Path(plan["runtime"])
    model = "sca-coder" if arm == "coder" else "local-qwen3-vl-8b"
    config = json.loads(json.dumps(plan["common_config"]))
    config["code"].update(model=model, base_url=f"http://127.0.0.1:{args.port}/v1")
    config["vision"].update(model="unused", base_url="http://127.0.0.1:1/v1")
    config_path = folder / "config.json"
    immutable_json(config_path, config)
    pending = []
    for row in plan["tasks"]:
        run_dir = folder / "runs" / row["task_id"]
        if (run_dir / "result.json").exists():
            result = read(run_dir / "result.json")
            if (result["task_id"] != row["task_id"] or not result.get("model_inference")
                    or read(run_dir / "config.json") != RunConfig.model_validate(config).model_dump()):
                raise ValueError("Completed result belongs to a different experiment")
            if interrupted_result(result):
                if not args.resume_interrupted:
                    raise ValueError(f"Interrupted attempt retained at {run_dir}; use --resume-interrupted to archive and retry")
            else:
                continue  # Includes completed model failures; never rerun them to cherry-pick success.
        if run_dir.exists():
            if not args.resume_interrupted:
                raise ValueError(f"Incomplete attempt retained at {run_dir}; use --resume-interrupted to archive and retry")
            archive = args.output / "interrupted_attempts" / args.split / arm / f"{row['task_id']}_{time.time_ns()}"
            archive.parent.mkdir(parents=True, exist_ok=True)
            run_dir.rename(archive)
        pending.append(row)
    if not pending:
        return
    service, env = start_service(c, args, runtime, arm, plan["resources"])
    try:
        active = []
        while pending or active:
            server_returncode = service["proc"].poll()
            if server_returncode is not None:
                receipt = record_server_exit(c, args, arm, service, active, server_returncode)
                raise RuntimeError(f"Model server exited during evaluation (returncode={server_returncode}, "
                                   f"signal={receipt['signal']}); inspect {service['log']}; "
                                   "completed task results retained")
            while pending and len(active) < args.workers:
                row = pending.pop(0)
                run_dir = folder / "runs" / row["task_id"]
                cmd = [sys.executable, "-u", str(runtime / "scripts/run_scale_4gpu.py"), "eval-one",
                       "--task", row["task_file"], "--config", str(config_path), "--run-dir", str(run_dir)]
                entry = c.launch(cmd, f"{args.split}_{arm}_{row['task_id']}", env=env, cwd=runtime)
                active.append((row, entry, time.monotonic()))
            for row, entry, started in list(active):
                if time.monotonic() - started > 4200:
                    terminate(entry["proc"])
                    raise RuntimeError(f"Task exceeded outer timeout; retained {entry['log']}")
                if entry["proc"].poll() is not None:
                    result_path = folder / "runs" / row["task_id"] / "result.json"
                    if entry["proc"].returncode or not result_path.exists():
                        raise RuntimeError(f"Evaluation worker crashed: {entry['log']}")
                    if interrupted_result(read(result_path)):
                        server_returncode = service["proc"].poll()
                        if server_returncode is not None:
                            record_server_exit(c, args, arm, service, active, server_returncode)
                        raise RuntimeError(f"Evaluation interrupted: {result_path}; "
                                           "use --resume-interrupted to archive and retry")
                    active.remove((row, entry, started))
            c.update(stage="evaluating", split=args.split, arm=arm, pending=len(pending), running=len(active),
                     total=len(plan["tasks"]))
            if active:
                time.sleep(1)
        make_report(folder)
    finally:
        terminate(service["proc"])


def comparison(output, split):
    folder = output / split
    plan = read(folder / "plan.json")
    expected = {r["task_id"] for r in plan["tasks"]}
    arms, outcomes = {}, {}
    for arm in ("baseline", "coder"):
        records = [read(folder / arm / "runs" / task / "result.json") for task in sorted(expected)
                   if (folder / arm / "runs" / task / "result.json").exists()]
        completed = [r for r in records if r.get("model_inference") and not interrupted_result(r)]
        if len(completed) != len(expected):
            raise ValueError(f"{arm} incomplete: {len(completed)}/{len(expected)}; no final paired report")
        arms[arm] = {"attempted": len(records), "resolved": sum(r["resolved"] for r in records),
                     "accuracy": sum(r["resolved"] for r in records) / len(records),
                     "mean_seconds": sum(r["elapsed_seconds"] for r in records) / len(records),
                     "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in records),
                     "completion_tokens": sum(r["usage"]["completion_tokens"] for r in records),
                     "exit_reasons": dict(Counter(r.get("exit_reason", r["status"]) for r in records)),
                     "by_repository": {repo: {"attempted": sum(r["group_id"] == repo for r in records),
                          "resolved": sum(r["resolved"] for r in records if r["group_id"] == repo)}
                          for repo in sorted({r["group_id"] for r in records})}}
        outcomes[arm] = {r["task_id"]: r["resolved"] for r in records}
    pairs = Counter()
    for task in sorted(expected):
        old, new = outcomes["baseline"][task], outcomes["coder"][task]
        pairs["retained" if old and new else "regressed" if old else "repaired" if new else "both_failed"] += 1
    result = {"complete": True, "split": split, "arms": arms, "paired": dict(pairs),
              "interpretation": plan["interpretation"]}
    atomic_json(folder / "comparison.json", result)
    lines = ["# 四卡 Coder 与现有模型的代码修复对照", "", plan["interpretation"], "",
             "| 模型 | 成功/总数 | 平均秒数 | 输入 token | 输出 token |", "|---|---:|---:|---:|---:|"]
    for arm, row in arms.items():
        lines.append(f"| {arm} | {row['resolved']}/{row['attempted']} | {row['mean_seconds']:.1f} | "
                     f"{row['prompt_tokens']} | {row['completion_tokens']} |")
    lines += ["", "配对结果：" + json.dumps(dict(pairs), ensure_ascii=False), "",
              "所有已尝试的模型失败均计入分母。环境预检失败另见数据目录 preflight_report.json。",
              "本轮是原始文本输入的代码侧评测；没有新增截图端到端成绩。"]
    (folder / "comparison.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "resources", "preflight", "run", "status", "stop", "report", "pipeline", "resume", "smoke", "probe-one",
                                       "audit-one", "eval-one"])
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--dev-count", type=int, default=100)
    p.add_argument("--test-count", type=int, default=300)
    p.add_argument("--gpus", default="4,5,6,7")
    p.add_argument("--allow-shared-gpus", action="store_true",
                   help="Allow existing GPU processes when each device has enough free memory plus headroom")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    p.add_argument("--gpu-headroom-mib", type=int, default=1536)
    p.add_argument("--wait-for-gpus", action="store_true",
                   help="Wait until selected GPUs pass two consecutive capacity checks, without stopping other jobs")
    p.add_argument("--port", type=int, default=18836)
    p.add_argument("--trace-server-signals", action="store_true",
                   help="Trace model-service termination signal sender PID/UID using strace; inference is unchanged")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--preflight-split", choices=["dev", "test", "all"], default="dev",
                   help="Prepare only this split; the default dev pipeline does not wait for test environments")
    p.add_argument("--arms", default="baseline,coder")
    p.add_argument("--agent-mode", choices=["v6", "v7"], default="v6",
                   help="v6 isolates tool fixes; v7 additionally uses evidence-backed state and compaction")
    p.add_argument("--workers", type=int, choices=[1, 2], default=2)
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--resume-interrupted", action="store_true",
                   help="Archive and retry cancelled/connection-interrupted/incomplete attempts; completed model failures remain")
    p.add_argument("--limit", type=int, default=0, help="Preflight-only partial check; never reduces formal evaluation")
    p.add_argument("--test-timeout", type=int, default=600)
    p.add_argument("--instance")
    p.add_argument("--task", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--receipt", type=Path, help="Internal probe-one result path; must not already exist")
    args = p.parse_args()
    args.output, args.data = args.output.resolve(), args.data.resolve()
    if args.trace_server_signals and args.action in {"pipeline", "resume", "smoke", "run"} and not shutil.which("strace"):
        p.error("--trace-server-signals requires strace")
    if not 0.5 <= args.gpu_memory_utilization <= 0.95 or args.gpu_headroom_mib < 0:
        p.error("GPU memory utilization must be 0.5..0.95 and headroom must be nonnegative")
    if args.limit < 0 or (args.limit and args.action != "preflight"):
        raise ValueError("--limit is only a positive preflight diagnostic limit; formal runs always use the full manifest")
    if args.action == "pipeline" and (args.split != "dev" or args.preflight_split != "dev"):
        raise ValueError("pipeline prepares and evaluates dev only; prepare and evaluate test explicitly")
    if args.action == "resume" and args.preflight_split != args.split:
        raise ValueError("resume requires --preflight-split to match --split")
    if args.action == "status":
        print(json.dumps(inspect_status(args.output, args.data), ensure_ascii=False, indent=2))
        return
    if args.action == "stop":
        state = read(args.output / "status.json")
        if alive(state.get("pid"), state.get("start_ticks")):
            os.kill(state["pid"], signal.SIGTERM)
            print("Cancellation requested for this controller only.")
        else:
            print("Controller is not live; no signal sent.")
        return
    if args.action == "prepare":
        value = prepare(ROOT, args.data, dev_count=args.dev_count, test_count=args.test_count)
        print(json.dumps({k: value[k] for k in ("targets", "reserves", "repositories", "input_mode", "gpu_started")}, indent=2))
        return
    if args.action == "audit-one":
        audit_one(ROOT, args.data, args.instance, args.test_timeout, args.retry_failed)
        return
    if args.action == "eval-one":
        if args.run_dir.exists():
            raise ValueError("Refusing to overwrite a previous task attempt")
        run_task(Task.from_file(args.task), RunConfig.model_validate(read(args.config)), args.run_dir,
                 reader_name="oracle", model_name="api")
        return
    if args.action == "probe-one":
        if args.task is None or args.receipt is None:
            p.error("probe-one requires --task and --receipt")
        probe_task_tools(args.task.resolve(), args.receipt.resolve())
        return
    if args.action == "report":
        print(json.dumps(comparison(args.output, args.split), ensure_ascii=False, indent=2))
        return
    def cancel(signum, frame):
        raise KeyboardInterrupt("Cancellation requested")
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, cancel)
    c = Controller(args.output, args.action)
    try:
        if args.action in {"pipeline", "resume", "smoke", "run"}:
            parse_gpus(args.gpus)
            immutable_json(args.output / "gpu_policy.json", gpu_policy(args))
            c.update(requested_gpus=args.gpus, gpu_policy=gpu_policy(args), wait_for_gpus=args.wait_for_gpus,
                     trace_server_signals=args.trace_server_signals)
        if args.action in {"resources", "pipeline"}:
            resources(c, args)
        if args.action == "resume":
            c.update(stage="resources_reused", resource_receipts=verify_resource_receipts(),
                     protocol_check="Repeated by each model service before task evaluation")
        if args.action == "smoke":
            smoke(c, args)
        if args.action in {"preflight", "pipeline", "resume"}:
            ready = ((args.data / "preflight_report.json").exists()
                     and preflight_is_ready(read(args.data / "preflight_report.json"), args.preflight_split))
            if args.action == "preflight" or not ready:
                preflight(c, args)
        if args.action in {"run", "pipeline", "resume"}:
            parse_gpus(args.gpus)
            arms = args.arms.split(",")
            if not arms or len(arms) != len(set(arms)) or set(arms) - {"baseline", "coder"}:
                raise ValueError("Select unique arms from baseline,coder")
            c.update(stage="checking_frozen_inputs")
            plan = plan_evaluation(args, progress=c.update)
            preflight_tools(c, args, plan)
            if args.action == "pipeline":
                smoke(c, args)
            for arm in arms:
                run_arm(c, args, plan, arm)
            if set(arms) == {"baseline", "coder"}:
                comparison(args.output, args.split)
        c.update(stage="preflight_incomplete" if c.state["stage"] == "preflight_incomplete" else "completed",
                 finished_at=time.time())
    except KeyboardInterrupt:
        c.update(stage="cancelled", finished_at=time.time())
        raise SystemExit(130) from None
    except Exception as error:
        c.update(stage="failed", error=str(error), finished_at=time.time())
        raise
    finally:
        c.close()


if __name__ == "__main__":
    main()
