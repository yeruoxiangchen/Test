#!/usr/bin/env python3
"""Prepare, run and inspect a frozen v5/v6 development comparison. No implicit GPU launch in prepare/status."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from screenshot_agent.cli import start_ticks
from screenshot_agent.config import Task
from screenshot_agent.control import atomic_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/v6_dev_20261024"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_runtime(root):
    state = read(root / "source_state.json")
    changed = [name for name, sha in state["sha256"].items() if digest(root/name) != sha]
    if changed:
        raise ValueError(f"Frozen runtime changed: {changed}")
    return state


def choose_tasks(rows, limit):
    if limit < 0 or limit > len(rows):
        raise ValueError(f"limit must be 0 (all) or 1..{len(rows)}")
    if not limit:
        return list(rows)
    by_id = {r["task_id"]: r for r in rows}
    # Known failure modes plus existing successes to check retention; NOT a fresh test set.
    priority = ["pytest-dev__pytest-7220", "mwaskom__seaborn-2848", "sphinx-doc__sphinx-8713",
                "pytest-dev__pytest-7373", "sympy__sympy-13471"]
    selected = [by_id[k] for k in priority if k in by_id][:limit]
    used = {r["task_id"] for r in selected}
    groups = defaultdict(list)
    for row in rows:
        if row["task_id"] not in used:
            groups[row["group_id"]].append(row)
    while len(selected) < limit:
        for group in sorted(groups):
            if groups[group] and len(selected) < limit:
                selected.append(groups[group].pop(0))
    return selected


def check_budgets(arms):
    expected = {"calls": 18, "tools": 50, "wall": 1200, "output": 2048,
                "vision_output": 2048, "context": 8192, "command_timeout": 180}
    for arm in arms:
        config = read(Path(arm["runtime"]) / arm["config"])
        limits = config["limits"]
        actual = {"calls": limits.get("max_model_calls", limits["max_steps"]),
                  "tools": limits["max_tool_calls"], "wall": limits["max_wall_seconds"],
                  "output": config["code"]["max_output_tokens"], "vision_output": config["vision"]["max_output_tokens"],
                  "context": config["max_context_tokens"], "command_timeout": limits["command_timeout_seconds"]}
        if actual != expected:
            raise ValueError(f"Matched-budget configuration mismatch in {arm['name']}: {actual}")


def prepare(output, limit=12, readers="oracle"):
    names = readers.split(",")
    if not names or len(set(names)) != len(names) or set(names) - {"oracle", "ocr", "vlm"}:
        raise ValueError("readers must be a unique comma-separated subset of oracle,ocr,vlm")
    baseline = ROOT / "outputs/v5_eval_source"
    verify_runtime(baseline)
    manifest = ROOT / "datasets/generated/swebench_extended_v5/tasks.jsonl"
    rows = [json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    selected = choose_tasks(rows, limit)
    output.mkdir(parents=True, exist_ok=False)
    (output / "tasks").mkdir()
    packed, inputs = [], {}
    for row in selected:
        path = (manifest.parent / row["task_file"]).resolve()
        task = Task.from_file(path)
        task.split = "dev"
        task.provenance += "_v6_development_regression"
        dest = output / "tasks" / (task.task_id + ".json")
        atomic_json(dest, task.model_dump())
        packed.append({**row, "task_file": str(dest), "split": "dev", "original_split": row["split"]})
        for p in [dest, *[Path(im.path) for im in task.images],
                  *[Path(im.reference) for im in task.images if im.reference],
                  *([Path(task.benchmark)] if task.benchmark else [])]:
            inputs[str(p)] = digest(p)
    manifest_copy = output / "tasks.jsonl"
    manifest_copy.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in packed))
    inputs[str(manifest_copy)] = digest(manifest_copy)
    runtime = output / "runtime_v6"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run([sys.executable, str(ROOT / "scripts/freeze_v5_source.py"), "--output", str(runtime)],
                   cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    plan = {"created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "task_count": len(selected), "readers": names, "runs_per_arm": len(selected)*len(names),
            "task_ids": [r["task_id"] for r in selected], "repositories": sorted({r["group_id"] for r in selected}),
            "manifest": str(manifest_copy), "inputs_sha256": inputs,
            "arms": [{"name": "v5", "runtime": str(baseline), "config": "configs/v5-local.json"},
                     {"name": "v6", "runtime": str(runtime), "config": "configs/v6-local.json"}],
            "shared_limits": {"code_and_summary_calls": 18, "tool_calls": 50, "wall_seconds": 1200,
                              "max_code_output": 2048, "service_context": 16384},
            "model_path": "/data/zjr/models/Qwen3-VL-8B-Instruct",
            "interpretation": "Previously observed cases selected for DEVELOPMENT regression, not new held-out accuracy. "
                              "Same model and readers; v6 changes tool feedback, stopping and context management. "
                              "v5 uses its frozen server/source; v6 uses a separate frozen runtime. "
                              "Repository snapshots and prepared dependency prefixes are reused read-only as inputs."}
    check_budgets(plan["arms"])
    atomic_json(output / "plan.json", plan)
    atomic_json(output / "status.json", {"stage": "prepared", "gpu_started": False})
    return plan


def status(output):
    if not (output / "plan.json").exists():
        return {"stage": "not_prepared", "output": str(output)}
    plan = read(output / "plan.json")
    state = read(output / "status.json")
    pid, ticks = state.get("pid"), state.get("start_ticks")
    state["controller_alive"] = bool(pid and ticks and start_ticks(pid) == ticks)
    state["progress"] = {}
    for arm in plan["arms"]:
        path = output / arm["name"] / "evaluation/status.json"
        progress = read(path) if path.exists() else {"status": "pending", "completed": 0}
        state["progress"][arm["name"]] = {**{k: progress[k] for k in ["status", "completed", "current"] if k in progress},
                                           "total": plan["runs_per_arm"]}
    state["report"] = str(output / "comparison.md") if (output / "comparison.md").exists() else None
    return state


def diagnostics(folder):
    counts = Counter()
    for line in (folder / "events.jsonl").read_text().split("\n"):
        if not line:
            continue
        event = json.loads(line)
        if event["event"] == "model_request" and event.get("kind") in {"code", "summary"}:
            counts["code_requests"] += 1
            if event.get("kind") == "summary" or (not event.get("tools") and any(
                    "Compress this conversation" in str(m.get("content", "")) for m in event.get("messages", []))):
                counts["summary_requests"] += 1
        if event["event"] == "tool_start" and event.get("name") == "edit_file":
            counts["edit_attempts"] += 1
        if event["event"] == "edit_failure":
            counts["failed_edit_requests"] += 1
            counts["blocked_repeats"] += bool(event.get("repeated"))
        if event["event"] == "tool_result" and event.get("name") == "edit_file":
            try:
                shell = json.loads(event["output"])
                worker = json.loads(shell["output"])
                if "match exactly once" in worker.get("error", ""):
                    counts["failed_edit_requests"] += 1
            except (ValueError, TypeError, KeyError):
                pass  # v6 records explicit edit_failure events instead.
    return counts


def report(output):
    plan = read(output / "plan.json")
    results, outcomes = {}, {}
    for arm in plan["arms"]:
        folder = output / arm["name"] / "evaluation"
        records = read(folder / "report.json")["records"]
        if len(records) != plan["runs_per_arm"] or not all(r["model_inference"] for r in records):
            raise ValueError("Incomplete comparison or replay results; no model accuracy report written")
        expected = {(t, r) for t in plan["task_ids"] for r in plan["readers"]}
        if {(r["task_id"], r["reader"]) for r in records} != expected:
            raise ValueError("Comparison contains different task-reader pairs")
        counts = Counter()
        for row in records:
            counts.update(diagnostics(folder / "runs" / (row["task_id"] + "_" + row["reader"])))
        n = len(records)
        outcomes[arm["name"]] = {(r["task_id"], r["reader"]): bool(r["resolved"]) for r in records}
        results[arm["name"]] = {"attempted": n, "resolved": sum(r["resolved"] for r in records),
            "with_patch": sum(bool(r.get("changed_files")) for r in records),
            "exit_reasons": dict(Counter(r.get("exit_reason", r["status"]) for r in records)),
            "mean_seconds": sum(r["elapsed_seconds"] for r in records)/n,
            "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in records),
            "completion_tokens": sum(r["usage"]["completion_tokens"] for r in records), **counts}
    paired = Counter()
    for key, old in outcomes["v5"].items():
        new = outcomes["v6"][key]
        paired["retained" if old and new else "regressed" if old else "repaired" if new else "both_failed"] += 1
    value = {"complete": True, "interpretation": plan["interpretation"], "arms": results,
             "paired": dict(paired), "actual_runs": 2*plan["runs_per_arm"]}
    atomic_json(output / "comparison.json", value)
    lines = ["# v6 开发回归对照", "", plan["interpretation"], "",
             "| 版本 | 成功/运行 | 产生修改 | 编辑失败请求/尝试 | 摘要/代码请求 | 平均秒数 |", "|---|---:|---:|---:|---:|---:|"]
    for name, row in results.items():
        lines.append(f"| {name} | {row['resolved']}/{row['attempted']} | {row['with_patch']} | "
                     f"{row.get('failed_edit_requests', 0)}/{row.get('edit_attempts', 0)} | "
                     f"{row.get('summary_requests', 0)}/{row.get('code_requests', 0)} | {row['mean_seconds']:.1f} |")
    lines += ["", "配对结果：" + json.dumps(dict(paired), ensure_ascii=False), "",
              "失败编辑请求包含被重复保护拦截的请求；不是独立任务数。所有失败/异常保留在分母中。",
              "完整退出原因与实际token消耗见 comparison.json。该批任务为开发回归，不用于声称未见任务泛化。"]
    (output / "comparison.md").write_text("\n".join(lines)+"\n")
    return value


def run(output, gpu, port):
    plan = read(output / "plan.json")
    if not gpu.isdecimal() or not 1024 <= port <= 65535:
        raise ValueError("Select one numeric physical GPU index and a port in 1024..65535")
    check_budgets(plan["arms"])
    if any((output / arm["name"]).exists() for arm in plan["arms"]):
        raise ValueError("Run directories already exist; preserve them and prepare a new --output for a retry")
    for name, sha in plan["inputs_sha256"].items():
        if digest(name) != sha:
            raise ValueError(f"Prepared input changed: {name}")
    for arm in plan["arms"]:
        verify_runtime(Path(arm["runtime"]))
    state = {"stage": "starting", "pid": os.getpid(), "start_ticks": start_ticks(os.getpid()),
             "gpu": gpu, "port": port, "started_at": time.time()}
    atomic_json(output / "status.json", state)
    active = None
    def cancel(signum, frame):
        raise KeyboardInterrupt("Stop requested")
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, cancel)
    try:
        for arm in plan["arms"]:
            runtime = Path(arm["runtime"])
            env = os.environ.copy()
            env.pop("SCA_CODEOCR_MANIFEST", None)  # Do not inherit an unrelated optional long evaluation.
            env.update(PYTHONPATH=str(runtime / "src"), SCA_GPU=gpu, SCA_PORT=str(port),
                       SCA_MANIFEST=plan["manifest"], SCA_CONFIG=str(runtime / arm["config"]),
                       SCA_READERS=",".join(plan["readers"]), SCA_LIMIT="0", SCA_SPLIT="dev",
                       SCA_RUN_DIR=str(output / arm["name"]), SCA_MAX_CONTEXT="16384", SCA_MAX_OUTPUT="2048",
                       SCA_MODEL_PATH=plan["model_path"],
                       SCA_REQUIRE_VLM_SUCCESS="0")
            log_path = output / (arm["name"] + ".log")
            with log_path.open("xb") as log:
                active = subprocess.Popen(["bash", str(runtime / "scripts/run_local_smoke.sh")],
                                          cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True)
                state.update(stage="evaluating", arm=arm["name"], child_pid=active.pid, log=str(log_path))
                atomic_json(output / "status.json", state)
                last = 0
                while active.poll() is None:
                    if time.monotonic()-last >= 15:
                        print(json.dumps(status(output), ensure_ascii=False), flush=True)
                        last = time.monotonic()
                    time.sleep(1)
                code = active.returncode
                if code:
                    raise RuntimeError(f"{arm['name']} failed (exit {code}); inspect {log_path}")
            active = None
            verify_runtime(runtime)
            # The shell sends TERM to its own model on exit; wait for that exact process to leave.
            pid_file = output / arm["name"] / "model_server.pid"
            model_pid = int(pid_file.read_text())
            model_ticks = start_ticks(model_pid)
            deadline = time.monotonic()+30
            while model_ticks and start_ticks(model_pid) == model_ticks and time.monotonic() < deadline:
                time.sleep(.5)
            if model_ticks and start_ticks(model_pid) == model_ticks:
                raise RuntimeError("Previous model service did not stop; inspect its process before starting another arm")
        state["stage"] = "reporting"
        atomic_json(output / "status.json", state)
        report(output)
        state.update(stage="complete", report=str(output / "comparison.md"))
    except KeyboardInterrupt:
        state["stage"] = "cancelled"
    except Exception as error:
        state.update(stage="failed", error=str(error))
        raise
    finally:
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait()
        state["finished_at"] = time.time()
        atomic_json(output / "status.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "status", "report", "stop"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--readers", default="oracle")
    parser.add_argument("--gpu", default="6")
    parser.add_argument("--port", type=int, default=18826)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        value = prepare(output, args.limit, args.readers)
        print(json.dumps({k: v for k, v in value.items() if k != "inputs_sha256"}, ensure_ascii=False, indent=2))
    elif args.command == "run":
        run(output, args.gpu, args.port)
    elif args.command == "status":
        print(json.dumps(status(output), ensure_ascii=False, indent=2))
    elif args.command == "report":
        print(json.dumps(report(output), ensure_ascii=False, indent=2))
    else:
        state = status(output)
        if state.get("controller_alive"):
            os.kill(state["pid"], signal.SIGTERM)
            print("Stop requested for the matching controller process.")
        else:
            print("No matching live controller; no signal sent.")


if __name__ == "__main__":
    main()
