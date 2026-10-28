"""CLI for data building, foreground/background runs, monitoring and evaluation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .config import RunConfig, Task, load_config
from .control import Events, RunCancelled, atomic_json
from .data import build_dataset
from .public_data import build_public_dataset, combine_manifests
from .report import make_report
from .runner import run_task
from .sandbox import BubblewrapSandbox


def new_run_id(prefix="run"):
    return f"{prefix}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{uuid.uuid4().hex[:6]}"


def start_ticks(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def launch(directory: Path, command: str) -> int:
    with (directory / "worker.log").open("ab") as log:
        proc = subprocess.Popen([sys.executable, "-m", "screenshot_agent", command,
                                 "--run-dir", str(directory)], stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, stdin=subprocess.DEVNULL)
    atomic_json(directory / "process.json", {"pid": proc.pid, "start_ticks": start_ticks(proc.pid),
                                            "background": True})
    return proc.pid


def execute_batch(directory: Path) -> dict:
    job = json.loads((directory / "batch_job.json").read_text())
    config = RunConfig.model_validate(job["config"])
    if not (directory / "process.json").exists():
        atomic_json(directory / "process.json", {"pid": os.getpid(), "start_ticks": start_ticks(os.getpid()),
                                                 "background": False})
    state = {"status": "running", "completed": 0, "total": len(job["tasks"])*len(job["readers"]),
             "pid": os.getpid(), "started_at": time.time()}
    atomic_json(directory / "status.json", state)
    total_cost = 0.0
    cap = job.get("batch_max_cost_usd")
    try:
        for task_file in job["tasks"]:
            task = Task.from_file(Path(task_file))
            for reader in job["readers"]:
                if cap is not None and total_cost >= cap:
                    state["status"] = "batch_cost_limit"
                    return state
                run_config = config.model_copy(deep=True)
                if cap is not None:
                    per_task = run_config.limits.max_cost_usd
                    run_config.limits.max_cost_usd = min(per_task or cap, cap-total_cost)
                path = directory / "runs" / f"{task.task_id}_{reader}"
                if path.exists():
                    raise ValueError(f"Duplicate task ID in batch: {task.task_id}")
                state["current"] = {"task_id": task.task_id, "reader": reader, "run_dir": str(path)}
                atomic_json(directory / "status.json", state)
                print(f"[{state['completed']}/{state['total']}] Starting {task.task_id} / {reader}", flush=True)
                result = run_task(task, run_config, path, reader_name=reader, model_name=job["model"])
                total_cost += result.get("usage", {}).get("cost_usd", 0)
                state["completed"] += 1
                state["cost_usd"] = total_cost
                atomic_json(directory / "status.json", state)
                print(f"[{state['completed']}/{state['total']}] {task.task_id} / {reader}: "
                      f"{result['status']}, {result.get('elapsed_seconds', 0):.1f}s", flush=True)
                if result["status"] == "cancelled":
                    state["status"] = "cancelled"
                    return state
        state["status"] = "complete"
    except (KeyboardInterrupt, RunCancelled):
        state["status"] = "cancelled"
    except Exception as error:  # top-level job boundary
        state.update(status="error", error=str(error))
    finally:
        state["finished_at"] = time.time()
        atomic_json(directory / "status.json", state)
        make_report(directory)
    return state


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Screenshot repair on CoreCoder with isolated execution")
    sub = p.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Check dependencies, credentials presence and actual isolation")
    doctor.add_argument("--output-root", type=Path, default=Path("outputs/doctor"))
    build = sub.add_parser("build-demo-data", help="Build self-authored synthetic fixtures and real error screenshots")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--limit", type=int, default=20, choices=range(1, 21))
    public = sub.add_parser("build-public-data", help="Restore pinned Boltons bugs and render synthetic screenshots")
    public.add_argument("--source", type=Path, required=True, help="Existing Boltons Git clone, read only")
    public.add_argument("--output", type=Path, required=True)
    combine = sub.add_parser("combine-manifests", help="Combine task manifests with group/split checks")
    combine.add_argument("--manifest", type=Path, action="append", required=True)
    combine.add_argument("--output", type=Path, required=True)
    codeocr = sub.add_parser("evaluate-codeocr", help="Read the author-released CodeOCR subset with OCR/VLM")
    codeocr.add_argument("--manifest", type=Path, required=True)
    codeocr.add_argument("--config", type=Path, required=True)
    codeocr.add_argument("--output", type=Path, required=True)
    codeocr.add_argument("--readers",default="ocr,vlm")
    codeocr.add_argument("--shard-index",type=int,default=0)
    codeocr.add_argument("--shard-count",type=int,default=1)
    codeocr.add_argument("--resume",action="store_true")
    run = sub.add_parser("run", help="Repair one task")
    run.add_argument("--task", type=Path, required=True)
    run.add_argument("--config", type=Path)
    run.add_argument("--reader", choices=["oracle", "ocr", "vlm"], default="vlm")
    run.add_argument("--model", choices=["api", "replay"], default="api")
    run.add_argument("--output-root", type=Path, default=Path("outputs/runs"))
    run.add_argument("--run-id")
    run.add_argument("--background", action="store_true")
    repair = sub.add_parser("repair", help="Repair directly from a repository, screenshots and a problem description")
    repair.add_argument("--repo", type=Path, required=True)
    repair.add_argument("--image", type=Path, action="append", required=True)
    repair.add_argument("--problem", required=True)
    repair.add_argument("--test-command", required=True)
    repair.add_argument("--protect", action="append", default=["tests"], help="Additional read-only test paths")
    repair.add_argument("--config", type=Path, required=True)
    repair.add_argument("--reader", choices=["ocr", "vlm"], default="vlm")
    repair.add_argument("--output-root", type=Path, default=Path("outputs/runs"))
    repair.add_argument("--run-id")
    repair.add_argument("--background", action="store_true")
    repair.set_defaults(model="api")
    for name in ["_worker", "_batch_worker", "status", "stop"]:
        child = sub.add_parser(name)
        child.add_argument("--run-dir", type=Path, required=True)
    evaluate = sub.add_parser("evaluate", help="Run the same tasks and code model with selected readers")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--config", type=Path)
    evaluate.add_argument("--readers", default="oracle,ocr,vlm")
    evaluate.add_argument("--model", choices=["api", "replay"], default="api")
    evaluate.add_argument("--split", choices=["dev", "test", "demo"])
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--batch-max-cost-usd", type=float, default=10.0)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--background", action="store_true")
    report = sub.add_parser("report")
    report.add_argument("--runs", type=Path, required=True)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "evaluate-codeocr":
            from .codeocr_benchmark import evaluate_codeocr
            report = evaluate_codeocr(args.manifest.resolve(), load_config(args.config), args.output.resolve(),
                tuple(args.readers.split(",")),shard_index=args.shard_index,shard_count=args.shard_count,resume=args.resume)
            print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
            return 0
        if args.command == "doctor":
            directory = args.output_root.resolve() / new_run_id("probe")
            (directory / "repo").mkdir(parents=True)
            report = {"python": sys.version, "corecoder": importlib.metadata.version("corecoder"),
                      "bwrap": shutil.which("bwrap"), "api_credentials_present": {
                          k: bool(os.environ.get(k)) for k in ["SCA_CODE_API_KEY", "SCA_VISION_API_KEY", "OPENAI_API_KEY"]}}
            try:
                sandbox = BubblewrapSandbox(directory / "repo", Events(directory / "events.jsonl"))
                report["sandbox_probe"] = sandbox.probe()
                report["sandbox"] = sandbox.description()
            except Exception as error:
                report["sandbox_error"] = str(error)
            atomic_json(directory / "doctor.json", report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return int("sandbox_error" in report)
        if args.command == "build-demo-data":
            result = build_dataset(args.output, args.limit)
            print(json.dumps({k: v for k, v in result.items() if k != "audit"}, indent=2))
            return 0
        if args.command == "build-public-data":
            result = build_public_dataset(args.source, args.output)
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "combine-manifests":
            print(json.dumps(combine_manifests(args.manifest, args.output), indent=2))
            return 0
        if args.command in {"run", "repair"}:
            config = load_config(args.config)
            if args.command == "repair":
                if not args.repo.is_dir() or any(not image.is_file() for image in args.image):
                    raise ValueError("Provide an existing repository directory and screenshot files")
                task = Task(task_id="user_screenshot", repo=str(args.repo.resolve()), problem=args.problem,
                            test_command=args.test_command, protected_paths=args.protect,
                            images=[{"image_id": f"shot_{i+1}", "path": str(image.resolve())}
                                    for i, image in enumerate(args.image)])
            else:
                task = Task.from_file(args.task.resolve())
            run_id = args.run_id or new_run_id(task.task_id)
            if Path(run_id).name != run_id or run_id in {".", ".."}:
                raise ValueError("run-id must be a single directory name")
            directory = args.output_root.resolve() / run_id
            directory.mkdir(parents=True, exist_ok=False)
            atomic_json(directory / "job.json", {"task": task.model_dump(), "config": config.model_dump(),
                        "reader": args.reader, "model": args.model})
            if args.background:
                pid = launch(directory, "_worker")
                print(json.dumps({"pid": pid, "run_dir": str(directory), "log": str(directory / "worker.log")}, indent=2))
                return 0
            atomic_json(directory / "process.json", {"pid": os.getpid(), "start_ticks": start_ticks(os.getpid()),
                                                     "background": False})
            result = run_task(task, config, directory, reader_name=args.reader, model_name=args.model)
            print(json.dumps({"run_dir": str(directory), **result}, ensure_ascii=False, indent=2))
            return 0 if result["resolved"] else 2
        if args.command == "_worker":
            job = json.loads((args.run_dir / "job.json").read_text())
            result = run_task(Task.model_validate(job["task"]), RunConfig.model_validate(job["config"]),
                              args.run_dir, reader_name=job["reader"], model_name=job["model"])
            print(json.dumps(result, ensure_ascii=False))
            return 0 if result["resolved"] else 2
        if args.command == "status":
            result = {"run_dir": str(args.run_dir)}
            for name in ["status.json", "process.json"]:
                if (args.run_dir / name).exists():
                    result[name.removesuffix(".json")] = json.loads((args.run_dir / name).read_text())
            process = result.get("process", {})
            if process:
                result["worker_alive"] = (process.get("start_ticks") is not None
                                          and start_ticks(process["pid"]) == process["start_ticks"])
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "stop":
            process = json.loads((args.run_dir / "process.json").read_text())
            pid = process["pid"]
            if process.get("start_ticks") is None or start_ticks(pid) != process["start_ticks"]:
                print("Worker is no longer running; no signal sent.")
                return 0
            # Signal only this registered worker; it closes its active sandbox process groups.
            os.kill(pid, signal.SIGTERM)
            print(f"Cancellation requested for worker {pid}.")
            return 0
        if args.command == "evaluate":
            config = load_config(args.config)
            readers = args.readers.split(",")
            if not readers or len(set(readers)) != len(readers) or set(readers)-{"oracle", "ocr", "vlm"}:
                raise ValueError("readers must be a unique comma-separated list of oracle,ocr,vlm")
            if args.batch_max_cost_usd <= 0:
                raise ValueError("Batch cost limit must be positive")
            entries = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
            selected = [v for v in entries if args.split is None or v["split"] == args.split]
            if args.limit is not None:
                selected = selected[:args.limit]
            if not selected:
                raise ValueError("No tasks selected")
            tasks = [str((args.manifest.parent / v["task_file"]).resolve()) for v in selected]
            directory = args.output.resolve()
            directory.mkdir(parents=True, exist_ok=False)
            atomic_json(directory / "batch_job.json", {"tasks": tasks, "readers": readers,
                        "model": args.model, "config": config.model_dump(),
                        "batch_max_cost_usd": args.batch_max_cost_usd})
            if args.background:
                pid = launch(directory, "_batch_worker")
                print(json.dumps({"pid": pid, "run_dir": str(directory), "log": str(directory / "worker.log")}, indent=2))
                return 0
            state = execute_batch(directory)
            print(json.dumps(state, ensure_ascii=False, indent=2))
            return 0 if state["status"] == "complete" else 2
        if args.command == "_batch_worker":
            state = execute_batch(args.run_dir)
            print(json.dumps(state, ensure_ascii=False))
            return 0 if state["status"] == "complete" else 2
        if args.command == "report":
            result = make_report(args.runs.resolve())
            print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
            return 0
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
