"""Serial, resumable dev comparison using the already-configured model endpoint."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from kb.cli import process_identity
from kb.config import Settings
from kb.benchmark.sources import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["launch", "work", "status", "stop", "report"])
    parser.add_argument("--output", type=Path, default=Settings().root / "artifacts/git_support_v1/dev_compare_20261028")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    owner = root / "queue.json"
    state = json.loads(owner.read_text()) if owner.exists() else {}
    alive = bool(state) and process_identity(state["pid"]) == state["start_ticks"]
    if args.action == "status":
        report_path = root / "comparison/comparison.json"
        worker_path = root / "queue_state.json"
        worker = json.loads(worker_path.read_text()) if worker_path.exists() else {}
        overall = "complete" if report_path.exists() else ("running" if alive else worker.get("status", "incomplete"))
        if not alive and overall == "running":
            overall = "interrupted"
        print(f"总状态: {overall} | 阶段: {worker.get('stage')} | 错误: {worker.get('error')}")
        print(f"队列存活: {alive} | PID: {state.get('pid')} | 目录: {root}")
        for arm in ("fixed-bm25", "agent-bm25"):
            path = root / arm / "status.json"
            s = json.loads(path.read_text()) if path.exists() else {}
            print(f"{arm:16} {s.get('status','pending'):12} {s.get('completed',0)}/{s.get('total',100)} 当前={s.get('current')} 错误={s.get('error')}")
            print("  日志: " + str(root / (arm + ".log")))
        print("配对报告: " + (str(report_path) if report_path.exists() else ("pending" if alive else "未完成；队列已退出，检查 queue.log")))
        print("队列日志: " + str(root / "queue.log"))
        print("推理串行；使用已配置模型服务，队列本身不额外加载模型。")
        return
    if args.action == "stop":
        if alive:
            command = Path(f"/proc/{state['pid']}/cmdline").read_bytes()
            if b"eval_queue.py" not in command or b"work" not in command:
                raise SystemExit("Queue process identity mismatch")
            os.killpg(state["pid"], signal.SIGTERM)
        print("已请求停止本项目评测队列；模型服务保持独立。")
        return
    if args.action == "launch":
        if alive:
            raise SystemExit("Queue already running")
        with (root / "queue.log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-B", __file__, "work", "--output", str(root)],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(owner, {"pid": process.pid, "start_ticks": process_identity(process.pid), "started_at": time.time()})
        print(json.dumps({"status":"launched", "pid":process.pid, "output":str(root)}, ensure_ascii=False))
        return
    if args.action == "report" and alive:
        raise SystemExit("Queue still running; cannot regenerate its report concurrently")
    stage = None
    def progress(status, error=None):
        write_json(root / "queue_state.json", {"status":status, "stage":stage,
            "error":error, "updated_at":time.time()})
    try:
        # report regenerates only CPU aggregation, never inference or gold labels.
        if args.action != "report":
            for arm in ("fixed-bm25", "agent-bm25"):
                stage = arm
                progress("running")
                command = [sys.executable, "-B", "-m", "kb.benchmark", "run", "--arm", arm, "--split", "dev", "--output", str(root / arm)]
                with (root / (arm + ".log")).open("a") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"{arm} exited {result.returncode}; inspect {root / (arm + '.log')}")
        stage = "report"
        progress("running")
        with (root / "comparison.log").open("a") as log:
            subprocess.run([sys.executable, "-B", "-m", "kb.benchmark", "compare", "--left", str(root / "fixed-bm25"),
                "--right", str(root / "agent-bm25"), "--output", str(root / "comparison")], stdout=log, stderr=subprocess.STDOUT, check=True)
        stage = None
        progress("complete")
        print("配对报告完成: " + str(root / "comparison/comparison.md"))
    except BaseException as exc:
        progress("interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed", str(exc))
        raise


if __name__ == "__main__":
    main()
