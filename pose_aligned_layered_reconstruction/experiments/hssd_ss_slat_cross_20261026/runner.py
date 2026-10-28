"""Serial cross-level queue; status/watch import only the standard library."""

import argparse
from collections import deque
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

# The direct-script entry avoids the parent package's Torch imports. It must
# also leave frozen source directories untouched when launched without run.sh.
sys.dont_write_bytecode = True
if __package__:
    from .common import (
        HERE, ROOT, OUT, PYTHON, MODULE, now, read, write, event,
        freeze_runtime, plan, new_jobs, case_dir, check_smoke,
    )
else:
    from common import (
        HERE, ROOT, OUT, PYTHON, MODULE, now, read, write, event,
        freeze_runtime, plan, new_jobs, case_dir, check_smoke,
    )


RUN_ACTIONS = ("prepare", "smoke", "all", "summary")
ACTIVE_STATES = ("starting", "running", "stopping")


def configure_environment():
    """Select only the explicitly configured GPU, including direct invocations."""
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("HSSD_CROSS_GPU", "4")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"


def check_gpu():
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    if not gpu.isascii() or not gpu.isdigit():
        raise ValueError("HSSD_CROSS_GPU must select one physical GPU index")
    # This is a memory query only: never initialize CUDA or probe other GPUs.
    output = subprocess.check_output(
        ["nvidia-smi", "-i", gpu, "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"],
        text=True, timeout=20,
    )
    used = int(output.strip())
    if used > 1024:
        raise RuntimeError(
            f"GPU {gpu} occupied ({used} MiB); set HSSD_CROSS_GPU explicitly"
        )


def process_start_ticks(pid):
    try:
        # comm may contain spaces or parentheses; field 22 follows the last ')'.
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19]
    except (OSError, ValueError, IndexError, TypeError):
        return None


def own_process(pid, expected_ticks=None):
    try:
        pid = int(pid)
        if pid <= 1:
            return False
        parts = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except (OSError, ValueError, TypeError):
        return False
    if str(HERE / "runner.py").encode() not in parts and (MODULE + ".runner").encode() not in parts:
        try:
            cwd = Path(f"/proc/{pid}/cwd").resolve(strict=True)
            matches = any(
                (cwd / os.fsdecode(part)).resolve() == (HERE / "runner.py").resolve()
                for part in parts if part.endswith(b"runner.py")
            )
        except OSError:
            return False
        if not matches:
            return False
    return expected_ticks is None or process_start_ticks(pid) == str(expected_ticks)


def state(value):
    write(OUT / "status.json", {
        **value,
        "time": now(),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pid": os.getpid(),
        "process_start_ticks": process_start_ticks(os.getpid()),
    })


def terminate_group(process):
    """Reap only the process group created for this stage, even on SIGTERM."""
    old_handlers = {
        sig: signal.signal(sig, signal.SIG_IGN)
        for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def stage(name, module, *args, timeout=21600):
    log = OUT / "logs" / f"{name}_{time.time_ns()}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    event(name, phase="starting", log=str(log))
    process = None
    try:
        with log.open("w") as stream:
            process = subprocess.Popen(
                [str(PYTHON), "-m", MODULE + "." + module, *args],
                cwd=str(ROOT), stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
            )
            while True:
                state({
                    "status": "running", "stage": name,
                    "child_pid": process.pid,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "log": str(log),
                })
                code = process.poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f"{name} exited {code}; see {log}")
                    event(name, phase="complete", elapsed_seconds=time.monotonic() - started)
                    return
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f"{name} exceeded {timeout} seconds; see {log}")
                time.sleep(3)
    finally:
        if process is not None:
            terminate_group(process)


def ensure_smoke():
    if not (OUT / "smoke.json").exists():
        stage("smoke", "worker", "smoke", timeout=7200)
    # Existing failed or stale smoke artifacts must not silently pass or be
    # replaced. common validates the runtime/input binding as well as success.
    check_smoke()


def open_lock(lock_fd=None):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "queue.lock"
    lock = os.fdopen(lock_fd, "a") if lock_fd is not None else path.open("a")
    try:
        if lock_fd is not None:
            actual, expected = os.fstat(lock.fileno()), path.stat()
            if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
                raise ValueError("Inherited lock descriptor does not belong to this queue")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock
    except BaseException:
        lock.close()
        raise


def run(action, lock_fd=None):
    # Closing an inherited descriptor leaves the launcher's shared flock held
    # by this process. Do not issue LOCK_UN while either process still uses it.
    with open_lock(lock_fd):
        state({"status": "starting", "stage": action})
        try:
            freeze_runtime()
            if action != "summary":
                stage("prepare", "prepare", timeout=1800)
            if action in ("smoke", "all"):
                check_gpu()
                ensure_smoke()
            if action == "all":
                stage("mesh", "worker", "mesh", timeout=86400)
            if action in ("all", "summary"):
                stage("summary", "summarize", timeout=14400)
            state({"status": action + "_complete_review_required", "stage": None})
            event("queue", phase="complete", action=action)
        except BaseException as error:
            previous = read(OUT / "status.json") if (OUT / "status.json").exists() else {}
            state({
                **previous,
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "child_pid": None,
                "error": str(error) or type(error).__name__,
            })
            raise


def start():
    with open_lock() as lock:
        # Reject an occupied selection before reporting a successful launch.
        check_gpu()
        log = OUT / f"launcher_{time.time_ns()}.log"
        with log.open("w") as stream:
            process = subprocess.Popen(
                [str(PYTHON), str(HERE / "runner.py"), "all", "--lock-fd", str(lock.fileno())],
                cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=(lock.fileno(),),
            )
        write(OUT / "launches" / f"{time.time_ns()}.json", {
            "time": now(), "action": "all", "pid": process.pid,
            "process_start_ticks": process_start_ticks(process.pid),
            "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "log": str(log),
        })
        print(
            f'Started SS/SLAT cross queue on GPU {os.environ["CUDA_VISIBLE_DEVICES"]}, '
            f"PID {process.pid}; log {log}", flush=True,
        )


def heartbeat_age(value):
    try:
        stamp = datetime.fromisoformat(value)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return max(0, (datetime.now(timezone.utc) - stamp).total_seconds())
    except (TypeError, ValueError):
        return None


def status():
    path = OUT / "status.json"
    value = read(path) if path.exists() else {"status": "not_started"}
    age = heartbeat_age(value.get("time"))
    print(f'[{now()}] HSSD SS/SLAT cross GPU {value.get("gpu", "-")}', flush=True)
    print(
        f'状态：{value.get("status", "unknown")}  阶段：{value.get("stage") or "无活动阶段"}  '
        f'心跳距今：{round(age) if age is not None else "-"} 秒', flush=True,
    )
    if value.get("status") in ACTIVE_STATES:
        alive = own_process(value.get("pid"), value.get("process_start_ticks"))
        print(f'队列 PID：{value.get("pid", "-")}；进程身份：{"匹配" if alive else "不存在或不匹配"}', flush=True)
        if not alive or age is None or age > 30:
            print("心跳或进程身份异常；旧 running 状态不能证明队列仍在运行。", flush=True)
    if (OUT / "plan.json").exists():
        jobs = list(new_jobs())
        cases = plan()["plan"]["cases"]
        references = sum(bool(case["is_reference"]) for case in cases)
        meshes = sum((case_dir(case) / "mesh.json").exists() for case in jobs)
        print(
            f"新增交叉 mesh={meshes}/{len(jobs)}；旧对角引用={references}；"
            f"逻辑结果={references + meshes}/{len(cases)}", flush=True,
        )
    else:
        print("新增交叉 mesh=0/12；计划旧对角引用=12，逻辑 case=24（尚未 prepare）。", flush=True)
    smoke_path = OUT / "smoke.json"
    if smoke_path.exists():
        # Large mesh/latent bindings are verified by ensure_smoke at launch,
        # rather than reread on every watch heartbeat.
        smoke = read(smoke_path)
        print(
            "Smoke：" + ("记录通过" if smoke.get("passed") else "记录未通过")
            + "（旧 table518 对角重放 + 真实 stool 交叉；执行时复核绑定）。",
            flush=True,
        )
    else:
        print("Smoke：尚未完成。", flush=True)
    if value.get("error"):
        print("错误：" + str(value["error"]), flush=True)
    if value.get("log"):
        log = Path(value["log"])
        print("日志：" + str(log), flush=True)
        # Only display logs owned by this experiment, including after a stale
        # status file has been manually copied into the directory.
        if log.is_file() and log.resolve().is_relative_to(OUT.resolve()):
            with log.open(errors="replace") as stream:
                tail = "".join(deque(stream, maxlen=2)).rstrip()
            if tail:
                print(tail, flush=True)
    if value.get("status") == "all_complete_review_required":
        print(f"计算与汇总已完成，请审阅 {OUT} 中的报告和图。", flush=True)


def stop():
    path = OUT / "status.json"
    value = read(path) if path.exists() else {}
    if value.get("status") not in ACTIVE_STATES:
        print("No active cross queue recorded.", flush=True)
        return
    pid, ticks = value.get("pid"), value.get("process_start_ticks")
    if not own_process(pid, ticks):
        if process_start_ticks(pid) is not None:
            raise ValueError("Recorded PID belongs to another process; refusing to signal")
        # Do not replace a concurrent launch's fresh status with this stale row.
        with open_lock():
            latest = read(path)
            if latest.get("pid") == pid and latest.get("process_start_ticks") == ticks:
                write(path, {**latest, "time": now(), "status": "interrupted",
                             "stage": None, "child_pid": None,
                             "error": "Queue process no longer exists"})
        print("Queue process no longer exists; recorded its interrupted state.", flush=True)
        return
    try:
        os.kill(int(pid), signal.SIGTERM)
    except ProcessLookupError:
        print("Queue exited before SIGTERM was delivered.", flush=True)
        return
    print(f"Sent SIGTERM to cross queue {pid}; waiting for its own child cleanup.", flush=True)
    deadline = time.monotonic() + 25
    while own_process(pid, ticks) and time.monotonic() < deadline:
        time.sleep(0.25)
    if own_process(pid, ticks):
        raise RuntimeError(f"Queue {pid} has not exited after SIGTERM; inspect {path}")
    print("Cross queue stopped; completed artifacts are retained.", flush=True)


def signal_stop(signum, frame):
    raise KeyboardInterrupt(f"Signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="status",
                        choices=(*RUN_ACTIONS, "start", "status", "watch", "stop"))
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    configure_environment()
    signal.signal(signal.SIGTERM, signal_stop)
    try:
        if args.action == "start":
            start()
        elif args.action == "stop":
            stop()
        elif args.action in ("status", "watch"):
            while True:
                status()
                if args.action == "status":
                    break
                time.sleep(10)
        else:
            run(args.action, args.lock_fd)
    except BlockingIOError:
        print(f"This cross queue is already locked: {OUT / 'queue.lock'}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        return 0 if args.action == "watch" else 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
