"""Visible first-surface oracle queue with read-only idle-GPU waiting."""

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
        HERE, ROOT, OUT, DATA_ROOT, PYTHON, MODULE, now, read, write, event,
        freeze_runtime, plan, new_jobs, case_dir, check_smoke, ARMS, STEPS,
    )
else:
    from common import (
        HERE, ROOT, OUT, DATA_ROOT, PYTHON, MODULE, now, read, write, event,
        freeze_runtime, plan, new_jobs, case_dir, check_smoke, ARMS, STEPS,
    )


RUN_ACTIONS = ("storage", "prepare", "evidence", "smoke", "train", "ss", "mesh", "all", "summary")
ACTIVE_STATES = ("starting", "running", "waiting_for_gpu", "stopping")
SELECTED_GPU = None
WAIT_SECONDS = 20


def configure_environment():
    """CPU preparation precedes selection; never pass 'auto' to CUDA."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    for key, suffix in {
        "TMPDIR": "tmp", "TORCH_HOME": "cache/torch", "HF_HOME": "cache/huggingface",
        "HF_HUB_CACHE": "cache/huggingface/hub", "TORCH_EXTENSIONS_DIR": "cache/torch_extensions",
        "MPLCONFIGDIR": "cache/matplotlib", "XDG_CACHE_HOME": "cache/xdg",
        "NUMBA_CACHE_DIR": "cache/numba",
    }.items():
        os.environ[key] = str(DATA_ROOT / suffix)
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    gpu_candidates()  # Reject malformed configuration before launching work.


def gpu_candidates():
    request = os.environ.get("HSSD_VISIBLE_ORACLE_GPU", "auto").strip()
    raw = os.environ.get("HSSD_VISIBLE_ORACLE_GPU_ALLOWLIST", "0,1,2,3,4,5,6,7")
    values = [x.strip() for x in raw.split(",")]
    if not values or any(not x.isascii() or not x.isdigit() for x in values):
        raise ValueError("GPU allowlist must contain comma-separated physical GPU indices")
    allowed = [int(x) for x in values]
    if len(set(allowed)) != len(allowed) or any(x not in range(8) for x in allowed):
        raise ValueError("GPU allowlist must contain distinct indices in 0..7")
    if request == "auto":
        return [str(x) for x in allowed]
    if not request.isascii() or not request.isdigit() or int(request) not in allowed:
        raise ValueError("HSSD_VISIBLE_ORACLE_GPU must be auto or one allowlisted index")
    return [str(int(request))]


def parse_gpu_snapshot(gpu_output, process_output):
    """Pure parser: a memory threshold alone is insufficient to claim a free GPU."""
    result, by_uuid = {}, {}
    for line in gpu_output.strip().splitlines():
        fields = [x.strip() for x in line.split(",")]
        if len(fields) != 3 or not fields[0].isdigit() or not fields[2].isdigit():
            raise ValueError("Invalid nvidia-smi GPU inventory")
        index, uuid, memory = fields
        if index in result or uuid in by_uuid:
            raise ValueError("Duplicate GPU inventory entry")
        result[index] = {"uuid": uuid, "memory_mib": int(memory), "compute_pids": []}
        by_uuid[uuid] = index
    if not result:
        raise ValueError("No GPUs returned by nvidia-smi")
    for line in process_output.strip().splitlines():
        fields = [x.strip() for x in line.split(",")]
        if len(fields) != 2 or fields[0] not in by_uuid or not fields[1].isdigit():
            raise ValueError("Invalid nvidia-smi compute process inventory")
        result[by_uuid[fields[0]]]["compute_pids"].append(int(fields[1]))
    for value in result.values():
        value["idle"] = value["memory_mib"] < 256 and not value["compute_pids"]
    return result


def gpu_snapshot():
    outputs = []
    for query in ("--query-gpu=index,uuid,memory.used", "--query-compute-apps=gpu_uuid,pid"):
        outputs.append(subprocess.check_output(
            ["nvidia-smi", query, "--format=csv,noheader,nounits"], text=True, timeout=20))
    return parse_gpu_snapshot(*outputs)


def wait_for_gpu(next_stage):
    """Choose once; subsequent stages wait for that same card without sharing."""
    global SELECTED_GPU
    while True:
        candidates = [SELECTED_GPU] if SELECTED_GPU is not None else gpu_candidates()
        inventory = gpu_snapshot()
        if any(gpu not in inventory for gpu in candidates):
            raise ValueError("An allowlisted physical GPU is absent")
        available = [gpu for gpu in candidates if inventory[gpu]["idle"]]
        if available:
            SELECTED_GPU = available[0]
            os.environ["CUDA_VISIBLE_DEVICES"] = SELECTED_GPU
            state({"status": "running", "stage": next_stage, "gpu_inventory": inventory,
                   "gpu_policy": "memory<256MiB and no compute PID; fixed after selection"})
            return SELECTED_GPU
        state({"status": "waiting_for_gpu", "stage": next_stage,
               "gpu_candidates": candidates, "gpu_inventory": inventory,
               "reason": "All eligible GPUs are occupied; no model or allocation probe started"})
        time.sleep(WAIT_SECONDS)


def process_start_ticks(pid):
    try:
        # comm may contain spaces or parentheses; field 22 follows the last ')'.
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
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
    ticks = process_start_ticks(pid)
    return ticks is not None and (expected_ticks is None or ticks == str(expected_ticks))


def state(value):
    write(OUT / "status.json", {
        **value,
        "time": now(),
        "gpu": SELECTED_GPU or "未分配",
        "gpu_request": os.environ.get("HSSD_VISIBLE_ORACLE_GPU", "auto"),
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


def stage(name, module, *args, timeout=21600, uses_gpu=False):
    if uses_gpu:
        wait_for_gpu(name)
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
        stage("smoke", "smoke", timeout=7200, uses_gpu=True)
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
            stage("storage", "storage", timeout=86400)
            if action == "storage":
                state({"status": "storage_complete_review_required", "stage": None})
                return
            freeze_runtime()
            if action != "summary":
                stage("prepare", "prepare", timeout=1800)
            if action in ("evidence", "smoke", "train", "ss", "mesh", "all"):
                stage("extract_visible_first_hit", "evidence", "extract", timeout=7200)
                stage("build_visible_evidence", "evidence", "build", timeout=3600)
            if action in ("smoke", "train", "ss", "mesh", "all"):
                ensure_smoke()
            if action in ("train", "all"):
                for arm in ARMS:
                    stage("train_" + arm, "train", "--arm", arm, timeout=86400, uses_gpu=True)
            if action in ("ss", "all"):
                for arm in ARMS:
                    stage("ss_" + arm, "evaluate", "ss", "--arm", arm, timeout=86400, uses_gpu=True)
            if action in ("mesh", "all"):
                for arm in ARMS:
                    stage("mesh_" + arm, "evaluate", "mesh", "--arm", arm, timeout=86400, uses_gpu=True)
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
            "gpu_request": os.environ.get("HSSD_VISIBLE_ORACLE_GPU", "auto"),
            "gpu_allowlist": gpu_candidates(), "log": str(log),
        })
        print(
            f'Started visible-surface oracle queue; GPU request {os.environ.get("HSSD_VISIBLE_ORACLE_GPU", "auto")}, '
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
    print(f'[{now()}] HSSD VisibleSurfaceOracle4 GPU {value.get("gpu", "-")}', flush=True)
    print(
        f'状态：{value.get("status", "unknown")}  阶段：{value.get("stage") or "无活动阶段"}  '
        f'{"心跳距今" if value.get("status") in ACTIVE_STATES else "状态更新距今"}：'
        f'{round(age) if age is not None else "-"} 秒', flush=True,
    )
    if value.get("status") in ACTIVE_STATES:
        alive = own_process(value.get("pid"), value.get("process_start_ticks"))
        print(f'队列 PID：{value.get("pid", "-")}；进程身份：{"匹配" if alive else "不存在或不匹配"}', flush=True)
        if not alive or age is None or age > 60:
            print("心跳或进程身份异常；旧 running 状态不能证明队列仍在运行。", flush=True)
    if value.get("status") == "waiting_for_gpu":
        print(f"等待空闲卡：{value.get('gpu_candidates')}；仅接受显存<256MiB且无compute进程。", flush=True)
        for gpu in value.get("gpu_candidates", []):
            info = value.get("gpu_inventory", {}).get(gpu, {})
            print(f"  GPU {gpu}: 显存={info.get('memory_mib')}MiB compute PID={info.get('compute_pids')}", flush=True)
        print("CPU准备已继续；等待期间未启动CUDA模型，不抢占其他任务。", flush=True)
    cases = plan()["plan"]["cases"] if (OUT / "plan.json").exists() else []
    new_ss = new_meshes = 0
    for arm in ARMS:
        folder = OUT / "train" / arm
        completed = read(folder / "complete.json") if (folder / "complete.json").exists() else {}
        cursor = int(completed.get("step", 0))
        last_loss = None
        history = folder / "history.jsonl"
        if history.exists():
            with history.open(errors="replace") as stream:
                tail = list(deque(stream, maxlen=3))
            for line in reversed(tail):
                try:
                    row = __import__("json").loads(line)
                    cursor = int(row["step"])
                    last_loss = row.get("flow_loss")
                    break
                except (ValueError, KeyError, TypeError):
                    continue
        counts = {"ss": 0, "mesh": 0}
        for case in cases:
            if case["arm"] != arm or case["is_reference"]:
                continue
            for kind in counts:
                result = case_dir(case) / (kind + ".json")
                if result.exists():
                    try:
                        counts[kind] += bool(read(result).get("success"))
                    except (OSError, ValueError, TypeError):
                        pass
        new_ss += counts["ss"]
        new_meshes += counts["mesh"]
        print(f"  {arm}: train={cursor}/{STEPS} loss={last_loss} SS={counts['ss']}/12 mesh={counts['mesh']}/12", flush=True)
    references = sum(bool(case["is_reference"]) for case in cases)
    print(f"新增SS={new_ss}/12 mesh={new_meshes}/12；旧两臂引用={references}/24；逻辑结果={references + new_meshes}/36", flush=True)
    smoke_path = OUT / "smoke.json"
    if smoke_path.exists():
        smoke = read(smoke_path)
        print("Smoke：" + ("记录通过" if smoke.get("passed") else "记录未通过")
              + "（启动/续跑时复核完整绑定）。", flush=True)
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
        print("队列已经结束，状态时间不再刷新；无需继续等待或再次启动。", flush=True)


def stop():
    path = OUT / "status.json"
    value = read(path) if path.exists() else {}
    if value.get("status") not in ACTIVE_STATES:
        print("No active visible-surface oracle queue recorded.", flush=True)
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
    print(f"Sent SIGTERM to visible-surface oracle queue {pid}; waiting for its own child cleanup.", flush=True)
    deadline = time.monotonic() + 25
    while own_process(pid, ticks) and time.monotonic() < deadline:
        time.sleep(0.25)
    if own_process(pid, ticks):
        raise RuntimeError(f"Queue {pid} has not exited after SIGTERM; inspect {path}")
    print("visible-surface oracle queue stopped; completed artifacts are retained.", flush=True)


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
        print(f"This visible-surface oracle queue is already locked: {OUT / 'queue.lock'}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        return 0 if args.action == "watch" else 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
