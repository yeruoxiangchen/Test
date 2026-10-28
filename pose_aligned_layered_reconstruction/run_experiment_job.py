"""Run an allowlisted local experiment module with durable stdout and status."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from .prepare_official_experiment import atomic_json


ALLOWED = {"prepare_official_experiment", "train_target_lari", "train_layered_ss",
           "train_distributed_round", "evaluate_ss_occupancy", "evaluate_target_lari", "assess_overfit"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--module", choices=sorted(ALLOWED), required=True)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.name) or not re.fullmatch(r"[0-7](,[0-7])*", args.gpus):
        raise ValueError("invalid job name or GPU list")
    if len(set(args.gpus.split(","))) != len(args.gpus.split(",")):
        raise ValueError("duplicate GPU")
    directory = Path(__file__).resolve().parent / "outputs/round1_20260909/jobs"
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / (args.name + ".log")
    status_path = directory / (args.name + ".json")
    if log_path.exists() or status_path.exists():
        raise FileExistsError("job names are immutable; use a new retry name")
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    command = [sys.executable, "-u"]
    if args.distributed:
        command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=" + str(len(args.gpus.split(",")))]
    command += ["-m", "pose_aligned_layered_reconstruction." + args.module, *arguments]
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=args.gpus, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               ATTN_BACKEND="flash_attn", SPCONV_ALGO="native", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", MAX_JOBS="4")
    env["PATH"] = "/home/zjr/anaconda3/envs/reconviagen/bin:/usr/local/cuda-12/bin:" + env.get("PATH", "")
    status = {"name": args.name, "command": command, "gpus": args.gpus, "log": str(log_path),
              "started_epoch": time.time(), "state": "starting"}
    with log_path.open("x", buffering=1) as stream:
        process = subprocess.Popen(command, cwd=Path(__file__).resolve().parent.parent, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT)
        status.update(state="running", pid=process.pid)
        atomic_json(status_path, status)
        print(f"job={args.name} pid={process.pid} log={log_path}", flush=True)
        return_code = process.wait()
    status.update(state="complete" if return_code == 0 else "failed", return_code=return_code,
                  elapsed_seconds=time.time() - status["started_epoch"])
    atomic_json(status_path, status)
    print(f"job={args.name} return_code={return_code} elapsed_seconds={status['elapsed_seconds']:.1f}", flush=True)
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
