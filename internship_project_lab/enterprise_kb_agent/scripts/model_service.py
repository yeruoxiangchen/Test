"""Own a single-GPU vLLM service without modifying shared environments or jobs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import hashlib
import httpx

from kb.cli import process_identity, write_private
from kb.config import Settings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["start", "status", "stop"])
    p.add_argument("--role", choices=["answer", "judge"], default="answer")
    p.add_argument("--gpu", default="1", choices=["0", "1", "6", "7"])
    p.add_argument("--port", type=int)
    p.add_argument("--memory-fraction", type=float, default=.78)
    p.add_argument("--model")
    p.add_argument("--python", default="/data/zjr/deepsearch_rl/venv/bin/python")
    args = p.parse_args()
    args.port = args.port or (18769 if args.role == "judge" else 18768)
    args.model = args.model or ("/data/zjr/models/Qwen3-VL-8B-Instruct" if args.role == "judge" else "/data/zjr/deepsearch_rl/models/Qwen3-8B")
    alias = "kb-judge-qwen3-vl-8b" if args.role == "judge" else "kb-qwen3-8b"
    context = 12288 if args.role == "judge" else 8192
    root = Settings().root
    state_path = root / ("judge_service.json" if args.role == "judge" else "model_service.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    alive = bool(state) and process_identity(state["pid"]) == state["start_ticks"]
    if args.action == "status":
        ready = False
        if alive:
            try:
                ready = httpx.get(state["url"] + "/health", timeout=2, trust_env=False).status_code == 200
            except httpx.HTTPError:
                pass
        print(json.dumps({**state, "alive": alive, "ready": ready}, ensure_ascii=False, indent=2))
        return
    if args.action == "stop":
        if alive:
            command = Path(f"/proc/{state['pid']}/cmdline").read_bytes()
            if state["model"].encode() not in command or b"vllm.entrypoints.openai.api_server" not in command:
                raise SystemExit("Process identity does not match; not stopped")
            os.killpg(state["pid"], signal.SIGTERM)
        print(json.dumps({"stop_requested": alive}))
        return
    if alive:
        raise SystemExit("Project model service already running; use status")
    if args.role == "judge":
        answer = json.loads((root / "model_service.json").read_text())
        answer_path = Path(answer["model_physical_path"])
        judge_path = Path(args.model).resolve()
        if judge_path == answer_path.resolve() or (judge_path / "model.safetensors.index.json").read_bytes() == (answer_path / "model.safetensors.index.json").read_bytes():
            raise SystemExit("Judge must have different weights, not only a different alias")
    if not .5 <= args.memory_fraction <= .8:
        raise SystemExit("Shared-GPU memory fraction must be between .5 and .8")
    stats = subprocess.check_output(["nvidia-smi", "-i", args.gpu, "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"], text=True)
    total, free = map(int, stats.strip().split(","))
    if free < total * args.memory_fraction + 2048:
        raise SystemExit(f"GPU {args.gpu} free {free} MiB below budget plus 2048 MiB headroom; not started")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", VLLM_USAGE_STATS="0", DO_NOT_TRACK="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               VLLM_CACHE_ROOT=str(root / "cache/vllm"), TRITON_CACHE_DIR=str(root / "cache/triton"),
               TORCHINDUCTOR_CACHE_DIR=str(root / "cache/torchinductor"))
    command = [args.python, "-B", "-u", "-m", "vllm.entrypoints.openai.api_server", "--model", args.model,
               "--served-model-name", alias, "--host", "127.0.0.1", "--port", str(args.port),
               "--dtype", "bfloat16", "--gpu-memory-utilization", str(args.memory_fraction),
               "--max-model-len", str(context), "--max-num-seqs", "1", "--max-num-batched-tokens", "2048",
               "--enforce-eager", "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
               "--disable-log-requests"]
    if args.role == "judge":
        overlay = root / "judge_packages"
        if not (overlay / "receipt.json").exists():
            raise SystemExit("Run judge-packages first; shared environment is not modified")
        env['PYTHONPATH'] = str(overlay) + os.pathsep + env.get('PYTHONPATH','')
        command += ["--limit-mm-per-prompt", '{"image":0,"video":0}']
    log_path = root / ("logs/judge.log" if args.role == "judge" else "logs/model.log")
    with log_path.open("a") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    state = {"pid": child.pid, "start_ticks": process_identity(child.pid), "gpu": args.gpu,
             "url": f"http://127.0.0.1:{args.port}", "model": alias, "dtype": "bfloat16", "role": args.role,
             "model_physical_path": str(Path(args.model).resolve()), "environment": args.python,
             "gpu_memory_fraction": args.memory_fraction, "context": context, "concurrency": 1,
             "started_at": time.time(), "log": str(log_path), "command": command}
    write_private(state_path, json.dumps(state, ensure_ascii=False, indent=2))
    # Separate config; user's private.env has priority. Existing app is not restarted here.
    if args.role == "judge":
        write_private(root / "judge.env", f"KB_JUDGE_BASE_URL=http://127.0.0.1:{args.port}/v1\nKB_JUDGE_MODEL={alias}\nKB_JUDGE_TOKENIZE_URL=http://127.0.0.1:{args.port}/tokenize\nKB_JUDGE_CONTEXT_TOKENS={context}\n")
    else:
        write_private(root / "model.env", f"KB_LLM_BASE_URL=http://127.0.0.1:{args.port}/v1\nKB_LLM_MODEL={alias}\nKB_TOKENIZE_URL=http://127.0.0.1:{args.port}/tokenize\nKB_CONTEXT_TOKENS={context}\n")
    print(json.dumps({"status": "starting", **state}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
