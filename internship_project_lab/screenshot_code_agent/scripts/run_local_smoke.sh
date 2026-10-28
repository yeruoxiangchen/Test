#!/usr/bin/env bash
set -euo pipefail
SCA_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCA_PROJECT_DIR"
SCA_PY="$SCA_PROJECT_DIR/.venv/bin/python"
SCA_MODEL_PY="$SCA_PROJECT_DIR/.model-env/bin/python"
SCA_MODEL_PATH="${SCA_MODEL_PATH:-/data/zjr/models/Qwen3-VL-8B-Instruct}"
SCA_GPU="${SCA_GPU:-6}"
SCA_PORT="${SCA_PORT:-18765}"
SCA_STARTUP_TIMEOUT="${SCA_STARTUP_TIMEOUT:-900}"
SCA_MANIFEST="${SCA_MANIFEST:-datasets/generated/v1/tasks.jsonl}"
SCA_CONFIG="${SCA_CONFIG:-configs/local-smoke.json}"
SCA_LIMIT="${SCA_LIMIT:-2}" # 0 selects every task in the manifest/split.
SCA_READERS="${SCA_READERS:-oracle,ocr,vlm}"
SCA_MAX_CONTEXT="${SCA_MAX_CONTEXT:-8192}"
SCA_MAX_OUTPUT="${SCA_MAX_OUTPUT:-1024}"
SCA_EVALUATE_ARGS=(--manifest "$SCA_MANIFEST" --config "$SCA_CONFIG" --readers "$SCA_READERS" --model api)
if [[ ! "$SCA_LIMIT" =~ ^[0-9]+$ ]]; then
  printf 'SCA_LIMIT must be a nonnegative integer (0 means all tasks).\n' >&2
  exit 1
fi
if (( SCA_LIMIT > 0 )); then
  SCA_EVALUATE_ARGS+=(--limit "$SCA_LIMIT")
fi
if [[ -n "${SCA_SPLIT:-}" ]]; then
  SCA_EVALUATE_ARGS+=(--split "$SCA_SPLIT")
fi
SCA_SMOKE_DIR="${SCA_RUN_DIR:-$SCA_PROJECT_DIR/outputs/local_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$(dirname -- "$SCA_SMOKE_DIR")"
mkdir "$SCA_SMOKE_DIR"
printf 'Smoke directory: %s\n' "$SCA_SMOKE_DIR"
"$SCA_PY" - "$SCA_MANIFEST" "$SCA_LIMIT" "${SCA_SPLIT:-}" "$SCA_READERS" "$SCA_CONFIG" "$SCA_SMOKE_DIR" "$SCA_GPU" <<'PY'
import json, sys
from pathlib import Path
manifest, limit, split, readers, config, output, gpu = sys.argv[1:]
rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
rows = [row for row in rows if not split or row['split'] == split]
if int(limit):
    rows = rows[:int(limit)]
reader_names = readers.split(',')
if not rows or set(reader_names) - {'oracle', 'ocr', 'vlm'} or len(set(reader_names)) != len(reader_names):
    raise SystemExit('Select at least one task and unique valid readers')
metadata = {'manifest': str(Path(manifest).resolve()), 'config': str(Path(config).resolve()),
            'task_count': len(rows), 'readers': reader_names, 'total_runs': len(rows)*len(reader_names),
            'split': split or 'all', 'gpu': gpu, 'model': 'Qwen3-VL-8B-Instruct', 'model_backend': 'api'}
Path(output, 'launch.json').write_text(json.dumps(metadata, indent=2))
print(f"Selected {len(rows)} tasks x {len(reader_names)} readers = {metadata['total_runs']} repair runs.", flush=True)
PY
SCA_USED_MB="$(nvidia-smi --id="$SCA_GPU" --query-gpu=memory.used --format=csv,noheader,nounits)"
if (( SCA_USED_MB > 1000 )); then
  printf 'GPU %s is already in use (%s MB); choose an idle GPU.\n' "$SCA_GPU" "$SCA_USED_MB" >&2
  exit 1
fi
export SCA_LOCAL_KEY
SCA_LOCAL_KEY="$($SCA_PY -c 'import secrets; print(secrets.token_urlsafe(32))')"
export SCA_CODE_MODEL=local-qwen3-vl-8b SCA_VISION_MODEL=local-qwen3-vl-8b
export SCA_CODE_BASE_URL="http://127.0.0.1:$SCA_PORT/v1" SCA_VISION_BASE_URL="http://127.0.0.1:$SCA_PORT/v1"
export SCA_CODE_API_KEY="$SCA_LOCAL_KEY" SCA_VISION_API_KEY="$SCA_LOCAL_KEY"
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy="$NO_PROXY"
CUDA_VISIBLE_DEVICES="$SCA_GPU" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_HOME="$SCA_PROJECT_DIR/.cache/huggingface" TORCH_HOME="$SCA_PROJECT_DIR/.cache/torch" \
  XDG_CACHE_HOME="$SCA_PROJECT_DIR/.cache" TRITON_CACHE_DIR="$SCA_PROJECT_DIR/.cache/triton" \
  CUDA_CACHE_PATH="$SCA_PROJECT_DIR/.cache/cuda" \
  TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
  "$SCA_MODEL_PY" -u scripts/local_model_api.py --model-path "$SCA_MODEL_PATH" --port "$SCA_PORT" \
  --max-context "$SCA_MAX_CONTEXT" --max-output "$SCA_MAX_OUTPUT" \
  > "$SCA_SMOKE_DIR/model_server.log" 2>&1 &
SCA_SERVER_PID=$!
printf '%s\n' "$SCA_SERVER_PID" > "$SCA_SMOKE_DIR/model_server.pid"
trap 'kill "$SCA_SERVER_PID" 2>/dev/null || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$SCA_PY" - "$SCA_PORT" "$SCA_SMOKE_DIR" "$SCA_SERVER_PID" "$SCA_STARTUP_TIMEOUT" <<'PY'
import json, os, sys, time, urllib.request
from pathlib import Path
url = f'http://127.0.0.1:{sys.argv[1]}/health'
directory = Path(sys.argv[2])
pid, timeout = int(sys.argv[3]), float(sys.argv[4])
if timeout <= 0:
    raise SystemExit('SCA_STARTUP_TIMEOUT must be positive')
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
started = time.monotonic()
next_progress = started
while time.monotonic() - started < timeout:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise SystemExit(f'Local model exited during startup; inspect {directory / "model_server.log"}')
    try:
        with opener.open(url, timeout=2) as response:
            data = json.load(response)
        if not data.get('ready'):
            raise ValueError('Model is not ready')
        (directory / 'ready.json').write_text(json.dumps(data))
        print(f'Local model ready after {time.monotonic() - started:.1f}s; starting evaluation.', flush=True)
        break
    except Exception:
        if time.monotonic() >= next_progress:
            print(f'Waiting for local model: {time.monotonic() - started:.0f}s / {timeout:.0f}s; log: {directory / "model_server.log"}', flush=True)
            next_progress = time.monotonic() + 15
        time.sleep(1)
else:
    raise SystemExit(f'Local model did not become ready within {timeout:.0f}s; inspect {directory / "model_server.log"}')
PY
"$SCA_PY" -u -m screenshot_agent evaluate "${SCA_EVALUATE_ARGS[@]}" \
  --output "$SCA_SMOKE_DIR/evaluation"
"$SCA_PY" -m screenshot_agent report --runs "$SCA_SMOKE_DIR/evaluation"
if [[ "${SCA_REQUIRE_VLM_SUCCESS:-1}" == "1" ]]; then
"$SCA_PY" - "$SCA_SMOKE_DIR/evaluation/report.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
if any(r['reader'] == 'vlm' for r in report['records']) and not any(r['reader'] == 'vlm' and r['resolved'] and r['usage']['prompt_tokens'] > 0 for r in report['records']):
    raise SystemExit('No real VLM-to-repair task passed; the report contains failures and must be reviewed.')
PY
fi
if [[ -n "${SCA_CODEOCR_MANIFEST:-}" ]]; then
  "$SCA_PY" -u -m screenshot_agent evaluate-codeocr --manifest "$SCA_CODEOCR_MANIFEST" \
    --config "$SCA_CONFIG" --output "$SCA_SMOKE_DIR/codeocr"
fi
"$SCA_PY" - "$SCA_PORT" "$SCA_SMOKE_DIR" <<'PY'
import json, sys, urllib.request
from pathlib import Path
with urllib.request.urlopen(f'http://127.0.0.1:{sys.argv[1]}/health') as response:
    Path(sys.argv[2], 'final_model_stats.json').write_text(json.dumps(json.load(response), indent=2))
PY
printf 'Real-model smoke report: %s/evaluation/report.md\n' "$SCA_SMOKE_DIR"
