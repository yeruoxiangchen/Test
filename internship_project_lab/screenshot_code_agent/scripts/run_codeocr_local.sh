#!/usr/bin/env bash
set -euo pipefail
SCA_READING_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCA_READING_ROOT"
SCA_GPU="${SCA_GPU:-7}"
SCA_PORT="${SCA_PORT:-18767}"
SCA_RUN_DIR="${SCA_RUN_DIR:-$SCA_READING_ROOT/outputs/codeocr_$(date -u +%Y%m%dT%H%M%SZ)}"
SCA_MANIFEST="${SCA_CODEOCR_MANIFEST:-datasets/generated/codeocr_v3_unicode/images.jsonl}"
SCA_CONFIG="${SCA_CONFIG:-configs/official-local.json}"
SCA_MAX_CONTEXT="${SCA_MAX_CONTEXT:-8192}"
SCA_MAX_OUTPUT="${SCA_MAX_OUTPUT:-2048}"
SCA_SHARD_INDEX="${SCA_SHARD_INDEX:-0}"
SCA_SHARD_COUNT="${SCA_SHARD_COUNT:-1}"
mkdir -p "$(dirname -- "$SCA_RUN_DIR")"
mkdir "$SCA_RUN_DIR"
printf 'CodeOCR directory: %s\n' "$SCA_RUN_DIR"
SCA_GPU_USED="$(nvidia-smi --id="$SCA_GPU" --query-gpu=memory.used --format=csv,noheader,nounits)"
if (( SCA_GPU_USED > 1000 )); then
  printf 'GPU %s is busy (%s MB).\n' "$SCA_GPU" "$SCA_GPU_USED" >&2
  exit 1
fi
export SCA_LOCAL_KEY
SCA_LOCAL_KEY="$(.venv/bin/python -c 'import secrets;print(secrets.token_urlsafe(32))')"
export SCA_VISION_API_KEY="$SCA_LOCAL_KEY" SCA_CODE_API_KEY="$SCA_LOCAL_KEY"
export SCA_VISION_MODEL=local-qwen3-vl-8b SCA_CODE_MODEL=local-qwen3-vl-8b
export SCA_VISION_BASE_URL="http://127.0.0.1:$SCA_PORT/v1" SCA_CODE_BASE_URL="http://127.0.0.1:$SCA_PORT/v1"
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy="$NO_PROXY"
CUDA_VISIBLE_DEVICES="$SCA_GPU" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
  HF_HOME="$SCA_READING_ROOT/.cache/huggingface" TORCH_HOME="$SCA_READING_ROOT/.cache/torch" \
  XDG_CACHE_HOME="$SCA_READING_ROOT/.cache" TRITON_CACHE_DIR="$SCA_READING_ROOT/.cache/triton" \
  CUDA_CACHE_PATH="$SCA_READING_ROOT/.cache/cuda" TOKENIZERS_PARALLELISM=false \
  .model-env/bin/python -u scripts/local_model_api.py --model-path /data/zjr/models/Qwen3-VL-8B-Instruct \
  --port "$SCA_PORT" --max-context "$SCA_MAX_CONTEXT" --max-output "$SCA_MAX_OUTPUT" > "$SCA_RUN_DIR/model_server.log" 2>&1 &
SCA_READING_PID=$!
printf '%s\n' "$SCA_READING_PID" > "$SCA_RUN_DIR/model_server.pid"
sca_reading_cleanup() {
  local sca_status=$?
  trap - EXIT
  set +e
  if [[ -f "$SCA_RUN_DIR/ready.json" ]]; then
    .venv/bin/python - "$SCA_PORT" "$SCA_RUN_DIR" <<'PY'
import json,sys,urllib.request
from pathlib import Path
try:
    with urllib.request.urlopen(f'http://127.0.0.1:{sys.argv[1]}/health',timeout=10) as response:
        Path(sys.argv[2],'final_model_stats.json').write_text(json.dumps(json.load(response),indent=2))
except Exception as error:
    Path(sys.argv[2],'stats_error.json').write_text(json.dumps({'error':str(error)}))
PY
  fi
  kill "$SCA_READING_PID" 2>/dev/null || true
  wait "$SCA_READING_PID" 2>/dev/null || true
  .venv/bin/python - "$SCA_RUN_DIR" "$sca_status" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1],'launcher_status.json').write_text(json.dumps({'exit_code':int(sys.argv[2]),'model_stopped':True}))
PY
  exit "$sca_status"
}
trap sca_reading_cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
.venv/bin/python - "$SCA_PORT" "$SCA_READING_PID" "$SCA_RUN_DIR" <<'PY'
import json,os,sys,time,urllib.request
from pathlib import Path
port,pid,output=sys.argv[1:]
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
started=time.monotonic()
while time.monotonic()-started<900:
    os.kill(int(pid),0)
    try:
        with opener.open(f'http://127.0.0.1:{port}/health',timeout=2) as response:
            state=json.load(response)
        if state.get('ready'):
            Path(output,'ready.json').write_text(json.dumps(state));break
    except Exception:
        pass
    print(f'Waiting for CodeOCR model: {time.monotonic()-started:.0f}s',flush=True)
    time.sleep(15)
else:
    raise SystemExit('Model startup timed out')
PY
.venv/bin/python - "$SCA_CONFIG" "$SCA_RUN_DIR" "$SCA_GPU" "$SCA_MANIFEST" "$SCA_SHARD_INDEX" "$SCA_SHARD_COUNT" "$SCA_MAX_CONTEXT" "$SCA_MAX_OUTPUT" <<'PY'
import json,sys
from pathlib import Path
from screenshot_agent.config import load_config
config,out,gpu,manifest,index,count,context,max_output=sys.argv[1:]
rows=[json.loads(s) for s in Path(manifest).read_text().splitlines() if s.strip()]
ids=sorted({r['source_id'] for r in rows})[int(index)::int(count)]
rows=[r for r in rows if r['source_id'] in set(ids)]
Path(out,'config.json').write_text(load_config(Path(config)).model_dump_json(indent=2))
Path(out,'launch.json').write_text(json.dumps({'gpu':gpu,'manifest':str(Path(manifest).resolve()),
    'max_context':int(context),'max_output':int(max_output),'source_files':len(ids),'image_variants':len(rows),
    'reading_attempts':len(rows)*2,'shard_index':int(index),'shard_count':int(count)},indent=2))
PY
.venv/bin/python -u -m screenshot_agent evaluate-codeocr --manifest "$SCA_MANIFEST" \
  --config "$SCA_CONFIG" --output "$SCA_RUN_DIR/codeocr" \
  --shard-index "$SCA_SHARD_INDEX" --shard-count "$SCA_SHARD_COUNT"
