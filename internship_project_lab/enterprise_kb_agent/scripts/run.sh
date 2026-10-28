#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export KB_DATA_DIR="${KB_DATA_DIR:-/data/zjr/enterprise_kb_agent}"
export TMPDIR="$KB_DATA_DIR/tmp"
export TORCH_HOME="$KB_DATA_DIR/cache/torch"
export HF_HOME="$KB_DATA_DIR/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_EXTENSIONS_DIR="$KB_DATA_DIR/cache/torch_extensions"
export MPLCONFIGDIR="$KB_DATA_DIR/cache/matplotlib"
export XDG_CACHE_HOME="$KB_DATA_DIR/cache"
export PIP_CACHE_DIR="$KB_DATA_DIR/cache/pip"
export PLAYWRIGHT_BROWSERS_PATH="$KB_DATA_DIR/cache/browsers"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
umask 077
mkdir -p "$TMPDIR" "$KB_DATA_DIR/logs" "$KB_DATA_DIR/artifacts"
if [[ -f "$KB_DATA_DIR/model.env" ]]; then
  set -a
  source "$KB_DATA_DIR/model.env"
  set +a
fi
if [[ -f "$KB_DATA_DIR/private.env" ]]; then
  set -a
  source "$KB_DATA_DIR/private.env"
  set +a
fi
if [[ -f "$KB_DATA_DIR/judge.env" ]]; then
  set -a
  source "$KB_DATA_DIR/judge.env"
  set +a
fi
PYTHON="$KB_DATA_DIR/venv/bin/python"
cd "$PROJECT_DIR"
ACTION="${1:-help}"
shift || true
case "$ACTION" in
  setup)
    "${KB_BOOTSTRAP_PYTHON:-python3}" -m venv "$KB_DATA_DIR/venv"
    if [[ -f requirements.lock ]]; then
      "$PYTHON" -m pip install -r requirements.lock
    else
      "$PYTHON" -m pip install -e '.[dev]'
    fi
    "$PYTHON" -m pip check
    "$PYTHON" -m pip freeze > "$KB_DATA_DIR/artifacts/installed-requirements.txt"
    ;;
  init|demo|user|evaluate|doctor|start|stop|status)
    exec "$PYTHON" -m kb.cli "$ACTION" "$@"
    ;;
  serve)
    exec "$PYTHON" -m uvicorn kb.app:app --host "${KB_HOST:-127.0.0.1}" --port "${KB_PORT:-8768}" --workers 1 "$@"
    ;;
  test)
    exec "$PYTHON" -m pytest --basetemp "$KB_DATA_DIR/tmp/pytest" "$@"
    ;;
  benchmark)
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
    exec nice -n 10 "$PYTHON" -m kb.benchmark "$@"
    ;;
  model)
    exec "$PYTHON" scripts/model_service.py "$@"
    ;;
  judge-model)
    exec "$PYTHON" scripts/model_service.py --role judge "$@"
    ;;
  judge-packages)
    exec "$PYTHON" scripts/setup_judge_packages.py
    ;;
  reranker-setup)
    export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 HF_HUB_DISABLE_XET=1
    exec "${KB_RERANK_PYTHON:-/data/zjr/deepsearch_rl/venv/bin/python}" -B scripts/reranker_setup.py "$@"
    ;;
  eval-queue)
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
    exec nice -n 10 "$PYTHON" scripts/eval_queue.py "$@"
    ;;
  eval-v3)
    export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
    exec nice -n 10 "$PYTHON" scripts/eval_v3.py "$@"
    ;;
  semantic-v4)
    exec nice -n 10 "$PYTHON" scripts/semantic_v4.py "$@"
    ;;
  eval-v4)
    export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
    exec nice -n 10 "$PYTHON" scripts/eval_v4.py "$@"
    ;;
  browser-test)
    exec "$PYTHON" scripts/browser_smoke.py "$@"
    ;;
  browser-setup)
    "$PYTHON" -m playwright install chromium
    exec "$PYTHON" scripts/browser_fonts.py
    ;;
  *)
    echo 'Usage: bash scripts/run.sh {setup|init|demo|user|doctor|test|evaluate|benchmark|model|eval-queue|serve|start|stop|status|browser-test}'
    ;;
esac
