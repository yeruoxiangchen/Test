#!/usr/bin/env bash
set -euo pipefail
PROBE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROBE_DIR/.."
PROBE_MODE="${1:-help}"
if [ "$#" -gt 0 ]; then shift; fi
PROBE_PYTHON="${VIDEO_COUNT_PYTHON:-$PROBE_DIR/.venv/bin/python}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
case "$PROBE_MODE" in
  setup)
    exec bash "$PROBE_DIR/setup.sh" "$@" ;;
  prepare|report|review)
    # Plans and reports use only the standard library.
    exec "${VIDEO_COUNT_PLAN_PYTHON:-python}" -u -m video_count_probe.cli "$PROBE_MODE" "$@" ;;
  test)
    exec "$PROBE_PYTHON" -u -m unittest discover -s video_count_probe/tests -v "$@" ;;
  download|check|smoke|pilot)
    if [ ! -x "$PROBE_PYTHON" ]; then
      echo "Missing experiment environment. Run: bash video_count_probe/run.sh setup" >&2
      exit 2
    fi
    exec "$PROBE_PYTHON" -u -m video_count_probe.cli "$PROBE_MODE" "$@" ;;
  *)
    echo 'Usage: bash video_count_probe/run.sh {setup|test|prepare|download|check|smoke|pilot|review|report} [options]'
    echo 'Defaults: data/output=/data/zjr/video_count_probe_20261025; weights=/data/zjr/models/Wan2.1-T2V-1.3B-Diffusers'
    exit 2 ;;
esac

