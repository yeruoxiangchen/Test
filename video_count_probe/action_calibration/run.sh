#!/usr/bin/env bash
set -euo pipefail
ACTION_SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ACTION_SOURCE_DIR/../.."
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
ACTION_PYTHON="${VIDEO_COUNT_PYTHON:-$PWD/video_count_probe/.venv/bin/python}"
exec "$ACTION_PYTHON" -B -u -m video_count_probe.action_calibration.experiment "$@"
