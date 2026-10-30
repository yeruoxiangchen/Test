#!/usr/bin/env bash
set -euo pipefail
PHASE_CONTROL_SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PHASE_CONTROL_SOURCE_DIR/../.."
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
PHASE_CONTROL_PYTHON="${VIDEO_COUNT_PYTHON:-$PWD/video_count_probe/.venv/bin/python}"
exec "$PHASE_CONTROL_PYTHON" -B -u -m video_count_probe.wan22_phase_control.cli "$@"
