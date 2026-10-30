#!/usr/bin/env bash
set -euo pipefail
PROBE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROBE_BASE_PYTHON="${VIDEO_COUNT_BASE_PYTHON:-/home/zjr/anaconda3/envs/hunyuan3d2mv/bin/python}"
if [ ! -x "$PROBE_DIR/.venv/bin/python" ]; then
  "$PROBE_BASE_PYTHON" -m venv --system-site-packages "$PROBE_DIR/.venv"
fi
# Inherit installed torch without modifying the source Conda environment.
"$PROBE_DIR/.venv/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
  'diffusers==0.35.1' 'transformers==4.49.0' 'accelerate==1.4.0' \
  'huggingface-hub==0.36.2' 'ftfy==6.3.1' 'sentencepiece==0.2.0' 'imageio-ffmpeg==0.6.0'
cd "$PROBE_DIR/.."
"$PROBE_DIR/.venv/bin/python" -u -m video_count_probe.cli check --without-model

