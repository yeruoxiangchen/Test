#!/usr/bin/env bash
set -euo pipefail
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SOURCE_DIR/scripts/env.sh"
COMFY_REVISION=54a4e56ad1cabd17871068f160d0e75a1557db34
COMFY_DIR="$AIGC_DATA_ROOT/vendor/ComfyUI"
BASE_PYTHON="${AIGC_TORCH_PYTHON:-/home/zjr/anaconda3/envs/chartvlm/bin/python}"
if [[ ! -x "$BASE_PYTHON" ]]; then
  printf 'Set AIGC_TORCH_PYTHON to an existing Python with CUDA-enabled PyTorch.\n'
  exit 1
fi
mkdir -p "$AIGC_DATA_ROOT/vendor" "$AIGC_DATA_ROOT/models" "$AIGC_DATA_ROOT/run"
if [[ ! -d "$COMFY_DIR/.git" ]]; then
  git init "$COMFY_DIR"
  git -C "$COMFY_DIR" remote add origin https://github.com/comfyanonymous/ComfyUI.git
  git -C "$COMFY_DIR" fetch --depth=1 origin "$COMFY_REVISION"
  git -C "$COMFY_DIR" checkout --detach FETCH_HEAD
fi
if [[ "$(git -C "$COMFY_DIR" rev-parse HEAD)" != "$COMFY_REVISION" ]]; then
  printf 'Existing ComfyUI revision differs; leaving it untouched.\n'
  exit 1
fi
if [[ ! -x "$AIGC_DATA_ROOT/comfy-venv/bin/python" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$AIGC_DATA_ROOT/comfy-venv"
fi
# Pin the inherited tensor libraries so pip cannot replace them in the overlay.
"$BASE_PYTHON" -B "$SOURCE_DIR/scripts/torch_constraints.py" "$AIGC_DATA_ROOT/runtime/torch-constraints.txt"
"$AIGC_DATA_ROOT/comfy-venv/bin/python" -m pip install -c "$AIGC_DATA_ROOT/runtime/torch-constraints.txt" -r "$COMFY_DIR/requirements.txt"
"$AIGC_DATA_ROOT/venv/bin/python" -m pip install 'huggingface_hub>=0.36,<1'
"$AIGC_DATA_ROOT/comfy-venv/bin/python" -m pip freeze --local > "$AIGC_DATA_ROOT/runtime/comfy-requirements.lock.txt"
printf 'ComfyUI ready: %s\nRevision: %s\n' "$COMFY_DIR" "$COMFY_REVISION"
