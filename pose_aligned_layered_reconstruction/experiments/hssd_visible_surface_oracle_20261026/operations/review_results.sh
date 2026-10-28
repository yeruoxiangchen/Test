#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../../.." && pwd)"
cd "$ROOT"
DATA_ROOT=/data/zjr/pose_aligned_layered_reconstruction
export TMPDIR="$DATA_ROOT/tmp" TORCH_HOME="$DATA_ROOT/cache/torch"
export HF_HOME="$DATA_ROOT/cache/huggingface" HF_HUB_CACHE="$DATA_ROOT/cache/huggingface/hub"
export TORCH_EXTENSIONS_DIR="$DATA_ROOT/cache/torch_extensions"
export MPLCONFIGDIR="$DATA_ROOT/cache/matplotlib" XDG_CACHE_HOME="$DATA_ROOT/cache/xdg"
export NUMBA_CACHE_DIR="$DATA_ROOT/cache/numba"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_THREADING_LAYER=GNU
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
PYTHON=/home/zjr/anaconda3/envs/reconviagen/bin/python
case "${1:-all}" in
  statistics) exec "$PYTHON" "$HERE/review_statistics.py" --write ;;
  integrity) exec "$PYTHON" "$HERE/review_integrity.py" --write ;;
  finalize) exec "$PYTHON" "$HERE/finalize_review.py" ;;
  all)
    "$PYTHON" "$HERE/review_statistics.py" --write
    "$PYTHON" "$HERE/review_integrity.py" --write
    exec "$PYTHON" "$HERE/finalize_review.py"
    ;;
  *) exit 2 ;;
esac
