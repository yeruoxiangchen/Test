#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 MANIFEST OUTPUT_ROOT" >&2
  exit 2
fi

MANIFEST_PATH=$1
SMOKE_ROOT=$2
PYTHON_BIN=${POSE_T2_PYTHON:-/home/zjr/anaconda3/envs/trellis2/bin/python}

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "manifest does not exist: $MANIFEST_PATH" >&2
  exit 2
fi
if [[ -e "$SMOKE_ROOT" ]]; then
  echo "refusing to overwrite existing smoke root: $SMOKE_ROOT" >&2
  exit 2
fi

"$PYTHON_BIN" -m pose_constrained_trellis2.train \
  --stage ss \
  --manifest "$MANIFEST_PATH" \
  --output-dir "$SMOKE_ROOT/ss" \
  --max-objects 1 \
  --max-steps 2 \
  --warmup-steps 1 \
  --save-every 1 \
  --log-every 1

"$PYTHON_BIN" -m pose_constrained_trellis2.train \
  --stage ss \
  --manifest "$MANIFEST_PATH" \
  --output-dir "$SMOKE_ROOT/ss" \
  --resume "$SMOKE_ROOT/ss/checkpoints/last.pt" \
  --max-objects 1 \
  --max-steps 3 \
  --warmup-steps 1 \
  --save-every 1 \
  --log-every 1

"$PYTHON_BIN" -m pose_constrained_trellis2.train \
  --stage shape \
  --manifest "$MANIFEST_PATH" \
  --output-dir "$SMOKE_ROOT/shape" \
  --max-objects 1 \
  --max-steps 2 \
  --warmup-steps 1 \
  --save-every 1 \
  --log-every 1

"$PYTHON_BIN" -m pose_constrained_trellis2.train \
  --stage shape \
  --manifest "$MANIFEST_PATH" \
  --output-dir "$SMOKE_ROOT/shape" \
  --resume "$SMOKE_ROOT/shape/checkpoints/last.pt" \
  --max-objects 1 \
  --max-steps 3 \
  --warmup-steps 1 \
  --save-every 1 \
  --log-every 1

"$PYTHON_BIN" -m pose_constrained_trellis2.infer \
  --manifest "$MANIFEST_PATH" \
  --sample-index 0 \
  --ss-checkpoint "$SMOKE_ROOT/ss/checkpoints/last.pt" \
  --shape-checkpoint "$SMOKE_ROOT/shape/checkpoints/last.pt" \
  --ss-steps 1 \
  --shape-steps 1 \
  --output "$SMOKE_ROOT/inference/object.ply"
