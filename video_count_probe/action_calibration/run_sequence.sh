#!/usr/bin/env bash
# Run one smoke clip, then the twelve-video pilot; stop if either stage fails.
set -euo pipefail
ACTION_SEQUENCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ACTION_SEQUENCE_DIR/../.."
export VIDEO_ACTION_ROOT="${VIDEO_ACTION_ROOT:-/data/zjr/video_count_action_calibration_20261027}"
case "$VIDEO_ACTION_ROOT" in
  /data/zjr/*) ;;
  *) printf '%s\n' 'VIDEO_ACTION_ROOT must be under /data/zjr/' >&2; exit 2 ;;
esac
: "${CUDA_VISIBLE_DEVICES:?Set one physical GPU before starting}"
export TMPDIR="$VIDEO_ACTION_ROOT/cache/tmp"
export TORCH_HOME="$VIDEO_ACTION_ROOT/cache/torch"
export HF_HOME="$VIDEO_ACTION_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_EXTENSIONS_DIR="$VIDEO_ACTION_ROOT/cache/extensions"
export MPLCONFIGDIR="$VIDEO_ACTION_ROOT/cache/matplotlib"
export XDG_CACHE_HOME="$VIDEO_ACTION_ROOT/cache/xdg"
mkdir -p "$VIDEO_ACTION_ROOT/logs" "$TMPDIR" "$TORCH_HOME" "$HF_HUB_CACHE" \
  "$TORCH_EXTENSIONS_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
exec 9>"$VIDEO_ACTION_ROOT/.sequence.lock"
if ! flock -n 9; then
  printf '%s\n' 'Another action-calibration sequence is already active.' >&2
  exit 1
fi
bash video_count_probe/action_calibration/run.sh smoke "$@"
exec bash video_count_probe/action_calibration/run.sh pilot "$@"
