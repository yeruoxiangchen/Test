#!/usr/bin/env bash
set -euo pipefail
DURATION_SEQUENCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DURATION_SEQUENCE_ROOT="${WAN22_DURATION_ROOT:-/data/zjr/video_count_wan22_duration_20261029}"
case "$DURATION_SEQUENCE_ROOT" in
  /data/zjr/*) ;;
  *) printf 'Expected WAN22_DURATION_ROOT under /data/zjr\n' >&2; exit 2 ;;
esac
mkdir -p "$DURATION_SEQUENCE_ROOT/logs" "$DURATION_SEQUENCE_ROOT/tmp"
export TMPDIR="$DURATION_SEQUENCE_ROOT/tmp"
export TORCH_HOME="$DURATION_SEQUENCE_ROOT/cache/torch"
export HF_HOME="$DURATION_SEQUENCE_ROOT/cache/huggingface"
export HF_HUB_CACHE="$DURATION_SEQUENCE_ROOT/cache/huggingface/hub"
export TORCH_EXTENSIONS_DIR="$DURATION_SEQUENCE_ROOT/cache/torch_extensions"
export MPLCONFIGDIR="$DURATION_SEQUENCE_ROOT/cache/matplotlib"
export XDG_CACHE_HOME="$DURATION_SEQUENCE_ROOT/cache/xdg"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec 9>"$DURATION_SEQUENCE_ROOT/.sequence.lock"
if ! flock -n 9; then
  printf 'Another duration sequence is already running in %s\n' "$DURATION_SEQUENCE_ROOT" >&2
  exit 75
fi
duration_sequence_exit() {
  local duration_sequence_code=$?
  trap - EXIT
  printf '%s sequence_exit=%s\n' "$(date -u +%FT%TZ)" "$duration_sequence_code"
  exit "$duration_sequence_code"
}
trap duration_sequence_exit EXIT
printf '%s sequence_started pid=%s gpu=%s root=%s\n' "$(date -u +%FT%TZ)" "$$" "${CUDA_VISIBLE_DEVICES:-unset}" "$DURATION_SEQUENCE_ROOT"
for duration_sequence_stage in encode smoke pilot; do
  printf '%s stage_start=%s\n' "$(date -u +%FT%TZ)" "$duration_sequence_stage"
  bash "$DURATION_SEQUENCE_DIR/run.sh" "$duration_sequence_stage" --root "$DURATION_SEQUENCE_ROOT"
  printf '%s stage_complete=%s\n' "$(date -u +%FT%TZ)" "$duration_sequence_stage"
done
