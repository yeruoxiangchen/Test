#!/usr/bin/env bash
set -euo pipefail
COUNT_RESPONSE_SEQUENCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COUNT_RESPONSE_SEQUENCE_ROOT="${WAN22_COUNT_RESPONSE_ROOT:-/data/zjr/video_count_wan22_count_response_20261029}"
case "$COUNT_RESPONSE_SEQUENCE_ROOT" in
  /data/zjr/*) ;;
  *) printf 'Expected WAN22_COUNT_RESPONSE_ROOT under /data/zjr\n' >&2; exit 2 ;;
esac
mkdir -p "$COUNT_RESPONSE_SEQUENCE_ROOT/logs" "$COUNT_RESPONSE_SEQUENCE_ROOT/tmp"
export TMPDIR="$COUNT_RESPONSE_SEQUENCE_ROOT/tmp"
export TORCH_HOME="$COUNT_RESPONSE_SEQUENCE_ROOT/cache/torch"
export HF_HOME="$COUNT_RESPONSE_SEQUENCE_ROOT/cache/huggingface"
export HF_HUB_CACHE="$COUNT_RESPONSE_SEQUENCE_ROOT/cache/huggingface/hub"
export TORCH_EXTENSIONS_DIR="$COUNT_RESPONSE_SEQUENCE_ROOT/cache/torch_extensions"
export MPLCONFIGDIR="$COUNT_RESPONSE_SEQUENCE_ROOT/cache/matplotlib"
export XDG_CACHE_HOME="$COUNT_RESPONSE_SEQUENCE_ROOT/cache/xdg"
export CUDA_CACHE_PATH="$COUNT_RESPONSE_SEQUENCE_ROOT/cache/cuda"
export TRITON_CACHE_DIR="$COUNT_RESPONSE_SEQUENCE_ROOT/cache/triton"
mkdir -p "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec 9>"$COUNT_RESPONSE_SEQUENCE_ROOT/.sequence.lock"
if ! flock -n 9; then
  printf 'Another count response sequence is already running in %s\n' "$COUNT_RESPONSE_SEQUENCE_ROOT" >&2
  exit 75
fi
count_response_sequence_exit() {
  local count_response_sequence_code=$?
  trap - EXIT
  printf '%s sequence_exit=%s\n' "$(date -u +%FT%TZ)" "$count_response_sequence_code"
  exit "$count_response_sequence_code"
}
trap count_response_sequence_exit EXIT
printf '%s sequence_started pid=%s gpu=%s root=%s\n' "$(date -u +%FT%TZ)" "$$" "${CUDA_VISIBLE_DEVICES:-unset}" "$COUNT_RESPONSE_SEQUENCE_ROOT"
for count_response_sequence_stage in encode smoke pilot; do
  printf '%s stage_start=%s\n' "$(date -u +%FT%TZ)" "$count_response_sequence_stage"
  bash "$COUNT_RESPONSE_SEQUENCE_DIR/run.sh" "$count_response_sequence_stage" --root "$COUNT_RESPONSE_SEQUENCE_ROOT"
  printf '%s stage_complete=%s\n' "$(date -u +%FT%TZ)" "$count_response_sequence_stage"
done
