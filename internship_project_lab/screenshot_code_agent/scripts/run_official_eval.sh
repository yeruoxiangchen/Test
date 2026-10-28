#!/usr/bin/env bash
set -euo pipefail
SCA_OFFICIAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCA_OFFICIAL_DIR"
export SCA_MANIFEST="${SCA_MANIFEST:-datasets/generated/swebench_lite_v3/tasks.jsonl}"
export SCA_CONFIG="${SCA_CONFIG:-configs/official-local.json}"
export SCA_LIMIT="${SCA_LIMIT:-0}"
export SCA_REQUIRE_VLM_SUCCESS=0
export SCA_MAX_OUTPUT=2048
export SCA_MAX_CONTEXT=16384
if [[ "${SCA_SKIP_CODEOCR:-0}" == "1" ]]; then
  unset SCA_CODEOCR_MANIFEST
else
  export SCA_CODEOCR_MANIFEST="${SCA_CODEOCR_MANIFEST:-datasets/generated/codeocr_v3/images.jsonl}"
fi
exec bash scripts/run_local_smoke.sh
