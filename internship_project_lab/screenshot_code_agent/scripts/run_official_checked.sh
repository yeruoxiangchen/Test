#!/usr/bin/env bash
set -euo pipefail
SCA_CHECKED_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCA_CHECKED_ROOT"
.venv/bin/python scripts/audit_screenshot_glyphs.py repair --dataset datasets/generated/swebench_lite_v3
export SCA_CODEOCR_MANIFEST="${SCA_CODEOCR_MANIFEST:-datasets/generated/codeocr_v3_unicode/images.jsonl}"
export SCA_RUN_DIR="${SCA_RUN_DIR:-$SCA_CHECKED_ROOT/outputs/official_$(date -u +%Y%m%dT%H%M%SZ)}"
bash scripts/run_official_eval.sh
.venv/bin/python scripts/clean_official_patches.py --run "$SCA_RUN_DIR"
