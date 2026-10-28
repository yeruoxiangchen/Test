#!/usr/bin/env bash
set -euo pipefail
SCA_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCA_PROJECT_DIR"
SCA_PY="$SCA_PROJECT_DIR/.venv/bin/python"
SCA_DATA="$SCA_PROJECT_DIR/datasets/generated/v1"
SCA_REPORT="$SCA_PROJECT_DIR/outputs/validation_$(date -u +%Y%m%dT%H%M%SZ)"
SCA_ANALYSIS_SCRIPTS=(scripts/archive_evaluation.py scripts/audit_csv_fields.py scripts/archive_official.py
  scripts/prepare_official.py scripts/official_grade.py scripts/audit_screenshot_glyphs.py
  scripts/recover_official_report.py scripts/clean_official_patches.py scripts/analyze_official_failures.py
  scripts/run_codeocr_full.py)
"$SCA_PY" -m ruff check src tests "${SCA_ANALYSIS_SCRIPTS[@]}"
"$SCA_PY" -m compileall -q src "${SCA_ANALYSIS_SCRIPTS[@]}"
bash -n scripts/run_local_smoke.sh scripts/run_local_eval.sh scripts/run_official_eval.sh \
  scripts/run_official_checked.sh scripts/run_codeocr_local.sh scripts/setup_official.sh scripts/setup_codeocr_full.sh
"$SCA_PY" -m screenshot_agent doctor
if [[ "${1:-}" == "--csv-audit" ]]; then
  shift
  "$SCA_PY" scripts/audit_csv_fields.py "$@"
  exit 0
fi
if [[ "${1:-}" == "--public-only" ]]; then
  SCA_PUBLIC_MANIFEST="${SCA_PUBLIC_MANIFEST:-$SCA_PROJECT_DIR/datasets/generated/boltons_v1/tasks.jsonl}"
  "$SCA_PY" -m screenshot_agent evaluate --manifest "$SCA_PUBLIC_MANIFEST" \
    --readers oracle,ocr --model replay --output "$SCA_REPORT"
  "$SCA_PY" -m screenshot_agent report --runs "$SCA_REPORT"
  printf 'Public known-patch replay report: %s/report.md\n' "$SCA_REPORT"
  exit 0
fi
if [[ ! -f "$SCA_DATA/tasks.jsonl" ]]; then
  "$SCA_PY" -m screenshot_agent build-demo-data --output "$SCA_DATA"
fi
"$SCA_PY" -m pytest tests -q
if [[ "${1:-}" == "--tests-only" ]]; then
  exit 0
fi
"$SCA_PY" -m screenshot_agent evaluate --manifest "$SCA_DATA/tasks.jsonl" \
  --readers oracle,ocr --model replay --output "$SCA_REPORT"
"$SCA_PY" -m screenshot_agent report --runs "$SCA_REPORT"
printf 'Validation report: %s/report.md\n' "$SCA_REPORT"
