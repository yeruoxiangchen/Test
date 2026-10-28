#!/usr/bin/env bash
set -euo pipefail
SCA_HARNESS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCA_HARNESS_ROOT"
export TMPDIR=/data/zjr/screenshot_code_agent/harness_compare_20261027/tmp
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"
if [[ ! -d vendor/mini-swe-agent ]]; then
  git clone --depth 1 --branch v2.4.6 --single-branch \
    https://github.com/SWE-agent/mini-swe-agent.git vendor/mini-swe-agent
fi
.venv/bin/python -m pip install --no-cache-dir -r requirements-harness.lock.txt
PYTHONPATH=src .venv/bin/python -c 'from screenshot_agent.harness import verify_mini; print(verify_mini()["commit"])'
