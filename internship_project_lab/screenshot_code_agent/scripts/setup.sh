#!/usr/bin/env bash
set -euo pipefail
SCA_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCA_PROJECT_DIR"
SCA_UPSTREAM_COMMIT=bd64cfe7ad2789fa0ab1b3a689ad160567b5cdb0
if [[ ! -d vendor/CoreCoder/.git ]]; then
  git clone --no-checkout https://github.com/he-yufeng/CoreCoder.git vendor/CoreCoder
  git -C vendor/CoreCoder checkout --detach "$SCA_UPSTREAM_COMMIT"
fi
SCA_ACTUAL_COMMIT="$(git -C vendor/CoreCoder rev-parse HEAD)"
if [[ "$SCA_ACTUAL_COMMIT" != "$SCA_UPSTREAM_COMMIT" ]]; then
  printf 'CoreCoder revision differs from UPSTREAM.json: %s\n' "$SCA_ACTUAL_COMMIT" >&2
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install --cache-dir .cache/pip -c requirements.lock.txt -c requirements-benchmark.lock.txt \
  -e vendor/CoreCoder -e '.[dev,ocr,benchmark]'
printf 'Installed isolated environment: %s/.venv\n' "$SCA_PROJECT_DIR"
