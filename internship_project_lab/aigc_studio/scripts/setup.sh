#!/usr/bin/env bash
set -euo pipefail
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SOURCE_DIR/scripts/env.sh"
mkdir -p "$AIGC_DATA_ROOT" "$TMPDIR" "$TORCH_HOME" "$HF_HUB_CACHE" "$TORCH_EXTENSIONS_DIR" "$MPLCONFIGDIR" "$PIP_CACHE_DIR" "$npm_config_cache" "$PLAYWRIGHT_BROWSERS_PATH" "$AIGC_DATA_ROOT/logs" "$AIGC_DATA_ROOT/assets" "$AIGC_DATA_ROOT/checks" "$AIGC_DATA_ROOT/runtime/frontend" "$AIGC_DATA_ROOT/models" "$AIGC_DATA_ROOT/vendor"
if [[ ! -x "$AIGC_DATA_ROOT/venv/bin/python" ]]; then
  python -m venv "$AIGC_DATA_ROOT/venv"
fi
"$AIGC_DATA_ROOT/venv/bin/python" -m pip install --disable-pip-version-check -q -r "$SOURCE_DIR/requirements.txt"
cp "$SOURCE_DIR/frontend/package.json" "$AIGC_DATA_ROOT/runtime/frontend/package.json"
if [[ -f "$SOURCE_DIR/frontend/package-lock.json" ]]; then
  cp "$SOURCE_DIR/frontend/package-lock.json" "$AIGC_DATA_ROOT/runtime/frontend/package-lock.json"
fi
npm install --prefix "$AIGC_DATA_ROOT/runtime/frontend" --no-audit --no-fund
if [[ ! -e "$SOURCE_DIR/frontend/node_modules" ]]; then
  ln -s "$AIGC_DATA_ROOT/runtime/frontend/node_modules" "$SOURCE_DIR/frontend/node_modules"
fi
cp "$AIGC_DATA_ROOT/runtime/frontend/package-lock.json" "$SOURCE_DIR/frontend/package-lock.json"
printf 'Application environment ready: %s\n' "$AIGC_DATA_ROOT/venv"

