#!/usr/bin/env bash
export AIGC_DATA_ROOT="${AIGC_DATA_ROOT:-/data/zjr/aigc_studio}"
export TMPDIR="$AIGC_DATA_ROOT/tmp"
export TORCH_HOME="$AIGC_DATA_ROOT/cache/torch"
export HF_HOME="$AIGC_DATA_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_EXTENSIONS_DIR="$AIGC_DATA_ROOT/cache/torch_extensions"
export MPLCONFIGDIR="$AIGC_DATA_ROOT/cache/matplotlib"
export XDG_CACHE_HOME="$AIGC_DATA_ROOT/cache"
export PIP_CACHE_DIR="$AIGC_DATA_ROOT/cache/pip"
export npm_config_cache="$AIGC_DATA_ROOT/cache/npm"
export PLAYWRIGHT_BROWSERS_PATH="$AIGC_DATA_ROOT/cache/playwright"
export PYTHONDONTWRITEBYTECODE=1
export AIGC_WEB_ROOT="$AIGC_DATA_ROOT/web-dist"

