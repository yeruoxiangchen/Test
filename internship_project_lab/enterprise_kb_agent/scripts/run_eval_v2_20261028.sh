#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
action="${1:-status}"
if [[ $# -gt 0 ]]; then shift; fi
exec bash "$PROJECT_DIR/scripts/run.sh" eval-queue "$action" --output "${KB_DATA_DIR:-/data/zjr/enterprise_kb_agent}/artifacts/git_support_v1/dev_compare_v2_20261028" "$@"
