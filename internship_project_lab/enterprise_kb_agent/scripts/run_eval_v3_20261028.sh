#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
action="${1:-status}"
if [[ $# -gt 0 ]]; then shift; fi
exec bash "$PROJECT_DIR/scripts/run.sh" eval-v3 "$action" "$@"
