#!/usr/bin/env bash
set -euo pipefail
SCA_MINI_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SCA_MINI_ACTION="${1:-status}"
if [[ $# -gt 0 ]]; then shift; fi
export VLLM_RPC_BASE_PATH=/data/zjr/screenshot_code_agent/workflow_v3_runtime
exec bash "$SCA_MINI_ROOT/scripts/run_harness_compare.sh" "$SCA_MINI_ACTION" \
  --output /data/zjr/screenshot_code_agent/mini_recovery_dev5_20261030_v2 \
  --reuse-prepared /data/zjr/screenshot_code_agent/v4_dev10_20261029 \
  --config "$SCA_MINI_ROOT/configs/mini-recovery-dev5.json" --limit 5 --arms mini \
  --task-ids marshmallow-code__marshmallow-2123,pylint-dev__astroid-984,pyvista__pyvista-4329,pvlib__pvlib-python-1832,pydicom__pydicom-793 \
  --gpus 2,3,6,7 --workers 2 --port 18899 \
  --allow-shared-gpus --wait-for-gpus --gpu-memory-utilization 0.74 \
  --gpu-headroom-mib 1536 --brief "$@"
