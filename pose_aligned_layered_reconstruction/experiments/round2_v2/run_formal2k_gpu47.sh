#!/usr/bin/env bash
# User-invoked GPU 4/7 launch; data preparation uses GPU 4, training uses both.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd -- "$WORKSPACE"
PYTHON=/home/zjr/anaconda3/envs/reconviagen/bin/python
RUN="$WORKSPACE/pose_aligned_layered_reconstruction/outputs/round2_v2_formal2k_20261018"
SERVICE=tracker-palr-v2-formal2k-20261018.service
MODULE=pose_aligned_layered_reconstruction.experiments.round2_v2.formal2k_runtime_gpu47

# Share the existing manual-launch lock to prevent competing launch commands.
exec 9>"$RUN/manual_gpu7_launch.lock"
if ! flock -n 9; then
    echo '已有启动命令正在运行；本次退出。' >&2
    exit 1
fi

check_service() {
    local service_state
    if ! service_state="$(systemctl --user show "$SERVICE" --property=ActiveState --value)"; then
        echo '无法查询用户后台服务状态；本次不启动。' >&2
        exit 1
    fi
    case "$service_state" in
        inactive|failed) ;;
        active|activating|reloading|deactivating)
            echo "服务当前为 $service_state；请监视现有服务，未重复启动。" >&2
            exit 1
            ;;
        *) echo "无法确认服务已停止：$service_state" >&2; exit 1 ;;
    esac
}

check_service
export PATH="/home/zjr/anaconda3/envs/reconviagen/bin:$PATH"
if ! command -v ninja >/dev/null 2>&1; then
    echo 'reconviagen 环境中找不到 ninja；未启动。' >&2
    exit 1
fi

if [[ ! -f "$RUN/memory_fix/regression_gpu7/report.json" ]]; then
    echo '缺少数据内存修复 GPU 回归报告；请完成验收后再运行，未启动。' >&2
    exit 1
fi
if [[ ! -f "$RUN/runtime_amendments/004_gpu47.json" ]]; then
    "$PYTHON" -u -m "$MODULE" --run-dir "$RUN" freeze-amendment \
        --previous-waiting-status "$RUN/migration_gpu47/before_stop_status.json"
fi
"$PYTHON" - "$RUN" <<'PY'
import sys
from pathlib import Path
from pose_aligned_layered_reconstruction.experiments.round2_v2.formal2k_runtime_gpu47 import verify_protocol
verify_protocol(Path(sys.argv[1]))
print('数据内存修复与双卡协议校验通过：GPU 4/7，全局有效 batch=2。')
PY

check_service
if [[ "$(systemctl --user show "$SERVICE" --property=ActiveState --value)" == failed ]]; then
    systemctl --user reset-failed "$SERVICE"
fi
systemd-run --user --unit=tracker-palr-v2-formal2k-20261018 \
    --description='PALR Formal2K memory-fixed data then unchanged GPU 4/7 training' \
    --service-type=exec --property="WorkingDirectory=$WORKSPACE" \
    --property=Restart=no --property=KillMode=control-group --property=TimeoutStopSec=90 \
    "$PYTHON" -u -m "$MODULE" --run-dir "$RUN" run --resume
echo "已提交：$SERVICE；缓存准备使用 GPU 4，四个训练阶段使用 GPU 4/7。"
echo "监视：$PYTHON -m $MODULE --run-dir $RUN monitor --watch 5"
