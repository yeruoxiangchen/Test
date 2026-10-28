#!/usr/bin/env bash
# User-invoked, single-attempt GPU 7 preparation and training handoff.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd -- "$WORKSPACE"
PYTHON=/home/zjr/anaconda3/envs/reconviagen/bin/python
RUN="$WORKSPACE/pose_aligned_layered_reconstruction/outputs/round2_v2_formal2k_20261018"
SERVICE=tracker-palr-v2-formal2k-20261018.service
MODULE=pose_aligned_layered_reconstruction.experiments.round2_v2.formal2k_runtime_memory
PROBE="$RUN/memory_probe/probe_largest.py"
PROBE_OUTPUT="$RUN/memory_probe/cap22_gpu7_envfixed"
PROBE_REPORT="$PROBE_OUTPUT/report.json"
PROBE_LOG="$RUN/memory_probe/cap22_gpu7_envfixed.log"

exec 9>"$RUN/manual_gpu7_launch.lock"
if ! flock -n 9; then
    echo '已有另一个手动启动命令正在运行；本次退出。' >&2
    exit 1
fi

check_service() {
    local service_state
    if ! service_state="$(systemctl --user show "$SERVICE" --property=ActiveState --value)"; then
        echo '无法查询用户后台服务状态；本次不启动。' >&2
        exit 1
    fi
    case "$service_state" in
        active|activating|reloading|deactivating)
            echo "训练服务当前为 $service_state；请直接监视现有服务，本脚本不会停止它。" >&2
            exit 1
            ;;
        inactive|failed) ;;
        *) echo "无法确认服务已停止：$service_state" >&2; exit 1 ;;
    esac
}
check_service

export PATH="/home/zjr/anaconda3/envs/reconviagen/bin:$PATH"
export CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 SPCONV_ALGO=native
export CUDA_HOME=/usr/local/cuda-12.6 TORCH_CUDA_ARCH_LIST=8.6
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if ! command -v ninja >/dev/null 2>&1; then
    echo 'reconviagen 环境的 PATH 中找不到 ninja；修复环境后再手动启动。' >&2
    exit 1
fi

if [[ ! -f "$PROBE_REPORT" ]]; then
    if [[ -e "$PROBE_OUTPUT" || -e "$PROBE_LOG" ]]; then
        echo "显存探测已有未完成记录，请先检查：$PROBE_LOG" >&2
        echo '本脚本不会覆盖记录或自动重试。' >&2
        exit 1
    fi
    "$PYTHON" - <<'PY'
from pose_aligned_layered_reconstruction.experiments.round2_v2.formal2k_runtime_memory import (
    admission_ready, gpu_usage, MINIMUM_FREE_MIB,
)
usage = gpu_usage(7)
if not admission_ready(usage, {'minimum_free_mib': MINIMUM_FREE_MIB}):
    raise SystemExit(f'GPU 7 尚不满足空闲准入条件：{usage}；未启动探测，请稍后手动重试。')
PY
    echo "开始一次 22 GiB 显存探测，日志：$PROBE_LOG"
    if "$PYTHON" -u "$PROBE" --data-dir "$RUN/data" --output-dir "$PROBE_OUTPUT" \
        --allocator-cap-mib 22528 --minimum-free-mib 23552 --gpu 7 >"$PROBE_LOG" 2>&1; then
        echo '探测进程正常退出，正在核对结果。'
    else
        code=$?
        echo "显存探测失败（退出码 $code）；未启动训练。查看：$PROBE_LOG" >&2
        tail -n 15 "$PROBE_LOG" >&2
        exit "$code"
    fi
fi

"$PYTHON" - "$RUN" "$PROBE_REPORT" <<'PY'
import sys
from pathlib import Path
from pose_aligned_layered_reconstruction.experiments.round2_v2.formal2k_runtime_memory import (
    read, verify_memory_probe,
)
root, report = map(Path, sys.argv[1:])
try:
    identity = read(root / 'data/plan.json')['identity_hash']
    verify_memory_probe(report, identity)
except Exception as error:
    print(f'显存探测尚未通过或证据不匹配：{error}\n报告：{report}\n未冻结新修订，未启动训练。', file=sys.stderr)
    raise SystemExit(1)
print('对象 35 / 1918、GPU 7、22 GiB 上限及数据身份已验证。')
PY

check_service
if [[ ! -f "$RUN/runtime_amendments/002_gpu7_memory.json" ]]; then
    "$PYTHON" -u -m "$MODULE" --run-dir "$RUN" freeze-amendment \
        --memory-probe-report "$PROBE_REPORT"
fi
"$PYTHON" - "$RUN" <<'PY'
import sys
from pathlib import Path
from pose_aligned_layered_reconstruction.experiments.round2_v2.formal2k_runtime_memory import verify_protocol
verify_protocol(Path(sys.argv[1]))
print('原协议、001、002 修订及冻结源码校验通过。')
PY

check_service
if [[ "$(systemctl --user show "$SERVICE" --property=ActiveState --value)" == failed ]]; then
    systemctl --user reset-failed "$SERVICE"
fi
systemd-run --user --unit=tracker-palr-v2-formal2k-20261018 \
    --property="WorkingDirectory=$WORKSPACE" --property=Restart=no \
    --property=KillMode=control-group --property=TimeoutStopSec=90 \
    "$PYTHON" -u -m "$MODULE" --run-dir "$RUN" run --resume
echo "已提交单卡后台服务：$SERVICE"
echo "监视：$PYTHON -m pose_aligned_layered_reconstruction.experiments.round2_v2.formal2k_run --run-dir $RUN monitor --watch 10"
