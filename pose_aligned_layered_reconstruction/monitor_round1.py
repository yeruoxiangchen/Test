"""Read-only Chinese progress display; --watch 5 refreshes every five seconds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from .background_round1 import ROOT


STAGES = {
    "adopting_existing_work": "接管已有任务", "overfit_training": "32对象过拟合训练",
    "overfit_final_evaluation": "32对象最终过拟合评估", "prepare_formal_data": "准备/校验2000对象缓存",
    "prepare_dev64": "准备Dev64数据", "formal_startup_recovery": "恢复正式训练启动",
    "formal_lari_training": "正式LaRI训练（含启动校验）", "formal_ss_training": "正式SS训练（含启动校验）",
    "formal_dev64_evaluation": "正式Dev64配对评估", "complete": "整轮实验完成",
}
STATES = {"running": "运行中", "blocked": "失败暂停，尚未完成", "stopped": "已停止，尚未完成", "complete": "整轮完成"}


def last_row(path):
    if not path.is_file():
        return {}
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 8192))
        rows = stream.read().splitlines()
    for row in reversed(rows):
        try:
            return json.loads(row)
        except json.JSONDecodeError:
            continue
    return {}


def render(status, root=ROOT):
    progress = status.get("progress", {})
    stage, state = status.get("stage", "unknown"), status.get("state", "unknown")
    lines = [f"总状态：{STATES.get(state, state)}",
             f"当前阶段：{STAGES.get(stage, stage)}",
             f"后台心跳：{status.get('updated_utc', '?')}（约30秒更新，下面训练文件直接读取）", "",
             f"训练数据：{progress.get('prepared_train2000_objects', 0)} / 2000 个（数量齐不等于校验结束）",
             f"过拟合评估：{progress.get('overfit32_candidate2000_records', 0)} / 64 条（32对象×2 seeds）"]
    gate_path = root / "overfit32_gate_final.json"
    if gate_path.is_file():
        gate = json.loads(gate_path.read_text())
        lines.append("过拟合检查：" + ("已通过" if gate.get("formal_training_allowed") else "未通过"))
    lines.append("Dev64数据：" + ("已准备" if (root / "data_v2/dev64.json").is_file() else "尚未准备完"))
    for key, label, default, budget in (("lari", "正式LaRI", "formal2000_lari_v1", 4000),
                                        ("ss", "正式SS", "formal2000_ss_v1", 3000)):
        name = status.get("formal_runs", {}).get(key, default)
        directory = root / name
        row = last_row(directory / "history.jsonl")
        step = row.get("step", 0)
        startup = [json.loads(p.read_text()) for p in directory.glob("startup_rank*.json")]
        if step:
            detail = f"{step} / {budget} updates（{100 * step / budget:.1f}%）"
        elif startup:
            checked = sum(p.get("verified_objects", 0) for p in startup)
            detail = f"0 / {budget} updates；启动校验 {checked} / 2000 个"
        elif directory.exists():
            detail = f"0 / {budget} updates；初始化中或启动失败，请结合总状态"
        else:
            detail = f"0 / {budget} updates；尚未开始"
        lines.append(f"{label}：{detail} [{name}]")
    count = sum(1 for _ in (root / "formal_dev64_eval/records").glob("*.json"))
    lines.append(f"正式Dev64评估：{count} / 256 条（64对象×2 seeds×父/候选模型）")
    active = [(name, job) for name, job in status.get("jobs", {}).items()
              if job.get("state") in ("starting", "running")]
    lines += ["", "当前子任务：" + (", ".join(name for name, _ in active) or "无")]
    for name, job in active:
        lines.append(f"  详细日志：{job.get('log', name)}")
    if status.get("error"):
        lines.append("错误：" + status["error"].strip().splitlines()[-1])
    lines += ["", "只有“总状态：整轮完成”才算结束；Ctrl+C 只退出此监视，不会停止后台训练。"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", type=float, default=0., metavar="SECONDS")
    args = parser.parse_args()
    if args.watch and args.watch < 1:
        parser.error("--watch must be at least one second")
    try:
        while True:
            status = json.loads((ROOT / "background_status.json").read_text())
            if args.watch and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(render(status), flush=True)
            if not args.watch:
                return
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("已退出监视；后台任务不受影响。")


if __name__ == "__main__":
    main()
