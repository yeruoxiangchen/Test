"""Archive completed evaluation evidence and write a reproducible Chinese analysis.

Optional --wait-seconds lets a CPU-only background job finish the report after
the GPU job. It never sends model requests or changes a task verdict.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from screenshot_agent.control import atomic_json
from screenshot_agent.report import make_report, summarize_rows


def wait_for_completion(root: Path, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        state_file = root / "status.json"
        state = json.loads(state_file.read_text()) if state_file.exists() else {}
        status = state.get("status")
        if status == "complete":
            if state.get("completed") != state.get("total"):
                raise ValueError("Batch claims complete but its completion counters disagree")
            # The batch writes its completion counters BEFORE generating its
            # report. The launcher then generates a report and records peak
            # CUDA usage. Wait for that work instead of racing either writer.
            pid_file = root.parent / "model_server.pid"
            server_alive = False
            if pid_file.exists():
                try:
                    fields = Path(f"/proc/{int(pid_file.read_text())}/stat").read_text().rsplit(")", 1)[1].split()
                    server_alive = fields[0] != "Z"
                except (OSError, ValueError, IndexError):
                    pass
            if (root.parent / "final_model_stats.json").exists() or not server_alive:
                return state
        elif status in {"cancelled", "error", "batch_cost_limit"}:
            raise ValueError(f"Batch terminated with {status}; raw report remains in {root}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Evaluation not complete: {state}; no final report claimed")
        time.sleep(2)


def paired_comparisons(records: list[dict]) -> list[dict]:
    cohorts = collections.defaultdict(dict)
    for row in records:
        key = (row["provenance"], row["split"], row["code_backend"])
        cohorts[key].setdefault(row["task_id"], {})[row["reader"]] = row["resolved"]
    result = []
    for (source, split, backend), tasks in sorted(cohorts.items()):
        for left, right in [("oracle", "ocr"), ("oracle", "vlm"), ("ocr", "vlm")]:
            common = [outcomes for outcomes in tasks.values() if left in outcomes and right in outcomes]
            if not common:
                continue
            result.append({"provenance": source, "split": split, "code_backend": backend,
                           "left": left, "right": right, "paired_tasks": len(common),
                           "both_pass": sum(r[left] and r[right] for r in common),
                           "left_only": sum(r[left] and not r[right] for r in common),
                           "right_only": sum(not r[left] and r[right] for r in common),
                           "both_fail": sum(not r[left] and not r[right] for r in common)})
    return result


def archive(root: Path, output: Path, report_path: Path) -> dict:
    state = json.loads((root / "status.json").read_text())
    if state.get("status") != "complete" or state.get("completed") != state.get("total"):
        raise ValueError("Archive only a fully completed batch; partial raw reports are retained separately")
    report = make_report(root)
    if any(r.get("code_backend") != "api" or not r.get("model_inference") for r in report["records"]):
        raise ValueError("This analysis describes model repair; known-patch replay reports must stay separate")
    if report["finished_runs"] != state["total"] or report["unfinished_or_not_attempted"]:
        raise ValueError("Finished result files do not cover every scheduled run")
    output.mkdir(parents=True, exist_ok=True)
    origin = output / "run_origin.json"
    if origin.exists() and json.loads(origin.read_text())["run_root"] != str(root):
        raise ValueError("This artifact directory belongs to another run; select a new output")
    atomic_json(origin, {"run_root": str(root)})
    report_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "report.json", output / "real_model_report.json")
    shutil.copy2(root / "report.md", output / "real_model_report.md")
    shutil.copy2(root / "status.json", output / "batch_status.json")
    for name in ("launch.json", "ready.json", "final_model_stats.json"):
        if (root.parent / name).exists():
            shutil.copy2(root.parent / name, output / name)
    job = json.loads((root / "batch_job.json").read_text())
    launch_file = root.parent / "launch.json"
    launch = json.loads(launch_file.read_text()) if launch_file.exists() else {}
    atomic_json(output / "execution_config.json", job["config"])
    profiles, data_roots = set(), set()
    for row in report["records"]:
        directory = Path(row["run_dir"])
        config = directory / "config.json"
        profiles.add(hashlib.sha256(config.read_bytes()).hexdigest())
        task = json.loads((directory / "task.json").read_text())
        data_roots.add(Path(task["repo"]).parent.parent)
        if (row["task_id"].startswith("boltons_") and row["reader"] == "vlm") or not row["resolved"]:
            target = output / f"{row['task_id']}_{row['reader']}"
            target.mkdir(exist_ok=True)
            for name in ("patch.diff", "result.json", "baseline_test.log", "verification_test.log", "events.jsonl", "error.log"):
                if (directory / name).exists():
                    shutil.copy2(directory / name, target / name)
            for name in ("provenance.json",):
                source = Path(task["repo"]).parent / name
                if source.exists():
                    shutil.copy2(source, target / name)
            image = directory / "screenshots/shot.png"
            if image.exists():
                shutil.copy2(image, target / "input.png")
    for i, directory in enumerate(sorted(data_roots)):
        source = directory / "build_report.json"
        if source.exists():
            shutil.copy2(source, output / f"dataset_{i}_build_report.json")
    comparisons = paired_comparisons(report["records"])
    atomic_json(output / "paired_comparisons.json", comparisons)
    trajectories = []
    for row in report["records"]:
        event_path = Path(row["run_dir"]) / "events.jsonl"
        events = [json.loads(line) for line in event_path.read_text().splitlines()] if event_path.exists() else []
        trajectories.append({"task_id": row["task_id"], "reader": row["reader"], "resolved": row["resolved"],
            "tools": dict(collections.Counter(e["name"] for e in events if e["event"] == "tool_start")),
            "model_errors": [e for e in events if e["event"] == "model_error"],
            "tool_errors": [e for e in events if e["event"] == "tool_error"],
            "max_request_prompt_tokens": max((e["response"].get("usage", {}).get("prompt_tokens", 0)
                                               for e in events if e["event"] == "model_response"), default=0)})
    atomic_json(output / "trajectory_summary.json", trajectories)
    project = Path(__file__).resolve().parents[1]
    review = project / "docs/failure_review_v2.md"
    if review.exists() and any(r["task_id"] in {"synthetic_leap", "synthetic_csv_fields"} for r in report["records"]):
        (output / "failure_review.md").write_text(review.read_text().replace("](artifacts/v2/", "]("))
    source_files = [*sorted((project / "src/screenshot_agent").glob("*.py")),
                    *sorted((project / "scripts").glob("*.py")), *sorted((project / "scripts").glob("*.sh")),
                    *sorted((project / "tests").glob("*.py")), project / "configs/local-smoke.json"]
    metadata = {"run_root": str(root), "completed_runs": len(report["records"]),
                "config_profiles": sorted(profiles), "same_config_across_runs": len(profiles) == 1,
                "code_and_vision_models": {k: job["config"][k]["model"] for k in ("code", "vision")},
                "source_sha256": {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in source_files},
                "artifact_sha256": {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(output.rglob("*")) if p.is_file()
                                    and p.name != "verification.json"}}
    atomic_json(output / "verification.json", metadata)
    short = os.path.relpath(output, report_path.parent)
    by_source = summarize_rows(report["records"], ("provenance", "code_backend", "reader"))
    failures = [r for r in report["records"] if not r["resolved"]]
    groups = {r["group_id"] for r in report["records"]}
    tasks = {r["task_id"] for r in report["records"]}
    limits = job["config"]["limits"]
    date = time.strftime("%Y-%m-%d", time.gmtime(state["started_at"]))
    task_definitions = {r["task_id"]: r for r in report["records"]}
    composition = collections.Counter((r["provenance"], r["split"]) for r in task_definitions.values())
    lines = ["# 扩展评测与第一版交付记录", "", f"日期：{date}。所有下述修复成绩均来自真实模型调用。", "",
             f"本轮共 {len(tasks)} 个原始任务、{len(groups)} 个任务或仓库分组、{len(report['records'])} 次运行；"
             f"{sum(r['resolved'] for r in report['records'])} 次通过独立补丁重放和固定测试。",
             "合成微型仓库与公开历史案例分别统计。三组为截图可见内容准确文本、RapidOCR CPU、Qwen3-VL 视觉工具。", "",
             "## 条件与可复现性", "",
             f"- 本机单张 RTX 3090（GPU {launch.get('gpu', '见模型日志')}），本地 Qwen3-VL-8B-Instruct BF16 / SDPA，贪心生成。",
             f"- 三组使用相同代码模型和配置：{job['config']['max_context_tokens']} 上下文，"
             f"单次最多 {job['config']['code']['max_output_tokens']} 新 token；每题 {limits['max_steps']} 次代码调用、"
             f"{limits['max_tool_calls']} 次工具调用、{limits['max_vision_calls']} 次截图读取、"
             f"{limits['max_wall_seconds']} 秒，单命令 {limits['command_timeout_seconds']} 秒。",
             f"- 所有逐题配置文件一致：`{metadata['same_config_across_runs']}`；原始任务、工具、模型、费用和测试事件保留。",
             "- 数据组成：" + "；".join(f"{source} / {split}: {count} 题" for (source, split), count in sorted(composition.items())) + "。",
             "- 所有截图为合成渲染；真实用户截图数为 0。公开案例保留完整历史版本和许可证。",
             "- 报告中的 API 价格为 0，不代表 GPU、时间与电力没有成本。", "",
             "## 按数据来源汇总", "",
             "| 来源 | 读取方式 | 独立验证通过 | 平均秒数 | 平均工具调用 | 平均读取 CER |",
             "|---|---|---:|---:|---:|---:|"]
    for row in by_source:
        source = "公开历史案例" if row["provenance"].startswith("public_") else "自编合成题"
        cer = "n/a" if row["reading_cer_mean"] is None else f"{100*row['reading_cer_mean']:.2f}%"
        lines.append(f"| {source} | {row['reader']} | {row['resolved']}/{row['attempted']} | "
                     f"{row['elapsed_seconds_mean']:.2f} | {row['tool_calls_mean']:.2f} | {cer} |")
    overall = {row["reader"]: row for row in report["summary"]}
    if {"ocr", "vlm"}.issubset(overall):
        left, right = overall["ocr"], overall["vlm"]
        lines += ["", f"全部任务合计：OCR {left['resolved']}/{left['attempted']}，"
                  f"视觉工具 {right['resolved']}/{right['attempted']}。"]
        if left["resolved_fraction"] == right["resolved_fraction"]:
            lines += ["本轮两者总体固定测试通过率相同。"]
            if (right["reading_cer_mean"] is not None and left["reading_cer_mean"] is not None
                    and right["reading_cer_mean"] < left["reading_cer_mean"]):
                lines += ["视觉转写 CER 更低，没有在本轮总体修复成绩中形成提升。"]
        lines += ["失败复核显示瓶颈还包括工具调用格式、重复检索/读取、精确替换及测试规格覆盖；"
                  "不能把修复失败统一归为识别错误，也不据此宣称跨仓库泛化或统计显著性。"]
    model_stats = output / "final_model_stats.json"
    if model_stats.exists():
        peak = json.loads(model_stats.read_text()).get("peak_cuda_memory_bytes")
        if peak:
            lines += ["", f"PyTorch 峰值已分配显存：{peak/(1024**3):.2f} GiB，驱动保留内存另计。"]
    if state.get("finished_at"):
        lines += [f"批次执行耗时：{(state['finished_at']-state['started_at'])/60:.2f} 分钟，"
                  "不含模型启动与后续报告生成。"]
    lines += ["", "## 按集合分开报告", "",
              "| 来源 | 集合 | 读取方式 | 通过 / 尝试 | 分组数 |",
              "|---|---|---|---:|---:|"]
    for row in report["summary_by_cohort"]:
        source = "公开历史案例" if row["provenance"].startswith("public_") else "自编合成题"
        lines.append(f"| {source} | {row['split']} | {row['reader']} | {row['resolved']}/{row['attempted']} | "
                     f"{row['independent_task_groups']} |")
    lines += ["", "## 同题配对", "",
              "| 来源 / 集合 | 比较 | 配对题数 | 都通过 | 仅左侧通过 | 仅右侧通过 | 都失败 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        source = "公开" if row["provenance"].startswith("public_") else "合成"
        lines.append(f"| {source} / {row['split']} | {row['left']} → {row['right']} | {row['paired_tasks']} | "
                     f"{row['both_pass']} | {row['left_only']} | {row['right_only']} | {row['both_fail']} |")
    lines += ["", "CER 对空格、换行、行号和标点敏感。转写差异与最终修复结果分别记录；"
              "读取更准确本身不证明修复成功率提高。顺序固定为 oracle、OCR、VLM，耗时含预热与调度影响。", "",
              "## 失败记录", ""]
    if not failures:
        lines += ["本轮没有未通过的任务。只能说明所选任务在当前预算下通过，无法据此识别复杂任务中的能力瓶颈。"]
    else:
        for row in failures:
            error = str(row.get("error", row.get("agent_error", "详见固定测试日志"))).replace("\n", " ")[:220]
            overlap = row.get("localization_diagnostic", {}).get("reference_file_overlap", "n/a")
            lines.append(f"- `{row['task_id']}` / {row['reader']}：{row['status']}；阶段 `{row.get('failure_stage')}`；"
                         f"结束原因 `{row.get('exit_reason', 'n/a')}`；参考文件重合 `{overlap}`；{error}")
        lines += ["", "以上是可观察的失败阶段；没有把 CER 或参考文件重合直接当作失败根因。"]
    if (output / "failure_review.md").exists():
        lines += ["", f"已人工检查的检索循环、未闭合工具调用和 CSV 边界案例见 [失败复核]({short}/failure_review.md)。"]
    audit_path = output / "posthoc_csv_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if audit.get("source_run") == str(root):
            lines += ["", "## 事后边界检查", "",
                      "CSV 补充检查针对连续、首尾逗号的空字段语义，按参考补丁核对。"
                      "这些测试在观察结果后加入，不改变本轮原评分和上述主指标。", "",
                      "| 读取方式 | 原测试通过 | 三项空字段检查通过 | 两者均通过 |",
                      "|---|---|---|---|"]
            for row in audit["records"]:
                lines.append(f"| {row['reader']} | {row['primary_resolved']} | {row['supplementary_passed']} | "
                             f"{row['primary_and_supplementary_passed']} |")
            lines += ["", "已知参考补丁通过全部补充检查。准确文本/视觉组通过原测试的 CSV 修改会丢失空字段；"
                      "这暴露了原固定测试的覆盖不足，不能将其成绩等同于完整语义正确性。",
                      f"[补充检查原始记录]({short}/posthoc_csv_audit.json)。"]
    lines += ["", "## 工程验证与贡献", "",
              "新增可配置完整评测、公开历史案例构建、仓库/截图/描述直接输入命令、逐题后台进度、"
              "按来源与集合报告及证据归档。CoreCoder 循环、现成模型和 OCR 模型为复用。",
              "构建公开任务时验证了 3 个错误版本与官方源码补丁；准确文本/OCR 的已知补丁回放另有 6 次独立验证。"
              "回放检查不计入这里的模型修复成绩。", "",
              "公开案例目前均为单文件小补丁，评分测试是可见且受保护的 unittest 适配器，未跑上游完整套件。"
              "模型可能见过公开历史修复。没有真实用户截图、隐藏测试、跨更多仓库的泛化或训练结果。", "",
              "## 运行问题", "",
              "早期本地模型启动使用 180 秒等待门限；失败记录显示权重仅加载到 50%。"
              "现改为默认 900 秒、每 15 秒显示等待进度，并在模型日志中区分导入、权重加载与处理器加载阶段。"
              "失败目录保留，没有将其计作某种截图读取方法的修复失败。", "",
              "## 可审阅材料", "",
              f"- [完整模型报告]({short}/real_model_report.md)",
              f"- [逐题结果、预算、测试、定位与读取诊断]({short}/real_model_report.json)",
              f"- [同题配对结果]({short}/paired_comparisons.json)",
              f"- [源码与产物指纹]({short}/verification.json)",
              "- [公开案例来源](public_cases.md)", "- [演示步骤和贡献边界](demo_guide.md)", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"report": str(report_path), "artifacts": str(output), "finished": len(report["records"]),
            "passed": sum(r["resolved"] for r in report["records"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=float, default=0)
    args = parser.parse_args()
    try:
        if args.wait_seconds:
            wait_for_completion(args.runs, args.wait_seconds)
        result = archive(args.runs.resolve(), args.output.resolve(), args.report.resolve())
    except (ValueError, TimeoutError) as error:
        atomic_json(args.runs.parent / "archive_status.json", {"status": "error", "error": str(error)})
        if args.report.exists() and args.report.read_text().startswith("# 扩展评测进度"):
            args.report.write_text(f"# 扩展评测尚未完成\n\n状态：{error}\n\n未生成最终成绩。"
                                   f"原始运行记录保留在 `{args.runs}`。\n")
        raise
    atomic_json(args.runs.parent / "archive_status.json", {"status": "complete", **result})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
