#!/usr/bin/env python3
"""Archive only completed official-source experiments, with preflight denominators."""

import argparse
import difflib
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

from PIL import Image


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def archive(run: Path, reading_run: Path, output: Path):
    root = Path(__file__).resolve().parents[1]
    run, reading_run, output = run.resolve(), reading_run.resolve(), output.resolve()
    status = load(run / "evaluation/status.json")
    if status.get("status") != "complete" or status["completed"] != status["total"]:
        raise ValueError("Repair batch is unfinished")
    repair_stats = run / "final_model_stats.json"
    repair_recovery = load(run/"completion_recovery.json") if (run/"completion_recovery.json").exists() else {}
    if not repair_stats.exists() and not (repair_recovery.get("report_complete") and
            repair_recovery.get("record_count_verified") and repair_recovery.get("model_released")):
        raise ValueError("Wait for repair cleanup or an explicit recorded reporting recovery")
    reading_stats = reading_run / "final_model_stats.json"
    recovery_path = reading_run / "completion_recovery.json"
    if not reading_stats.exists():
        recovery = load(recovery_path) if recovery_path.exists() else {}
        if not (recovery.get("report_complete") and recovery.get("record_count_verified")
                and recovery.get("model_released")):
            raise ValueError("Wait for reading cleanup or an explicit recorded recovery")
    report = load(run / "evaluation/report.json")
    if not repair_stats.exists() and len(report["records"]) != repair_recovery["completed_runs"]:
        raise ValueError("Repair recovery count differs from registered results")
    clean = load(run / "clean_patch_audit.json")
    if not clean.get("complete") or clean["attempts"] != status["total"]:
        raise ValueError("Wait for independent cache-free patch verification")
    reading = load(reading_run / "codeocr/report.json")
    if not reading.get("complete"):
        raise ValueError("CodeOCR reconstruction batch is unfinished")
    if not reading_stats.exists() and len(reading["records"]) != recovery["completed_readings"]:
        raise ValueError("Reading recovery record count differs from the saved report")
    if any(not r.get("model_inference") or r.get("code_backend") != "api" for r in report["records"]):
        raise ValueError("Known patch replays cannot count as model quality")
    launch_state = load(run / "source_state.json")
    runtime_paths = [name for name in launch_state["sha256"] if name.startswith("src/") or name in {
        "scripts/local_model_api.py", "scripts/run_local_smoke.sh", "scripts/run_official_eval.sh",
        "scripts/official_grade.py", "configs/official-local.json"}]
    changed = [name for name in runtime_paths if hashlib.sha256((root/name).read_bytes()).hexdigest()
               != launch_state["sha256"][name]]
    if changed:
        raise ValueError(f"Runtime sources changed during the comparison: {changed}")
    origin = {"repair_run": str(run), "reading_run": str(reading_run)}
    if (output / "run_origin.json").exists() and load(output / "run_origin.json") != origin:
        raise ValueError("Archive belongs to another experiment")
    write(output / "run_origin.json", origin)
    write(output / "real_model_report.json", report)
    write(output / "codeocr_report.json", reading)
    write(output / "codeocr_build_report.json", reading["scope"])
    write(output / "batch_status.json", status)
    write(output / "launch_source_state.json", launch_state)
    if (root / "outputs/official_v3_setup/engineering_checks.json").exists():
        shutil.copy2(root / "outputs/official_v3_setup/engineering_checks.json", output / "engineering_checks.json")
    build_root = root / "datasets/generated/swebench_lite_v3"
    build = load(build_root / "build_report.json")
    write(output / "task_build_report.json", build)
    shutil.copy2(build_root / "selection.json", output / "selection.json")
    shutil.copy2(root / "BENCHMARK_UPSTREAM.json", output / "upstream.json")
    if repair_stats.exists():
        shutil.copy2(repair_stats, output / "model_stats.json")
    else:
        shutil.copy2(run / "completion_recovery.json", output / "repair_completion_recovery.json")
        recovered_source = Path(repair_recovery["reporting_source"])
        if hashlib.sha256(recovered_source.read_bytes()).hexdigest() != repair_recovery["reporting_source_sha256"]:
            raise ValueError("The reporting recovery source has changed")
        shutil.copy2(recovered_source, output / "recovered_report.py")
        before = (root/"src/screenshot_agent/report.py").read_text().splitlines(keepends=True)
        after = recovered_source.read_text().splitlines(keepends=True)
        (output/"reporting_recovery.patch").write_text("".join(difflib.unified_diff(before,after,
            fromfile="a/src/screenshot_agent/report.py",tofile="b/src/screenshot_agent/report.py")))
    shutil.copy2(run / "clean_patch_audit.json", output / "clean_patch_audit.json")
    if (run / "trace_diagnostics.json").exists():
        shutil.copy2(run / "trace_diagnostics.json", output / "trace_diagnostics.json")
    if reading_stats.exists():
        shutil.copy2(reading_stats, output / "reading_model_stats.json")
    else:
        shutil.copy2(recovery_path, output / "reading_completion_recovery.json")
    if (run / "input_glyph_audit.json").exists():
        shutil.copy2(run / "input_glyph_audit.json", output / "pre_exposure_glyph_audit.json")
    shutil.copy2(build_root / "glyph_audit.json", output / "glyph_audit.json")
    glyph_audit = load(build_root / "glyph_audit.json")
    if not glyph_audit["all_visible_glyphs_supported"]:
        raise ValueError("Input glyph coverage is incomplete")
    old_reading = root / "outputs/official_v3_pilot/codeocr/report.json"
    if old_reading.exists():
        shutil.copy2(old_reading, output / "old_font_diagnostic_report.json")
        write(output / "old_font_diagnostic_scope.json", {"primary_result": False,
            "reason": "Four source files had unsupported CJK characters. Old images/results retained, not used as primary reconstruction scores."})
    shutil.copy2(run / "launch.json", output / "launch.json")
    shutil.copy2(run / "evaluation/report.md", output / "real_model_report.md")
    shutil.copy2(reading_run / "codeocr/report.md", output / "codeocr_report.md")
    configs = set()
    corrected_exposures = 0
    for directory in (run / "evaluation/runs").iterdir():
        if not directory.is_dir():
            continue
        configs.add(hashlib.sha256((directory / "config.json").read_bytes()).hexdigest())
        trial = load(directory / "result.json")
        instance = trial["task_id"]
        correction = glyph_audit["tasks"].get(instance,{})
        image = directory / "screenshots/shot.png"
        if correction.get("correction_before_model_exposure") and image.exists():
            with Image.open(image) as actual, Image.open(build_root/instance/"screenshot_light.png") as expected:
                if actual.size != expected.size or actual.convert("RGB").tobytes() != expected.convert("RGB").tobytes():
                    raise ValueError("A compared trial did not receive the glyph-corrected input")
            corrected_exposures += 1
        destination = output / "trials" / directory.name
        destination.mkdir(parents=True, exist_ok=True)
        for name in ["result.json", "patch.diff", "patch.clean.diff", "baseline_test.log", "verification_test.log", "agent_error.log"]:
            if (directory / name).exists():
                shutil.copy2(directory / name, destination / name)
        if not trial["resolved"] or trial["reader"] == "vlm":
            shutil.copy2(directory / "events.jsonl", destination / "events.jsonl")
        clean_source = run / "clean_patch_checks" / directory.name
        for name in ["check.json", "verification_test.log"]:
            if (clean_source/name).exists():
                shutil.copy2(clean_source/name, destination/("clean_"+name))
    if len(configs) != 1:
        raise ValueError("The compared trials do not share exactly one configuration")
    source_records = {}
    for row in build["audits"]:
        instance = row["instance_id"]
        source = build_root / instance
        destination = output / "tasks" / instance
        destination.mkdir(parents=True, exist_ok=True)
        for name in ["audit.json", "baseline_test.log", "reference_test.log", "screenshot_light.png", "reference.patch"]:
            if (source / name).exists():
                shutil.copy2(source / name, destination / name)
        source_records[instance] = {"status": row["status"], "repo": row["repo"]}
    stats = load(repair_stats) if repair_stats.exists() else None
    records = report["records"]
    lines = ["# 官方来源任务与截图读取评测", "", "日期：2026-10-22。", "",
             "本轮接入 SWE-bench Lite 原始任务、测试补丁、官方日志解析器/评分器，以及 CodeOCR 作者发布的代码数据、渲染和重建指标。",
             "执行使用独立 Python 3.9.20 与 Bubblewrap。Docker Hub 连接失败、公开缓存站返回 403，未运行官方 Docker 镜像；成绩属于截图派生子集。",
             "官方测试已安装在 Agent 可见仓库中，并设为只读保护；本轮不是隐藏测试协议。参考源码补丁和官方评分元数据在仓库外。", "",
             "## 数据和环境预检", "",
             f"预先固定 {build['original_selection']} 个任务、{build['groups']} 个仓库组；{build['validated']} 个通过错误版本复现与参考补丁评分。",
             "任务 ID 和选择规则先于完整模型对照确定；Pytest 分为开发组、SymPy 分为测试组，再渲染两种主题。原始来源均为官方 test split。",
             "保留原 Issue 周围描述，仅将预先选定的一个代码/输出块替换为截图；未声称每题都必须依赖图像。", "",
             "| 原始任务 | 预检状态 |", "|---|---|"]
    for instance, row in source_records.items():
        lines.append(f"| `{instance}` | {row['status']} |")
    lines += ["", "## 真实模型修复", "",
              f"完成 {status['completed']}/{status['total']} 次运行；通过独立重放与官方评分的运行共 {sum(r['resolved'] for r in records)} 次。",
              f"三组配置文件一致：{len(configs) == 1}。模型为 Qwen3-VL-8B-Instruct；代码输出上限 1024 token，视觉转写 2048 token；18 次代码/压缩调用、50 次工具调用、1200 秒总预算、180 秒单命令。",
              "Agent 以 4096 token 估计阈值压缩上下文，本地服务最大上下文 16384。该配置在预备运行发现原上限不足后确定，预备运行另存，不合并成绩。", "",
              "| 读取方式 | 通过 / 尝试 | 平均秒数 | 平均工具调用 | 截图原始 CER |", "|---|---:|---:|---:|---:|"]
    for name in ["oracle", "ocr", "vlm"]:
        group = [r for r in records if r["reader"] == name]
        lines.append(f"| {name} | {sum(r['resolved'] for r in group)}/{len(group)} | "
                     f"{sum(r['elapsed_seconds'] for r in group)/len(group):.2f} | "
                     f"{sum(r['usage']['tool_calls'] for r in group)/len(group):.2f} | "
                     f"{next(s['reading_cer_mean'] for s in report['summary'] if s['reader']==name):.2%} |")
    lines += ["", "截图 CER 对首轮整图读取逐字符比较，包含空白差异；与下表 CodeOCR 的归一化 CER 定义不同。两种指标不合并。",
              "评分同时检查官方 FAIL_TO_PASS 和 PASS_TO_PASS；退出码为零、Agent 声称完成均不能单独判为成功。",
              "", "| 内部分组 | 读取方式 | 通过 / 尝试 |", "|---|---|---:|"]
    for split in ["dev","test"]:
        for name in ["oracle","ocr","vlm"]:
            group = [r for r in records if r["reader"]==name and r["split"]==split]
            lines.append(f"| {split} | {name} | {sum(r['resolved'] for r in group)}/{len(group)} |")
    lines += ["", f"测试缓存剔除后，独立重放 {clean['attempts']} 份补丁，{clean['clean_passed']} 次通过；与原始判定全部一致：{clean['all_verdicts_preserved']}。空补丁保留为失败。",
              "清理不使用参考补丁、不做语义修改，也不调用模型；原始补丁、原始成绩与清理后测试日志分别保留。",
              "本轮模型比较使用冻结的修复前工程版本。文件检索、缓存导出和 Unicode 日志解析问题的后续修复与开发组回归单独记录，不混入这 27 次成绩。", "",
              "## CodeOCR 重建子集", "",
              "从作者发布的 100 份代码中，按固定 ID 顺序选择 10 份不超过 80 行、2200 字符的完整文件，来自 10 个仓库。",
              "调用上游渲染函数和 1×/2×退化函数生成 20 张图片，比较 CPU OCR 与视觉模型，共 40 次读取；没有运行这些代码。",
              "归一化 CER 使用上游原函数，忽略空白行和行尾空格，保留缩进。另存原始 CER、WER、行相似度和 Python 语法检查。", "",
              "| 读取方式 | 退化倍数 | 完成 / 尝试 | 归一化 CER | 语法通过数 |", "|---|---:|---:|---:|---:|"]
    for row in reading["summary"]:
        cer = f"{row['normalized_cer_mean']:.2%}" if row["normalized_cer_mean"] is not None else "N/A"
        lines.append(f"| {row['reader']} | {row['ratio']} | {row['completed']}/{row['attempted']} | {cer} | {row['syntax_valid']} |")
    lines += ["", "这是代码读取测量，不能当作修复成功率。上游 API 辅助模块缺少配置常量；适配器直接加载两个未修改的纯函数 AST，渲染模块正常导入，上游源码保持原样。", "",
              "初版 DejaVu Sans Mono 字体使四份代码的中文注释显示为方框；其 40 次读取仅保留作数据质量诊断。主表使用 Noto Sans Mono CJK、经字形覆盖审计的同样十份代码重新渲染，参考文本和任务 ID 没有改变。",
              f"SWE-bench 的一个尚未送入模型的截图含缺字 ℯ，已更换为支持该字符的字体。已核对 {corrected_exposures} 次实际输入与修正后的图像像素完全一致；原图及修正前后指纹保留。", "",
              "## 失败与范围", ""]
    for r in records:
        if not r["resolved"]:
            lines.append(f"- `{r['task_id']}` / {r['reader']}：{r['status']}；阶段 {r.get('failure_stage')}；"
                         f"结束原因 {r.get('exit_reason')}；{r.get('agent_error',r.get('error',''))[:200]}")
    lines += ["", "本轮包含两个仓库的小规模派生任务；全部截图合成，真实用户截图数为零。没有原始 SWE-bench 排行榜成绩、SWE-bench Multimodal 成绩或训练结果。",
              "初始环境失败、依赖缺失、长测试超时和预备运行的上下文错误分别保留；环境预检失败不描述为模型修复失败。",
              "本地 API 标价为零，GPU 时间、电力与计算资源有实际成本。",
              (f"修复批次 PyTorch 峰值已分配显存：{stats['peak_cuda_memory_bytes']/2**30:.2f} GiB。" if stats else
               "修复批次汇总器错误地将 Unicode NEL 当成 JSONL 记录分隔符，导致完成 27 次运行后汇总失败，最终 CUDA 峰值未采集。用独立修正的汇总器恢复报告，原始事件、补丁与逐题判定 SHA-256 均保持一致。"), "",
              "源码固定提交、数据指纹、逐题补丁、完整固定测试日志与评分细节见 `docs/artifacts/v3/`。"]
    if not reading_stats.exists():
        lines += ["", "读取批次的 40 条结果与报告完整保存，但启动脚本在结束采集显存时因运行中修订发生解析错误；模型已退出。读取批次的最终 CUDA 峰值未采集，未用其他批次的值补填。"]
    (root / "docs/experiment_report_v3.md").write_text("\n".join(lines) + "\n")
    if (root / "docs/failure_review_v3.md").exists():
        shutil.copy2(root / "docs/failure_review_v3.md", output / "failure_review.md")
    source_hashes = {}
    for folder in [root / "src", root / "scripts", root / "configs", root / "tests"]:
        for p in folder.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                source_hashes[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    with zipfile.ZipFile(output/"source_snapshot.zip","w",zipfile.ZIP_DEFLATED) as snapshot:
        for relative in sorted(source_hashes):
            snapshot.write(root/relative,relative)
        for name in ["pyproject.toml","requirements.lock.txt","requirements-benchmark.lock.txt","UPSTREAM.json",
                     "BENCHMARK_UPSTREAM.json","README.md"]:
            snapshot.write(root/name,name)
    artifact_hashes = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in output.rglob("*") if p.is_file() and p.name != "verification.json"}
    write(output / "verification.json", {"same_config": len(configs) == 1,
        "runtime_sources_unchanged": not changed, "corrected_input_exposures_checked": corrected_exposures,
        "finished": status["completed"], "passed": sum(r["resolved"] for r in records),
        "clean_patch_verdicts_preserved": clean["all_verdicts_preserved"],
        "source_sha256": source_hashes, "artifact_sha256": artifact_hashes})
    return {"complete": True, "repair_runs": len(records), "passed": sum(r["resolved"] for r in records),
            "reading_runs": len(reading["records"]), "report": str(root / "docs/experiment_report_v3.md")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reading-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/artifacts/v3"))
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    started = time.monotonic()
    while True:
        if ((args.run / "final_model_stats.json").exists() or (args.run / "completion_recovery.json").exists()) and (args.run / "clean_patch_audit.json").exists() and (args.reading_run / "codeocr/report.json").exists() and ((args.reading_run / "final_model_stats.json").exists() or (args.reading_run / "completion_recovery.json").exists()):
            break
        if time.monotonic() - started >= args.wait_seconds:
            raise SystemExit("Reports are unfinished; no final results were archived")
        print("Waiting for completed repair and CodeOCR reports...", flush=True)
        time.sleep(30)
    try:
        result = archive(args.run, args.reading_run, args.output)
        write(args.run / "archive_status.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as error:
        write(args.run / "archive_status.json", {"complete": False, "error": str(error)})
        raise
