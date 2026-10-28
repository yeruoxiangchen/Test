"""Controlled development-only suite; serial inference, existing LLM + CPU rerank."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import fcntl
from kb.cli import process_identity
from kb.config import Settings
from kb.benchmark.runner import source_fingerprint
from kb.benchmark.sources import write_json, read_jsonl, digest
from kb.benchmark.review import export_review, jobs_for

ARMS = ["retrieval-bm25", "fixed-bm25", "agent-bm25", "agent-native-bm25", "agent-evidence-bm25",
        "retrieval-rerank", "fixed-rerank", "agent-evidence-rerank"]
PAIRS = [("retrieval", "retrieval-bm25", "retrieval-rerank"), ("rerank", "fixed-bm25", "fixed-rerank"),
         ("messages", "agent-bm25", "agent-native-bm25"), ("workflow", "fixed-bm25", "agent-evidence-bm25"),
         ("workflow_rerank", "fixed-rerank", "agent-evidence-rerank"),
         ("evidence_rerank", "agent-evidence-bm25", "agent-evidence-rerank")]


def run_command(root, stage, args):
    with (root / (stage + ".log")).open("a") as log:
        p = subprocess.run([sys.executable, "-B", "-m", "kb.benchmark", *args], stdout=log, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f"{stage} exited {p.returncode}; inspect {root / (stage + '.log')}")


def report_inputs(root, requested_protocol):
    """Validate original inference identity; reporting fixes may have new code.

    Never rewrite protocol/config/predictions or relax inference resume checks.
    """
    manifest = root / "protocol.json"
    if not manifest.exists():
        raise ValueError("Report recovery requires the original protocol.json")
    original = json.loads(manifest.read_text())
    for key in ("arms", "limit", "split"):
        if original[key] != requested_protocol[key]:
            raise ValueError("Report configuration differs from original protocol: " + key)
    inputs = {"protocol.json": digest(manifest.read_bytes())}
    common = None
    for arm in original["arms"]:
        directory = root / arm
        cfg = json.loads((directory / "config.json").read_text())
        if cfg["arm"] != arm or cfg["source_fingerprint"] != original["source_fingerprint"]:
            raise ValueError("Inference identity differs from frozen protocol: " + arm)
        if cfg["split"] != original["split"] or cfg["limit"] != original["limit"]:
            raise ValueError("Inference split/limit differs from frozen protocol: " + arm)
        shared = {k: cfg[k] for k in ("case_ids", "dataset_receipt")}
        if common is not None and shared != common:
            raise ValueError("Cases or dataset differ between arms")
        common = shared
        rows = read_jsonl(directory / "predictions.jsonl")
        ids = [r["id"] for r in rows]
        report = json.loads((directory / "report.json").read_text())
        if len(ids) != len(set(ids)) or set(ids) != set(cfg["case_ids"]) or report["records"] != len(ids) or report["expected"] != len(ids):
            raise ValueError("Incomplete or duplicate inference records: " + arm)
        for name in ("config.json", "predictions.jsonl", "report.json", "grading.jsonl"):
            path = directory / name
            inputs[str(path.relative_to(root))] = digest(path.read_bytes())
    return {"original_protocol": original, "reporting_protocol": requested_protocol, "inputs_sha256": inputs,
            "note": "Report-only recovery. Original inference protocol and all arm files remain unchanged."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["launch", "work", "status", "stop", "report"])
    parser.add_argument("--output", type=Path, default=Settings().root / "artifacts/git_support_v1/dev_ablation_v3_20261028")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    launch_lock = None
    if args.action == "launch":
        launch_lock = (root / "launch.lock").open("a")
        fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner_path, state_path = root / "queue.json", root / "queue_state.json"
    owner = json.loads(owner_path.read_text()) if owner_path.exists() else {}
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    alive = bool(owner) and process_identity(owner["pid"]) == owner["start_ticks"]
    if args.action == "status":
        status = state.get("status", "pending")
        if not alive and status == "running":
            status = "interrupted"
        print(f"总状态: {status} | 阶段: {state.get('stage')} | PID: {owner.get('pid')} | 存活: {alive}")
        for arm in owner.get("arms", args.arms):
            p = root / arm / "status.json"
            row = json.loads(p.read_text()) if p.exists() else {}
            print(f"{arm:24} {row.get('status','pending'):12} {row.get('completed',0)}/{row.get('total',owner.get('limit') or 100)} {row.get('current') or ''}")
        print("错误:", state.get("error"))
        print("语义评分:", state.get("semantic", "pending"))
        model_dir = Settings().root / "models/bge-reranker-base"
        if not (model_dir / "receipt.json").exists():
            downloaded = sum(p.stat().st_size for p in model_dir.glob('.cache/huggingface/download/*.incomplete'))
            print(f"重排权重准备中，已下载约 {downloaded / 1024**2:.1f} MiB；日志: {root / 'reranker_setup.log'}")
        print("报告:", root / "report.md")
        print("日志:", root / "queue.log")
        print("仅开发集；复用已配置 Qwen 服务，重排使用 CPU；不训练、不加载新 GPU 模型。")
        return
    if args.action == "report" and alive:
        raise RuntimeError("Queue still running; cannot regenerate its report concurrently")
    if args.action == "stop":
        if alive:
            command = Path(f"/proc/{owner['pid']}/cmdline").read_bytes()
            if b"eval_v3.py" not in command or b"work" not in command:
                raise RuntimeError("Queue identity mismatch")
            os.killpg(owner["pid"], signal.SIGTERM)
        print("已请求停止本队列；独立模型服务继续运行。")
        return
    if args.action == "launch":
        if alive:
            raise RuntimeError("Queue already running")
        cmd = [sys.executable, "-B", __file__, "work", "--output", str(root), "--arms", *args.arms]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        with (root / "queue.log").open("a") as log:
            child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(owner_path, {"pid": child.pid, "start_ticks": process_identity(child.pid), "arms": args.arms,
                               "limit": args.limit, "started_at": time.time()})
        print(json.dumps({"pid": child.pid, "output": str(root)}, ensure_ascii=False))
        return
    lock = (root / "suite.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stage, semantic = None, "pending_configuration"
    def progress(status, error=None):
        write_json(state_path, {"status": status, "stage": stage, "semantic": semantic, "error": error, "updated_at": time.time()})
    def interrupted(sig, frame):
        raise SystemExit("Received signal " + str(sig))
    signal.signal(signal.SIGTERM, interrupted)
    protocol = {"arms": args.arms, "limit": args.limit, "split": "dev", "source_fingerprint": source_fingerprint(),
                "queue_sha256": digest(Path(__file__).read_bytes())}
    try:
        manifest = root / "protocol.json"
        recovery = None
        if args.action == "report":
            recovery = report_inputs(root, protocol)
            recovery["started_at"] = time.time()
            write_json(root / "reporting_manifest.json", recovery)
        else:
            if manifest.exists() and json.loads(manifest.read_text()) != protocol:
                raise ValueError("Suite code/config changed; use a new output directory")
            write_json(manifest, protocol)
        if args.action != "report":
            reranker_checked = False
            for arm in args.arms:
                if source_fingerprint() != protocol["source_fingerprint"]:
                    raise ValueError("Source changed during suite; stopping before the next arm")
                if arm.endswith("rerank") and not (Settings().root / "models/bge-reranker-base/receipt.json").exists():
                    stage = "reranker_setup"
                    progress("running")
                    entry = Path(__file__).resolve().parent / "run.sh"
                    with (root / "reranker_setup.log").open("a") as log:
                        subprocess.run(["bash", str(entry), "reranker-setup"], stdout=log, stderr=subprocess.STDOUT, check=True)
                if arm.endswith("rerank") and not reranker_checked:
                    stage = "reranker_check"
                    progress("running")
                    with (root / "reranker_check.log").open("a") as log:
                        subprocess.run([sys.executable, "-B", str(Path(__file__).resolve().parent / "reranker_check.py")],
                                       stdout=log, stderr=subprocess.STDOUT, check=True)
                    reranker_checked = True
                stage = arm
                progress("running")
                cmd = ["run", "--arm", arm, "--split", "dev", "--output", str(root / arm)]
                if args.limit:
                    cmd += ["--limit", str(args.limit)]
                run_command(root, arm, cmd)
        stage = "report"
        progress("running")
        comparisons = []
        for name, left, right in PAIRS:
            if left in args.arms and right in args.arms:
                run_command(root, name, ["compare", "--left", str(root / left), "--right", str(root / right), "--output", str(root / "comparison" / name)])
                comparisons.append(name)
        generative = [root / arm for arm in args.arms if not arm.startswith("retrieval-")]
        export_review(Settings().root / "datasets/git_support_zh_v1", root / "label_review", generative)
        if os.getenv("KB_JUDGE_BASE_URL") and os.getenv("KB_JUDGE_MODEL"):
            semantic = "running"
            for run in generative:
                stage = "semantic_" + run.name
                progress("running")
                run_command(root, stage, ["judge", "--run-dir", str(run), "--review-dir", str(root / "label_review"), "--output", str(root / "semantic" / run.name)])
            semantic = "complete_model_grades_pending_human_calibration"
        else:
            semantic = "pending_independent_judge; reviewed grading.jsonl exported"
        lines = ["# 知识库 v3 开发集对照", "", "只用于开发诊断；不是独立人工语义准确率，也未运行冻结测试集。", "",
                 "| 分支 | 记录 | 失败 | 全事实关键词诊断（原标签） | 动作匹配 | 模型调用 | 输入 tokens | p50 秒 |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for arm in args.arms:
            r = json.loads((root / arm / "report.json").read_text())
            m = r["metrics"]
            def value(key):
                return f"{100*m[key]['mean']:.2f}%" if m.get(key) else "—"
            lines.append(f"| {arm} | {r['records']}/{r['expected']} | {value('failed')} | {value('all_required_facts_diagnostic')} | {value('action_correct')} | {r['model_calls']} | {r['prompt_tokens']} | {r['latency_seconds']['p50']:.2f} |")
        lines += ["", "配对报告（含题族 bootstrap 区间）："] + [f"- [{n}](comparison/{n}/comparison.md)" for n in comparisons]
        lines += ["", "- messages：相同 bounded 控制策略，规范化状态消息 vs 最近真实 assistant/tool 批次＋唯一证据状态；消息角色和上一轮参数呈现共同变化，不宣称只改一个 role 字段。",
                  "- rerank：固定检索与生成不变，只在 BM25 前20候选上重排取前5；未添加按开发集调出的分数阈值。",
                  "- workflow：对比先取证、最多一次缺口补查；策略和提示词共同变化，按整个工作流评估。",
                  "- 修订标签诊断见 label_review；原报告仍保留原冻结标签分数。",
                  "- 语义评分状态：" + semantic,
                  "- CPU 重排首题含模型加载开销；共享 GPU 时延不是专用吞吐。"]
        (root / "report.md").write_text("\n".join(lines) + "\n")
        if recovery:
            for name, expected in recovery["inputs_sha256"].items():
                if digest((root / name).read_bytes()) != expected:
                    raise RuntimeError("Original input changed during report recovery: " + name)
            recovery.update(completed_at=time.time(), original_inputs_unchanged=True)
            write_json(root / "reporting_manifest.json", recovery)
        stage = None
        progress("complete")
    except BaseException as exc:
        progress("interrupted" if isinstance(exc, (SystemExit, KeyboardInterrupt)) else "failed", str(exc))
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
