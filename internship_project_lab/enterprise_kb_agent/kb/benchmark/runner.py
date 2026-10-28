"""Isolated corpus/index/run directories; only the oracle arm receives gold spans."""
from dataclasses import replace
import fcntl
import json
from pathlib import Path
import secrets
import shutil
import time
import httpx
from ..agent import Harness
from ..auth import create_user
from ..config import Settings
from ..db import Database, uid, encode
from ..ingestion import register_document, ingest
from ..retrieval import Retriever
from .dataset import inside, verify_freeze
from .fixed import FixedRAG, retrieval_query
from .scoring import retrieval_metrics, answer_metrics, summary, paired_difference, metric_keys
from .sources import digest, read_jsonl, write_json

ARMS = ["retrieval-bm25", "retrieval-rerank", "fixed-bm25", "fixed-hybrid", "fixed-rerank",
        "agent-bm25", "agent-hybrid", "agent-legacy-bm25", "agent-native-bm25",
        "agent-evidence-bm25", "agent-evidence-rerank", "agent-guarded-bm25", "agent-guarded-rerank",
        "agent-guarded-no-followup-bm25", "agent-guarded-no-followup-rerank", "oracle"]


def resume_rows(path):
    """Recover only an unterminated final write; do not hide corruption elsewhere."""
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines, rows, offset = raw.splitlines(keepends=True), [], 0
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeError):
            if i != len(lines)-1 or line.endswith(b"\n"):
                raise ValueError("Corrupt predictions file; not silently skipped") from None
            path.with_suffix(".interrupted_tail").write_bytes(line)
            with path.open("r+b") as f:
                f.truncate(offset)
            return rows
        offset += len(line)
    if raw and not raw.endswith(b"\n"):
        with path.open("ab") as f:
            f.write(b"\n")
    return rows


def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    return digest("".join(str(p.relative_to(root)) + digest(p.read_bytes()) for p in sorted(root.rglob("*.py"))).encode())


def index(dataset, cfg, target):
    """Never touch live application DB. Reuse identical index on resume."""
    cfg = replace(cfg, root=Path(target) / "workspace", mode="extractive")
    cfg.prepare()
    db = Database(cfg.db_path)
    db.initialize()
    for role in ("admin", "member"):
        if not db.one("SELECT id FROM users WHERE username=?", (role,)):
            create_user(db, role, secrets.token_urlsafe(24), role)
    for base, visibility in (("public", "all"),):
        db.execute("INSERT OR IGNORE INTO bases VALUES(?,?,?,?,?)", (base, base, "evaluation corpus", visibility, 0))
    documents = read_jsonl(dataset / "documents.jsonl")
    fixtures = {}
    for item in read_jsonl(dataset / "catalog.jsonl"):
        fixture = item.get("fixture")
        if fixture:
            fixtures[fixture["id"]] = fixture
    for fixture in fixtures.values():
        db.execute("INSERT OR IGNORE INTO bases VALUES(?,?,?,?,?)", (fixture["id"], fixture["id"], "synthetic control fixture", fixture["visibility"], 0))
        documents.append({"id": fixture["id"], "title": fixture["title"], "text": fixture["text"], "fixture": True,
                          "visibility": fixture["visibility"]})
    mapping = {}
    for doc in documents:
        text = doc.get("text") if doc.get("fixture") else inside(dataset, doc["path"]).read_text(encoding="utf-8")
        base = doc["id"] if doc.get("fixture") else "public"
        row, _ = register_document(db, cfg, base, doc["id"] + "_" + doc["title"] + ".txt", text.encode())
        if row["status"] != "ready":
            db.execute("UPDATE documents SET status='queued' WHERE id=?", (row["id"],))
            ingest(db, cfg, row["id"])
        status = db.one("SELECT status,error FROM documents WHERE id=?", (row["id"],))
        if status["status"] != "ready" or (cfg.embedding_url and status["error"]):
            raise RuntimeError("Index failed: " + doc["id"] + ": " + str(status))
        start = 0
        for chunk in db.all("SELECT * FROM chunks WHERE document_id=? ORDER BY ordinal", (row["id"],)):
            a = text.find(chunk["text"], start)
            if a < 0:
                raise ValueError("Cannot map chunk to canonical source: " + doc["id"])
            mapping[chunk["id"]] = {"document_id": doc["id"], "start": a, "end": a + len(chunk["text"]),
                "text": chunk["text"], "visibility": doc["visibility"], "fixture": bool(doc.get("fixture"))}
            start = a + 1
    write_json(Path(target) / "index_manifest.json", {"chunks": len(mapping), "documents": len(documents),
        "embedding_profile": cfg.embedding_profile if cfg.embedding_url else None,
        "chunk_size": 900, "overlap": 120, "text_only": True})
    return db, cfg, mapping


def map_hits(rows, mapping):
    return [{**mapping[r["id"]], "chunk_id": r["id"], **({"label": r["label"]} if "label" in r else {})} for r in rows]


def run(dataset, cfg, target, arm, split="dev", limit=None, evaluation_profile="natural"):
    dataset, target = Path(dataset), Path(target)
    receipt = verify_freeze(dataset)
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    policy = {"agent-legacy-bm25": "legacy_v1", "agent-native-bm25": "bounded_native_v3",
              "agent-evidence-bm25": "evidence_first_v3", "agent-evidence-rerank": "evidence_first_v3"}.get(arm, "bounded_v2")
    if arm.startswith('agent-guarded'):policy='guarded_v4'
    cfg=replace(cfg,guarded_followup='no-followup' not in arm)
    intervention={}
    if evaluation_profile!='natural':
        spec=json.loads((dataset/'authoring_spec.json').read_text())
        if evaluation_profile!='withheld_first' or spec.get('kind')!='synthetic_gap_v1' or split!='dev':
            raise ValueError('Evidence withholding is restricted to the synthetic dev stress set')
        intervention=spec['withhold_on_first_search']
    cfg = replace(cfg, agent_policy=policy, reranker_path=str(cfg.root / "models/bge-reranker-base") if arm.endswith("rerank") else "")
    reranker_receipt = None
    if cfg.reranker_path:
        path = Path(cfg.reranker_path) / "receipt.json"
        if not path.exists():
            raise ValueError("Run reranker-setup first; no silent BM25 fallback")
        reranker_receipt = json.loads(path.read_text())
        reranker_receipt["worker_sha256"] = digest((Path(__file__).resolve().parents[2] / "scripts/reranker_worker.py").read_bytes())
    retrieval_only = arm.startswith("retrieval-")
    if arm.endswith("hybrid") and not (cfg.embedding_url and cfg.embedding_model):
        raise ValueError("Hybrid arm requires a real embedding service; no silent BM25 fallback")
    if not arm.endswith("hybrid"):
        cfg = replace(cfg, embedding_url="", embedding_model="", embedding_key="")
    if not retrieval_only and not (cfg.llm_url and cfg.llm_model and cfg.tokenize_url):
        raise ValueError("Model evaluation requires LLM endpoint, model and exact /tokenize endpoint")
    if not retrieval_only:
        headers = {"Authorization": "Bearer " + cfg.llm_key} if cfg.llm_key else {}
        try:
            response = httpx.get(cfg.llm_url.rstrip("/") + "/models", headers=headers, timeout=10, trust_env=False)
            response.raise_for_status()
            if cfg.llm_model not in {m["id"] for m in response.json()["data"]}:
                raise ValueError("Configured model absent from endpoint")
        except (httpx.HTTPError, KeyError):
            raise RuntimeError("Model endpoint not ready; no evaluation cases were consumed") from None
    cases = read_jsonl(dataset / (split + ".jsonl"))
    if limit:
        # Round-robin families: a small smoke run must not be just four copies of the same topic.
        cases = sorted(cases, key=lambda c: (c["variant"] != "direct", c["id"]))[:limit]
    target.mkdir(parents=True, exist_ok=True)
    lock = (target / "runner.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("This output directory already has a runner") from None
    model_receipt = None
    service_path = Settings().root / "model_service.json"
    if not retrieval_only and service_path.exists():
        service = json.loads(service_path.read_text())
        if cfg.llm_url.rstrip("/") == service.get("url", "") + "/v1":
            model_receipt = {k: service[k] for k in ("model_physical_path", "dtype", "environment", "gpu_memory_fraction", "context", "concurrency")}
            model_path = Path(service["model_physical_path"])
            model_receipt["small_metadata_sha256"] = {p.name: digest(p.read_bytes()) for p in
                [model_path / "config.json", model_path / "tokenizer_config.json", model_path / "model.safetensors.index.json"] if p.exists()}
    config = {"dataset": str(dataset.resolve()), "dataset_receipt": receipt["files"], "arm": arm, "split": split,
        "case_ids": [c["id"] for c in cases], "source_fingerprint": source_fingerprint(),
        "llm_model": cfg.llm_model if not retrieval_only else None,
        "llm_url": cfg.llm_url if not retrieval_only else None,
        "reranker_receipt": reranker_receipt, "reranker_candidates": cfg.reranker_candidates if cfg.reranker_path else None,
        "embedding_model": cfg.embedding_model, "embedding_url": cfg.embedding_url,
        "max_turns": cfg.max_turns, "max_tool_calls": cfg.max_tool_calls, "context_tokens": cfg.context_tokens,
        "agent_policy": cfg.agent_policy if arm.startswith("agent") else None,
        "guarded_followup":cfg.guarded_followup if policy=='guarded_v4' else None,"evaluation_profile":evaluation_profile,
        "effective_max_model_calls": min(4,cfg.max_turns) if policy=='guarded_v4' else min(3, cfg.max_turns) if policy == "evidence_first_v3" else cfg.max_turns if arm.startswith("agent") else 0 if retrieval_only else 1,
        "output_tokens": cfg.output_tokens, "temperature": 0, "enable_thinking": cfg.enable_thinking,
        "retrieval_top_k": 5, "seed": 0, "limit": limit, "model_receipt": model_receipt,
        "latency_scope": "shared_server_observed_latency_not_dedicated_gpu_throughput"}
    if (target / "config.json").exists():
        if json.loads((target / "config.json").read_text()) != config:
            raise ValueError("Run config/code/dataset changed; choose a new output directory")
    else:
        write_json(target / "config.json", config)
        source_root = Path(__file__).resolve().parents[1]
        for path in source_root.rglob("*.py"):
            dest = target / "source_snapshot" / path.relative_to(source_root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    rows_path = target / "predictions.jsonl"
    existing = resume_rows(rows_path)
    completed = {r["id"] for r in existing}
    if len(completed) != len(existing) or not completed.issubset(config["case_ids"]):
        raise ValueError("Duplicate or unexpected existing predictions")
    def progress(status, current=None, error=None):
        write_json(target / "status.json", {"status": status, "arm": arm, "split": split, "completed": len(completed),
            "total": len(cases), "current": current, "error": error, "updated_at": time.time()})
    progress("indexing")
    try:
        db, evaluation, mapping = index(dataset, cfg, target)
        if cfg.embedding_url:
            evaluation.strict_embedding = True
        for case in cases:
            if case["id"] in completed:
                continue
            progress("running", case["id"])
            started = time.monotonic()
            user = db.one("SELECT id,role,username FROM users WHERE username=?", (case["user_role"],))
            bases = ["public"]
            if case["fixture"] and (case["fixture"]["visibility"] == "all" or user["role"] == "admin"):
                bases.append(case["fixture"]["id"])
            query = retrieval_query(case["question"], case["history"])
            row = {"id": case["id"], "family_id": case["family_id"], "category": case["category"],
                   "variant": case["variant"], "question": case["question"], "history": case["history"], "arm": arm}
            try:
                if retrieval_only:
                    hits, warnings = Retriever(db, evaluation).search(user, query, bases, limit=10)
                    row.update(status="retrieved", hits=map_hits(hits, mapping), warnings=warnings,
                               metrics=retrieval_metrics(case, map_hits(hits, mapping)))
                else:
                    now, conv, run_id = time.time(), uid(), uid()
                    db.execute("INSERT INTO conversations VALUES(?,?,?,?)", (conv, user["id"], case["question"], now))
                    for i, message in enumerate(case["history"]):
                        db.execute("INSERT INTO messages VALUES(?,?,?,?,?,?)", (uid(), conv, message["role"], message["content"], None, now - 10 + i))
                    db.execute("INSERT INTO runs(id,conversation_id,user_id,question,base_ids,mode,status,created_at,updated_at) VALUES(?,?,?,?,?,?,'queued',?,?)",
                        (run_id, conv, user["id"], case["question"], encode(bases), "agent", now, now))
                    if arm.startswith("agent"):
                        harness=Harness(db,evaluation)
                        if case['id'] in intervention:
                            from .gap_stress import FirstSearchFilter
                            hidden=[i for i,v in mapping.items() if v['document_id']==intervention[case['id']]]
                            harness.retriever=FirstSearchFilter(harness.retriever,hidden)
                            db.event(run_id,'stress_intervention',{'profile':evaluation_profile,'hidden_document':intervention[case['id']], 'agent_sees_hidden_id':False})
                        harness.execute(run_id)
                        if case['id'] in intervention:
                            db.event(run_id,'stress_observation',{'removed_count':harness.retriever.removed_count,
                                'intervention_effective':harness.retriever.removed_count>0})
                    else:
                        oracle = None
                        if arm == "oracle":
                            oracle = [chunk_id for chunk_id, v in mapping.items() if any(
                                e["document_id"] == v["document_id"] and v["start"] < e["end"] and v["end"] > e["start"]
                                for e in case["evidence"]) and (v["visibility"] == "all" or user["role"] == "admin")]
                        FixedRAG(db, evaluation).execute_fixed(run_id, case["history"], oracle)
                    result = db.one("SELECT status,answer,citations,error,state FROM runs WHERE id=?", (run_id,))
                    trace = [{"kind": e["kind"], "at": e["created_at"], **json.loads(e["payload"])}
                             for e in db.all("SELECT * FROM events WHERE run_id=? ORDER BY id", (run_id,))]
                    row.update(status=result["status"], answer=result["answer"], error=result["error"],
                               citations=map_hits(json.loads(result["citations"]), mapping), trace=trace,
                               model_calls=sum(e["kind"] == "model" for e in trace),
                               tool_calls=sum(e["kind"] == "tool" for e in trace) + int(result["status"] in {"answered", "insufficient", "clarify"}),
                               prompt_tokens=sum(e.get("usage", {}).get("prompt_tokens", 0) for e in trace),
                               completion_tokens=sum(e.get("usage", {}).get("completion_tokens", 0) for e in trace))
                    row["metrics"] = answer_metrics(case, row)
            except Exception as exc:
                row.update(status="error", answer=None, citations=[], error=str(exc)[:300], hits=[])
                row["metrics"] = retrieval_metrics(case, []) if retrieval_only else answer_metrics(case, row)
            row["latency_seconds"] = time.monotonic() - started
            with rows_path.open("a", encoding="utf-8") as f:
                f.write(encode(row) + "\n")
                f.flush()
            existing.append(row)
            completed.add(case["id"])
            print(encode({"arm": arm, "id": case["id"], "status": row["status"], "progress": f"{len(completed)}/{len(cases)}"}), flush=True)
        report = summary(existing, len(cases))
        report["config"] = config
        write_json(target / "report.json", report)
        write_report(target, report)
        export_grading(dataset, target, cases, existing)
        progress("complete")
        return report
    except BaseException as exc:
        progress("interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed", error=str(exc)[:300])
        raise
    finally:
        lock.close()


def write_report(target, report):
    lines = ["# Knowledge Desk 评测", "", f"- 分支：{report['config']['arm']}；划分：{report['config']['split']}",
        f"- 完成：{report['records']}/{report['expected']}；模型：{report['config']['llm_model'] or '无模型，CPU 检索'}",
        "- 问题为代理编写、原文证据自动核对的候选集，尚未独立人工复核。每题族 4 个变体；置信区间按题族重采样。",
        "- 事实关键词命中、引用与标准证据重合只作诊断，不能宣称为真实回答准确率或语义忠实度。", "",
        "| 指标 | 均值 | 95% 题族 bootstrap 区间 |", "|---|---:|---|" ]
    for key, score in report["metrics"].items():
        if score:
            lo, hi = score["ci95_family_bootstrap"]
            lines.append(f"| {key} | {score['mean']:.4f} | {lo:.4f}–{hi:.4f} |")
    lines.extend(["", f"调用成本：模型 {report['model_calls']} 次，工具 {report['tool_calls']} 次，输入 {report['prompt_tokens']} / 输出 {report['completion_tokens']} tokens。",
        "", "详见 report.json 的分类统计，以及 predictions.jsonl 的逐题结果与完整工具轨迹。",
        "", "评测方法参考 [BEIR](https://github.com/beir-cellar/beir) 的检索分层评测，以及 [ALCE](https://github.com/princeton-nlp/ALCE) 的答案/引用分开评估。此报告不是这两个官方基准的成绩。"])
    (Path(target) / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_grading(dataset, target, cases, predictions):
    # Gold is added AFTER inference and is never passed to normal inference arms.
    from .sources import write_jsonl
    by_id = {r["id"]: r for r in predictions}
    rows = []
    for c in cases:
        p = by_id[c["id"]]
        if p["arm"].startswith("retrieval-"):
            continue
        rows.append({"id": c["id"], "family_id": c["family_id"], "question": c["question"], "history": c["history"],
            "reference_answer": c["reference_answer"], "required_facts": c["required_facts"], "expected_action": c["expected_action"],
            "gold_evidence": c["evidence"], "answer": p.get("answer"), "citations": p.get("citations", []),
            "instructions": "分别评估答案事实正确/完整、是否由引用原文支持、正确拒答或澄清；引用存在不等于支持。不给模型名称。",
            "reviewer": "", "correctness": None, "faithfulness": None, "citation_support": None, "rationale": ""})
    write_jsonl(Path(target) / "grading.jsonl", rows)


def compare(left, right, target):
    a, b = [json.loads((Path(p) / "config.json").read_text()) for p in (left, right)]
    left_label, right_label = a.get("arm", Path(left).name), b.get("arm", Path(right).name)
    for key in ("dataset_receipt", "split", "case_ids", "llm_model", "context_tokens", "output_tokens", "temperature", "source_fingerprint"):
        if a[key] != b[key]:
            raise ValueError("Comparison config mismatch: " + key)
    if a.get('evaluation_profile','natural')!=b.get('evaluation_profile','natural'):
        raise ValueError('Comparison config mismatch: evaluation_profile')
    ra, rb = [read_jsonl(Path(p) / "predictions.jsonl") for p in (left, right)]
    if len(ra) != len(a["case_ids"]) or len(rb) != len(b["case_ids"]):
        raise ValueError("Cannot compare incomplete runs")
    for config, rows in ((a, ra), (b, rb)):
        ids = [row["id"] for row in rows]
        if len(set(ids)) != len(ids) or set(ids) != set(config["case_ids"]):
            raise ValueError("Duplicate or unexpected case ids in comparison")
    right_by_id = {r["id"]: r for r in rb}
    if any(r["family_id"] != right_by_id[r["id"]]["family_id"] for r in ra):
        raise ValueError("Paired case family mismatch")
    keys = metric_keys(ra + rb)
    deltas, availability, unpaired = {}, {}, {}
    for key in keys:
        pairs = [(r["metrics"].get(key), right_by_id[r["id"]]["metrics"].get(key)) for r in ra]
        counts = {"total": len(pairs),
            "left_defined": sum(v is not None for v, w in pairs),
            "right_defined": sum(w is not None for v, w in pairs),
            "both_defined": sum(v is not None and w is not None for v, w in pairs),
            "both_missing": sum(v is None and w is None for v, w in pairs),
            "left_only": sum(v is not None and w is None for v, w in pairs),
            "right_only": sum(v is None and w is not None for v, w in pairs)}
        availability[key] = counts
        # Precision is intentionally undefined without emitted citations. Its
        # denominator depends on model output, unlike gold-conditioned metrics.
        # Preserve frozen per-case scores; neither impute zeros nor discard failed
        # cases from other metrics to manufacture a paired precision improvement.
        if key == "citation_gold_evidence_precision" and (counts["left_only"] or counts["right_only"]):
            unpaired[key] = {"reason": "output_conditioned_unequal_availability",
                "note": "Citation precision is undefined without citations or gold evidence. Different eligible case sets; no paired delta or CI."}
            continue
        deltas[key] = paired_difference(ra, rb, key)
    result = {"left": str(left), "right": str(right), "direction": "right_minus_left",
        "deltas": deltas, "metric_availability": availability, "unpaired_metrics": unpaired,
        "left_summary": summary(ra, len(a["case_ids"])), "right_summary": summary(rb, len(b["case_ids"])),
        "inference_source_fingerprint": a["source_fingerprint"],
        "reporting_source_fingerprint": source_fingerprint(),
        "note": "Paired by task id, resampled by family. Automatic diagnostics, not semantic accuracy."}
    Path(target).mkdir(parents=True, exist_ok=True)
    write_json(Path(target) / "comparison.json", result)
    lines = ["# 开发集配对结果", "", f"- 左组 {left_label}：{left}", f"- 右组 {right_label}：{right}",
        f"- 两组各 {len(ra)} 条；差值为右组减左组。所有失败记录保留。",
        "- 原始预测和分数未改写。本报告单独记录推理与汇总代码指纹。",
        "- 指标为自动诊断，不等于独立人工确认的回答准确率。", "",
        f"| 指标 | {left_label} | {right_label} | 配对差值 | 95% 题族 bootstrap 区间 |",
        "|---|---:|---:|---:|---|"]
    for key, delta in deltas.items():
        if delta is not None:
            l = result["left_summary"]["metrics"][key]["mean"]
            r = result["right_summary"]["metrics"][key]["mean"]
            lo, hi = delta["ci95_family_bootstrap"]
            lines.append(f"| {key} | {l:.4f} | {r:.4f} | {delta['mean']:+.4f} | {lo:+.4f}–{hi:+.4f} |")
    if unpaired:
        lines.extend(["", "## 无法进行配对差值的条件指标", ""])
        for key in unpaired:
            c = availability[key]
            lines.append(f"- {key}：{left_label} 有定义 {c['left_defined']}/{c['total']}，{right_label} 有定义 {c['right_defined']}/{c['total']}；两组统计对象不同，不计算配对差值。无引用时精度未定义，不补成 0，也不据此剔除其他指标的失败样本。")
    lines.extend(["", "## 调用成本", "", "| 分支 | 模型调用 | 工具调用 | 输入 tokens | 输出 tokens |",
        "|---|---:|---:|---:|---:|"])
    for label, name in ((left_label, "left_summary"), (right_label, "right_summary")):
        s = result[name]
        lines.append(f"| {label} | {s['model_calls']} | {s['tool_calls']} | {s['prompt_tokens']} | {s['completion_tokens']} |")
    (Path(target) / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
