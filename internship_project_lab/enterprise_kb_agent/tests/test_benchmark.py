import json
from pathlib import Path
import pytest
from kb.config import Settings
from kb.context import ContextBudget
from kb.benchmark.dataset import build, freeze, verify_freeze
from kb.benchmark.sources import digest, write_json, write_jsonl
from kb.benchmark.scoring import covered, retrieval_metrics, answer_metrics, paired_difference, summary
from kb.benchmark.fixed import FixedRAG
from kb.benchmark.runner import index, run
from kb.db import uid, encode


def dataset(tmp_path):
    root = tmp_path / "dataset"
    (root / "corpus").mkdir(parents=True)
    text = "# 凭证配置\ncache 将凭证存放内存，默认 15 分钟。\nstore 使用明文磁盘存储。"
    (root / "corpus/doc.txt").write_text(text)
    write_jsonl(root / "documents.jsonl", [{"id":"doc", "title":"凭证配置", "path":"corpus/doc.txt", "sha256":digest(text.encode()),
        "question_split":"dev", "visibility":"all", "kind":"public_document"}])
    c = {"id":"fact", "split":"dev", "questions":["cache 存多久？","凭证缓存多久？","cache 凭证保存期限？","默认时间？"],
        "context":"讨论 cache 凭证缓存。", "reference_answer":"内存 15 分钟", "required_facts":[["15"],["内存"]],
        "evidence":[{"document_id":"doc", "quote":"cache 将凭证存放内存，默认 15 分钟。"}]}
    acl = {"id":"acl", "split":"dev", "questions":["内部评测码？","评测码是多少？","给出内部测试码？","它的值？"],
        "context":"问内部评测码", "reference_answer":"无权限", "expected_action":"insufficient", "evidence":[],
        "fixture":{"id":"private", "title":"内部评测码", "text":"内部评测码是 PRIVATE_CANARY", "visibility":"admin"},
        "forbidden_strings":["PRIVATE_CANARY"]}
    write_jsonl(root / "catalog.jsonl", [c, acl])
    write_json(root / "source_manifest.json", {"fixture":True})
    write_json(root / "authoring_spec.json", {"fixture":True})
    build(root)
    freeze(root)
    return root


def test_spans_overlap_not_double_count_and_require_multiple_passages():
    e = {"document_id":"a","start":10,"end":30}
    hits = [{"document_id":"a","start":0,"end":20}, {"document_id":"a","start":15,"end":40}]
    assert covered(e, hits) == 1
    assert covered(e, hits[:1]) == .5
    m = retrieval_metrics({"evidence":[e]}, hits)
    assert m["evidence_recall_at_1"] == 0 and m["evidence_recall_at_3"] == 1
    assert m["mrr_at_10"] == .5


def test_freeze_detects_label_and_document_mutation(tmp_path):
    root = dataset(tmp_path)
    with pytest.raises(ValueError, match="frozen"):
        build(root)
    with (root / "corpus/doc.txt").open("a") as f:
        f.write("changed")
    with pytest.raises(ValueError, match="corpus changed"):
        verify_freeze(root)


def test_split_leakage_rejected(tmp_path):
    root = dataset(tmp_path)
    # The validator must detect a family copied to another split, even with a new id.
    c = json.loads((root / "dev.jsonl").read_text().splitlines()[0])
    c.update(id="other-id", split="test")
    write_jsonl(root / "test.jsonl", [c])
    from kb.benchmark.dataset import validate
    with pytest.raises(ValueError, match="family leaks"):
        validate(root)


def test_context_drops_history_without_orphaning_tool_calls():
    cfg = Settings(tokenize_url="", context_tokens=1600, output_tokens=200)
    budget = ContextBudget(cfg, counter=lambda m,t: sum(len(x.get("content") or "") for x in m) + len(t)*10)
    messages = [{"role":"system","content":"system"}, {"role":"user","content":"x"*1300},
        {"role":"assistant","content":"history"}, {"role":"user","content":"current"},
        {"role":"assistant","content":None,"tool_calls":[{"id":"1"}]}, {"role":"tool","content":"evidence","tool_call_id":"1"}]
    result, receipt = budget.prepare(messages, [{}], 3)
    assert receipt["history_messages_removed"] == 2
    assert result[-3:] == messages[-3:] and messages[1]["content"] == "x"*1300
    with pytest.raises(RuntimeError, match="预算不足"):
        budget.prepare([{"role":"system","content":"sys"},{"role":"user","content":"z"*1800}], [])


def test_exact_context_counter_passes_identical_tool_schema(monkeypatch):
    import httpx
    requests = []
    def handler(req):
        requests.append(json.loads(req.content))
        return httpx.Response(200, json={"count":222,"max_model_len":8192})
    real = httpx.Client
    monkeypatch.setattr("kb.context.httpx.Client", lambda **kw:real(transport=httpx.MockTransport(handler)))
    cfg = Settings(tokenize_url="http://local/tokenize")
    messages=[{"role":"user","content":"你好"}]
    _, receipt = ContextBudget(cfg).prepare(messages, [{"function":{"name":"search"}}])
    assert receipt["counter"] == "server_tokenizer" and receipt["input_tokens"] == 222
    assert requests[0]["tools"][0]["function"]["name"] == "search"


def test_cpu_run_has_no_model_calls_keeps_denominators_and_acl(tmp_path, monkeypatch):
    root = dataset(tmp_path)
    cfg = Settings(root=tmp_path, embedding_url="", embedding_model="", tokenize_url="")
    def forbidden(*a, **k):
        raise AssertionError("CPU retrieval must not call model")
    monkeypatch.setattr("kb.agent.ModelClient.complete", forbidden)
    target = tmp_path / "run"
    report = run(root, cfg, target, "retrieval-bm25")
    assert report["complete"] and report["records"] == 8 and report["model_calls"] == 0
    rows = [json.loads(s) for s in (target / "predictions.jsonl").read_text().splitlines()]
    assert not any(h["visibility"] == "admin" for row in rows for h in row["hits"])
    before = (target / "predictions.jsonl").read_bytes()
    run(root, cfg, target, "retrieval-bm25")
    assert before == (target / "predictions.jsonl").read_bytes()
    cfg.output_tokens += 1
    with pytest.raises(ValueError, match="changed"):
        run(root, cfg, target, "retrieval-bm25")


def test_fixed_rag_gets_only_question_retrieved_sources_and_finish(tmp_path):
    root = dataset(tmp_path)
    cfg = Settings(root=tmp_path, tokenize_url="", embedding_url="", embedding_model="")
    target = tmp_path / "fixed"
    target.mkdir()
    db, cfg, mapping = index(root, cfg, target)
    user = db.one("SELECT * FROM users WHERE username='member'")
    conv, run_id = uid(), uid()
    db.execute("INSERT INTO conversations VALUES(?,?,?,?)", (conv,user["id"],"cache",0))
    db.execute("INSERT INTO runs(id,conversation_id,user_id,question,base_ids,mode,status,created_at,updated_at) VALUES(?,?,?,?,?,?,'queued',0,0)",
        (run_id,conv,user["id"],"cache 凭证多久",encode(["public"]),"agent"))
    class Model:
        def complete(self, messages, tools):
            assert [t["function"]["name"] for t in tools] == ["finish"]
            assert "PRIVATE_CANARY" not in encode(messages)
            assert "reference_answer" not in encode(messages) and "expected_action" not in encode(messages)
            return {"role":"assistant","content":None,"tool_calls":[{"id":"f","function":{"name":"finish","arguments":encode(
                {"answer":"cache 在内存保存 15 分钟 [S1]","citations":["S1"],"status":"answered"})}}]}, {"prompt_tokens":10,"completion_tokens":5}
    FixedRAG(db,cfg,Model()).execute_fixed(run_id,[])
    assert db.one("SELECT status FROM runs WHERE id=?",(run_id,))["status"] == "answered"


def test_diagnostics_do_not_pretend_to_be_semantic_and_failures_count():
    c={"expected_action":"answered","required_facts":[["15"]],"forbidden_strings":[],"evidence":[]}
    m=answer_metrics(c,{"status":"error","answer":None,"citations":[]})
    assert m["failed"] and not m["action_correct"] and not m["citation_format_valid"]
    assert m["semantic_correctness"] is None and m["unsupported_answer"] is None
    c["expected_action"]="insufficient"
    m=answer_metrics(c,{"status":"answered","answer":"15 [S1]","citations":[{"label":"S1"}]})
    assert m["unsupported_answer"] and m["false_refusal"] is None
    left=[{"id":str(i),"family_id":str(i//4),"metrics":{"x":0}} for i in range(8)]
    right=[{**r,"metrics":{"x":1}} for r in left]
    d=paired_difference(left,right,"x")
    assert d["mean"]==1 and d["families"]==2
    with pytest.raises(ValueError, match="same completed"):
        paired_difference(left,right[:-1],"x")


def test_resume_repairs_only_an_incomplete_final_record(tmp_path):
    from kb.benchmark.runner import resume_rows
    p=tmp_path/'predictions.jsonl'
    p.write_bytes(b'{"id":"a"}\n{"id":')
    assert resume_rows(p)==[{"id":"a"}]
    assert p.read_bytes()==b'{"id":"a"}\n'
    assert p.with_suffix('.interrupted_tail').read_bytes()==b'{"id":'
    p.write_bytes(b'{bad}\n{"id":"b"}\n')
    with pytest.raises(ValueError,match="Corrupt"):
        resume_rows(p)


def test_import_grades_requires_full_coverage_and_real_judgments(tmp_path):
    from kb.benchmark.grading import apply_grades
    root=tmp_path/'run';root.mkdir()
    write_json(root/'config.json',{"case_ids":["a","b"]})
    rows=[{"id":x,"family_id":x,"category":"lookup","status":"error","metrics":{},"latency_seconds":1} for x in ['a','b']]
    write_jsonl(root/'predictions.jsonl',rows)
    grades=tmp_path/'grades.jsonl'
    g={"id":"a","correctness":0,"faithfulness":0,"citation_support":0,"rationale":"无答案","reviewer":"tester","reviewer_type":"human"}
    write_jsonl(grades,[g])
    with pytest.raises(ValueError,match="every case"):
        apply_grades(root,grades,tmp_path/'scores')
    write_jsonl(grades,[g,{**g,"id":"b","correctness":1}])
    with pytest.raises(ValueError,match="failed/no-answer"):
        apply_grades(root,grades,tmp_path/'scores')
    write_jsonl(grades,[g,{**g,"id":"b"}])
    report=apply_grades(root,grades,tmp_path/'scores')
    assert report['metrics']['semantic_correctness']['mean']==0


def test_comparison_keeps_failures_and_reports_undefined_citation_precision(tmp_path):
    from kb.benchmark.runner import compare
    left, right = tmp_path/'left', tmp_path/'right'
    config = {"dataset_receipt":{}, "split":"dev", "case_ids":["a","b"], "llm_model":"test",
              "context_tokens":8192, "output_tokens":1400, "temperature":0, "source_fingerprint":"frozen"}
    base = [{"id":x, "family_id":x, "category":"lookup", "latency_seconds":1,
             "metrics":{"failed":False, "action_correct":True, "citation_gold_evidence_recall":1.,
                        "citation_gold_evidence_precision":1.}} for x in config["case_ids"]]
    changed = [{**r, "metrics":dict(r["metrics"])} for r in base]
    changed[1]["metrics"].update(failed=True, action_correct=False, citation_gold_evidence_recall=0.,
                                citation_gold_evidence_precision=None)
    for path, rows in ((left,base), (right,changed)):
        path.mkdir()
        write_json(path/'config.json',config)
        write_jsonl(path/'predictions.jsonl',rows)
    originals=[(p/'predictions.jsonl').read_bytes() for p in (left,right)]
    result=compare(left,right,tmp_path/'report')
    assert result['deltas']['failed']['mean']==.5
    assert result['deltas']['action_correct']['mean']==-.5
    assert result['deltas']['citation_gold_evidence_recall']['mean']==-.5
    key='citation_gold_evidence_precision'
    assert key not in result['deltas'] and key in result['unpaired_metrics']
    assert result['metric_availability'][key]['left_defined']==2
    assert result['metric_availability'][key]['right_defined']==1
    assert result['left_summary']['records']==result['right_summary']['records']==2
    assert originals==[(p/'predictions.jsonl').read_bytes() for p in (left,right)]
    assert (tmp_path/'report/comparison.md').exists()
    # Missing a gold-conditioned score remains an error, not a silent omission.
    changed[1]['metrics']['action_correct']=None
    write_jsonl(right/'predictions.jsonl',changed)
    with pytest.raises(ValueError,match='Unequal metric availability: action_correct'):
        compare(left,right,tmp_path/'bad')
    write_jsonl(right/'predictions.jsonl',[changed[0],changed[0]])
    with pytest.raises(ValueError,match='Duplicate or unexpected'):
        compare(left,right,tmp_path/'bad')


def test_paired_difference_rejects_duplicates_and_wrong_families():
    left=[{'id':'a','family_id':'one','metrics':{'x':1}}]
    with pytest.raises(ValueError,match='Duplicate'):
        paired_difference(left+left,left,'x')
    with pytest.raises(ValueError,match='family mismatch'):
        paired_difference(left,[{**left[0],'family_id':'two'}],'x')


def test_retrieval_comparison_excludes_evaluable_metadata_keeps_denominators(tmp_path):
    from kb.benchmark.runner import compare
    left, right = tmp_path / 'bm25', tmp_path / 'rerank'
    cfg = {'dataset_receipt': {}, 'split': 'dev', 'case_ids': ['a','b','c'], 'llm_model': None,
           'context_tokens': 8192, 'output_tokens': 1400, 'temperature': 0, 'source_fingerprint': 'original'}
    rows = [{'id': x, 'family_id': x, 'category': 'lookup', 'latency_seconds': 1,
             'metrics': {'evaluable': x != 'c', 'returned': 5,
                         **({'evidence_recall_at_5': 0.5} if x != 'c' else {})}} for x in cfg['case_ids']]
    changed = [{**r, 'metrics': dict(r['metrics'])} for r in rows]
    changed[0]['metrics']['evidence_recall_at_5'] = 1.
    for path, data in ((left, rows), (right, changed)):
        path.mkdir()
        write_json(path/'config.json', cfg)
        write_jsonl(path/'predictions.jsonl', data)
    report = compare(left, right, tmp_path/'comparison')
    assert 'evaluable' not in report['deltas']
    assert report['deltas']['evidence_recall_at_5']['mean'] == .25
    assert report['metric_availability']['evidence_recall_at_5']['both_defined'] == 2
    assert report['left_summary']['records'] == report['right_summary']['records'] == 3
    assert (tmp_path/'comparison/comparison.md').exists()
