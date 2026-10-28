import json
import pytest
from kb.agent import Harness, ModelClient
from kb.db import encode
from test_application import upload


def call(name,args):
    return {"role":"assistant","content":None,"tool_calls":[{"id":"call_"+name,"type":"function","function":{"name":name,"arguments":json.dumps(args,ensure_ascii=False)}}]}, {"total_tokens":10}


class ScriptedModel:
    def complete(self,messages):
        tool_messages=[m for m in messages if m["role"]=="tool"]
        if not tool_messages:
            return call("search",{"query":"住宿报销"})
        last=json.loads(tool_messages[-1]["content"])
        if "results" in last:
            return call("read_chunk",{"chunk_id":last["results"][0]["chunk_id"]})
        return call("finish",{"answer":"住宿标准每晚 600 元 [S1]","citations":["S1"],"status":"answered"})


def run(system,model):
    cfg,db,client,base,*_=system
    cfg.agent_policy="legacy_v1"
    upload(system)
    row=client.post("/api/chat",json={"question":"住宿报销标准", "base_ids":[base]}).json()
    db.execute("UPDATE runs SET mode='agent' WHERE id=?",(row["run_id"],))
    Harness(db,cfg,model).execute(row["run_id"])
    return row,client.get("/api/runs/"+row["run_id"]).json()


def test_model_driven_search_read_finish_with_verified_sources(system):
    row,result=run(system,ScriptedModel())
    assert result["status"]=="answered"
    assert result["citations"][0]["label"]=="S1"
    assert [e["payload"]["name"] for e in result["events"] if e["kind"]=="tool"]==["search","read_chunk"]
    cfg,db,*_=system
    assert db.one("SELECT count(*) AS n FROM messages WHERE run_id=? AND role='assistant'",(row["run_id"],))["n"]==1
    Harness(db,cfg,ScriptedModel()).execute(row["run_id"])
    assert db.one("SELECT count(*) AS n FROM messages WHERE run_id=? AND role='assistant'",(row["run_id"],))["n"]==1


def test_fabricated_citation_gets_tool_feedback_then_recovery(system):
    class Model(ScriptedModel):
        def complete(self,messages):
            if len(messages)==2:
                return call("finish",{"answer":"不存在的答案 [S999]","citations":["S999"],"status":"answered"})
            if "引用编号不存在" in messages[-1].get("content",""):
                return call("search",{"query":"住宿报销"})
            return super().complete(messages)
    _,result=run(system,Model())
    assert result["status"]=="answered" and "S999" not in result["answer"]
    assert any(e["kind"]=="tool" and "error" in e["payload"]["result"] for e in result["events"])


def test_missing_inline_citation_rejected_and_budget_bounded(system):
    class Model(ScriptedModel):
        def complete(self,messages):
            if any(m["role"]=="tool" and '"citation": "S1"' in m["content"] for m in messages):
                return call("finish",{"answer":"住宿标准为 600 元","citations":["S1"],"status":"answered"})
            return super().complete(messages)
    _,result=run(system,Model())
    assert result["status"]=="error" and not result["answer"]
    assert sum(e["kind"]=="model" for e in result["events"])==system[0].max_turns


def test_unknown_tool_and_invalid_json_do_not_execute_code(system):
    class Model:
        def complete(self,messages):
            return call("exec",{"code":"dangerous()"})
    _,result=run(system,Model())
    assert result["status"]=="error"
    assert any(e["kind"]=="tool" and e["payload"]["result"].get("error")=="未知工具" for e in result["events"])


def test_checkpoint_recovery_preserves_budget_and_complete_tool_messages(system):
    class Crash(ScriptedModel):
        def complete(self,messages):
            if any(m["role"]=="tool" for m in messages):
                raise SystemExit("simulated process death")
            return super().complete(messages)
    with pytest.raises(SystemExit):
        run(system,Crash())
    cfg,db,client,*_=system
    stored=db.one("SELECT * FROM runs WHERE status='running'")
    assert json.loads(stored["state"])["turns"]==1
    assert json.loads(stored["state"])["messages"][-1]["role"]=="tool"
    db.execute("UPDATE runs SET status='queued' WHERE id=?",(stored["id"],))
    Harness(db,cfg,ScriptedModel()).execute(stored["id"])
    result=client.get("/api/runs/"+stored["id"]).json()
    assert result["status"]=="answered"
    assert json.loads(db.one("SELECT state FROM runs WHERE id=?",(stored["id"],))["state"])["turns"]==3


def test_cancellation_during_model_request_blocks_answer(system):
    cfg,db,*_=system
    class Cancel(ScriptedModel):
        def complete(self,messages):
            db.execute("UPDATE runs SET cancel_requested=1 WHERE status='running'")
            return super().complete(messages)
    _,result=run(system,Cancel())
    assert result["status"]=="cancelled" and result["answer"] is None


def test_model_http_transport_uses_tools_and_no_secret_in_error(system,monkeypatch):
    import httpx
    cfg=system[0];cfg.llm_url="http://local.invalid/v1";cfg.llm_model="test";cfg.llm_key="secret-test"
    requests=[]
    def handler(request):
        requests.append(request)
        return httpx.Response(200,json={"choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{"total_tokens":1}})
    real=httpx.Client
    monkeypatch.setattr("kb.agent.httpx.Client",lambda **kwargs:real(transport=httpx.MockTransport(handler)))
    result,usage=ModelClient(cfg).complete([{"role":"user","content":"hello"}])
    payload=json.loads(requests[0].content)
    assert requests[0].url.path=="/v1/chat/completions" and len(payload["tools"])==4
    assert result["content"]=="ok" and usage["total_tokens"]==1


def test_failure_not_silently_replaced_by_extractive_answer(system):
    class Model:
        def complete(self,messages):
            raise RuntimeError("模型服务连接失败或超时，请检查服务后重试")
    _,result=run(system,Model())
    assert result["status"]=="error" and result["answer"] is None
    assert "模型服务" in result["error"]
