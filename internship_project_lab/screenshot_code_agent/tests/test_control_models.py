import json

import httpx
import pytest
from openai import APITimeoutError, OpenAI

from screenshot_agent.config import Limits, ModelConfig, RunConfig
from screenshot_agent.control import Budget, BudgetExceeded, Events, RunStopped
from screenshot_agent.models import APIModel


def make_model(tmp_path, monkeypatch, handler, **kwargs):
    monkeypatch.setenv("SCA_UNIT_KEY", "unit-test-not-a-real-key")
    config = ModelConfig(model="unit-test", base_url="http://unit.test/v1", api_key_env="SCA_UNIT_KEY",
                         input_usd_per_million=1, output_usd_per_million=2, max_output_tokens=32)
    events = Events(tmp_path / "events.jsonl")
    budget = Budget(Limits(**kwargs), events)
    model = APIModel(config, budget, events)
    model.client = OpenAI(api_key="unit-test-not-a-real-key", base_url=config.base_url, max_retries=0,
                          http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return model, budget


def response(message, tokens=5):
    return httpx.Response(200, json={"id": "unit", "object": "chat.completion", "created": 0,
        "model": "unit-test", "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        "usage": {"prompt_tokens": 10, "completion_tokens": tokens, "total_tokens": 10+tokens}})


def test_api_tool_contract_and_charging(tmp_path, monkeypatch):
    def handler(request):
        payload = json.loads(request.content)
        assert "tools" in payload
        assert payload["stream"] is False
        return response({"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "read_file", "arguments": '{"file_path":"a.py"}'}}]})
    model, budget = make_model(tmp_path, monkeypatch, handler)
    value = model.chat([{"role": "user", "content": "fix"}], tools=[{"type": "function"}])
    assert value.tool_calls[0].arguments == {"file_path": "a.py"}
    assert value.message["tool_calls"][0]["function"]["name"] == "read_file"
    assert budget.cost_usd == pytest.approx(.00002)
    assert budget.reserved_usd == 0
    assert "unit-test-not-a-real-key" not in (tmp_path / "events.jsonl").read_text()


def test_preflight_budget_blocks_network_call(tmp_path, monkeypatch):
    called = []
    model, _ = make_model(tmp_path, monkeypatch, lambda r: called.append(r), max_cost_usd=.000001)
    with pytest.raises(BudgetExceeded):
        model.chat([{"role": "user", "content": "fix"}])
    assert not called


def test_bad_json_is_recoverable_and_billed(tmp_path, monkeypatch):
    model, budget = make_model(tmp_path, monkeypatch, lambda r: response(
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "read_file", "arguments": "{"}}]}))
    result = model.chat([{"role": "user", "content": "fix"}])
    assert result.tool_calls[0].name == "invalid_tool_arguments"
    assert budget.prompt_tokens == 10


def test_http_failure_keeps_reservation(tmp_path, monkeypatch):
    model, budget = make_model(tmp_path, monkeypatch, lambda r: httpx.Response(500, json={"error": "unit failure"}))
    with pytest.raises(Exception):
        model.chat([{"role": "user", "content": "fix"}])
    assert budget.cost_usd > 0
    assert budget.estimated_calls == 1
    assert budget.reserved_usd == 0


def test_request_timeout_is_recorded_once_and_keeps_cost_reservation(tmp_path,monkeypatch):
    requests=[]
    def fail(request):
        requests.append(request)
        raise httpx.ReadTimeout('simulated slow endpoint',request=request)
    model,budget=make_model(tmp_path,monkeypatch,fail)
    with pytest.raises(APITimeoutError):
        model.chat([{'role':'user','content':'fix'}])
    assert len(requests)==1
    assert budget.code_calls==1 and budget.estimated_calls==1
    assert budget.cost_usd>0 and budget.reserved_usd==0
    assert 'APITimeoutError' in (tmp_path/'events.jsonl').read_text()


def test_terminal_signal_is_not_recoverable_tool_error():
    assert issubclass(BudgetExceeded, RunStopped)
    assert not issubclass(BudgetExceeded, Exception)


def test_live_cap_requires_prices(monkeypatch):
    monkeypatch.setenv("SCA_UNIT_KEY", "fake")
    with pytest.raises(ValueError, match="prices"):
        ModelConfig(model="unit", api_key_env="SCA_UNIT_KEY").require_live(2)
    with pytest.raises(ValueError):
        RunConfig(limits={"max_steps": 0})
