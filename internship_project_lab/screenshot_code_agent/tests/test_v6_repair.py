import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from corecoder.context import estimate_tokens
from openai import OpenAI

from screenshot_agent.config import AgentPolicy, Limits, ModelConfig, RunConfig, Task
from screenshot_agent.context import RepairContext
from screenshot_agent.control import Budget, BudgetExceeded, Events, NoProgress, SummaryUnavailable
from screenshot_agent.data import build_dataset
from screenshot_agent.models import APIModel, ModelOutputError
from screenshot_agent.runner import run_task
from screenshot_agent.tools import build_tools, filesystem_result


def tools_for(sandbox, tmp_path):
    events = Events(tmp_path / "v6_events.jsonl")
    budget = Budget(Limits(max_steps=30), events, separate_summaries=True)
    images = SimpleNamespace(crop=lambda **kwargs: {})
    return {t.name: t for t in build_tools(sandbox, images, None, budget, events, "python3 -m unittest",
                                         agent_policy=AgentPolicy(mode="v6", read_file_max_lines=20))}


@pytest.mark.sandbox
def test_real_multiline_feedback_and_read_range(sandbox, tmp_path):
    source = "\n".join(f"value_{i} = {i}" for i in range(80)) + "\n"
    (sandbox.repo / "many.py").write_text(source)
    tool = tools_for(sandbox, tmp_path)["read_file"]
    text = tool.execute(file_path="many.py", limit=500)
    assert "1\tvalue_0 = 0\n2\tvalue_1 = 1" in text
    assert "20\tvalue_19" in text and "21\tvalue_20" not in text
    assert "next_offset: 21" in text
    assert "total_lines: 80" in text


@pytest.mark.sandbox
def test_failed_edit_feedback_recovery_and_file_hash_guard(sandbox, tmp_path):
    tools = tools_for(sandbox, tmp_path)
    wrong = dict(file_path="sample.py", old_string="value = 7", new_string="value = 9")
    first = tools["edit_file"].execute(**wrong)
    assert 'error: "edit_match_failed"' in first and "1\tvalue = 1" in first
    tools["edit_file"].execute(**wrong)
    blocked = tools["edit_file"].execute(**wrong)
    assert "repeated_failed_edit" in blocked
    assert (sandbox.repo / "sample.py").read_text() == "value = 1\n"
    # A real change, including a shell edit, permits the previously failed arguments again.
    sandbox.run("printf 'value = 7\\n' > sample.py")
    success = tools["edit_file"].execute(**wrong)
    assert "python_syntax:" in success and '"valid": true' in success
    assert (sandbox.repo / "sample.py").read_text() == "value = 9\n"


@pytest.mark.sandbox
def test_unrecovered_identical_failure_stops_instead_of_spinning(sandbox, tmp_path):
    tool = tools_for(sandbox, tmp_path)["edit_file"]
    args = dict(file_path="sample.py", old_string="absent", new_string="other")
    for _ in range(3):
        tool.execute(**args)
    with pytest.raises(NoProgress):
        tool.execute(**args)


@pytest.mark.sandbox
def test_syntax_feedback_does_not_hide_or_rollback_edit(sandbox, tmp_path):
    result = tools_for(sandbox, tmp_path)["edit_file"].execute(
        file_path="sample.py", old_string="value = 1", new_string="value = (")
    assert '"valid": false' in result
    assert (sandbox.repo / "sample.py").read_text() == "value = (\n"


@pytest.mark.sandbox
def test_enhanced_tools_still_protect_tests_and_require_unique_match(sandbox, tmp_path):
    tool = tools_for(sandbox, tmp_path)["edit_file"]
    result = tool.execute(file_path="tests/test_fixed.py", old_string="immutable", new_string="changed")
    assert "returncode: 1" in result
    assert (sandbox.repo / "tests/test_fixed.py").read_text() == "immutable\n"
    (sandbox.repo / "repeat.py").write_text("x = 1\nx = 1\n")
    result = tool.execute(file_path="repeat.py", old_string="x = 1", new_string="x = 2")
    assert "match_count: 2" in result
    assert (sandbox.repo / "repeat.py").read_text() == "x = 1\nx = 1\n"


def test_worker_unwrapping_never_decodes_source_as_protocol():
    source = '{"output": "This is file content"}\nnext_line'
    raw = json.dumps({"returncode": 0, "output": json.dumps({"output": source, "total_lines": 2})})
    assert filesystem_result(raw)["output"] == source


def test_separate_summary_budget_retains_shared_cap(tmp_path):
    budget = Budget(Limits(max_steps=3, max_summary_calls=1, max_model_calls=3),
                    Events(tmp_path / "events"), separate_summaries=True)
    config = ModelConfig(input_usd_per_million=0, output_usd_per_million=0)
    budget.reserve("summary", config, [])
    with pytest.raises(SummaryUnavailable):
        budget.reserve("summary", config, [])
    budget.reserve("code", config, [])
    budget.reserve("code", config, [])
    with pytest.raises(BudgetExceeded):
        budget.reserve("code", config, [])
    assert budget.code_calls == 3 and budget.summary_calls == 1 and budget.decision_calls == 2


def test_summary_cannot_spend_last_shared_decision(tmp_path):
    budget = Budget(Limits(max_model_calls=1), Events(tmp_path / "events"), separate_summaries=True)
    config = ModelConfig(input_usd_per_million=0, output_usd_per_million=0)
    with pytest.raises(SummaryUnavailable):
        budget.reserve("summary", config, [])
    budget.reserve("code", config, [])


def test_compaction_preserves_task_and_tool_pairs_with_summary_fallback(tmp_path):
    class NoSummary:
        def summarize(self, messages):
            raise SummaryUnavailable("used up")
    anchor = "Fix the original issue. Tests: fixed_test. Screenshot: original evidence."
    messages = [{"role": "user", "content": anchor}]
    for i in range(8):
        messages.extend([
            {"role": "assistant", "content": "", "tool_calls": [{"id": str(i), "function": {
                "name": "read_file", "arguments": "{}"}, "type": "function"}]},
            {"role": "tool", "tool_call_id": str(i), "content": "actual source\n" * 400},
        ])
    context = RepairContext(4096, anchor, Events(tmp_path / "events"))
    assert context.maybe_compress(messages, NoSummary())
    assert messages[0]["content"] == anchor
    assert estimate_tokens(messages) <= 4096
    seen = set()
    for row in messages:
        seen.update(c["id"] for c in row.get("tool_calls", []))
        if row["role"] == "tool":
            assert row["tool_call_id"] in seen
    assert "Extracted earlier observations" in str(messages)


def make_model(tmp_path, monkeypatch, finish="stop"):
    monkeypatch.setenv("SCA_V6_TEST_KEY", "test-only")
    config = ModelConfig(model="mock", api_key_env="SCA_V6_TEST_KEY", max_output_tokens=2048,
                         input_usd_per_million=0, output_usd_per_million=0)
    events = Events(tmp_path / "events")
    budget = Budget(Limits(), events, separate_summaries=True)
    model = APIModel(config, budget, events, policy=AgentPolicy(mode="v6"))
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
            "model": "mock", "choices": [{"index": 0, "finish_reason": finish,
                "message": {"role": "assistant", "content": "Observed edits and failed tests."}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}})
    model.client = OpenAI(api_key="test-only", base_url="http://unit.test/v1", max_retries=0,
                         http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return model, budget, requests


def test_model_summary_is_short_and_accounted_separately(tmp_path, monkeypatch):
    model, budget, requests = make_model(tmp_path, monkeypatch)
    model.summarize([{"role": "user", "content": "summarize"}])
    assert requests[0]["max_tokens"] == 384 and "tools" not in requests[0]
    assert budget.summary_calls == 1 and budget.decision_calls == 0


def test_truncated_output_is_not_a_submission(tmp_path, monkeypatch):
    model, budget, requests = make_model(tmp_path, monkeypatch, finish="length")
    with pytest.raises(ModelOutputError, match="output_truncated"):
        model.chat([{"role": "user", "content": "fix"}])
    assert budget.code_calls == 1


def test_finish_reason_distinguishes_eos_from_output_limit():
    path = Path(__file__).resolve().parents[1] / "scripts/local_model_api.py"
    spec = importlib.util.spec_from_file_location("v6_local", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    finish = module.generation_finish_reason
    assert finish([1, 2, 3], 3, [9], False) == "length"
    assert finish([1, 2, 9], 3, [9], False) == "stop"
    assert finish([1, 2, 9], 3, [9], True) == "tool_calls"
    assert finish([1, 2, 3], 3, [9], True) == "length"


@pytest.mark.sandbox
def test_v6_complete_replay_uses_independent_verification(tmp_path):
    folder = tmp_path / "data"
    build_dataset(folder, limit=1)
    task = Task.from_file(folder / "synthetic_add/task_light.json")
    config = RunConfig(agent=AgentPolicy(mode="v6"), limits=Limits(max_wall_seconds=120))
    result = run_task(task, config, tmp_path / "run", reader_name="oracle", model_name="replay")
    assert result["resolved"] and result["validation"]["patch_replay"]["applied"]
    assert result["model_inference"] is False


@pytest.mark.sandbox
@pytest.mark.parametrize("exception,reason", [(NoProgress("repeat"), "no_progress"),
                                              (ModelOutputError("output_truncated"), "output_truncated")])
def test_early_stop_still_exports_and_verifies_patch(tmp_path, monkeypatch, exception, reason):
    class Interrupted:
        def __init__(self, llm, tools, **kwargs):
            self.tools, self.messages = tools, []
        def chat(self, prompt):
            next(t for t in self.tools if t.name == "edit_file").execute(
                file_path="miniapp.py", old_string="return a - b", new_string="return a + b")
            raise exception
    monkeypatch.setattr("screenshot_agent.runner.Agent", Interrupted)
    folder = tmp_path / "data"
    build_dataset(folder, limit=1)
    task = Task.from_file(folder / "synthetic_add/task_light.json")
    result = run_task(task, RunConfig(agent=AgentPolicy(mode="v6")), tmp_path / "run",
                      reader_name="oracle", model_name="replay")
    assert result["exit_reason"] == reason
    assert result["resolved"]
