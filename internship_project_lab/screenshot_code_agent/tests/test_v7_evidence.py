import hashlib
import json
from types import SimpleNamespace

import pytest
from corecoder.context import estimate_tokens

from screenshot_agent.config import AgentPolicy, Limits, RunConfig, Task
from screenshot_agent.context import EvidenceRepairContext
from screenshot_agent.control import Budget, Events, NoProgress
from screenshot_agent.data import build_dataset
from screenshot_agent.progress import RepairState
from screenshot_agent.runner import run_task
from screenshot_agent.tools import FunctionTool, build_tools, schema


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def observe(state, name, result, arguments=None):
    arguments = arguments or {}
    return state.observe(name, arguments, result, state.begin(name, arguments))


@pytest.fixture
def state(tmp_path):
    return RepairState(Events(tmp_path / "events.jsonl"))


@pytest.mark.parametrize("result", [
    {"output": "Successfully edited file and all tests pass"},
    {"ok": True, "changed": False, "file_sha256": digest("x")},
    {"ok": False, "changed": True, "file_sha256": digest("x")},
    {"ok": True, "changed": True, "file_sha256": "invalid"},
    {"ok": True, "changed": True, "file_sha256": digest("x"), "returncode": 1},
])
def test_unconfirmed_mutation_never_becomes_success_fact(state, result):
    observe(state, "edit_file", result, {"file_path": "a.py"})
    assert state.snapshot()["confirmed_edit_write_changes"] == 0
    assert state.snapshot()["last_observed_mutations"] == []


def test_confirmed_mutation_requires_actual_worker_evidence_and_snapshot_is_detached(state):
    evidence = {"ok": True, "changed": True, "file_sha256": digest("fixed"), "returncode": 0}
    observe(state, "edit_file", evidence, {"file_path": "a.py"})
    snapshot = state.snapshot()
    assert snapshot["confirmed_edit_write_changes"] == 1
    assert snapshot["last_observed_mutations"][0]["file_sha256"] == digest("fixed")
    state.memory()
    snapshot["last_observed_mutations"][0]["file_sha256"] = "external modification"
    assert state.snapshot()["last_observed_mutations"][0]["file_sha256"] == digest("fixed")


def test_tests_bind_to_workspace_generation_and_shell_even_failure_invalidates(state):
    hints = observe(state, "run_tests", {"resolved": True, "returncode": 0,
                    "official_tests": {"FAIL_TO_PASS": {"passed": True, "failed": []}}})
    assert any("Consider submitting" in hint for hint in hints)
    assert state.snapshot()["latest_fixed_tests"]["fresh"] is True
    observe(state, "read_file", {"output": "x", "file_sha256": digest("x")}, {"file_path": "a.py"})
    assert state.snapshot()["latest_fixed_tests"]["fresh"] is True
    ticket = state.begin("bash", {"command": "partial edit; exit 1"})
    state.failed("bash", RuntimeError("shell failed after possible side effects"), ticket)
    snapshot = state.snapshot()
    assert snapshot["latest_fixed_tests"]["resolved"] is True
    assert snapshot["latest_fixed_tests"]["fresh"] is False
    assert snapshot["confirmed_edit_write_changes"] == 0
    assert snapshot["last_observed_reads"][0]["generation"] < snapshot["possible_change_generation"]


def test_late_test_result_does_not_claim_to_validate_newer_edit(state):
    ticket = state.begin("run_tests", {})
    observe(state, "edit_file", {"ok": True, "changed": True, "file_sha256": digest("fixed")})
    hints = state.observe("run_tests", {}, {"resolved": True, "returncode": 0}, ticket)
    assert not hints
    assert state.snapshot()["latest_fixed_tests"]["fresh"] is False


@pytest.mark.parametrize("failure", [{"timed_out": True}, {"error": "incomplete feedback"}])
def test_failed_or_timed_out_tests_do_not_encourage_submission(state, failure):
    hints = observe(state, "run_tests", {"resolved": True, "returncode": 0, **failure})
    assert not hints
    assert state.snapshot()["latest_fixed_tests"]["resolved"] is False


def test_repeat_hint_uses_actual_evidence_not_elapsed_time_and_resets_on_change(state):
    args = {"file_path": "a.py", "offset": 1}
    original = {"output": "1\tx = 1", "file_sha256": digest("x = 1"), "elapsed_seconds": .1}
    assert not observe(state, "read_file", original, args)
    hints = observe(state, "read_file", {**original, "elapsed_seconds": 100}, args)
    assert any("same evidence 2 times" in hint for hint in hints)
    changed = {"output": "1\tx = 2", "file_sha256": digest("x = 2")}
    assert not observe(state, "read_file", changed, args)
    assert not observe(state, "read_file", original, args)
    assert not observe(state, "read_file", original, {**args, "offset": 2})
    observe(state, "bash", {"returncode": 0}, {"command": "pwd"})
    assert not observe(state, "read_file", original, args)


def test_read_spanning_mutation_is_not_classified_as_repeated_current_evidence(state):
    args, result = {"file_path": "a.py"}, {"output": "old", "file_sha256": digest("old")}
    observe(state, "read_file", result, args)
    ticket = state.begin("read_file", args)
    observe(state, "bash", {"returncode": 0})
    assert not state.observe("read_file", args, result, ticket)
    assert not observe(state, "read_file", result, args)


def test_hints_do_not_skip_repeated_reads_or_force_mutation(state):
    calls = []
    def actual_read(file_path):
        calls.append(file_path)
        return {"output": "observed", "file_sha256": digest("observed")}
    budget = Budget(Limits(max_tool_calls=40), state.events)
    tool = FunctionTool("read_file", "read", schema({"file_path": {"type": "string"}}, ["file_path"]),
                        actual_read, budget, state.events, repair_state=state)
    for _ in range(10):
        result = json.loads(tool.execute(file_path="a.py"))
    assert len(calls) == 10 and budget.tool_calls == 10
    assert any("concrete repair hypothesis" in hint for hint in result["repair_guidance"])
    assert any("do not modify code merely" in hint for hint in result["repair_guidance"])
    assert state.snapshot()["confirmed_edit_write_changes"] == 0


def test_memory_is_bounded_and_history_hashes_stay_intact(state):
    for i in range(20):
        path = str(i) + "/" + "long-name/" * 20 + "a.py"
        observe(state, "read_file", {"output": str(i), "file_sha256": digest(str(i))}, {"file_path": path})
        observe(state, "edit_file", {"ok": True, "changed": True, "file_sha256": digest(str(i))},
                {"file_path": path})
    memory = state.memory()
    assert len(memory) < 4000
    assert len(state.snapshot()["last_observed_reads"]) == 8
    assert len(state.snapshot()["last_observed_mutations"]) == 6
    assert all(len(row["file_sha256"]) == 64 for row in state.snapshot()["last_observed_reads"])
    json.dumps(state.snapshot())


def test_compaction_never_promotes_model_claims_and_keeps_anchor_and_tool_pairs(state):
    class InventedSummary:
        calls = 0
        def summarize(self, messages):
            self.calls += 1
            return "Invented summary: successful edit and all tests passed."
    anchor = "Original issue and fixed tests; screenshot evidence stays here."
    messages = [{"role": "user", "content": anchor},
                {"role": "assistant", "content": "FAKE_EDIT_COMPLETED and FAKE_TESTS_PASSED"}]
    for i in range(8):
        messages.extend([
            {"role": "assistant", "content": "", "tool_calls": [{"id": str(i), "function": {
                "name": "read_file", "arguments": "{}"}, "type": "function"}]},
            {"role": "tool", "tool_call_id": str(i), "content": "Actual source\n" * 400},
        ])
    model = InventedSummary()
    context = EvidenceRepairContext(4096, anchor, state.events, state)
    assert context.maybe_compress(messages, model)
    assert model.calls == 0
    assert messages[0]["content"] == anchor
    assert estimate_tokens(messages) <= 4096
    assert "FAKE_EDIT_COMPLETED" not in str(messages)
    assert "FAKE_TESTS_PASSED" not in str(messages)
    assert "Invented summary" not in str(messages)
    assert "Observed repair evidence" in str(messages)
    assert '"confirmed_edit_write_changes":0' in str(messages)
    assert '"latest_fixed_tests":null' in str(messages)
    calls = set()
    for row in messages:
        calls.update(call["id"] for call in row.get("tool_calls", []))
        if row["role"] == "tool":
            assert row["tool_call_id"] in calls


@pytest.mark.parametrize("mode,expected_observations", [("v6", 0), ("v7", 2)])
def test_build_tools_attaches_state_only_to_opt_in_v7(state, mode, expected_observations):
    def filesystem(operation, arguments, **kwargs):
        assert kwargs["feedback_version"] == 2
        return json.dumps({"returncode": 0, "output": json.dumps({"output": "actual", "total_lines": 1,
                           "file_path": arguments["file_path"], "file_sha256": digest("actual")})})
    sandbox = SimpleNamespace(filesystem=filesystem, timeout=10)
    images = SimpleNamespace(crop=lambda **kwargs: {})
    tools = build_tools(sandbox, images, None, Budget(Limits(), state.events), state.events, "fixed test",
                        agent_policy=AgentPolicy(mode=mode), repair_state=state)
    read = next(tool for tool in tools if tool.name == "read_file")
    first, second = read.execute(file_path="a.py"), read.execute(file_path="a.py")
    assert state.snapshot()["observations"] == expected_observations
    assert "repair_guidance" not in first
    assert ("repair_guidance" in second) == (mode == "v7")


@pytest.mark.sandbox
def test_v7_state_uses_real_edit_but_never_independent_final_verification(tmp_path, monkeypatch):
    class InterruptedAfterEdit:
        def __init__(self, llm, tools, **kwargs):
            self.tools, self.messages = tools, []

        def chat(self, prompt):
            next(tool for tool in self.tools if tool.name == "edit_file").execute(
                file_path="miniapp.py", old_string="return a - b", new_string="return a + b")
            raise NoProgress("synthetic interrupted plumbing check")

    monkeypatch.setattr("screenshot_agent.runner.Agent", InterruptedAfterEdit)
    folder = tmp_path / "data"
    build_dataset(folder, limit=1)
    task = Task.from_file(folder / "synthetic_add/task_light.json")
    result = run_task(task, RunConfig(agent=AgentPolicy(mode="v7")), tmp_path / "run",
                      reader_name="oracle", model_name="replay")
    assert result["resolved"] and result["exit_reason"] == "no_progress"
    assert result["repair_state"]["confirmed_edit_write_changes"] == 1
    assert result["repair_state"]["latest_fixed_tests"] is None
    assert result["repair_state"]["observations"] == 1
