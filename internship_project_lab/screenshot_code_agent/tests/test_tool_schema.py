import pytest

from screenshot_agent.config import Limits
from screenshot_agent.control import Budget, Events
from screenshot_agent.tools import FunctionTool, schema


def test_large_tool_feedback_is_bounded_while_full_event_is_preserved(tmp_path):
    import json
    events = Events(tmp_path / "events.jsonl")
    budget = Budget(Limits(max_tool_output_chars=512), events)
    payload = "start_" + '\\"\n' * 2000 + "_end"
    tool = FunctionTool("read_file", "fixture", schema({}), lambda: payload, budget, events)
    encoded = tool.execute()
    assert len(encoded) <= 512
    output = json.loads(encoded)
    assert output["truncated"] and output["original_chars"] == len(payload)
    assert output["output"].startswith("start_") and output["output"].endswith("_end")
    assert len(output["output"]) < 700
    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert next(r for r in records if r["event"] == "tool_result")["output"] == payload
    assert any(r["event"] == "tool_output_truncated" for r in records)


def test_unadvertised_arguments_cannot_change_tool_operation(tmp_path):
    called = []
    events = Events(tmp_path / "events.jsonl")
    budget = Budget(Limits(), events)
    tool = FunctionTool("read_file", "read only", schema({"file_path": {"type": "string"}}, ["file_path"]),
                        lambda **kwargs: called.append(kwargs), budget, events)
    with pytest.raises(ValueError, match="schema"):
        tool.execute(file_path="a.py", _operation="write_file", content="forged")
    with pytest.raises(ValueError, match="type"):
        tool.execute(file_path=17)
    assert called == []


def test_ocr_reads_are_not_counted_as_vision_api_calls(tmp_path):
    from screenshot_agent.control import BudgetExceeded
    from screenshot_agent.images import Reading
    from screenshot_agent.tools import build_tools
    class Reader:
        name = "ocr"
        def read(self, image_id):
            return Reading(raw_text="observed text")
    class Images:
        def crop(self, image_id, bbox):
            return {}
    events = Events(tmp_path / "events.jsonl")
    budget = Budget(Limits(max_vision_calls=1), events)
    tools = build_tools(None, Images(), Reader(), budget, events, "unused")
    reader = next(t for t in tools if t.name == "read_screenshot")
    reader.execute(image_id="shot")
    assert budget.screenshot_calls == 1
    assert budget.vision_calls == 0
    with pytest.raises(BudgetExceeded):
        reader.execute(image_id="shot")
