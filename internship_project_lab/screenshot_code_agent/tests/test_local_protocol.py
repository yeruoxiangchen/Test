import importlib.util
import json
from pathlib import Path


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/local_model_api.py"
    spec = importlib.util.spec_from_file_location("local_smoke_api", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_local_qwen_tool_parser_preserves_structured_arguments():
    text = 'Inspecting.\n<tool_call>\n{"name":"read_file","arguments":{"file_path":"a.py"}}\n</tool_call>'
    content, calls = module().parse_tools(text)
    assert content == "Inspecting."
    assert calls[0]["function"]["name"] == "read_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {"file_path": "a.py"}


def test_incomplete_tool_call_is_not_executed():
    text = '<tool_call>{"name":"bash"'
    content, calls = module().parse_tools(text)
    assert content == text
    assert not calls


def test_all_qwen_processor_messages_use_content_blocks():
    rows = module().normalize_messages([
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "fix"},
        {"role": "assistant", "content": None, "tool_calls": [{"type": "function", "id": "a",
            "function": {"name": "read_file", "arguments": '{"file_path":"a.py"}'}}]},
        {"role": "tool", "content": "result", "tool_call_id": "a"},
    ])
    assert all(isinstance(r["content"], list) for r in rows)
    assert rows[0]["content"] == [{"type": "text", "text": "rules"}]
    assert rows[2]["tool_calls"][0]["function"]["arguments"] == {"file_path": "a.py"}
