"""CoreCoder Tool adapters; repository access always crosses the sandbox boundary."""

from __future__ import annotations

import json
import posixpath
import threading

from corecoder.tools.base import Tool

from .config import AgentPolicy, ScreenshotPolicy
from .control import Budget, Events, NoProgress
from .images import ImageStore
from .sandbox import BubblewrapSandbox


def schema(properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties, "required": required or [], "additionalProperties": False}


STRING = {"type": "string"}
INTEGER = {"type": "integer"}


def filesystem_result(raw) -> dict:
    """Unwrap the trusted worker protocol once; source text is never JSON-decoded again."""
    envelope = json.loads(raw) if isinstance(raw, str) else raw
    payload = envelope.get("output", "")
    try:
        payload = json.loads(payload)
    except (ValueError, TypeError):
        payload = {"output": payload}
    if not isinstance(payload, dict):
        payload = {"output": str(payload)}
    return {**payload, **{k: v for k, v in envelope.items() if k != "output"}}


def readable_result(value) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    metadata = []
    for key, item in value.items():
        if key != "output":
            metadata.append(f"{key}: {json.dumps(item, ensure_ascii=False)}")
    body = value.get("output", "")
    return "\n".join(metadata) + ("\n--- output ---\n" + str(body) if body else "")


class FunctionTool(Tool):
    def __init__(self, name: str, description: str, parameters: dict, function,
                 budget: Budget, events: Events, lock=None, *, readable=False, repair_state=None):
        self.name, self.description, self.parameters = name, description, parameters
        self.function, self.budget, self.events, self.lock = function, budget, events, lock
        self.readable = readable
        self.repair_state = repair_state

    def execute(self, **kwargs) -> str:
        self.budget.tool(self.name, kwargs)
        ticket = None
        try:
            properties = self.parameters["properties"]
            if kwargs.keys() - properties.keys() or set(self.parameters["required"]) - kwargs.keys():
                raise ValueError("Tool arguments do not match its declared schema")
            for key, value in kwargs.items():
                expected = properties[key].get("type")
                if ((expected == "string" and not isinstance(value, str))
                        or (expected == "integer" and type(value) is not int)
                        or (expected == "array" and not isinstance(value, list))):
                    raise ValueError(f"Invalid argument type for {key}: expected {expected}")
            def invoke():
                nonlocal ticket
                if self.repair_state is not None:
                    ticket = self.repair_state.begin(self.name, kwargs)
                value = self.function(**kwargs)
                if self.repair_state is not None:
                    hints = self.repair_state.observe(self.name, kwargs, value, ticket)
                    if hints and isinstance(value, dict):
                        value = {**value, "repair_guidance": hints}
                return value
            if self.lock is not None:
                with self.lock:
                    result = invoke()
            else:
                result = invoke()
            if self.readable:
                result = readable_result(result)
            elif not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)
            self.events.emit("tool_result", name=self.name, output=result)
            cap = self.budget.limits.max_tool_output_chars
            if len(result) > cap:
                # Full feedback remains in events; bound the text added to the
                # model context and make pagination/truncation observable.
                self.events.emit("tool_output_truncated", name=self.name, original_chars=len(result), limit=cap)
                if self.readable:
                    marker = "\n... output truncated; use a narrower read_file range ...\n"
                    available = cap - len(marker)
                    head = available * 2 // 3
                    return result[:head] + marker + result[-(available-head):]
                size = cap // 2
                while True:
                    head = size // 2
                    clipped = json.dumps({"truncated": True, "original_chars": len(result),
                        "output": result[:head] + "\n... output truncated; request a narrower range ...\n"
                                  + result[-(size-head):]}, ensure_ascii=False)
                    if len(clipped) <= cap:
                        return clipped
                    size = max(1, size * 4 // 5)
            return result
        except BaseException as error:
            if ticket is not None:
                self.repair_state.failed(self.name, error, ticket)
            self.events.emit("tool_error", name=self.name, error_type=type(error).__name__, error=str(error))
            raise


def build_tools(sandbox: BubblewrapSandbox, images: ImageStore, reader,
                budget: Budget, events: Events, test_command: str, test_runner=None,
                screenshot_policy: ScreenshotPolicy | None = None,
                agent_policy: AgentPolicy | None = None, repair_state=None) -> list[Tool]:
    screenshot_policy = screenshot_policy or ScreenshotPolicy()
    agent_policy = agent_policy or AgentPolicy()
    enhanced = agent_policy.mode in {"v6", "v7"}
    if agent_policy.mode != "v7":
        repair_state = None
    failed_edits = {}
    mutation_lock = threading.RLock()
    definitions = [
        ("read_file", "Read UTF-8 source with line numbers inside /workspace.",
         schema({"file_path": STRING, "offset": INTEGER, "limit": INTEGER}, ["file_path"])),
        ("edit_file", "Replace exactly one matching source snippet; include enough context for uniqueness.",
         schema({"file_path": STRING, "old_string": STRING, "new_string": STRING},
                ["file_path", "old_string", "new_string"])),
        ("write_file", "Create or replace a UTF-8 file inside /workspace.",
         schema({"file_path": STRING, "content": STRING}, ["file_path", "content"])),
        ("glob", "Find repository filenames using a relative glob pattern.", schema({"pattern": STRING}, ["pattern"])),
        ("grep", "Search repository text using a Python regular expression.",
         schema({"pattern": STRING, "path": STRING}, ["pattern"])),
    ]
    tools = []
    def filesystem_function(operation):
        def call(**arguments):
            if not enhanced:
                return sandbox.filesystem(operation, arguments, timeout=budget.remaining_seconds())
            if operation == "read_file":
                arguments["limit"] = min(max(1, arguments.get("limit", agent_policy.read_file_max_lines)),
                                         agent_policy.read_file_max_lines)
            signature = None
            guard = None
            if operation == "edit_file":
                name = posixpath.normpath(arguments["file_path"]).removeprefix("/workspace/")
                signature = (name, arguments["old_string"], arguments["new_string"])
                guard = failed_edits.get(signature)
            raw = sandbox.filesystem(operation, arguments, timeout=budget.remaining_seconds(),
                                     feedback_version=2, edit_guard=guard)
            result = filesystem_result(raw)
            if signature is not None and result.get("error") in {"edit_match_failed", "repeated_failed_edit"}:
                same = guard is not None and guard["file_sha256"] == result["file_sha256"]
                count = guard["failures"]+1 if same else 1
                blocks = guard.get("blocks", 0) if same else 0
                if result["error"] == "repeated_failed_edit":
                    blocks += 1
                failed_edits[signature] = {"file_sha256": result["file_sha256"], "failures": count,
                                          "limit": agent_policy.repeated_edit_limit, "blocks": blocks}
                events.emit("edit_failure", file_path=arguments["file_path"], error=result["error"],
                            match_count=result["match_count"], repeated=blocks > 0, attempts=count)
                if blocks >= 2:
                    raise NoProgress("Identical failed edit repeated on unchanged contents after recovery feedback")
            elif signature is not None and result.get("ok"):
                failed_edits.pop(signature, None)
            return result
        return call

    for name, description, parameters in definitions:
        call = filesystem_function(name)
        tools.append(FunctionTool(name, description, parameters, call, budget, events,
                                  mutation_lock if name in {"edit_file", "write_file"} else None,
                                  readable=enhanced, repair_state=repair_state))

    def bash(command: str, timeout: int | None = None):
        return sandbox.run(command, timeout=min(timeout or sandbox.timeout, budget.remaining_seconds()))

    def run_tests():
        if test_runner is not None:
            return test_runner()
        return sandbox.run(test_command, timeout=budget.remaining_seconds())

    def read_screenshot(image_id: str):
        budget.check()
        with budget.lock:
            if budget.screenshot_calls >= budget.limits.max_vision_calls:
                from .control import BudgetExceeded
                raise BudgetExceeded("Screenshot reading call limit reached")
            budget.screenshot_calls += 1
        reading = reader.read(image_id)
        events.emit("screenshot_read", image_id=image_id, reader=reader.name, reading=reading.model_dump())
        return {"image_id": image_id, "reader": reader.name, **reading.model_dump()}

    tools += [
        FunctionTool("bash", "Run a command in the isolated repository. cwd resets to /workspace for each call.",
                     schema({"command": STRING, "timeout": INTEGER}, ["command"]), bash, budget, events, mutation_lock,
                     readable=enhanced, repair_state=repair_state),
        FunctionTool("run_tests", "Execute the task's fixed test command in the isolated repository.",
                     schema({}), run_tests, budget, events, mutation_lock, readable=enhanced, repair_state=repair_state),
        FunctionTool("read_screenshot", "Read an allowed screenshot/crop; returns visible text and uncertainties.",
                     schema({"image_id": STRING}, ["image_id"]), read_screenshot, budget, events),
        FunctionTool("crop_screenshot", "Crop an image using [left, top, right, bottom] pixel coordinates; returns a new ID.",
                     schema({"image_id": STRING, "bbox": {"type": "array", "items": INTEGER,
                                                          "minItems": 4, "maxItems": 4}}, ["image_id", "bbox"]),
                     images.crop, budget, events),
    ]
    if not screenshot_policy.allow_crop:
        tools = [t for t in tools if t.name!="crop_screenshot"]
    return tools
