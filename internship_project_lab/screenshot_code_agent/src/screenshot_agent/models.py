"""Model adapters preserving CoreCoder's response contract and a shared run budget."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from corecoder.llm import LLMResponse, ToolCall
from openai import OpenAI

from .config import AgentPolicy, ModelConfig
from .control import Budget, Events


class ModelOutputError(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class APIModel:
    def __init__(self, config: ModelConfig, budget: Budget, events: Events, kind: str = "code",
                 policy: AgentPolicy | None = None):
        config.require_live(budget.limits.max_cost_usd)
        self.config, self.budget, self.events, self.kind = config, budget, events, kind
        self.policy = policy or AgentPolicy()
        # Loopback services must not accidentally go through a globally configured HTTP proxy.
        local = urlsplit(config.base_url).hostname in {"localhost", "127.0.0.1", "::1"}
        client = httpx.Client(trust_env=False) if local else None
        self.client = OpenAI(api_key=os.environ[config.api_key_env], base_url=config.base_url,
                             max_retries=0, http_client=client)

    def complete(self, messages: list, tools: list | None = None, *, purpose: str | None = None):
        kind = purpose or self.kind
        config = self.config.model_copy(update={"max_output_tokens": min(
            self.config.max_output_tokens, self.policy.summary_max_tokens)}) if kind == "summary" else self.config
        reserve = self.budget.reserve(kind, config, messages, tools)
        self.events.emit("model_request", kind=kind, model=config.model, messages=messages, tools=tools)
        settled = False
        try:
            kwargs = dict(config.model_kwargs)
            kwargs.update(model=config.model, messages=messages,
                          max_tokens=config.max_output_tokens, stream=False)
            if tools:
                kwargs["tools"] = tools
            response = self.client.with_options(
                timeout=min(self.budget.limits.request_timeout_seconds, self.budget.remaining_seconds())
            ).chat.completions.create(**kwargs)
            self.budget.settle(config, reserve, response.usage)
            settled = True
            self.events.emit("model_response", kind=kind, response=response.model_dump())
            return response
        except BaseException as error:
            self.events.emit("model_error", kind=kind, error_type=type(error).__name__, error=str(error)[:1000])
            raise
        finally:
            if not settled:
                self.budget.settle(config, reserve, None)

    def summarize(self, messages: list) -> str:
        response = self.complete(messages, purpose="summary")
        if response.choices[0].finish_reason == "length":
            raise ModelOutputError("summary_truncated")
        return response.choices[0].message.content or ""

    def chat(self, messages, tools=None, on_token=None, on_reasoning=None) -> LLMResponse:
        response = self.complete(messages, tools)
        msg = response.choices[0].message
        if self.policy.mode in {"v6", "v7"}:
            reason = response.choices[0].finish_reason
            if reason == "length":
                raise ModelOutputError("output_truncated")
            if reason == "content_filter":
                raise ModelOutputError("output_filtered")
            if not msg.tool_calls and "<tool_call" in (msg.content or ""):
                raise ModelOutputError("malformed_tool_output")
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments)
                if not isinstance(args, dict):
                    raise ValueError("tool arguments must be a JSON object")
            except (ValueError, TypeError):
                calls.append(ToolCall(tc.id, "invalid_tool_arguments", {"raw": tc.function.arguments}))
            else:
                calls.append(ToolCall(tc.id, tc.function.name, args))
        return LLMResponse(content=msg.content or "", tool_calls=calls,
                           prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                           completion_tokens=response.usage.completion_tokens if response.usage else 0)


class ReplayModel:
    """Deterministic plumbing check with a known patch. NEVER an agent quality measurement."""

    def __init__(self, script: Path, budget: Budget, events: Events):
        self.steps = json.loads(script.read_text(encoding="utf-8"))
        self.index = 0
        self.budget, self.events = budget, events

    def chat(self, messages, tools=None, **kwargs) -> LLMResponse:
        self.budget.check()
        if not tools:  # context compaction fallback in long synthetic tests
            return LLMResponse(content="Continue the repair; consult recent tool outputs.")
        if self.budget.code_calls >= self.budget.limits.max_steps:
            from .control import BudgetExceeded
            raise BudgetExceeded("Model step limit reached")
        self.budget.code_calls += 1
        self.budget.decision_calls += 1
        if self.index >= len(self.steps):
            return LLMResponse(content="Recorded plumbing sequence finished.")
        step = self.steps[self.index]
        self.index += 1
        self.events.emit("replay_response", step=self.index, response=step, knows_reference_patch=True)
        if "tool" in step:
            return LLMResponse(tool_calls=[ToolCall(f"replay_{self.index}", step["tool"], step.get("arguments", {}))])
        return LLMResponse(content=step.get("content", "Recorded sequence finished."))
