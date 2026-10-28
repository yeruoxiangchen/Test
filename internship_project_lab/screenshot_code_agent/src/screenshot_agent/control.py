"""Shared limits and append-only events, including calls made from parallel tools."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

from .config import Limits, ModelConfig


class RunStopped(BaseException):
    """A terminal control signal; CoreCoder's recoverable tool errors must not swallow it."""


class BudgetExceeded(RunStopped):
    pass


class ContextWindowExceeded(BudgetExceeded):
    """Required prompt content cannot fit even after bounded compaction."""


class RunCancelled(RunStopped):
    pass


class NoProgress(RunStopped):
    """Repeated identical failure on unchanged file contents after a recovery warning."""


class SummaryUnavailable(RuntimeError):
    """Use local extractive compaction instead of spending another model call."""


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def public_payload(value):
    """Keep image identity in traces without writing base64 image payloads repeatedly."""
    if isinstance(value, str) and value.startswith("data:image/"):
        return {"image_payload_sha256": hashlib.sha256(value.encode()).hexdigest(), "bytes": len(value)}
    if isinstance(value, list):
        return [public_payload(v) for v in value]
    if isinstance(value, dict):
        return {k: public_payload(v) for k, v in value.items()}
    return value


class Events:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()

    def emit(self, event: str, **fields) -> None:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"time": time.time(), "event": event, **public_payload(fields)},
                                        ensure_ascii=False) + "\n")


class Budget:
    def __init__(self, limits: Limits, events: Events, *, separate_summaries: bool = False):
        self.limits = limits
        self.events = events
        self.started = time.monotonic()
        self.cancelled = threading.Event()
        self.lock = threading.RLock()
        self.code_calls = self.vision_calls = self.tool_calls = 0
        self.decision_calls = self.summary_calls = 0
        self.separate_summaries = separate_summaries
        self.screenshot_calls = 0
        self.prompt_tokens = self.completion_tokens = 0
        self.cost_usd = self.reserved_usd = 0.0
        self.estimated_calls = 0

    def check(self) -> None:
        if self.cancelled.is_set():
            raise RunCancelled("Task cancelled")
        if time.monotonic() - self.started >= self.limits.max_wall_seconds:
            raise BudgetExceeded("Wall time limit reached")

    def remaining_seconds(self) -> float:
        self.check()
        return max(0.1, self.limits.max_wall_seconds - (time.monotonic() - self.started))

    def tool(self, name: str, arguments: dict) -> None:
        with self.lock:
            self.check()
            if self.tool_calls >= self.limits.max_tool_calls:
                raise BudgetExceeded("Tool call limit reached")
            self.tool_calls += 1
            self.events.emit("tool_start", name=name, arguments=arguments, tool_call=self.tool_calls)

    def reserve(self, kind: str, config: ModelConfig, messages: list, tools: list | None = None) -> float:
        with self.lock:
            self.check()
            if kind == "vision":
                if self.vision_calls >= self.limits.max_vision_calls:
                    raise BudgetExceeded("Screenshot reading call limit reached")
            else:
                if self.separate_summaries and kind == "summary":
                    # Keep the last shared call available for an actual decision.
                    if (self.summary_calls >= self.limits.max_summary_calls or
                            (self.limits.max_model_calls is not None and
                             self.code_calls >= self.limits.max_model_calls - 1)):
                        raise SummaryUnavailable("Summary allowance exhausted; use extractive compaction")
                if self.limits.max_model_calls is not None and self.code_calls >= self.limits.max_model_calls:
                    raise BudgetExceeded("Shared model call limit reached (decisions + summaries)")
                steps = self.decision_calls if self.separate_summaries else self.code_calls
                if kind != "summary" and steps >= self.limits.max_steps:
                    raise BudgetExceeded("Decision call limit reached" if self.separate_summaries else
                                         "Model step limit reached (includes context summaries)")
            safe = public_payload({"messages": messages, "tools": tools})
            # Byte-length bound + protocol margin. Image accounting is configurable per provider.
            input_reserve = len(json.dumps(safe, ensure_ascii=False).encode()) + 1024
            image_count = json.dumps(safe).count("image_payload_sha256")
            input_reserve += image_count * config.image_token_reserve
            price_in = config.input_usd_per_million or 0.0
            price_out = config.output_usd_per_million or 0.0
            reserve = (input_reserve * price_in + config.max_output_tokens * price_out) / 1_000_000
            cap = self.limits.max_cost_usd
            if cap is not None and self.cost_usd + self.reserved_usd + reserve > cap + 1e-12:
                raise BudgetExceeded("Insufficient cost budget for the next request's reserved input/output")
            self.reserved_usd += reserve
            if kind == "vision":
                self.vision_calls += 1
            else:
                self.code_calls += 1
                if kind == "summary":
                    self.summary_calls += 1
                else:
                    self.decision_calls += 1
            return reserve

    def settle(self, config: ModelConfig, reserve: float, usage) -> None:
        with self.lock:
            self.reserved_usd = max(0.0, self.reserved_usd - reserve)
            if usage is None:
                self.cost_usd += reserve  # failed/usage-less calls keep the conservative reservation
                self.estimated_calls += 1
            else:
                prompt = int(usage.prompt_tokens or 0)
                completion = int(usage.completion_tokens or 0)
                self.prompt_tokens += prompt
                self.completion_tokens += completion
                self.cost_usd += (prompt * (config.input_usd_per_million or 0)
                                  + completion * (config.output_usd_per_million or 0)) / 1_000_000
            self.events.emit("usage", **self.snapshot())

    def snapshot(self) -> dict:
        return {
            "code_model_calls": self.code_calls, "vision_model_calls": self.vision_calls,
            "decision_model_calls": self.decision_calls, "summary_model_calls": self.summary_calls,
            "screenshot_read_calls": self.screenshot_calls,
            "tool_calls": self.tool_calls, "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens, "cost_usd": round(self.cost_usd, 8),
            "reserved_usd": round(self.reserved_usd, 8), "estimated_or_failed_calls": self.estimated_calls,
            "elapsed_seconds": round(time.monotonic() - self.started, 3),
        }
