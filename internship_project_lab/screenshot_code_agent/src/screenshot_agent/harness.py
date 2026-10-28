"""Matched resource limits and the pinned mini-SWE-agent protocol adapters.

The upstream DefaultAgent loop is imported unchanged. Commands always use our
existing sandbox; upstream LocalEnvironment (host execution) is never used.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Literal

from openai import APIConnectionError, APITimeoutError
from pydantic import Field, model_validator

from .config import RunConfig
from .context import compact_request
from .control import Budget, BudgetExceeded, ContextWindowExceeded
from .models import APIModel

PROTOCOL = "hidden_reference_tests_harness_v1"
PROJECT = Path(__file__).resolve().parents[2]


class HarnessConfig(RunConfig):
    framework: Literal["corecoder", "mini"] = "corecoder"
    protocol: Literal["hidden_reference_tests_harness_v1"] = PROTOCOL
    tokenizer_path: str
    service_context_tokens: int = Field(default=32768, ge=4096)
    max_prompt_tokens: int = Field(default=24576, ge=2048)
    total_prompt_tokens: int = Field(default=1_500_000, ge=1)
    total_completion_tokens: int = Field(default=100_000, ge=1)

    @model_validator(mode="after")
    def context_fits(self):
        if self.max_prompt_tokens + self.code.max_output_tokens > self.service_context_tokens:
            raise ValueError("Prompt and output allowances exceed the model service context")
        if self.max_context_tokens > self.max_prompt_tokens:
            raise ValueError("Agent history window must fit the full prompt allowance")
        return self


class QwenTokenCounter:
    """Use the pinned tokenizer and chat template locally, including tool schemas."""
    def __init__(self, path):
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer
        path = Path(path)
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        meta = json.loads((path / "tokenizer_config.json").read_text())
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = lambda value, **kw: json.dumps(value, ensure_ascii=False, **kw)
        self.template = env.from_string(meta["chat_template"])
        self.meta = meta

    def __call__(self, messages, tools):
        # The serving layer converts OpenAI JSON argument strings to mappings
        # before rendering the Qwen chat template.
        messages = copy.deepcopy(messages)
        for row in messages:
            for call in row.get("tool_calls") or []:
                value = call["function"].get("arguments")
                if isinstance(value, str):
                    call["function"]["arguments"] = json.loads(value)
        rendered = self.template.render(messages=messages, tools=tools or [], add_generation_prompt=True,
                                        bos_token=self.meta.get("bos_token"), eos_token=self.meta.get("eos_token"))
        return len(self.tokenizer.encode(rendered, add_special_tokens=False).ids)


class HarnessBudget(Budget):
    def __init__(self, config, events, counter):
        super().__init__(config.limits, events, separate_summaries=True)
        self.config, self.counter = config, counter
        self.accounted_prompt = self.accounted_completion = 0
        self.pending_tokens = None

    def reserve(self, kind, config, messages, tools=None):
        count = self.counter(messages, tools)
        if count > self.config.max_prompt_tokens:
            raise ContextWindowExceeded("Full prompt (including tools) exceeds the matched context allowance")
        if self.accounted_prompt + count > self.config.total_prompt_tokens:
            raise BudgetExceeded("Total prompt token allowance exhausted")
        if self.accounted_completion + config.max_output_tokens > self.config.total_completion_tokens:
            raise BudgetExceeded("Insufficient completion token allowance for another full request")
        reservation = super().reserve(kind, config, messages, tools)
        self.pending_tokens = (count, config.max_output_tokens)
        self.events.emit("prompt_token_check", tokens=count, limit=self.config.max_prompt_tokens)
        return reservation

    def settle(self, config, reserve, usage):
        prompt, completion = self.pending_tokens or (0, config.max_output_tokens)
        self.pending_tokens = None
        self.accounted_prompt += int(usage.prompt_tokens) if usage else prompt
        self.accounted_completion += int(usage.completion_tokens) if usage else completion
        super().settle(config, reserve, usage)

    def snapshot(self):
        return {**super().snapshot(), "accounted_prompt_tokens": self.accounted_prompt,
                "accounted_completion_tokens": self.accounted_completion}


class HarnessAPIModel(APIModel):
    """One explicit transport retry, charged against the same token/call budget."""
    def complete(self, messages, tools=None, *, purpose=None):
        messages = compact_request(messages, tools, self.budget.counter,
                                   self.budget.config.max_prompt_tokens, self.events)
        for attempt in range(2):
            try:
                return super().complete(messages, tools, purpose=purpose)
            except (APIConnectionError, APITimeoutError) as error:
                if attempt:
                    raise
                self.budget.check()
                self.events.emit("transport_retry", attempt=attempt + 1, error_type=type(error).__name__)


def verify_mini():
    receipt = json.loads((PROJECT / "MINI_UPSTREAM.json").read_text())
    root = PROJECT / "vendor/mini-swe-agent"
    for name, digest in receipt["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Pinned mini-SWE-agent source changed: {name}")
    return receipt


def load_mini(run_dir):
    verify_mini()
    os.environ["MSWEA_SILENT_STARTUP"] = "1"
    # Never load an unrelated user's global .env from home.
    os.environ["MSWEA_GLOBAL_CONFIG_DIR"] = str(Path(run_dir) / "mini_config")
    sys.path.insert(0, str(PROJECT / "vendor/mini-swe-agent/src"))
    from minisweagent.agents.default import DefaultAgent
    return DefaultAgent


class MiniModel:
    def __init__(self, api, templates):
        self.api, self.templates = api, templates

    def query(self, messages):
        from minisweagent.exceptions import FormatError
        from minisweagent.models.utils.actions_toolcall import BASH_TOOL, parse_toolcall_actions
        prepared = [{k: v for k, v in row.items() if k != "extra"} for row in messages]
        prior_cost = self.api.budget.cost_usd
        response = self.api.complete(prepared, [BASH_TOOL])
        cost = self.api.budget.cost_usd - prior_cost
        try:
            actions = parse_toolcall_actions(response.choices[0].message.tool_calls or [],
                format_error_template=self.templates["format_error_template"],
                template_kwargs={"finish_reason": response.choices[0].finish_reason})
        except FormatError as error:
            error.messages[0].setdefault("extra", {}).update(cost=cost, response=response.model_dump(mode="json"))
            raise
        message = response.choices[0].message.model_dump(exclude_none=True)
        message["extra"] = {"actions": actions, "cost": cost}
        return message

    def format_message(self, **kwargs):
        return kwargs

    def format_observation_messages(self, message, outputs, template_vars=None):
        from minisweagent.models.utils.actions_toolcall import format_toolcall_observation_messages
        return format_toolcall_observation_messages(actions=message["extra"]["actions"], outputs=outputs,
            observation_template=self.templates["observation_template"], template_vars=template_vars)

    def get_template_vars(self):
        return {}

    def serialize(self):
        return {"info": {"config": {"model": self.api.config.model_dump(),
                                    "adapter": "shared_sca_APIModel"}}}


class MiniEnvironment:
    def __init__(self, sandbox, budget, events):
        self.sandbox, self.budget, self.events = sandbox, budget, events

    def execute(self, action, cwd=""):
        from minisweagent.exceptions import Submitted
        command = action.get("command")
        if not isinstance(command, str):
            raise ValueError("bash command must be a string")
        self.budget.tool("bash", {"command": command})
        output = self.sandbox.run(command, timeout=self.budget.remaining_seconds())
        self.events.emit("tool_result", name="bash", output=output)
        lines = output["output"].lstrip().splitlines(keepends=True)
        if (lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
                and output["returncode"] == 0 and not output["timed_out"]):
            raise Submitted({"role": "exit", "content": "".join(lines[1:]),
                             "extra": {"exit_status": "Submitted", "submission": "".join(lines[1:])}})
        output["exception_info"] = "Command timed out" if output["timed_out"] else ""
        return output

    def get_template_vars(self):
        return {"cwd": self.sandbox.description()["repository_mount"]}

    def serialize(self):
        return {"info": {"sandbox": self.sandbox.description()}}


def make_mini(config, budget, events, sandbox, run_dir, api=None):
    import yaml
    agent_class = load_mini(run_dir)
    upstream = yaml.safe_load((PROJECT / "vendor/mini-swe-agent/src/minisweagent/config/benchmarks/swebench.yaml").read_text())
    prompt = upstream["agent"]["instance_template"].split("    ## Submission")[0]
    # YAML dedents its text. The adapter exports the same workspace diff for both
    # frameworks, rather than taking an agent-supplied patch file as authoritative.
    prompt = prompt.split("## Submission")[0] + """## Submission
When finished, run `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` as a separate command.
The evaluator exports your workspace diff; do not create a patch.txt file.
The network is disabled and dependencies are preinstalled. Use existing repository
tests and your own reproduction scripts. Reference evaluation tests are unavailable.
</instructions>
"""
    prompt = prompt.replace("/testbed", sandbox.description()["repository_mount"])
    api = api or HarnessAPIModel(config.code, budget, events, policy=config.agent)
    return agent_class(MiniModel(api, upstream["model"]), MiniEnvironment(sandbox, budget, events),
                       system_template=upstream["agent"]["system_template"], instance_template=prompt,
                       step_limit=config.limits.max_steps, cost_limit=0,
                       wall_time_limit_seconds=config.limits.max_wall_seconds,
                       output_path=Path(run_dir) / "mini_trajectory.json")
