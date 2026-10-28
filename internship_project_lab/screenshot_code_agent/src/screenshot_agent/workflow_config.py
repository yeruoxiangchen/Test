"""Optional product workflow settings; legacy run configs remain unchanged."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .control import Budget, BudgetExceeded, RunStopped, atomic_json
from .harness import HarnessBudget, HarnessConfig


class PhaseEnded(RunStopped):
    pass


class WorkflowPolicy(BaseModel):
    model_config = ConfigDict(extra='forbid')
    locate_calls: int = Field(default=8, ge=1, le=40)
    reproduce_calls: int = Field(default=8, ge=1, le=40)
    repair_calls: int = Field(default=32, ge=1, le=100)
    retry_calls: int = Field(default=12, ge=1, le=100)
    max_retries: int = Field(default=2, ge=0, le=3)
    map_chars: int = Field(default=7000, ge=1000, le=20000)
    retrieval_top_k: int = Field(default=6, ge=1, le=15)
    public_test_timeout_seconds: int = Field(default=120, ge=1, le=600)
    reproduction_timeout_seconds: int = Field(default=60, ge=1, le=300)
    max_reproductions: int = Field(default=3, ge=1, le=6)


class WorkflowConfig(HarnessConfig):
    framework: Literal['corecoder'] = 'corecoder'
    workflow: WorkflowPolicy = Field(default_factory=WorkflowPolicy)


def load_workflow_config(path):
    text = Path(path).read_text()
    text = re.sub(r'\$\{([A-Z0-9_]+)\}', lambda m:os.environ.get(m[1],m[0]), text)
    return WorkflowConfig.model_validate_json(text)


class WorkflowBudget(HarnessBudget):
    """Persist shared usage, including a conservative charge for interrupted HTTP calls."""
    def __init__(self, config, events, counter, path, *, resume=False):
        super().__init__(config, events, counter)
        self.path = Path(path)
        self.phase_limit = None
        if resume and self.path.exists():
            old = json.loads(self.path.read_text())
            for field in ('code_calls','vision_calls','tool_calls','decision_calls','summary_calls',
                          'screenshot_calls','prompt_tokens','completion_tokens','cost_usd',
                          'estimated_calls','accounted_prompt','accounted_completion'):
                setattr(self,field,old.get(field,0))
            pending = old.get('pending_tokens')
            if pending:
                self.accounted_prompt += pending[0]
                self.accounted_completion += pending[1]
                self.cost_usd += old.get('reserved_usd',0)
                self.estimated_calls += 1
            self.started = time.monotonic()-old.get('elapsed_seconds',0)
        self.save()

    def save(self):
        with self.lock:
            fields = ('code_calls','vision_calls','tool_calls','decision_calls','summary_calls',
                      'screenshot_calls','prompt_tokens','completion_tokens','cost_usd',
                      'reserved_usd','estimated_calls','accounted_prompt','accounted_completion','pending_tokens')
            atomic_json(self.path, {**{name:getattr(self,name) for name in fields},
                                   'elapsed_seconds':time.monotonic()-self.started, 'usage':self.snapshot()})

    def reserve(self, kind, config, messages, tools=None):
        if kind != 'vision' and self.phase_limit is not None and self.code_calls >= self.phase_limit:
            raise PhaseEnded('Stage model allowance reached; continue with observed evidence')
        if kind == 'vision':
            # Image inputs use the vision model's accounting; never render them
            # through the text-only Qwen-Coder chat template.
            if self.accounted_prompt + config.image_token_reserve > self.config.total_prompt_tokens:
                raise BudgetExceeded('Total prompt token allowance exhausted')
            if self.accounted_completion + config.max_output_tokens > self.config.total_completion_tokens:
                raise BudgetExceeded('Insufficient completion token allowance for vision request')
            reservation = Budget.reserve(self,kind,config,messages,tools)
            self.pending_tokens = (config.image_token_reserve, config.max_output_tokens)
        else:
            reservation = super().reserve(kind,config,messages,tools)
        self.save()
        return reservation

    def settle(self, config, reserve, usage):
        super().settle(config,reserve,usage)
        self.save()

    def tool(self, name, arguments):
        super().tool(name,arguments)
        self.save()

    def stage(self, calls):
        self.phase_limit = self.code_calls+calls
        self.check()
        if self.code_calls >= (self.limits.max_model_calls or self.limits.max_steps):
            raise BudgetExceeded('Shared model allowance exhausted')
