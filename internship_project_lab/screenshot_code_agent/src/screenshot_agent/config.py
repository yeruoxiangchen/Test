"""Validated, secret-free model and execution configuration."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = ""
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    input_usd_per_million: float | None = Field(default=None, ge=0)
    output_usd_per_million: float | None = Field(default=None, ge=0)
    max_output_tokens: int = Field(default=2048, ge=1, le=32768)
    image_token_reserve: int = Field(default=8192, ge=1024)
    model_kwargs: dict = Field(default_factory=dict)

    def require_live(self, cost_cap: float | None) -> None:
        if not self.model or "${" in self.model or "${" in self.base_url:
            raise ValueError("Configure model name and base URL before using a live model")
        if not os.environ.get(self.api_key_env):
            raise ValueError(f"Missing credential environment variable: {self.api_key_env}")
        if cost_cap is not None and (
            self.input_usd_per_million is None or self.output_usd_per_million is None
        ):
            raise ValueError("A cost cap requires explicit model input/output prices; local endpoints may use 0")
        forbidden = {"model", "messages", "tools", "stream", "max_tokens", "max_completion_tokens"}
        if forbidden & self.model_kwargs.keys():
            raise ValueError("model_kwargs cannot override model, input, tools, stream or token limits")


class Limits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_steps: int = Field(default=20, ge=1)
    max_summary_calls: int = Field(default=2, ge=0)
    # Optional shared code-decision + summary cap for matched-budget comparisons.
    max_model_calls: int | None = Field(default=None, ge=1)
    max_tool_calls: int = Field(default=80, ge=1)
    max_tool_output_chars: int = Field(default=12000, ge=256, le=1000000)
    max_vision_calls: int = Field(default=3, ge=1)
    max_wall_seconds: int = Field(default=600, ge=1)
    command_timeout_seconds: int = Field(default=60, ge=1)
    request_timeout_seconds: int = Field(default=60, ge=1)
    max_cost_usd: float | None = Field(default=2.0, gt=0)


class ScreenshotPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    allow_crop: bool = True
    allow_reread: bool = True


class AgentPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["legacy", "v6", "v7"] = "legacy"
    read_file_max_lines: int = Field(default=120, ge=10, le=500)
    repeated_edit_limit: int = Field(default=2, ge=1, le=10)
    summary_max_tokens: int = Field(default=384, ge=64, le=1024)


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: ModelConfig = Field(default_factory=ModelConfig)
    vision: ModelConfig = Field(default_factory=ModelConfig)
    limits: Limits = Field(default_factory=Limits)
    max_context_tokens: int = Field(default=32768, ge=2048)
    screenshots: ScreenshotPolicy = Field(default_factory=ScreenshotPolicy)
    agent: AgentPolicy = Field(default_factory=AgentPolicy)


def load_config(path: Path | None) -> RunConfig:
    if path is None:
        return RunConfig()
    value = path.read_text(encoding="utf-8")
    value = re.sub(r"\$\{([A-Z0-9_]+)\}", lambda m: os.environ.get(m[1], m[0]), value)
    return RunConfig.model_validate_json(value)


class ImageInput(BaseModel):
    image_id: str
    path: str
    # Exact visible transcription. Only oracle/recorded readers may access it.
    reference: str | None = None
    layout: str | None = None


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: Literal["bubblewrap", "venv", "oci", "docker"] = "bubblewrap"
    prefix: str | None = None
    rootfs: str | None = None
    image: str | None = None
    image_digest: str | None = None
    python: str = "/opt/miniconda3/envs/testbed/bin/python"
    workdir: str = "/testbed"
    memory_gib: int = Field(default=8, ge=1, le=32)
    max_repo_files: int = Field(default=10000, ge=1, le=100000)
    max_repo_bytes: int = Field(default=100_000_000, ge=1, le=2_000_000_000)
    allowed_symlinks: dict[str, str] = Field(default_factory=dict)
    python_overlay: str | None = None
    python_overlay_manifest_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    library_overlay: str | None = None
    library_overlay_manifest_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def require_overlay_receipts(self):
        for name in ("python_overlay", "library_overlay"):
            path, checksum = getattr(self, name), getattr(self, name + "_manifest_sha256")
            if bool(path) != bool(checksum):
                raise ValueError(f"{name} requires both a directory and its manifest SHA-256")
            if path and self.backend != "oci":
                raise ValueError("Compatibility overlays are supported only by the OCI runtime")
        return self


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    repo: str
    problem: str
    test_command: str = "python3 -m unittest discover -s tests -v"
    images: list[ImageInput] = Field(default_factory=list)
    protected_paths: list[str] = Field(default_factory=lambda: ["tests"])
    split: str = "demo"
    group_id: str = ""
    provenance: str = "user_provided"
    reference_patch: str | None = None
    replay: str | None = None
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    # Trusted grading metadata lives OUTSIDE the repository and is never a tool input.
    benchmark: str | None = None

    @classmethod
    def from_file(cls, path: Path) -> "Task":
        data = json.loads(path.read_text(encoding="utf-8"))
        for name in ("repo", "reference_patch", "replay", "benchmark"):
            if data.get(name):
                data[name] = str((path.parent / data[name]).resolve())
        for item in data.get("images", []):
            for name in ("path", "reference", "layout"):
                if item.get(name):
                    item[name] = str((path.parent / item[name]).resolve())
        for name in ("rootfs", "prefix", "python_overlay", "library_overlay"):
            if data.get("runtime", {}).get(name):
                data["runtime"][name] = str((path.parent / data["runtime"][name]).resolve())
        return cls.model_validate(data)
