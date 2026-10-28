from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass
class Settings:
    root: Path = field(default_factory=lambda: Path(os.getenv("KB_DATA_DIR", "/data/zjr/enterprise_kb_agent")))
    mode: str = field(default_factory=lambda: os.getenv("KB_MODE", "extractive"))
    llm_url: str = field(default_factory=lambda: os.getenv("KB_LLM_BASE_URL", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("KB_LLM_MODEL", ""))
    llm_key: str = field(default_factory=lambda: os.getenv("KB_LLM_API_KEY", ""))
    tokenize_url: str = field(default_factory=lambda: os.getenv("KB_TOKENIZE_URL", ""))
    context_tokens: int = field(default_factory=lambda: int(os.getenv("KB_CONTEXT_TOKENS", "8192")))
    output_tokens: int = field(default_factory=lambda: int(os.getenv("KB_OUTPUT_TOKENS", "1400")))
    enable_thinking: bool = field(default_factory=lambda: os.getenv("KB_ENABLE_THINKING", "0") == "1")
    embedding_url: str = field(default_factory=lambda: os.getenv("KB_EMBED_BASE_URL", ""))
    embedding_model: str = field(default_factory=lambda: os.getenv("KB_EMBED_MODEL", ""))
    embedding_key: str = field(default_factory=lambda: os.getenv("KB_EMBED_API_KEY", ""))
    max_turns: int = 6
    max_tool_calls: int = 12
    agent_policy: str = field(default_factory=lambda: os.getenv("KB_AGENT_POLICY", "bounded_v2"))
    reranker_path: str = ""
    reranker_python: str = field(default_factory=lambda: os.getenv("KB_RERANK_PYTHON", "/data/zjr/deepsearch_rl/venv/bin/python"))
    reranker_candidates: int = 20
    guarded_followup: bool = True
    timeout: float = 45
    strict_embedding: bool = False
    max_upload_bytes: int = 20 * 1024 * 1024
    secure_cookie: bool = field(default_factory=lambda: os.getenv("KB_SECURE_COOKIE", "0") == "1")

    def prepare(self):
        if self.mode not in {"extractive", "agent"}:
            raise ValueError("KB_MODE must be extractive or agent")
        if self.mode == "agent" and not (self.llm_url and self.llm_model):
            raise ValueError("agent 模式需要 KB_LLM_BASE_URL 和 KB_LLM_MODEL")
        if bool(self.embedding_url) != bool(self.embedding_model):
            raise ValueError("向量服务地址和模型名必须一起设置")
        self.root = self.root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        for name in ("uploads", "tmp", "logs", "cache", "artifacts"):
            (self.root / name).mkdir(exist_ok=True, mode=0o700)

    @property
    def db_path(self):
        return self.root / "knowledge.sqlite3"

    @property
    def embedding_profile(self):
        # Both endpoint and model identify the index. Different profiles never mix.
        return f"{self.embedding_url.rstrip('/')}|{self.embedding_model}"
