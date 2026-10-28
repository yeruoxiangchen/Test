"""Budget the complete chat + tool schema, retaining tool-call/result pairs."""
import copy
import json
import httpx


class ContextBudget:
    def __init__(self, settings, counter=None):
        self.settings, self.counter = settings, counter

    def count(self, messages, tools):
        if self.counter:
            return self.counter(messages, tools), "test_counter"
        cfg = self.settings
        if cfg.tokenize_url:
            headers = {"Authorization": "Bearer " + cfg.llm_key} if cfg.llm_key else {}
            try:
                with httpx.Client(timeout=cfg.timeout, trust_env=False) as client:
                    r = client.post(cfg.tokenize_url, headers=headers, json={"model": cfg.llm_model,
                        "messages": messages, "tools": tools,
                        "chat_template_kwargs": {"enable_thinking": cfg.enable_thinking}})
                    r.raise_for_status()
                    body = r.json()
                if body.get("max_model_len", cfg.context_tokens) < cfg.context_tokens:
                    raise RuntimeError("配置的上下文预算超过服务上限")
                return int(body["count"]), "server_tokenizer"
            except (httpx.HTTPError, KeyError, ValueError):
                raise RuntimeError("Token 计数服务失败；未发送未经预算检查的模型请求") from None
        # Explicit fallback for generic endpoints without /tokenize; not an exact count.
        count = len(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False).encode("utf-8")) + 256
        return count, "conservative_utf8_estimate"

    def prepare(self, messages, tools, question_index=1):
        limit = self.settings.context_tokens - self.settings.output_tokens - 256
        result = copy.deepcopy(messages)
        removed = 0
        count, mode = self.count(result, tools)
        # Only conversation history before this task may be dropped. Never drop the
        # current question, assistant tool calls, or their corresponding tool outputs.
        while count > limit and question_index > 1:
            end = 2
            while end < question_index and result[end]["role"] != "user":
                end += 1
            removed += end - 1
            del result[1:end]
            question_index -= end - 1
            count, mode = self.count(result, tools)
        if count > limit:
            raise RuntimeError(f"上下文预算不足：输入 {count}，可用 {limit}；未截断问题或证据")
        return result, {"input_tokens": count, "counter": mode, "history_messages_removed": removed,
                        "input_limit": limit, "output_reserved": self.settings.output_tokens}
