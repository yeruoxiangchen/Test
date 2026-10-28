"""Bounded compaction for repair trajectories, without changing vendored CoreCoder."""

from __future__ import annotations

import copy

from corecoder.context import ContextManager, estimate_tokens

from .control import ContextWindowExceeded


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "\n... earlier output omitted; re-read a narrow range if needed ...\n"
    available = max(0, limit-len(marker))
    head = available // 2
    return text[:head] + marker + text[-(available-head):]


class RepairContext(ContextManager):
    def __init__(self, max_tokens: int, anchor: str, events):
        super().__init__(max_tokens)
        self.anchor = anchor
        self.events = events
        self.request = None

    def bind_request(self, counter, limit, prefix, tools):
        """Count the actual system message and current tool schemas as well."""
        self.request = counter, limit, prefix, tools

    @staticmethod
    def _extract(messages: list[dict]) -> str:
        # Deliberately labelled excerpts, not invented statements of task completion.
        rows = []
        for row in messages[-6:]:
            content = str(row.get("content") or "")
            if row.get("tool_calls"):
                content = str(row["tool_calls"]) + "\n" + content
            if content:
                rows.append(f"{row['role']}: {clip(content, 450)}")
        return "Extracted earlier observations (may be incomplete):\n" + clip("\n".join(rows), 1800)

    def _summary(self, old: list[dict], llm) -> str:
        fallback = self._extract(old)
        if not hasattr(llm, "summarize"):
            return fallback
        observed = []
        for row in old:
            content = str(row.get("content") or "")
            if row.get("tool_calls"):
                content = str(row["tool_calls"]) + "\n" + content
            if content:
                observed.append(f"[{row['role']}] {clip(content, 1200)}")
        prompt = [
            {"role": "system", "content": (
                "Summarize ONLY observed repair progress in at most 8 short bullets. "
                "Preserve file paths, successful edits, latest test failures and next unresolved action. "
                "Do not write code, complete source listings, invent outcomes, or claim tests passed without evidence.")},
            {"role": "user", "content": clip("\n".join(observed), 10000)},
        ]
        try:
            text = llm.summarize(prompt).strip()
            if not text or len(text) > 2200:
                raise ValueError("Summary was empty or too long")
            self.events.emit("context_summary", source="model", characters=len(text))
            return text
        except Exception as error:
            # Shared time/cost/total-call limits are BaseException signals and still terminate.
            self.events.emit("context_summary", source="extractive", reason=type(error).__name__)
            return fallback

    def maybe_compress(self, messages: list[dict], llm=None) -> bool:
        if self.request is not None:
            counter, limit, prefix, tools = self.request
            leading = prefix()
            full = leading + messages
            fitted = compact_request(full, tools(), counter, limit, self.events)
            if fitted is full:
                return False
            messages[:] = fitted[len(leading):]
            return True
        before = estimate_tokens(messages)
        if before <= int(self.max_tokens * .70):
            return False
        # Never compress away the original issue, fixed tests or initial screenshot evidence.
        if not messages or messages[0].get("role") != "user" or messages[0].get("content") != self.anchor:
            raise ValueError("Repair context lost its original task anchor")
        for row in messages[1:-4]:
            if row.get("role") == "tool":
                row["content"] = clip(str(row.get("content") or ""), 1400)
        split = self._safe_split(messages, 4)
        if estimate_tokens(messages) > int(self.max_tokens * .70) and split > 1:
            summary = self._summary(messages[1:split], llm)
            tail = messages[split:]
            messages[:] = [messages[0], {"role": "user", "content": "Earlier repair state:\n" + summary}, *tail]
        # Bound large recent tool results too. No assistant/tool pair is detached.
        if estimate_tokens(messages) > int(self.max_tokens * .85):
            for row in messages[1:]:
                if row.get("role") == "tool":
                    row["content"] = clip(str(row.get("content") or ""), 1000)
        if estimate_tokens(messages) > self.max_tokens:
            split = self._safe_split(messages, 1)
            if split > 1:
                messages[:] = [messages[0],
                               {"role": "user", "content": self._extract(messages[1:split])},
                               *messages[split:]]
        after = estimate_tokens(messages)
        self.events.emit("context_compacted", before_estimated_tokens=before, after_estimated_tokens=after,
                         task_anchor_preserved=messages[0]["content"] == self.anchor)
        if after > self.max_tokens:
            raise ValueError("Task plus latest tool group exceeds context budget; use a larger context configuration")
        return after != before


def compact_request(messages, tools, counter, limit, events):
    """Fit an exact full request without model calls or detached tool replies.

    Preserve system instructions, the original issue, the latest user request,
    and the latest assistant/tool group. Old tool observations are clipped before
    removing complete groups. Required content that still cannot fit fails
    explicitly, before an API reservation or transport request is made.
    """
    before = counter(messages, tools)
    if before <= limit:
        return messages
    work = copy.deepcopy(messages)
    users = [i for i, row in enumerate(work) if row.get('role') == 'user']
    protected = {i for i, row in enumerate(work) if row.get('role') in {'system', 'developer'}}
    if users:
        protected.update((users[0], users[-1]))
    # Every assistant tool-call message and its consecutive replies is one unit.
    groups = []
    for i, row in enumerate(work):
        if row.get('role') == 'tool' and groups:
            groups[-1].append(i)
        else:
            groups.append([i])
    if groups:
        protected.update(groups[-1])

    def finish(candidate):
        after = counter(candidate, tools)
        if after > limit:
            return None
        events.emit('context_compacted', counting='full_request_tokenizer',
                    before_tokens=before, after_tokens=after, limit=limit,
                    task_anchor_preserved=True, model_call=False)
        return candidate

    for bound in (4000, 1600, 600):
        for i, row in enumerate(work):
            if row.get('role') == 'tool' or (row.get('role') == 'assistant' and i not in protected):
                if isinstance(row.get('content'), str):
                    row['content'] = clip(row['content'], bound)
        fitted = finish(work)
        if fitted is not None:
            return fitted
    removed = set()
    for group in groups:
        if protected.intersection(group):
            continue
        removed.update(group)
        candidate = [row for i, row in enumerate(work) if i not in removed]
        fitted = finish(candidate)
        if fitted is not None:
            return fitted
    raise ContextWindowExceeded(
        'Full prompt cannot fit after compaction; original issue, current request, '
        'system instructions, tools and latest tool group are preserved')


class EvidenceRepairContext(RepairContext):
    """v7 compaction: retain observed facts without asking a model to invent state."""

    def __init__(self, max_tokens: int, anchor: str, events, state):
        super().__init__(max_tokens, anchor, events)
        self.state = state

    def _extract(self, messages: list[dict]) -> str:
        # Assistant prose and prior summaries are deliberately not carried forward
        # as evidence. Recent tool text is a quoted observation, not a completion claim.
        tools = [row for row in messages if row.get("role") == "tool"][-3:]
        excerpts = [f"tool[{row.get('tool_call_id', '?')}]: {clip(str(row.get('content') or ''), 300)}"
                    for row in tools]
        tail = "\nRecent tool excerpts (possibly incomplete repository content):\n" + "\n".join(excerpts) if tools else ""
        return self.state.memory() + tail

    def _summary(self, old: list[dict], llm) -> str:
        text = self._extract(old)
        self.events.emit("context_summary", source="tool_evidence", characters=len(text), model_call=False)
        return text
