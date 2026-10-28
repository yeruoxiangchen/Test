"""Bounded repair evidence recorded from trusted tool return values, never model prose."""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import re
import threading

READ_TOOLS = {"read_file", "grep", "glob"}
WRITE_TOOLS = {"edit_file", "write_file"}


def bounded(value, limit=180):
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


class RepairState:
    """Last observed evidence; shell operations make workspace freshness uncertain.

    A generation is a possible-change boundary, not an assertion that files changed.
    Successful mutations are counted separately and require the worker's explicit
    changed flag and content hash. No reference patch or independent verifier is read.
    """

    def __init__(self, events):
        self.events = events
        self.lock = threading.RLock()
        self.sequence = self.generation = self.confirmed_mutations = 0
        self.tools_since_mutation = self.shell_calls = 0
        self.reads = collections.OrderedDict()
        self.mutations = collections.deque(maxlen=6)
        self.repeats = collections.OrderedDict()
        self.latest_test = None
        self.latest_error = None

    def begin(self, name, arguments):
        with self.lock:
            self.sequence += 1
            if name in WRITE_TOOLS or name == "bash":
                # Even a failed shell/edit may have partially changed a file.
                self.generation += 1
                self.repeats.clear()
            if name == "bash":
                self.shell_calls += 1
            return {"observation_id": self.sequence, "generation": self.generation}

    def observe(self, name, arguments, result, ticket):
        with self.lock:
            self.tools_since_mutation += 1
            hints = []
            value = result if isinstance(result, dict) else {}
            ok = value.get("returncode", 0) == 0 and not value.get("timed_out") and not value.get("error")
            digest = str(value.get("file_sha256", ""))
            if name in READ_TOOLS and ok:
                stable = {key: value.get(key) for key in (
                    "output", "file_sha256", "total_lines", "start_line", "end_line", "next_offset")}
                signature = (name, json.dumps(arguments, sort_keys=True, ensure_ascii=False))
                fingerprint = hashlib.sha256(json.dumps(stable, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                previous = self.repeats.pop(signature, None)
                count = previous[1] + 1 if previous and previous[0] == fingerprint else 1
                # Reads can run in parallel with a mutation. Keep the observation,
                # but do not classify such a read as evidence of unchanged contents.
                if ticket["generation"] == self.generation:
                    self.repeats[signature] = (fingerprint, count)
                    while len(self.repeats) > 128:
                        self.repeats.popitem(last=False)
                else:
                    count = 1
                if count in {2, 4, 8}:
                    hints.append(f"This exact {name} request returned the same evidence {count} times since the "
                                 "last possible file change. Reuse that evidence, or state what new information "
                                 "another read should establish; this request was still executed normally.")
                if name == "read_file":
                    path = bounded(value.get("file_path", arguments.get("file_path", "")))
                    self.reads.pop(path, None)
                    self.reads[path] = {"path": path, "start_line": value.get("start_line"),
                                        "end_line": value.get("end_line"), "file_sha256": digest or None,
                                        "generation": ticket["generation"],
                                        "observation_id": ticket["observation_id"]}
                    while len(self.reads) > 8:
                        self.reads.popitem(last=False)
            if name in WRITE_TOOLS and ok and value.get("ok") is True and value.get("changed") is True:
                if re.fullmatch(r"[0-9a-f]{64}", digest):
                    self.confirmed_mutations += 1
                    self.tools_since_mutation = 0
                    self.mutations.append({"path": bounded(value.get("file_path", arguments.get("file_path", ""))),
                                           "file_sha256": digest, "generation": ticket["generation"],
                                           "observation_id": ticket["observation_id"]})
            if name == "run_tests":
                resolved = value.get("resolved")
                if type(resolved) is not bool:
                    resolved = value.get("returncode") == 0
                resolved = resolved and not value.get("timed_out", False) and not bool(value.get("error"))
                groups = {}
                for group, status in value.get("official_tests", {}).items():
                    groups[bounded(group, 80)] = {"passed": status.get("passed"),
                                                 "failed_count": len(status.get("failed", []))}
                self.latest_test = {"observation_id": ticket["observation_id"],
                                    "generation": ticket["generation"], "resolved": resolved,
                                    "returncode": value.get("returncode"),
                                    "timed_out": bool(value.get("timed_out", False)), "groups": groups}
                if resolved and ticket["generation"] == self.generation:
                    hints.append("The fixed tests just reported a resolved result. Consider submitting the current "
                                 "patch instead of unrelated exploration; the independent final verifier still decides.")
            if value.get("error") or value.get("returncode", 0) != 0:
                self.latest_error = {"observation_id": ticket["observation_id"], "tool": name,
                                     "error": bounded(value.get("error", "nonzero command/test result"), 240),
                                     "returncode": value.get("returncode")}
            if name in READ_TOOLS | {"bash", "run_tests"} and self.tools_since_mutation in {10, 20, 30}:
                hints.append(f"{self.tools_since_mutation} tool results since the last confirmed edit/write change. "
                             "State a concrete repair hypothesis and the next missing evidence or minimal edit. "
                             "Shell changes are not tracked as confirmed edits; do not modify code merely to satisfy "
                             "this reminder.")
            self.events.emit("repair_progress", tool=name, observation_id=ticket["observation_id"],
                             generation=self.generation, confirmed_mutations=self.confirmed_mutations,
                             hints=hints)
            return hints

    def failed(self, name, error, ticket):
        with self.lock:
            self.tools_since_mutation += 1
            self.latest_error = {"observation_id": ticket["observation_id"], "tool": name,
                                 "error": bounded(str(error), 240), "error_type": type(error).__name__}
            if name == "run_tests":
                self.latest_test = {"observation_id": ticket["observation_id"],
                                    "generation": ticket["generation"], "resolved": False,
                                    "error_type": type(error).__name__}

    def snapshot(self):
        with self.lock:
            test = copy.deepcopy(self.latest_test)
            if test is not None:
                test["fresh"] = test["generation"] == self.generation
            return {"protocol": "tool_evidence_v1", "observations": self.sequence,
                    "possible_change_generation": self.generation,
                    "confirmed_edit_write_changes": self.confirmed_mutations,
                    "tools_since_confirmed_change": self.tools_since_mutation, "shell_calls": self.shell_calls,
                    "last_observed_reads": copy.deepcopy(list(self.reads.values())),
                    "last_observed_mutations": copy.deepcopy(list(self.mutations)), "latest_fixed_tests": test,
                    "latest_tool_error": copy.deepcopy(self.latest_error),
                    "scope": "Tool-return evidence only. Shell side effects and current net diff are not inferred; "
                             "old hashes/edits are historical observations, not proof of present contents."}

    def memory(self):
        data = self.snapshot()
        # Bounded facts, not a generative summary of what the agent intended to do.
        for read in data["last_observed_reads"]:
            if read["file_sha256"]:
                read["file_sha256"] = read["file_sha256"][:12]
        for change in data["last_observed_mutations"]:
            change["file_sha256"] = change["file_sha256"][:12]
        # Drop the oldest historical entries rather than truncating a factual
        # sentence/JSON field halfway. Full bounded state remains in result.json.
        omitted = {"reads": 0, "mutations": 0}
        while len(json.dumps(data, ensure_ascii=False)) > 3600:
            if data["last_observed_reads"]:
                data["last_observed_reads"].pop(0)
                omitted["reads"] += 1
            elif data["last_observed_mutations"]:
                data["last_observed_mutations"].pop(0)
                omitted["mutations"] += 1
            else:
                break
        if any(omitted.values()):
            data["older_observations_omitted_from_memory"] = omitted
        return "Observed repair evidence (tool results, not model claims):\n" + json.dumps(
            data, ensure_ascii=False, separators=(",", ":"))
