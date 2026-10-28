"""Exercise the real filesystem worker in a tiny disposable repository.

The target runtime/interpreter is preserved. A passing test command alone cannot
establish that Agent file tools work in an old project environment.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from .sandbox import SandboxUnavailable

PROTOCOL = "filesystem_tools_v1"
REQUIRED_CHECKS = {
    *(f"v{version}.{operation}" for version in (1, 2)
      for operation in ("write_file", "read_file", "glob", "grep", "edit_file")),
    "ambiguous_edit", "repeated_edit_guard", "protected_write", "path_escape",
    "symlink_escape", "git_access", "syntax_feedback", "v2.mutation_feedback", "v2.read_feedback",
}


def validate_receipt(receipt, *, python_version, worker_sha256):
    """Reject partial or stale capability claims, even when the probe process exited zero."""
    if not isinstance(receipt, dict) or receipt.get("protocol") != PROTOCOL:
        raise ValueError("Missing or unsupported filesystem contract protocol")
    checks = receipt.get("checks")
    if not isinstance(checks, dict) or any(checks.get(name) is not True for name in REQUIRED_CHECKS):
        raise ValueError("Incomplete filesystem contract checks")
    if receipt.get("python_version") != python_version:
        raise ValueError("Filesystem AST feedback used a different Python interpreter")
    if receipt.get("worker_sha256") != worker_sha256:
        raise ValueError("Filesystem contract belongs to different worker code")
    if receipt.get("cleanup_complete") is not True:
        raise ValueError("Filesystem probe cleanup was not confirmed")


def worker_reply(raw: str, *, operation: str, expected_returncode: int = 0) -> dict:
    """Fail clearly on worker startup/serialization errors, including old Python."""
    try:
        envelope = json.loads(raw)
        if not isinstance(envelope, dict):
            raise ValueError("worker envelope is not an object")
        if envelope.get("timed_out") or envelope.get("returncode") != expected_returncode:
            raise ValueError(f"unexpected returncode={envelope.get('returncode')}, "
                             f"timed_out={envelope.get('timed_out')}: "
                             f"{str(envelope.get('output', ''))[-1200:]}")
        output = json.loads(envelope["output"])
        if not isinstance(output, dict):
            raise ValueError("worker output is not an object")
        return output
    except (ValueError, TypeError, KeyError) as error:
        raise SandboxUnavailable(f"Filesystem tool preflight failed for {operation}: {error}") from error


def filesystem_probe(sandbox) -> dict:
    checks = {}
    receipt = {"protocol": PROTOCOL, "checks": checks,
               "worker_sha256": hashlib.sha256(Path(__file__).with_name("worker.py").read_bytes()).hexdigest()}
    child = None
    sandbox.events.emit("filesystem_probe_started", protocol=PROTOCOL)

    def require(condition, name):
        if not condition:
            raise SandboxUnavailable(f"Filesystem tool preflight failed: {name}")
        checks[name] = True

    try:
        with tempfile.TemporaryDirectory(prefix="sca-tool-probe-") as temporary:
            repo = Path(temporary) / "repo"
            repo.mkdir()
            protected_text = "protected = 'unchanged'\n"
            (repo / "protected_test.py").write_text(protected_text)
            (repo / ".git").mkdir()
            (repo / ".git/config").write_text("protected git sentinel\n")
            (repo / "escape.py").symlink_to("/etc/passwd")
            kwargs = {"protected_root": repo, "timeout": min(30, sandbox.timeout)}
            if hasattr(sandbox, "runtime"):
                kwargs["runtime"] = sandbox.runtime
            child = type(sandbox)(repo, sandbox.events, ["protected_test.py"], **kwargs)

            def call(operation, arguments, *, version=2, rc=0, guard=None):
                return worker_reply(child.filesystem(operation, arguments, feedback_version=version,
                                                     edit_guard=guard),
                                    operation=operation, expected_returncode=rc)

            for version in (1, 2):
                prefix, name = f"v{version}", f"fixture_{version}.py"
                source = "value = 1\nrepeated = 1\nrepeated = 1\n"
                result = call("write_file", {"file_path": name, "content": source}, version=version)
                require((repo / name).read_text() == source, prefix + ".write_file")
                if version == 2:
                    syntax = result.get("python_syntax", {})
                    require(result.get("ok") is True and result.get("changed") is True
                            and syntax.get("valid") is True and bool(syntax.get("interpreter")),
                            "v2.mutation_feedback")
                    receipt["python_version"] = syntax["interpreter"]
                result = call("read_file", {"file_path": name, "offset": 1, "limit": 1}, version=version)
                require(result.get("output") == "1\tvalue = 1" and result.get("total_lines") == 3,
                        prefix + ".read_file")
                if version == 2:
                    require(result.get("next_offset") == 2
                            and result.get("file_sha256") == hashlib.sha256(source.encode()).hexdigest(),
                            "v2.read_feedback")
                result = call("glob", {"pattern": name}, version=version)
                require(result.get("output") == name, prefix + ".glob")
                result = call("grep", {"pattern": "^value = 1$", "path": name}, version=version)
                require(result.get("output") == f"{name}:1:value = 1", prefix + ".grep")
                result = call("edit_file", {"file_path": name, "old_string": "value = 1",
                                            "new_string": "value = 2"}, version=version)
                require((repo / name).read_text() == source.replace("value = 1", "value = 2"),
                        prefix + ".edit_file")

            unchanged = (repo / "fixture_2.py").read_bytes()
            args = {"file_path": "fixture_2.py", "old_string": "repeated = 1", "new_string": "repeated = 2"}
            result = call("edit_file", args, rc=1)
            require(result.get("error") == "edit_match_failed" and result.get("match_count") == 2
                    and (repo / "fixture_2.py").read_bytes() == unchanged, "ambiguous_edit")
            result = call("edit_file", args, rc=1,
                          guard={"file_sha256": hashlib.sha256(unchanged).hexdigest(), "failures": 2, "limit": 2})
            require(result.get("error") == "repeated_failed_edit"
                    and (repo / "fixture_2.py").read_bytes() == unchanged, "repeated_edit_guard")
            result = call("write_file", {"file_path": "protected_test.py", "content": "changed\n"}, rc=1)
            require("Read-only file system" in result.get("error", "")
                    and (repo / "protected_test.py").read_text() == protected_text, "protected_write")
            for name, label in (("../outside.py", "path_escape"), ("escape.py", "symlink_escape"),
                                (".git/config", "git_access")):
                result = call("read_file", {"file_path": name}, rc=1)
                require(result.get("error_type") == "ValueError" and "Path must stay" in result.get("error", ""),
                        label)
            result = call("write_file", {"file_path": "invalid.py", "content": "value = (\n"})
            require(result.get("ok") is True and result.get("python_syntax", {}).get("valid") is False,
                    "syntax_feedback")
            child.cancel()
            child = None
        receipt["cleanup_complete"] = True
        sandbox.events.emit("filesystem_probe_completed", **receipt)
        return receipt
    except BaseException as error:
        sandbox.events.emit("filesystem_probe_failed", protocol=PROTOCOL,
                            error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if child is not None:
            child.cancel()
