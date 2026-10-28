import json
from pathlib import Path

import pytest

from screenshot_agent.control import Events
from screenshot_agent.sandbox import BubblewrapSandbox, SandboxUnavailable
from screenshot_agent.tool_probe import PROTOCOL, worker_reply


@pytest.mark.parametrize("envelope", [
    {"returncode": 1, "output": "SyntaxError: future feature annotations is not defined", "timed_out": False},
    {"returncode": 1, "output": "AttributeError: 'PosixPath' object has no attribute 'is_relative_to'"},
    {"returncode": 0, "output": "{\"ok\": true}", "timed_out": True},
    {"returncode": 0, "output": "worker did not return JSON"},
    {"returncode": 0, "output": "[]"},
    [],
])
def test_preflight_rejects_worker_startup_and_protocol_errors(envelope):
    with pytest.raises(SandboxUnavailable, match="Filesystem tool preflight failed for write_file"):
        worker_reply(json.dumps(envelope), operation="write_file")


def test_expected_failure_still_requires_structured_worker_reply():
    raw = json.dumps({"returncode": 1, "output": "SyntaxError: broken worker"})
    with pytest.raises(SandboxUnavailable):
        worker_reply(raw, operation="protected_write", expected_returncode=1)
    raw = json.dumps({"returncode": 1, "output": json.dumps({"error": "blocked", "error_type": "ValueError"})})
    assert worker_reply(raw, operation="protected_write", expected_returncode=1)["error"] == "blocked"


def test_failed_filesystem_contract_is_propagated_and_fixture_cleaned(tmp_path, monkeypatch):
    repo = tmp_path / "original"
    repo.mkdir()
    (repo / "original.py").write_text("original = 1\n")
    events_path = tmp_path / "events.jsonl"
    sandbox = BubblewrapSandbox(repo, Events(events_path))
    probe_repos = []

    def broken_worker(self, operation, arguments, **kwargs):
        probe_repos.append(self.repo)
        return json.dumps({"returncode": 1, "output": "SyntaxError: future feature annotations is not defined"})

    monkeypatch.setattr(BubblewrapSandbox, "filesystem", broken_worker)
    with pytest.raises(SandboxUnavailable, match="future feature annotations"):
        sandbox.filesystem_probe()
    assert probe_repos and probe_repos[0] != repo
    assert not probe_repos[0].parent.exists()
    assert list(repo.iterdir()) == [repo / "original.py"]
    assert (repo / "original.py").read_text() == "original = 1\n"
    assert '"filesystem_probe_failed"' in events_path.read_text()
    assert '"filesystem_probe_completed"' not in events_path.read_text()


@pytest.mark.sandbox
def test_real_probe_exercises_tools_feedback_and_protection_without_task_mutation(tmp_path):
    repo = tmp_path / "original"
    repo.mkdir()
    (repo / "original.py").write_text("original = 1\n")
    events_path = tmp_path / "events.jsonl"
    sandbox = BubblewrapSandbox(repo, Events(events_path), timeout=5)
    try:
        receipt = sandbox.probe()["filesystem_probe"]
    finally:
        sandbox.cancel()
    assert receipt["protocol"] == PROTOCOL
    for version in (1, 2):
        for operation in ("glob", "grep", "read_file", "write_file", "edit_file"):
            assert receipt["checks"][f"v{version}.{operation}"] is True
    for name in ("v2.mutation_feedback", "v2.read_feedback", "ambiguous_edit", "repeated_edit_guard",
                 "protected_write", "path_escape", "symlink_escape", "git_access", "syntax_feedback"):
        assert receipt["checks"][name] is True
    assert receipt["python_version"].startswith("3.")
    assert len(receipt["worker_sha256"]) == 64
    assert receipt["cleanup_complete"] is True
    assert list(repo.iterdir()) == [repo / "original.py"]
    assert (repo / "original.py").read_text() == "original = 1\n"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    completed = [event for event in events if event.get("event") == "filesystem_probe_completed"]
    assert len(completed) == 1
    assert completed[0]["checks"] == receipt["checks"]


def test_worker_containment_uses_relative_to_and_keeps_boundaries():
    # This helper must work before Path.is_relative_to was introduced in Python 3.9.
    from screenshot_agent.worker import contained

    class OldPath:
        def __init__(self, value):
            self.value = Path(value)

        def relative_to(self, root):
            return self.value.relative_to(root)

    assert contained(OldPath("/workspace/sub/file.py"), Path("/workspace"))
    assert not contained(OldPath("/workspace-other/file.py"), Path("/workspace"))
    assert not contained(OldPath("/etc/passwd"), Path("/workspace"))
