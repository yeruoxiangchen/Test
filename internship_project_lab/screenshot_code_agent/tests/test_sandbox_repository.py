import json
import time

import pytest

from screenshot_agent.repository import export_patch, prepare_repository, replay_patch

pytestmark = pytest.mark.sandbox


def test_all_file_operations_are_isolated(sandbox, tmp_path):
    payload = sandbox.filesystem("edit_file", {"file_path": "sample.py", "old_string": "value = 1", "new_string": "value = 2"})
    assert json.loads(payload)["returncode"] == 0
    assert (sandbox.repo / "sample.py").read_text() == "value = 2\n"
    outside = tmp_path / "host_secret"
    outside.write_text("host-only-value")
    result = sandbox.run(["python3", "-c", f"from pathlib import Path; print(Path({str(outside)!r}).exists())"])
    assert result["output"].strip() == "False"
    result = sandbox.filesystem("read_file", {"file_path": "../host_secret"})
    assert "Path must stay inside" in result
    assert "host-only-value" not in result


def test_environment_network_and_protected_tests(sandbox, monkeypatch):
    monkeypatch.setenv("SCA_SENTINEL_SECRET", "never-forward-this")
    result = sandbox.run("python3 -c 'import os; print(os.environ.get(\"SCA_SENTINEL_SECRET\"))'")
    assert result["output"].strip() == "None"
    result = sandbox.filesystem("write_file", {"file_path": "tests/test_fixed.py", "content": "forged"})
    assert json.loads(result)["returncode"] != 0
    assert (sandbox.repo / "tests/test_fixed.py").read_text() == "immutable\n"
    result = sandbox.run("python3 -c 'import socket; s=socket.socket(); s.settimeout(.2); s.connect((\"1.1.1.1\",53))'")
    assert result["returncode"] != 0


def test_timeout_kills_descendants(sandbox):
    result = sandbox.run("(sleep 1; touch survived) & sleep 20", timeout=.1)
    assert result["timed_out"]
    time.sleep(1.2)
    assert not (sandbox.repo / "survived").exists()
    assert not sandbox.active


def test_unique_edit_and_path_escape(sandbox):
    (sandbox.repo / "repeat.py").write_text("x = 1\nx = 1\n")
    result = sandbox.filesystem("edit_file", {"file_path": "repeat.py", "old_string": "x = 1", "new_string": "x = 2"})
    assert json.loads(result)["returncode"] != 0
    result = sandbox.filesystem("glob", {"pattern": "../../*"})
    assert json.loads(result)["returncode"] != 0


def test_patch_contains_new_files_and_deletions(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "old.py").write_text("old = 1\n")
    (source / "stay.py").write_text("value = 1\n")
    output = tmp_path / "run"
    output.mkdir()
    base, work = prepare_repository(source, output)
    (work / "old.py").unlink()
    (work / "new.py").write_text("new = 2\n")
    (work / ".gitignore").write_text("new.py\n")
    patch = output / "patch.diff"
    meta = export_patch(base, work, patch, [])
    assert "new.py" in meta["changed_files"]
    assert "old.py" in meta["changed_files"]
    target = output / "verify"
    assert replay_patch(base, target, patch)["applied"]
    assert not (target / "old.py").exists()
    assert (target / "new.py").read_text() == "new = 2\n"
    assert (source / "old.py").exists()


def test_symlink_patch_cannot_escape_snapshot(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "leak").symlink_to("/etc/passwd")
    output = tmp_path / "run"
    output.mkdir()
    with pytest.raises(ValueError, match="symbolic links"):
        prepare_repository(source, output)
