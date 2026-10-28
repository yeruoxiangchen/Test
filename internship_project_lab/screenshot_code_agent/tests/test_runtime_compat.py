import hashlib
import json

import pytest

from screenshot_agent.config import RuntimeConfig, Task
from screenshot_agent.control import Events
from screenshot_agent.runtime import OCISandbox, verified_overlay
from screenshot_agent.sandbox import SandboxUnavailable


def overlay(tmp_path, name):
    path = tmp_path / name
    path.mkdir()
    manifest = path / "manifest.json"
    manifest.write_text(json.dumps({"packages": ["fixture==1"], "python_version": "3.9"}))
    return path, hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_overlays_require_receipts_and_an_oci_runtime(tmp_path):
    path, checksum = overlay(tmp_path, "dependency")
    with pytest.raises(ValueError, match="both a directory"):
        RuntimeConfig(backend="oci", python_overlay=str(path))
    with pytest.raises(ValueError, match="both a directory"):
        RuntimeConfig(backend="oci", library_overlay_manifest_sha256=checksum)
    with pytest.raises(ValueError, match="only by the OCI"):
        RuntimeConfig(backend="venv", python_overlay=str(path), python_overlay_manifest_sha256=checksum)
    with pytest.raises(ValueError):
        RuntimeConfig(backend="oci", python_overlay=str(path), python_overlay_manifest_sha256="changed")


def test_overlay_changed_manifest_and_escaping_links_are_rejected(tmp_path):
    path, checksum = overlay(tmp_path, "dependency")
    assert verified_overlay(str(path), checksum) == path
    with pytest.raises(SandboxUnavailable, match="manifest changed"):
        verified_overlay(str(path), "0" * 64)
    (path / "external").symlink_to(tmp_path / "private")
    with pytest.raises(SandboxUnavailable, match="escapes"):
        verified_overlay(str(path), checksum)


def test_task_resolves_overlay_paths_relative_to_manifest(tmp_path):
    path, checksum = overlay(tmp_path, "dependency")
    task_file = tmp_path / "task.json"
    task_file.write_text(json.dumps({"task_id": "fixture", "repo": "repo", "problem": "fix",
                                    "runtime": {"backend": "oci", "python_overlay": "dependency",
                                                "python_overlay_manifest_sha256": checksum}}))
    assert Task.from_file(task_file).runtime.python_overlay == str(path)


def test_runtime_mounts_dependencies_readonly_and_preserves_testbed_alias(tmp_path):
    python, python_sha = overlay(tmp_path, "python")
    libs, libs_sha = overlay(tmp_path, "libraries")
    rootfs = tmp_path / "rootfs"
    (rootfs / "opt/miniconda3/envs/testbed/bin").mkdir(parents=True)
    (rootfs / "opt/miniconda3/envs/testbed/bin/python").touch()
    repo = tmp_path / "repo"
    repo.mkdir()
    runtime = RuntimeConfig(backend="oci", rootfs=str(rootfs), python_overlay=str(python),
                            python_overlay_manifest_sha256=python_sha, library_overlay=str(libs),
                            library_overlay_manifest_sha256=libs_sha)
    sandbox = OCISandbox(repo, Events(tmp_path / "events.jsonl"), runtime=runtime)
    command = sandbox.command([runtime.python, "-m", "pytest"])
    for source, target in ((python, "/tmp/sca-python"), (libs, "/tmp/sca-libs")):
        index = command.index(str(source))
        assert command[index - 1:index + 2] == ["--ro-bind", str(source), target]
    assert "/tmp/sca-python:/testbed/src:/testbed" in command
    assert "LD_LIBRARY_PATH" in command
    assert command[command.index("--chdir") + 1] == "/testbed"
    assert sandbox.description()["python_overlay_manifest_sha256"] == python_sha
