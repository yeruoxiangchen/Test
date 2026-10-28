import json
from pathlib import Path

import pytest

from screenshot_agent.control import atomic_json
from screenshot_agent.scale_compat import file_sha, python_overlay
from screenshot_agent.scale_data import finalize


def test_python_overlay_uses_image_abi_and_reuses_sealed_packages(tmp_path, monkeypatch):
    rootfs = tmp_path / "rootfs"
    (rootfs / "opt/miniconda3/envs/testbed/lib/python3.10/encodings").mkdir(parents=True)
    calls = []
    def install(command, **kwargs):
        calls.append(command)
        folder = Path(command[command.index("--target") + 1])
        folder.mkdir()
        (folder / "pytest.py").write_text("# immutable package fixture\n")
    monkeypatch.setattr("screenshot_agent.scale_compat.subprocess.run", install)
    overlay = python_overlay(tmp_path, "pydicom/pydicom", rootfs)
    assert calls[0][calls[0].index("--abi") + 1] == "cp310"
    manifest = json.loads((overlay / "manifest.json").read_text())
    assert manifest["requirements"] == ["pytest==7.4.4"]
    assert manifest["files_sha256"]["pytest.py"] == file_sha(overlay / "pytest.py")
    assert python_overlay(tmp_path, "pydicom/pydicom", rootfs) == overlay
    assert len(calls) == 1
    manifest["requirements"] = ["pytest==8.3.5"]
    atomic_json(overlay / "manifest.json", manifest)
    with pytest.raises(ValueError, match="fixed profile"):
        python_overlay(tmp_path, "pydicom/pydicom", rootfs)


def test_unknown_python_abi_is_rejected_before_download(tmp_path):
    with pytest.raises(ValueError, match="wheel ABI"):
        python_overlay(tmp_path, "pyvista/pyvista", tmp_path / "missing")


def test_legacy_pydicom_keeps_its_python36_compatible_pytest(tmp_path):
    rootfs = tmp_path / "rootfs"
    (rootfs / "opt/miniconda3/envs/testbed/lib/python3.6/encodings").mkdir(parents=True)
    assert python_overlay(tmp_path, "pydicom/pydicom", rootfs) is None


def test_complete_dev_can_freeze_without_using_test_results(tmp_path):
    rows = [{"instance_id": name, "repo": "a/b", "sca_split": split, "in_verified": False}
            for name, split in [("d1", "dev"), ("d2", "dev"), ("d3", "dev"), ("t1", "test")]]
    atomic_json(tmp_path / "selection.json", {"targets": {"dev": 2, "test": 1}})
    atomic_json(tmp_path / "selected_records.json", rows)
    for name, state in [("d1", "validated"), ("d2", "environment_failed"), ("d3", "validated")]:
        atomic_json(tmp_path / "tasks" / name / "audit.json", {"status": state})
    report = finalize(tmp_path, split="dev")
    assert report["ready_by_split"] == {"dev": True, "test": False}
    assert report["ready"]
    assert (tmp_path / "dev.jsonl").is_file()
    assert not (tmp_path / "test.jsonl").exists()
    assert report["validated_selected"]["dev"] == 2
