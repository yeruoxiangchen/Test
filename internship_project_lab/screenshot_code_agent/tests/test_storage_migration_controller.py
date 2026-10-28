import fcntl
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_scale_4gpu.py"
spec = importlib.util.spec_from_file_location("storage_controller", SCRIPT)
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def test_migration_prevents_new_controller_and_output_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    with (tmp_path / ".storage-migration.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Storage migration is active"):
            controller.Controller(tmp_path / "run", "resume")
        assert not (tmp_path / "run").exists()
    running = controller.Controller(tmp_path / "run", "resume")
    running.close()


def test_active_controllers_block_switch_until_all_finish(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    first = controller.Controller(tmp_path / "first", "resume")
    second = controller.Controller(tmp_path / "second", "resume")
    with (tmp_path / ".storage-migration.lock").open("a") as lock:
        with pytest.raises(BlockingIOError):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        first.close()
        with pytest.raises(BlockingIOError):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        second.close()
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_source_snapshot_uses_original_project_storage_lock(tmp_path, monkeypatch):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "source_state.json").write_text(json.dumps({"original_project": str(tmp_path)}))
    monkeypatch.setattr(controller, "ROOT", frozen)
    with (tmp_path / ".storage-migration.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Storage migration is active"):
            controller.Controller(tmp_path / "run", "resume")
    assert not (frozen / ".storage-migration.lock").exists()


def test_failed_controller_releases_its_shared_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    (tmp_path / "run").write_text("not a directory")
    with pytest.raises(FileExistsError):
        controller.Controller(tmp_path / "run", "resume")
    with (tmp_path / ".storage-migration.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
