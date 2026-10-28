import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "storage_migration", Path(__file__).resolve().parents[1] / "scripts/migrate_coder_storage.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def migration(tmp_path, monkeypatch):
    project = tmp_path / "project"
    for source, _ in m.ITEMS:
        folder = project / source
        folder.mkdir(parents=True)
        (folder / "tiny.txt").write_text(source)
    model = project / m.MODEL
    (model / "download_complete.json").write_text(json.dumps({
        "files": {"tiny.txt": {"bytes": (model / "tiny.txt").stat().st_size,
                               "sha256": m.digest(model / "tiny.txt")}}}))
    (project / ".coder-env/setup_complete.json").write_text(json.dumps({
        "vllm_version": "0.10.2", "transformers_version": "4.55.2"}))
    value = m.Migration(project, tmp_path / "target")
    monkeypatch.setattr(m, "assert_idle", lambda *_: None)
    monkeypatch.setattr(value, "smoke", lambda: {"tested": True})
    return value


def test_plan_reads_small_manifest_and_does_not_create_target(migration):
    planned = migration.plan()
    assert planned["model_bytes"] > 0 and not planned["copies_started"]
    assert not migration.target.exists()


def test_apply_preserves_content_original_prefix_and_backups_then_rolls_back(migration):
    before = [(old, (old / "tiny.txt").read_bytes()) for old in (
        migration.project / source for source, _ in m.ITEMS)]
    migration.apply()
    assert m.read(migration.state_path)["stage"] == "completed"
    for (old, destination, backup), (_, original) in zip(migration.paths(), before):
        assert old.is_symlink() and old.resolve() == destination
        assert (old / "tiny.txt").read_bytes() == original
        assert (backup / "tiny.txt").read_bytes() == original
    migration.rollback()
    assert m.read(migration.state_path)["stage"] == "rolled_back"
    for old, destination, backup in migration.paths():
        assert old.is_dir() and not old.is_symlink()
        assert destination.is_dir() and not backup.exists()


def test_refuses_unrelated_nonempty_target(migration):
    migration.target.mkdir()
    (migration.target / "unrelated").write_text("retain")
    with pytest.raises(ValueError, match="unrelated"):
        migration.apply()
    assert (migration.target / "unrelated").read_text() == "retain"


def test_refuses_marker_from_other_project(migration):
    migration.owned(create=True)
    owner = m.read(migration.marker)
    owner["project"] = "/another/project"
    m.atomic(migration.marker, owner)
    with pytest.raises(ValueError, match="another migration"):
        migration.apply()


def test_bad_model_hash_never_switches_source(migration):
    receipt = migration.project / m.MODEL / "download_complete.json"
    bad = m.read(receipt)
    bad["files"]["tiny.txt"]["sha256"] = "0" * 64
    m.atomic(receipt, bad)
    with pytest.raises(ValueError, match="checksum mismatch"):
        migration.apply()
    assert m.read(migration.state_path)["stage"] == "failed_restored"
    assert all(not old.is_symlink() and not backup.exists() for old, _, backup in migration.paths())


def test_failed_import_restores_every_original_and_keeps_copies(migration, monkeypatch):
    def failed():
        raise RuntimeError("import failed")
    monkeypatch.setattr(migration, "smoke", failed)
    with pytest.raises(RuntimeError, match="import failed"):
        migration.apply()
    assert all(old.is_dir() and not old.is_symlink() and destination.is_dir() and not backup.exists()
               for old, destination, backup in migration.paths())


def test_interrupted_copy_can_resume_without_removing_source(migration, monkeypatch):
    original = migration.copy_verify
    calls = []

    def interrupted(old, destination, backup):
        calls.append(old)
        if len(calls) == 2:
            raise KeyboardInterrupt("copy interrupted")
        original(old, destination, backup)

    monkeypatch.setattr(migration, "copy_verify", interrupted)
    with pytest.raises(KeyboardInterrupt):
        migration.apply()
    assert all(not old.is_symlink() for old, _, _ in migration.paths())
    monkeypatch.setattr(migration, "copy_verify", original)
    migration.apply()
    assert all(old.is_symlink() for old, _, _ in migration.paths())


def test_crash_after_rename_recovers_on_next_apply(migration):
    migration.owned(create=True)
    for row in migration.paths():
        migration.copy_verify(*row)
    old, _, backup = next(migration.paths())
    old.rename(backup)
    migration.apply()
    assert all(old.is_symlink() for old, _, _ in migration.paths())


def test_rollback_refuses_unexpected_symlink_before_touching_any_item(migration, tmp_path):
    migration.apply()
    rows = list(migration.paths())
    wrong = tmp_path / "unrelated"
    wrong.mkdir()
    old = rows[-1][0]
    old.unlink()
    old.symlink_to(wrong)
    with pytest.raises(ValueError, match="Unexpected source symlink"):
        migration.rollback()
    assert rows[0][0].is_symlink()
    assert wrong.is_dir()


def test_refuses_live_process_before_creating_target(migration, monkeypatch):
    def active(*args):
        raise RuntimeError("still active")
    monkeypatch.setattr(m, "assert_idle", active)
    with pytest.raises(RuntimeError, match="still active"):
        migration.apply()
    assert not migration.target.exists()


def test_exclusive_lock_conflicts_with_shared_controller(migration):
    lock = migration.project / ".storage-migration.lock"
    with lock.open("a") as handle:
        m.fcntl.flock(handle, m.fcntl.LOCK_SH)
        with pytest.raises(RuntimeError, match="already running"):
            with m.locks(migration.project):
                pytest.fail("must not acquire")


def test_target_inside_project_and_symlink_target_are_rejected(migration, tmp_path):
    with pytest.raises(ValueError, match="separate"):
        m.Migration(migration.project, migration.project / "copied")
    link = tmp_path / "linked"
    link.symlink_to(migration.target)
    with pytest.raises(ValueError, match="real directory"):
        m.Migration(migration.project, link)


def test_rejects_destination_symlink_before_copy(migration, tmp_path):
    migration.owned(create=True)
    row = next(migration.paths())
    other = tmp_path / "other"
    other.mkdir()
    row[1].symlink_to(other)
    with pytest.raises(ValueError, match="redirect"):
        migration.copy_verify(*row)
    assert list(other.iterdir()) == []


def test_dangling_backup_is_not_overwritten(migration, tmp_path):
    migration.owned(create=True)
    row = next(migration.paths())
    row[2].symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="conflict"):
        migration.layout(*row)
    assert row[0].is_dir() and row[2].is_symlink()


def test_completed_apply_is_idempotent_and_does_not_overwrite_used_environment(migration, monkeypatch):
    migration.apply()
    used = migration.target / "coder-env/tiny.txt"
    used.write_text("deliberate later change")
    monkeypatch.setattr(migration, "copy_verify", lambda *_: pytest.fail("must not restore historical backup"))
    migration.apply()
    assert used.read_text() == "deliberate later change"


def test_failure_before_source_validation_does_not_trigger_rollback(migration, monkeypatch, tmp_path):
    migration.owned(create=True)
    rows = list(migration.paths())
    rows[-1][2].symlink_to(tmp_path / "missing")
    monkeypatch.setattr(migration, "copy_verify", lambda *_: pytest.fail("must validate all sources first"))
    monkeypatch.setattr(migration, "rollback", lambda: pytest.fail("must leave preexisting conflict untouched"))
    with pytest.raises(ValueError, match="conflict"):
        migration.apply()


@pytest.mark.parametrize("comm,arguments,permitted", [
    ("(sd-pam)", b"(sd-pam)\0\0", True),
    ("(sd-pam)", b"python\0train.py\0", False),
    ("python", b"(sd-pam)\0", False),
    ("python", b"python\0train.py\0", False),
])
def test_idle_guard_only_skips_exact_systemd_pam_helper(tmp_path, monkeypatch, comm, arguments, permitted):
    proc = tmp_path / "proc"
    (proc / "1").mkdir(parents=True)
    (proc / "1/comm").write_text("systemd\n")
    (proc / "1/cmdline").write_bytes(b"/sbin/init\0")
    helper = proc / "352469"
    helper.mkdir()
    (helper / "comm").write_text(comm + "\n")
    (helper / "cmdline").write_bytes(arguments)

    def protected_link(path):
        if Path(path).parent == helper:
            raise PermissionError("non-dumpable process")
        return "/sbin/init" if Path(path).name == "exe" else "/"

    monkeypatch.setattr(m.os, "readlink", protected_link)
    if permitted:
        m.assert_idle(tmp_path / "project", tmp_path / "target", proc)
    else:
        with pytest.raises(RuntimeError, match="Cannot inspect owned processes"):
            m.assert_idle(tmp_path / "project", tmp_path / "target", proc)


@pytest.mark.parametrize("comm,title,permitted", [
    ("sshd", "sshd: testuser@pts/11", True),
    ("sshd", "sshd: testuser@notty", True),
    ("sshd", "sshd: another@pts/11", False),
    ("python", "sshd: testuser@pts/11", False),
    ("sshd", "python train.py", False),
])
def test_idle_guard_exact_owned_ssh_session_only(tmp_path, monkeypatch, comm, title, permitted):
    proc = tmp_path / "proc"
    (proc / "1").mkdir(parents=True)
    (proc / "1/comm").write_text("systemd\n")
    (proc / "1/cmdline").write_bytes(b"/sbin/init\0")
    helper = proc / "1836434"
    helper.mkdir()
    (helper / "comm").write_text(comm + "\n")
    (helper / "cmdline").write_bytes(title.encode() + b"\0")
    monkeypatch.setattr(m.pwd, "getpwuid", lambda _: type("User", (), {"pw_name": "testuser"})())

    def protected_link(path):
        if Path(path).parent == helper:
            raise PermissionError("non-dumpable process")
        return "/sbin/init" if Path(path).name == "exe" else "/"

    monkeypatch.setattr(m.os, "readlink", protected_link)
    if permitted:
        m.assert_idle(tmp_path / "project", tmp_path / "target", proc)
    else:
        with pytest.raises(RuntimeError, match="Cannot inspect owned processes"):
            m.assert_idle(tmp_path / "project", tmp_path / "target", proc)


def test_idle_guard_collects_every_unreadable_owned_process(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    for pid, comm in ((1, "systemd"), (352469, "python"), (352470, "unknown-runtime")):
        folder = proc / str(pid)
        folder.mkdir(parents=True)
        (folder / "comm").write_text(comm)
        (folder / "cmdline").write_bytes(comm.encode() + b"\0")

    def protected_link(path):
        if Path(path).parent.name != "1":
            raise PermissionError("cannot inspect")
        return "/sbin/init" if Path(path).name == "exe" else "/"

    monkeypatch.setattr(m.os, "readlink", protected_link)
    with pytest.raises(RuntimeError) as error:
        m.assert_idle(tmp_path / "project", tmp_path / "target", proc)
    assert all(value in str(error.value) for value in ("352469", "352470", "python", "unknown-runtime"))


@pytest.mark.parametrize("bad_key", [None, "prefix", "base_prefix", "vllm", "torch_file", "vllm_file"])
def test_smoke_checks_versions_and_physical_environment_paths(migration, monkeypatch, bad_key):
    migration.target.mkdir()
    info = {"prefix": str(migration.target / "coder-env"), "base_prefix": str(migration.target / "model-env"),
            "vllm": "0.10.2", "transformers": "4.55.2", "torch": "2.8.0",
            "torch_file": str(migration.target / "coder-env/lib/torch/__init__.py"),
            "vllm_file": str(migration.target / "coder-env/lib/vllm/__init__.py")}
    if bad_key:
        info[bad_key] = "wrong"

    def fake_run(command, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        assert kwargs["timeout"] == 120
        return type("Result", (), {"stdout": "SCA_STORAGE_SMOKE=" + json.dumps(info)})()

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    if bad_key:
        with pytest.raises(RuntimeError, match="physical environment"):
            m.Migration.smoke(migration)
    else:
        assert m.Migration.smoke(migration) == info


@pytest.mark.parametrize("error", [
    m.subprocess.CalledProcessError(1, ["python"], output="ImportError: example failure"),
    m.subprocess.TimeoutExpired(["python"], 120, output=b"partial import output"),
])
def test_smoke_failure_preserves_captured_output_in_named_log(migration, monkeypatch, error):
    migration.target.mkdir()

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(m.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="cpu_import_check.log"):
        m.Migration.smoke(migration)
    expected = error.output.decode() if isinstance(error.output, bytes) else error.output
    assert (migration.target / "cpu_import_check.log").read_text() == expected
