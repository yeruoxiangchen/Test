import os

import pytest

from screenshot_agent.config import RuntimeConfig
from screenshot_agent.repository import export_patch, git, prepare_repository, replay_patch, validate_tree


def fixture_repo(tmp_path):
    source = tmp_path / "source"
    fixture = source / "test/fixtures/linter/sqlfluffignore"
    fixture.mkdir(parents=True)
    (source / "app.py").write_text("value = 1\n")
    (fixture / "query.sql").write_text("SELECT 1;\n")
    (fixture / "path_c").symlink_to(".", target_is_directory=True)
    allowed = {"test/fixtures/linter/sqlfluffignore/path_c": "."}
    return source, allowed


def test_baseline_self_directory_link_is_an_entry_not_a_recursive_walk(tmp_path):
    source, allowed = fixture_repo(tmp_path)
    with pytest.raises(ValueError, match="symbolic links"):
        validate_tree(source)
    paths = validate_tree(source, allowed_symlinks=allowed)
    assert len(paths) == 3
    link = source / next(iter(allowed))
    assert link in paths and link.is_symlink()
    with pytest.raises(ValueError, match="exceeds limit"):
        validate_tree(source, allowed_symlinks=allowed, max_files=2)
    assert RuntimeConfig().allowed_symlinks == {}


@pytest.mark.parametrize("mutation", ["changed_target", "deleted", "replaced_regular", "new_link", "ignored_directory_link"])
def test_baseline_link_allowlist_rejects_every_structural_change(tmp_path, mutation):
    source, allowed = fixture_repo(tmp_path)
    link = source / next(iter(allowed))
    if mutation == "new_link":
        (source / "new_link").symlink_to("app.py")
    elif mutation == "ignored_directory_link":
        (source / ".venv").symlink_to("test", target_is_directory=True)
    else:
        link.unlink()
        if mutation == "changed_target":
            link.symlink_to("./")  # Same resolved target is insufficient: preserve the original text.
        elif mutation == "replaced_regular":
            link.write_text(".")
    with pytest.raises(ValueError, match="symbolic link"):
        validate_tree(source, allowed_symlinks=allowed)


def test_allowlist_cannot_authorize_external_or_absolute_links(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("private")
    link = source / "link"
    link.symlink_to("../outside")
    with pytest.raises(ValueError, match="escapes repository"):
        validate_tree(source, allowed_symlinks={"link": "../outside"})
    link.unlink()
    link.symlink_to(str(source))
    with pytest.raises(ValueError, match="must be relative"):
        validate_tree(source, allowed_symlinks={"link": str(source)})


def test_allowlisted_chained_link_cannot_escape_through_changed_ancestor(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "link").symlink_to("bridge/private")
    (source / "bridge").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="escapes repository"):
        validate_tree(source, allowed_symlinks={"link": "bridge/private"})


def test_prepare_export_replay_preserves_cycle_fixture_without_mutating_source(tmp_path):
    source, allowed = fixture_repo(tmp_path)
    baseline, workspace = prepare_repository(source, tmp_path / "run", allowed_symlinks=allowed)
    name = next(iter(allowed))
    assert (baseline / name).is_symlink() and os.readlink(baseline / name) == "."
    assert (workspace / name).is_symlink() and os.readlink(workspace / name) == "."
    (workspace / "app.py").write_text("value = 2\n")
    patch = tmp_path / "patch.diff"
    result = export_patch(baseline, workspace, patch, ["test"], allowed_symlinks=allowed)
    assert result["changed_files"] == ["app.py"]
    replay = tmp_path / "replay"
    assert replay_patch(baseline, replay, patch, allowed_symlinks=allowed)["applied"]
    assert (replay / "app.py").read_text() == "value = 2\n"
    assert (replay / name).is_symlink() and os.readlink(replay / name) == "."
    assert (source / "app.py").read_text() == "value = 1\n"


def test_replay_rejects_patch_that_adds_a_new_link(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("value = 1\n")
    baseline, workspace = prepare_repository(source, tmp_path / "run")
    (workspace / "new_link").symlink_to("app.py")
    git(workspace, "add", "new_link")
    patch = tmp_path / "unsafe.diff"
    patch.write_bytes(git(workspace, "diff", "--cached", "--binary").stdout)
    result = replay_patch(baseline, tmp_path / "replay", patch)
    assert not result["applied"] and "symbolic links" in result["error"]
