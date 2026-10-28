import httpx
import pytest

import screenshot_agent.scale_images as images
from screenshot_agent.control import atomic_json
from screenshot_agent.repository import git


def fixture_image(tmp_path):
    root, output = tmp_path / "project", tmp_path / "data"
    digest = "sha256:" + "1" * 64
    directory = root / ".cache/oci/images" / ("1" * 64)
    rootfs = directory / "bundle/rootfs"
    repo = rootfs / "testbed"
    repo.mkdir(parents=True)
    git(repo, "init", "-q")
    (repo / "value.py").write_text("value = 'donor'\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "donor")
    before = git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    (repo / "value.py").write_text("value = 'target base, not gold'\n")
    (repo / "new_file.py").write_text("target_only = True\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "target")
    target = git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    git(repo, "checkout", "--detach", before)
    image = {"requested_image": "swebench/fixture-donor:latest", "platform_digest": digest,
             "rootfs": str(rootfs)}
    atomic_json(directory / "image.json", image)
    atomic_json(output / "tasks/org__repo-1/benchmark.json", {
        "record": {"repo": "org/repo", "instance_id": "org__repo-1", "version": "1"}})
    atomic_json(output / "tasks/org__repo-1/image.json", image)
    record = {"instance_id": "org__repo-2", "repo": "org/repo", "base_commit": target,
              "version": "2", "image": "swebench/fixture-target:latest"}
    return root, output, repo, image, record


def test_unavailable_image_uses_exact_target_commit_and_freezes_adapted_pin(tmp_path, monkeypatch):
    root, output, donor, image, record = fixture_image(tmp_path)
    before = git(donor, "rev-parse", "HEAD").stdout
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(args[0])
        raise images.RegistryImageUnavailableError("fixture absent image")

    monkeypatch.setattr(images, "pull_image", unavailable)
    resolved, source, provenance = images.resolve_task_image(root, output, record, output / "attempt1")
    assert resolved == image
    assert (source / "value.py").read_text() == "value = 'target base, not gold'\n"
    assert (source / "new_file.py").exists()
    assert not (source / ".git").exists()
    assert git(donor, "rev-parse", "HEAD").stdout == before
    assert (donor / "value.py").read_text() == "value = 'donor'\n"
    assert provenance["environment_adapted"]
    assert provenance["target_commit"] == record["base_commit"]
    assert provenance["source_method"] == "donor_git_exact_commit"
    assert provenance["target_tree"] == git(donor, "rev-parse", record["base_commit"] + "^{tree}").stdout.decode().strip()
    assert not (output / "image_pins/org__repo-2.json").exists()
    _, second_source, second = images.resolve_task_image(root, output, record, output / "attempt2")
    assert len(calls) == 1  # A pinned adapted image is not reselected or looked up on the registry.
    assert second["archive_sha256"] == provenance["archive_sha256"]
    assert second_source != source


def test_original_pinned_cache_is_reused_without_network(tmp_path, monkeypatch):
    root, output, donor, image, record = fixture_image(tmp_path)
    atomic_json(output / "image_pins/org__repo-2.json", {
        "image": record["image"], "platform_digest": image["platform_digest"]})
    monkeypatch.setattr(images, "pull_image", lambda *a, **k: pytest.fail("No network lookup for a complete pinned cache"))
    result, source, provenance = images.resolve_task_image(root, output, record, output / "attempt")
    assert result == image and source == donor
    assert not provenance["environment_adapted"]
    # The caller still validates the original image's checkout tree against base_commit.


def test_transport_error_does_not_silently_switch_to_donor(tmp_path, monkeypatch):
    root, output, _, _, record = fixture_image(tmp_path)

    def network_error(*args, **kwargs):
        raise httpx.ConnectError("temporary network error")

    monkeypatch.setattr(images, "pull_image", network_error)
    with pytest.raises(httpx.ConnectError):
        images.resolve_task_image(root, output, record, output / "attempt")
    assert not (output / "adapted_image_pins/org__repo-2.json").exists()


def test_original_pin_cannot_change_or_fall_back_to_another_runtime(tmp_path, monkeypatch):
    root, output, _, _, record = fixture_image(tmp_path)
    pin = output / "image_pins/org__repo-2.json"
    atomic_json(pin, {"image": record["image"], "platform_digest": "sha256:" + "2" * 64})
    original = pin.read_bytes()

    def unavailable(*args, **kwargs):
        raise images.RegistryImageUnavailableError("fixture absent pinned image")

    monkeypatch.setattr(images, "pull_image", unavailable)
    with pytest.raises(ValueError, match="cannot silently replace"):
        images.resolve_task_image(root, output, record, output / "attempt")
    assert pin.read_bytes() == original
    assert not (output / "adapted_image_pins/org__repo-2.json").exists()


def test_donor_must_be_from_the_same_repository(tmp_path, monkeypatch):
    root, output, _, _, record = fixture_image(tmp_path)
    record["repo"] = "different/repo"

    def unavailable(*args, **kwargs):
        raise images.RegistryImageUnavailableError("fixture absent image")

    monkeypatch.setattr(images, "pull_image", unavailable)
    with pytest.raises(images.RegistryImageUnavailableError, match="no complete same-repository donor"):
        images.resolve_task_image(root, output, record, output / "attempt")


def test_adapted_source_commit_cannot_change_on_resume(tmp_path, monkeypatch):
    root, output, _, _, record = fixture_image(tmp_path)

    def unavailable(*args, **kwargs):
        raise images.RegistryImageUnavailableError("fixture absent image")

    monkeypatch.setattr(images, "pull_image", unavailable)
    images.resolve_task_image(root, output, record, output / "attempt1")
    record["base_commit"] = "a" * 40
    with pytest.raises(ValueError, match="pin differs"):
        images.resolve_task_image(root, output, record, output / "attempt2")


def test_export_ignore_cannot_silently_omit_target_source(tmp_path):
    _, _, donor, _, _ = fixture_image(tmp_path)
    (donor / ".gitattributes").write_text("value.py export-ignore\n")
    git(donor, "add", ".gitattributes")
    git(donor, "commit", "-qm", "export ignore")
    commit = git(donor, "rev-parse", "HEAD").stdout.decode().strip()
    with pytest.raises(ValueError, match="omitted target source files"):
        images._export_source(donor, commit, tmp_path / "export")


def test_untrusted_repository_path_is_rejected_before_fetch(tmp_path):
    root, output, _, _, record = fixture_image(tmp_path)
    record["repo"] = "../repo"
    with pytest.raises(ValueError, match="GitHub owner/repository"):
        images.resolve_task_image(root, output, record, output / "attempt")
