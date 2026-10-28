"""Pinned task images, with an explicit same-repository fallback for unavailable images.

An adapted task uses a donor runtime but exports only the target base commit.
Donor history and its checked-out source are never exposed as the task source.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import tarfile
from pathlib import Path

from .control import atomic_json
from .oci import RegistryImageUnavailableError, check_digest, pull_image
from .repository import git


def _read(path):
    return json.loads(Path(path).read_text())


def _pin(path, data):
    if path.exists() and _read(path) != data:
        raise ValueError(f"Image/source pin changed; preserve the old experiment: {path}")
    atomic_json(path, data)


def _cached_image(root, digest):
    directory = root / ".cache/oci/images" / check_digest(digest)
    path = directory / "image.json"
    if not path.is_file() or not (directory / "bundle/rootfs").is_dir():
        return None
    image = _read(path)
    if image["platform_digest"] != digest:
        raise ValueError("Cached image metadata differs from the pinned digest")
    if Path(image["rootfs"]).resolve() != (directory / "bundle/rootfs").resolve():
        raise ValueError("Cached image rootfs points outside its digest directory")
    return image


def _available_donors(root, output, record):
    donors = []
    for task in sorted((output / "tasks").glob("*/benchmark.json")):
        image_path = task.parent / "image.json"
        if not image_path.is_file():
            continue
        previous = _read(task).get("record", {})
        if previous.get("repo") != record["repo"] or previous.get("instance_id") == record["instance_id"]:
            continue
        image = _read(image_path)
        image = _cached_image(root, image["platform_digest"])
        if image is None or not (Path(image["rootfs"]) / "testbed/.git").exists():
            continue
        donors.append((previous.get("version") != record.get("version"), previous["instance_id"], image))
    donors.sort(key=lambda item: (item[0], item[1], item[2]["platform_digest"]))
    return donors


def _contains(repo, commit):
    return git(repo, "cat-file", "-e", commit + "^{commit}", check=False).returncode == 0


def _exact_source(root, image, record):
    donor_repo = Path(image["rootfs"]) / "testbed"
    commit = record["base_commit"]
    if _contains(donor_repo, commit):
        return donor_repo, "donor_git_exact_commit"
    cache = root / ".cache/source_repos"
    cache.mkdir(parents=True, exist_ok=True)
    name = record["repo"].replace("/", "__")
    repo = cache / (name + ".git")
    url = "https://github.com/" + record["repo"] + ".git"
    with (cache / (name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not repo.exists():
            repo.mkdir()
            git(repo, "init", "--bare", "-q")
            git(repo, "remote", "add", "origin", url)
        if git(repo, "remote", "get-url", "origin").stdout.decode().strip() != url:
            raise ValueError("Source cache origin differs from the declared GitHub repository")
        if _contains(repo, commit):
            return repo, "source_cache_exact_commit"
        # Preserve the user's proxy environment while disabling credential helpers and prompts.
        env = os.environ.copy()
        env.update(GIT_TERMINAL_PROMPT="0", GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null")
        subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "credential.helper=",
                        "-C", str(repo), "fetch", "--no-tags", "--depth=1", "origin", commit],
                       env=env, check=True, timeout=300)
        if not _contains(repo, commit):
            raise ValueError("Exact source commit was not fetched")
        return repo, "github_exact_commit_fetch"


def _export_source(repo, commit, folder):
    destination = folder / "fallback_source"
    archive = folder / "fallback_source.tar"
    if destination.exists() or archive.exists():
        raise ValueError("Refusing to overwrite a previous fallback source export; archive the attempt first")
    folder.mkdir(parents=True, exist_ok=True)
    git(repo, "archive", "--format=tar", "--output", str(archive.resolve()), commit)
    expected = {}
    for row in git(repo, "ls-tree", "-rz", "--full-tree", commit).stdout.split(b"\0"):
        if not row:
            continue
        meta, name = row.split(b"\t", 1)
        mode, kind, checksum = meta.decode().split()
        if kind != "blob":
            raise ValueError("Exact source export requires materializing submodules separately")
        expected[os.fsdecode(name)] = (mode, checksum)
    with tarfile.open(archive) as source:
        seen = set()
        for entry in source.getmembers():
            path = Path(entry.name)
            if path.is_absolute() or ".." in path.parts or ".git" in path.parts:
                raise ValueError("Unsafe path in exact-commit source archive")
            if entry.isdir():
                continue
            if entry.name not in expected or entry.name in seen:
                raise ValueError("Archive has unexpected or duplicate paths")
            mode, checksum = expected[entry.name]
            if entry.issym() and mode == "120000":
                content = os.fsencode(entry.linkname)
            elif entry.isfile() and mode in ("100644", "100755"):
                content = source.extractfile(entry).read()
                if bool(entry.mode & 0o111) != (mode == "100755"):
                    raise ValueError("Archive executable mode differs from the target tree")
            else:
                raise ValueError("Archive entry type differs from the target tree")
            digest = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            if digest != checksum:
                raise ValueError("Archive content differs from the target source blob")
            seen.add(entry.name)
        if seen != set(expected):
            raise ValueError("Archive omitted target source files, possibly due to export-ignore attributes")
        source.extractall(destination, filter="data")
    h = hashlib.sha256()
    with archive.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(block)
    return destination, h.hexdigest()


def resolve_task_image(root: Path, output: Path, record: dict, folder: Path):
    """Return (runtime image, exact source directory, explicit provenance).

    Existing official and adapted pins are immutable. Only the specific fresh-token
    no-access exception enables a fallback; TLS, timeout, checksum and other errors
    remain failures of the original image rather than silently changing protocols.
    """
    root, output, folder = Path(root).resolve(), Path(output).resolve(), Path(folder).resolve()
    name, repo_name, commit = record["instance_id"], record["repo"], record["base_commit"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
        raise ValueError("Unsafe task instance ID")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_name) \
            or any(part in (".", "..") for part in repo_name.split("/")):
        raise ValueError("Source must be a declared GitHub owner/repository")
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("Source base_commit must be a full SHA-1 commit ID")
    original_pin = output / "image_pins" / (name + ".json")
    adapted_pin = output / "adapted_image_pins" / (name + ".json")
    if original_pin.exists() and adapted_pin.exists():
        raise ValueError("Task has both original and adapted image pins")
    identity = {"requested_image": record["image"], "repo": repo_name, "target_commit": commit}
    chosen, image, source_method = None, None, None
    if adapted_pin.exists():
        chosen = _read(adapted_pin)
        if any(chosen.get(key) != value for key, value in identity.items()):
            raise ValueError("Adapted image/source pin differs from the requested task")
        image = _cached_image(root, chosen["actual_image_digest"])
        if image is None:
            raise ValueError("Pinned adapted runtime is missing; restore its cache, do not select a different donor")
        source_method = chosen["source_method"]
    else:
        reference = record["image"]
        if original_pin.exists():
            pin = _read(original_pin)
            if pin["image"] != reference:
                raise ValueError("Original image pin differs from the requested image")
            image = _cached_image(root, pin["platform_digest"])
            reference = reference.split("@")[0].rsplit(":", 1)[0] + "@" + pin["platform_digest"]
        else:
            matches = []
            for path in (root / ".cache/oci/images").glob("*/image.json"):
                metadata = _read(path)
                if metadata.get("requested_image") == reference:
                    cached = _cached_image(root, metadata["platform_digest"])
                    if cached:
                        matches.append(cached)
            if len({item["platform_digest"] for item in matches}) > 1:
                raise ValueError("Multiple cached digests for this image; pin one explicitly")
            image = matches[0] if matches else None
        if image is None:
            try:
                image = pull_image(reference, root / ".cache/oci", root / ".cache/umoci", workers=2)
            except RegistryImageUnavailableError:
                if original_pin.exists():
                    raise ValueError("Pinned original image is unavailable; cannot silently replace its runtime") from None
                donors = _available_donors(root, output, record)
                if not donors:
                    raise RegistryImageUnavailableError(
                        f"Original image unavailable and no complete same-repository donor for {name}") from None
                _, donor_id, image = donors[0]
                chosen = {**identity, "donor_instance_id": donor_id,
                          "actual_image": image["requested_image"], "actual_image_digest": image["platform_digest"],
                          "reason": "Original image absent or unavailable to anonymous registry clients"}
        if chosen is None:
            _pin(original_pin, {"image": record["image"], "platform_digest": image["platform_digest"]})
            return image, Path(image["rootfs"]) / "testbed", {
                "kind": "original_task_image", **identity, "actual_image": image["requested_image"],
                "actual_image_digest": image["platform_digest"], "environment_adapted": False}
    source_repo, acquisition = _exact_source(root, image, record)
    tree = git(source_repo, "rev-parse", commit + "^{tree}").stdout.decode().strip()
    if chosen.get("target_tree") and chosen["target_tree"] != tree:
        raise ValueError("Exact source tree differs from the adapted pin")
    chosen = {**chosen, "target_tree": tree, "source_method": source_method or acquisition}
    _pin(adapted_pin, chosen)
    source, archive_hash = _export_source(source_repo, commit, folder)
    provenance = {"kind": "same_repository_donor_exact_commit", **chosen,
                  "environment_adapted": True, "archive_sha256": archive_hash,
                  "interpretation": "Donor runtime with exact target base source; not the original task image. "
                                    "Requires independent buggy-base and reference-patch validation."}
    atomic_json(folder / "image_provenance.json", provenance)
    return image, source, provenance
