"""Fetch digest-verified public Docker Hub images into a rootless OCI cache.

The image's original registry manifest is retained. umoci performs layer/whiteout
handling and rootless unpacking; task code is never executed during image setup.
"""

from __future__ import annotations

import base64
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

from .control import atomic_json

MANIFEST_TYPES = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def check_digest(value: str) -> str:
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", value):
        raise ValueError("Only full SHA-256 digests are accepted")
    return value.split(":", 1)[1]


def _request(client, url, **kwargs):
    for attempt in range(5):
        try:
            response = client.get(url, **kwargs)
            response.raise_for_status()
            return response
        except (httpx.TransportError, httpx.HTTPStatusError, OSError) as error:
            if isinstance(error, httpx.HTTPStatusError) and error.response.status_code not in (408, 429) \
                    and error.response.status_code < 500:
                raise  # Refresh authentication separately; permanent 4xx errors do not improve with sleeps.
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 15))


class RegistryImageUnavailableError(PermissionError):
    """A fresh anonymous token has no pull grant; the image may be absent or private."""


class _RegistrySession:
    """Refresh short-lived registry tokens for both manifests and concurrent layer downloads."""

    def __init__(self, client, repo, realm, service):
        self.client, self.repo, self.realm, self.service = client, repo, realm, service
        self.token, self.expires_at = None, 0.0
        self.lock = threading.Lock()

    def refresh(self, rejected=None):
        if not self.realm:
            return
        if not self.realm.startswith("https://"):
            raise ValueError("Registry authentication must use HTTPS")
        with self.lock:
            current = "Bearer " + self.token if self.token else None
            if rejected is not None and current != rejected:
                return  # Another layer worker already refreshed the rejected token.
            if rejected is None and self.token and time.monotonic() < self.expires_at:
                return
            data = _request(self.client, self.realm, params={
                "service": self.service, "scope": f"repository:{self.repo}:pull"},
                headers={"Cache-Control": "no-cache", "Pragma": "no-cache"}).json()
            token = data.get("token") or data.get("access_token")
            if not isinstance(token, str) or not token:
                raise ValueError("Registry did not return an authentication token")
            # This is diagnostic only: the registry still verifies the actual token on every request.
            # Opaque tokens are supported; Docker Hub JWTs explicitly expose an empty grant for
            # repositories that are absent or unavailable to anonymous clients.
            claims = {}
            try:
                payload = token.split(".")[1]
                claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
            except (IndexError, ValueError, UnicodeError):
                pass
            grants = claims.get("access") if isinstance(claims, dict) else None
            if isinstance(grants, list) and not any(
                    isinstance(grant, dict) and grant.get("name") == self.repo
                    and "pull" in grant.get("actions", []) for grant in grants):
                raise RegistryImageUnavailableError(
                    f"Fresh registry token grants no anonymous pull access to {self.repo}; "
                    "image may be absent or private. Retrying an unchanged token cannot fix this.")
            lifetime = float(data.get("expires_in", 300))
            if lifetime <= 0:
                raise ValueError("Registry returned an expired authentication token")
            self.token = token
            self.expires_at = time.monotonic() + max(0.0, lifetime - min(30.0, lifetime / 10))

    def headers(self):
        self.refresh()
        headers = {"Accept": MANIFEST_TYPES}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        return headers

    def get(self, url):
        for attempt in range(2):
            headers = self.headers()
            try:
                return _request(self.client, url, headers=headers)
            except httpx.HTTPStatusError as error:
                if error.response.status_code != 401 or not self.realm or attempt:
                    raise
                self.refresh(rejected=headers.get("Authorization"))
        raise AssertionError("Unreachable registry retry")

    @contextmanager
    def stream(self, url):
        for attempt in range(2):
            headers = self.headers()
            with self.client.stream("GET", url, headers=headers) as response:
                if getattr(response, "status_code", 200) == 401 and self.realm and not attempt:
                    self.refresh(rejected=headers.get("Authorization"))
                    continue
                response.raise_for_status()
                yield response
                return
        raise AssertionError("Unreachable registry stream retry")


def pull_image(image: str, cache: Path, umoci: Path, *, workers: int = 3,
               registry: str = "https://registry-1.docker.io") -> dict:
    """Download one public linux/amd64 Docker Hub image, preserving every digest."""
    match = re.fullmatch(r"(?:docker.io/)?([a-z0-9_.-]+/[a-z0-9_./-]+)(?::([A-Za-z0-9_.-]+)|@(sha256:[a-f0-9]{64}))?", image)
    if not match:
        raise ValueError("Expected a public Docker Hub namespace/image[:tag|@sha256:digest]")
    repo, tag, pinned = match.groups()
    reference = pinned or tag or "latest"
    cache = cache.resolve()
    blobs = cache / "blobs" / "sha256"
    blobs.mkdir(parents=True, exist_ok=True)
    headers = {"Accept": MANIFEST_TYPES}
    if not registry.startswith("https://"):
        raise ValueError("A registry must use HTTPS")
    with httpx.Client(timeout=httpx.Timeout(600, connect=30), follow_redirects=True) as client:
        base = f"{registry.rstrip('/')}/v2/{repo}"
        if registry == "https://registry-1.docker.io":
            realm, service = "https://auth.docker.io/token", "registry.docker.io"
        else:
            challenge = client.get(f"{base}/manifests/{reference}", headers=headers)
            if challenge.status_code == 401:
                auth = dict(re.findall(r'(\w+)="([^"]+)"', challenge.headers.get("www-authenticate", "")))
                realm, service = auth["realm"], auth.get("service", "")
            else:
                challenge.raise_for_status()
                realm, service = None, None
        session = _RegistrySession(client, repo, realm, service)
        response = session.get(f"{base}/manifests/{reference}")
        origin_bytes = response.content
        origin_digest = digest(origin_bytes)
        if pinned and origin_digest != pinned:
            raise ValueError("Registry response does not match pinned image digest")
        origin = response.json()
        if "manifests" in origin:
            candidates = [x for x in origin["manifests"] if x.get("platform", {}).get("os") == "linux"
                          and x.get("platform", {}).get("architecture") == "amd64"]
            if len(candidates) != 1:
                raise ValueError("Image index must select exactly one linux/amd64 manifest")
            selected = candidates[0]["digest"]
            check_digest(selected)
            response = session.get(f"{base}/manifests/{selected}")
            if digest(response.content) != selected:
                raise ValueError("Image platform manifest digest mismatch")
            manifest = response.json()
        else:
            selected = origin_digest
            manifest = origin
        directory = cache / "images" / check_digest(selected)
        directory.mkdir(parents=True, exist_ok=True)
        metadata_path = directory / "image.json"
        if metadata_path.exists():
            saved = json.loads(metadata_path.read_text())
            if (directory / "bundle/rootfs").is_dir():
                return saved
        (directory / "registry_manifest.json").write_bytes(response.content)
        (directory / "registry_index_or_manifest.json").write_bytes(origin_bytes)
        descriptors = [manifest["config"], *manifest["layers"]]
        total = sum(x["size"] for x in descriptors)
        print(f"Image {image}: {len(manifest['layers'])} layers, {total / 2**30:.2f} GiB compressed; {selected}", flush=True)

        def fetch(descriptor):
            checksum = check_digest(descriptor["digest"])
            target = blobs / checksum
            with (blobs / (checksum + ".lock")).open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if target.exists() and target.stat().st_size == descriptor["size"]:
                    return
                for attempt in range(5):
                    temporary = blobs / (checksum + f".{os.getpid()}.part")
                    try:
                        hasher = hashlib.sha256()
                        with session.stream(f"{base}/blobs/{descriptor['digest']}") as stream:
                            with temporary.open("wb") as output:
                                for chunk in stream.iter_bytes(1024 * 1024):
                                    output.write(chunk)
                                    hasher.update(chunk)
                        if hasher.hexdigest() != checksum or temporary.stat().st_size != descriptor["size"]:
                            raise ValueError("Image blob checksum or size mismatch")
                        temporary.replace(target)
                        print(f"  verified layer {checksum[:12]} ({descriptor['size'] / 2**20:.1f} MiB)", flush=True)
                        return
                    except (httpx.HTTPError, OSError) as error:
                        if isinstance(error, RegistryImageUnavailableError) or (
                                isinstance(error, httpx.HTTPStatusError)
                                and error.response.status_code not in (408, 429)
                                and error.response.status_code < 500):
                            raise
                        if attempt == 4:
                            raise
                        time.sleep(min(2 ** attempt, 15))
                    finally:
                        temporary.unlink(missing_ok=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(fetch, descriptors))
        config = json.loads((blobs / check_digest(manifest["config"]["digest"])).read_text())
        if config.get("architecture") != "amd64" or config.get("os") != "linux":
            raise ValueError("Only Linux amd64 execution is supported")
        converted = json.loads(json.dumps(manifest))
        converted["mediaType"] = "application/vnd.oci.image.manifest.v1+json"
        converted["config"]["mediaType"] = "application/vnd.oci.image.config.v1+json"
        for layer in converted["layers"]:
            if layer["mediaType"] == "application/vnd.docker.image.rootfs.diff.tar.gzip":
                layer["mediaType"] = "application/vnd.oci.image.layer.v1.tar+gzip"
            elif layer["mediaType"] == "application/vnd.docker.image.rootfs.diff.tar":
                layer["mediaType"] = "application/vnd.oci.image.layer.v1.tar"
        payload = json.dumps(converted, separators=(",", ":")).encode()
        oci_digest = digest(payload)
        (blobs / check_digest(oci_digest)).write_bytes(payload)
        layout = directory / "oci"
        layout.mkdir(exist_ok=True)
        (layout / "blobs").mkdir(exist_ok=True)
        link = layout / "blobs/sha256"
        if not link.exists():
            try:
                link.symlink_to(blobs, target_is_directory=True)
            except FileExistsError:
                if link.resolve() != blobs:
                    raise ValueError("OCI blob directory points to a different cache") from None
        atomic_json(layout / "oci-layout", {"imageLayoutVersion": "1.0.0"})
        atomic_json(layout / "index.json", {"schemaVersion": 2, "manifests": [{
            "mediaType": converted["mediaType"], "digest": oci_digest, "size": len(payload),
            "annotations": {"org.opencontainers.image.ref.name": "selected"}}]})
        with (directory / "unpack.lock").open("a") as unpack_lock:
            fcntl.flock(unpack_lock, fcntl.LOCK_EX)
            if metadata_path.exists() and (directory / "bundle/rootfs").is_dir():
                return json.loads(metadata_path.read_text())
            bundle = directory / "bundle"
            if bundle.exists():
                # Preserve an interrupted unpack instead of repeatedly failing on a nonempty target.
                bundle.rename(directory / f"bundle.incomplete.{time.time_ns()}")
            print(f"Unpacking {image} with umoci --rootless", flush=True)
            subprocess.run([str(umoci.resolve()), "unpack", "--rootless", "--image", str(layout) + ":selected",
                            str(bundle)], check=True)
            rootfs = directory / "bundle/rootfs"
            for name in ("workspace", "grader"):
                (rootfs / name).mkdir(exist_ok=True)
            (rootfs / "sca_worker.py").touch(exist_ok=True)
            metadata = {"requested_image": image, "registry": registry,
                        "registry_digest": origin_digest, "platform_digest": selected,
                        "oci_manifest_digest": oci_digest, "compressed_bytes": total,
                        "rootfs": str(directory / "bundle/rootfs"), "runtime_config": config.get("config", {}),
                        "source_manifest": str(directory / "registry_manifest.json"),
                        "layers": [x["digest"] for x in manifest["layers"]],
                        "unpacker": "umoci --rootless", "unpacker_sha256": digest(umoci.read_bytes())}
            atomic_json(metadata_path, metadata)
            return metadata
