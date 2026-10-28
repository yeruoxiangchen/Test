import hashlib
import json
from pathlib import Path

import pytest

import screenshot_agent.oci as oci


def fake_registry(monkeypatch, tmp_path):
    config = json.dumps({"architecture": "amd64", "os": "linux", "config": {}}).encode()
    layer = b"fake compressed layer for an offline downloader test"
    def descriptor(data):
        return {"digest": "sha256:" + hashlib.sha256(data).hexdigest(), "size": len(data)}
    manifest = {"schemaVersion": 2, "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
                "config": {**descriptor(config), "mediaType": "application/vnd.docker.container.image.v1+json"},
                "layers": [{**descriptor(layer), "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip"}]}
    payload = json.dumps(manifest).encode()
    blobs = {descriptor(config)["digest"]: config, descriptor(layer)["digest"]: layer}

    class Response:
        def __init__(self, data):
            self.content = data

        def json(self):
            return json.loads(self.content)

        def raise_for_status(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def iter_bytes(self, size):
            yield self.content

    class Client(Response):
        def __init__(self, *args, **kwargs):
            pass

        def get(self, url, **kwargs):
            return Response(b'{"token":"fixture"}' if "auth.docker.io" in url else payload)

        def stream(self, method, url, **kwargs):
            return Response(blobs[url.rsplit("/", 1)[1]])

    unpacked = []
    def unpack(command, **kwargs):
        bundle = Path(command[-1])
        assert not bundle.exists()
        (bundle / "rootfs").mkdir(parents=True)
        unpacked.append(str(bundle))
    monkeypatch.setattr(oci.httpx, "Client", Client)
    monkeypatch.setattr(oci.subprocess, "run", unpack)
    umoci = tmp_path / "umoci"
    umoci.write_text("test-only unpacker")
    return payload, umoci, unpacked


def test_incomplete_unpack_is_archived_and_completed_cache_reused(tmp_path, monkeypatch):
    payload, umoci, unpacked = fake_registry(monkeypatch, tmp_path)
    cache = tmp_path / "cache"
    directory = cache / "images" / hashlib.sha256(payload).hexdigest()
    (directory / "bundle").mkdir(parents=True)
    (directory / "bundle/partial").write_text("old interrupted unpack")
    result = oci.pull_image("fixture/task:latest", cache, umoci)
    assert len(unpacked) == 1
    archived = list(directory.glob("bundle.incomplete.*"))
    assert len(archived) == 1 and (archived[0] / "partial").read_text() == "old interrupted unpack"
    assert Path(result["rootfs"]).is_dir()
    assert oci.pull_image("fixture/task:latest", cache, umoci) == result
    assert len(unpacked) == 1


def test_registry_manifest_must_match_requested_digest(tmp_path, monkeypatch):
    _, umoci, unpacked = fake_registry(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="pinned image digest"):
        oci.pull_image("fixture/task@sha256:" + "0" * 64, tmp_path / "cache", umoci)
    assert not unpacked
