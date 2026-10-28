import base64
import hashlib
import json
from pathlib import Path

import httpx
import pytest

import screenshot_agent.oci as oci


def test_manifest_and_blob_401_refresh_token_and_preserve_digest_validation(tmp_path, monkeypatch):
    config = json.dumps({"architecture": "amd64", "os": "linux", "config": {}}).encode()
    layer = b"offline image layer fixture"

    def descriptor(data):
        return {"digest": "sha256:" + hashlib.sha256(data).hexdigest(), "size": len(data)}

    manifest = {"schemaVersion": 2, "config": descriptor(config),
                "layers": [{**descriptor(layer), "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip"}]}
    tokens, attempts = [], []

    def handler(request):
        attempts.append((request.url.path, request.headers.get("authorization")))
        if request.url.host == "auth.docker.io":
            tokens.append(f"opaque-fixture-{len(tokens) + 1}")
            return httpx.Response(200, json={"token": tokens[-1], "expires_in": 300})
        if "/manifests/" in request.url.path:
            if request.headers["authorization"] == "Bearer opaque-fixture-1":
                return httpx.Response(401)
            return httpx.Response(200, json=manifest)
        if request.url.path.endswith(descriptor(config)["digest"]):
            if request.headers["authorization"] == "Bearer opaque-fixture-2":
                return httpx.Response(401)
            return httpx.Response(200, content=config)
        assert request.url.path.endswith(descriptor(layer)["digest"])
        return httpx.Response(200, content=layer)

    real_client = httpx.Client
    monkeypatch.setattr(oci.httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs))

    def unpack(command, **kwargs):
        (Path(command[-1]) / "rootfs").mkdir(parents=True)

    monkeypatch.setattr(oci.subprocess, "run", unpack)
    unpacker = tmp_path / "umoci"
    unpacker.write_text("offline fixture")
    cache = tmp_path / "cache"
    result = oci.pull_image("fixture/task:latest", cache, unpacker, workers=1)
    assert len(tokens) == 3
    assert len([p for p, _ in attempts if "/manifests/" in p]) == 2
    assert (cache / "blobs/sha256" / hashlib.sha256(layer).hexdigest()).read_bytes() == layer
    assert result["layers"] == [descriptor(layer)["digest"]]


def test_expiring_token_is_refreshed_before_next_layer_without_waiting_for_401(monkeypatch):
    clock, tokens = [0.0], []
    monkeypatch.setattr(oci.time, "monotonic", lambda: clock[0])

    def handler(request):
        tokens.append("token-" + str(len(tokens) + 1))
        return httpx.Response(200, json={"access_token": tokens[-1], "expires_in": 300})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        session = oci._RegistrySession(client, "fixture/task", "https://auth.docker.io/token", "registry.docker.io")
        assert session.headers()["Authorization"] == "Bearer token-1"
        clock[0] = 269
        assert session.headers()["Authorization"] == "Bearer token-1"
        clock[0] = 271
        assert session.headers()["Authorization"] == "Bearer token-2"
        # A second worker holding the first token must not repeat the refresh.
        session.refresh(rejected="Bearer token-1")
        assert len(tokens) == 2


def test_fresh_token_with_empty_access_marks_image_unavailable_not_transient_auth_failure():
    payload = base64.urlsafe_b64encode(json.dumps({"access": []}).encode()).decode().rstrip("=")
    token = "fixture." + payload + ".fixture"
    calls = []

    def handler(request):
        calls.append(request.url.host)
        return httpx.Response(200, json={"token": token, "expires_in": 300})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        session = oci._RegistrySession(client, "fixture/missing", "https://auth.docker.io/token", "registry.docker.io")
        with pytest.raises(oci.RegistryImageUnavailableError, match="image may be absent or private") as error:
            session.get("https://registry-1.docker.io/v2/fixture/missing/manifests/latest")
    assert calls == ["auth.docker.io"]
    assert token not in str(error.value)


def test_401_after_refresh_is_bounded_and_not_retried_with_same_token(monkeypatch):
    calls = []
    monkeypatch.setattr(oci.time, "sleep", lambda delay: pytest.fail("Permanent 401 must not sleep/retry same token"))

    def handler(request):
        calls.append(request.url.host)
        if request.url.host == "auth.docker.io":
            return httpx.Response(200, json={"token": "opaque-fixture-" + str(len(calls)), "expires_in": 300})
        return httpx.Response(401)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        session = oci._RegistrySession(client, "fixture/task", "https://auth.docker.io/token", "registry.docker.io")
        with pytest.raises(httpx.HTTPStatusError):
            session.get("https://registry-1.docker.io/v2/fixture/task/manifests/latest")
    assert calls == ["auth.docker.io", "registry-1.docker.io"] * 2


def test_tls_eof_on_token_request_is_retried(monkeypatch):
    calls, sleeps = [], []
    monkeypatch.setattr(oci.time, "sleep", sleeps.append)

    def handler(request):
        calls.append(request.url.host)
        if len(calls) == 1:
            raise OSError("SSL: UNEXPECTED_EOF_WHILE_READING")
        return httpx.Response(200, json={"token": "opaque-fixture", "expires_in": 300})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        session = oci._RegistrySession(client, "fixture/task", "https://auth.docker.io/token", "registry.docker.io")
        assert session.headers()["Authorization"] == "Bearer opaque-fixture"
    assert len(calls) == 2 and sleeps == [1]
