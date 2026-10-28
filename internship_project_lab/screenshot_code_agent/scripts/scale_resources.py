#!/usr/bin/env python3
"""Explicit, resumable resource setup. Never starts a GPU model or evaluates tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from screenshot_agent.control import atomic_json

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs/scale_4gpu_resources.json"
DATA = ROOT / "datasets/swebench_full_pinned"
MODEL = ROOT / ".cache/models/Qwen3-Coder-30B-A3B-Instruct"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metadata():
    import pyarrow.parquet as pq

    pins = json.loads(LOCK.read_text())
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = {"resources_lock_sha256": sha256(LOCK), "datasets": {}}
    for name in ("full", "verified"):
        pin = pins[name]
        files = {}
        for split, count in pin["splits"].items():
            target = DATA / f"{name}_{split}.parquet"
            receipt = target.with_suffix(".receipt.json")
            url = (f"https://huggingface.co/datasets/{pin['repo_id']}/resolve/{pin['revision']}/"
                   f"data/{split}-00000-of-00001.parquet")
            if receipt.exists():
                old = json.loads(receipt.read_text())
                if old["url"] != url or not target.exists() or old["sha256"] != sha256(target):
                    raise ValueError(f"Pinned dataset cache changed: {target}")
            else:
                subprocess.run(["curl", "-fL", "--retry", "3", "--retry-all-errors", "--connect-timeout", "20",
                                "--max-time", "300", url, "-o", str(target) + ".part"], check=True)
                downloaded = Path(str(target) + ".part")
                if pq.ParquetFile(downloaded).metadata.num_rows != count:
                    raise ValueError(f"Unexpected number of rows in {url}")
                downloaded.replace(target)
                atomic_json(receipt, {"url": url, "sha256": sha256(target), "rows": count})
            files[split] = {"path": str(target), **json.loads(receipt.read_text())}
        manifest["datasets"][name] = {**pin, "files": files}
    atomic_json(DATA / "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)


def model():
    # Runs in the already available model environment, which supplies huggingface_hub.
    from huggingface_hub import snapshot_download

    pin = json.loads(LOCK.read_text())["coder"]
    MODEL.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=pin["repo_id"], revision=pin["revision"], local_dir=str(MODEL),
                      allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "LICENSE", "README.md"],
                      max_workers=2)
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())
    paths = sorted(set(index["weight_map"].values()) | {
        "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
        "model.safetensors.index.json", "generation_config.json"})
    files = {}
    for name in paths:
        p = MODEL / name
        if not p.is_file() or p.stat().st_size == 0:
            raise ValueError(f"Missing model file: {p}")
        files[name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    atomic_json(MODEL / "download_complete.json", {**pin, "files": files})
    print(json.dumps({"model_ready": str(MODEL), "files": len(files)}), flush=True)


def setup():
    pin = json.loads(LOCK.read_text())
    env = ROOT / ".coder-env"
    if not (env / "bin/python").exists():
        subprocess.run([str(ROOT / ".model-env/bin/python"), "-m", "venv", str(env)], check=True)
    python = str(env / "bin/python")
    subprocess.run([python, "-m", "pip", "install", f"vllm=={pin['vllm_version']}",
                    "transformers==4.55.2", "huggingface-hub==0.36.2"], check=True)
    # Validate the parser registered in THIS pinned vLLM; don't borrow a name from newer docs.
    code = ("from vllm.entrypoints.openai.tool_parsers import ToolParserManager; "
            f"assert '{pin['tool_call_parser']}' in ToolParserManager.tool_parsers; "
            "import importlib.metadata as m; print(m.version('vllm'))")
    subprocess.run([python, "-c", code], check=True)
    frozen = subprocess.check_output([python, "-m", "pip", "freeze"], text=True)
    (env / "requirements.actual.txt").write_text(frozen)
    atomic_json(env / "setup_complete.json", {"vllm_version": pin["vllm_version"],
                "transformers_version": "4.55.2", "huggingface_hub_version": "0.36.2",
                "tool_call_parser": pin["tool_call_parser"], "freeze_sha256": sha256(env / "requirements.actual.txt")})


def registry_probe(selection):
    """Read two image manifests and at most 1KB of a blob; never unpack/download an image."""
    import httpx

    from screenshot_agent.oci import MANIFEST_TYPES

    rows = json.loads(selection.read_text())
    examples = [next(r for r in rows if r["sca_split"] == s) for s in ("dev", "test")]
    reports = []
    try:
        with httpx.Client(timeout=httpx.Timeout(20, connect=10), follow_redirects=True) as client:
            for row in examples:
                repo, tag = row["image"].removeprefix("docker.io/").rsplit(":", 1)
                response = client.get("https://auth.docker.io/token", params={
                    "service": "registry.docker.io", "scope": f"repository:{repo}:pull"})
                response.raise_for_status()
                headers = {"Authorization": "Bearer " + response.json()["token"], "Accept": MANIFEST_TYPES}
                base = f"https://registry-1.docker.io/v2/{repo}"
                response = client.get(f"{base}/manifests/{tag}", headers=headers)
                response.raise_for_status()
                manifest = response.json()
                if "manifests" in manifest:
                    item = next(x for x in manifest["manifests"] if x.get("platform", {}).get("architecture") == "amd64"
                                and x.get("platform", {}).get("os") == "linux")
                    response = client.get(f"{base}/manifests/{item['digest']}", headers=headers)
                    response.raise_for_status()
                    manifest = response.json()
                layer = manifest["layers"][0]
                with client.stream("GET", f"{base}/blobs/{layer['digest']}",
                                   headers={**headers, "Range": "bytes=0-1023"}) as stream:
                    stream.raise_for_status()
                    chunk = next(stream.iter_bytes(1024))
                    if not chunk:
                        raise ValueError("Empty registry blob")
                reports.append({"instance_id": row["instance_id"], "image": row["image"],
                                "compressed_bytes": sum(x["size"] for x in manifest["layers"]),
                                "manifest_sha256": hashlib.sha256(response.content).hexdigest(), "blob_probe": True})
                print(json.dumps(reports[-1]), flush=True)
    except Exception as error:
        atomic_json(selection.parent / "registry_probe.json", {"ready": False, "checked": reports,
                    "error_type": type(error).__name__, "error": str(error)})
        raise
    atomic_json(selection.parent / "registry_probe.json", {"ready": True, "checked": reports,
                "interpretation": "Connectivity probe only, not environment validation."})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["metadata", "model", "setup", "registry-probe"])
    p.add_argument("--selection", type=Path, default=ROOT / "datasets/generated/swebench_full_20261024/selected_records.json")
    a = p.parse_args()
    if a.action == "model" and Path(sys.executable).resolve() != (ROOT / ".model-env/bin/python").resolve():
        raise SystemExit("Use .model-env/bin/python for model downloads")
    {"metadata": metadata, "model": model, "setup": setup,
     "registry-probe": lambda: registry_probe(a.selection)}[a.action]()
