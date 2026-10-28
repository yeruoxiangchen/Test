"""Copy only this experiment's bound dependencies to /data, preserving home.

The frozen logical paths and hashes are unchanged. An immutable mapping records
their physical locations. Large home files are read once, while copying and
hashing; the destination is independently hashed before publication. No source
tree, raw scene asset graph or VGGT predictor checkpoint is traversed.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True

SCHEMA = "visible_surface_oracle_storage_v1"
REVISION = "647659a5ad5fbf67e22793e7b5e2cee4b30c5d13"
HF_RELATIVE = Path("models--Stable-X--trellis-vggt-v0-2/snapshots") / REVISION
HOME_HUB = Path("/home/zjr/.cache/huggingface/hub")
CHUNK = 8 * 1024 * 1024
SMALL_JSON_LIMIT = 16 * 1024 * 1024


def _common():
    if __package__:
        from . import common
    else:
        import common
    return common


def _absolute(path):
    # Do not resolve symlinks: existing frozen paths are logical identifiers.
    value = Path(os.path.abspath(os.fspath(path)))
    if not value.is_absolute():
        raise ValueError("Storage inventory requires absolute logical paths")
    return str(value)


def _sha_stream(handle, output=None):
    digest = hashlib.sha256()
    total = 0
    while True:
        block = handle.read(CHUNK)
        if not block:
            break
        digest.update(block)
        total += len(block)
        if output is not None:
            output.write(block)
    return digest.hexdigest(), total


def _hash_file(path):
    with Path(path).open("rb") as handle:
        return _sha_stream(handle)


def _stat(path):
    s = Path(path).stat()
    return {"size": s.st_size, "mtime_ns": s.st_mtime_ns, "device": s.st_dev, "inode": s.st_ino}


class Inventory:
    """Explicit dependency inventory, with in-memory small metadata reuse."""
    def __init__(self, common):
        self.common = common
        self.items = {}
        self.blobs = {}
        self.json_cache = {}
        self.old_manifest = None
        manifest_path = Path(common.OUT) / "storage.json"
        if manifest_path.exists():
            # Direct JSON avoids common.physical() -> storage.json recursion.
            self.old_manifest = json.loads(manifest_path.read_text())

    def add(self, binding, reason):
        if not isinstance(binding, dict) or not {"path", "sha256"}.issubset(binding):
            raise ValueError("Expected a frozen path/sha256 binding: " + reason)
        path, sha = _absolute(binding["path"]), binding["sha256"]
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("Invalid SHA256: " + path)
        if "vggt-object-v0-1" in path or "/datasets/scene_occlusion_pilot_20261020/" in path:
            raise ValueError("Unnecessary predictor/raw scene asset entered inventory: " + path)
        existing = self.items.get(path)
        if existing is not None:
            if existing["sha256"] != sha:
                raise ValueError("Conflicting frozen hashes for " + path)
            existing["reasons"].add(reason)
        else:
            self.items[path] = {"path": path, "sha256": sha, "reasons": {reason}}
        return {"path": path, "sha256": sha}

    def source(self, logical):
        """Prefer an already verified /data copy when reopening small JSON."""
        logical = _absolute(logical)
        if self.old_manifest:
            row = self.old_manifest.get("mapping", {}).get(logical)
            if row and Path(row["physical"]).is_file():
                return Path(row["physical"])
        return Path(logical)

    def json(self, path_or_binding, reason):
        if isinstance(path_or_binding, dict):
            bound = self.add(path_or_binding, reason)
            logical, expected = bound["path"], bound["sha256"]
        else:
            logical, expected = _absolute(path_or_binding), None
        if logical not in self.json_cache:
            source = self.source(logical)
            if source.stat().st_size > SMALL_JSON_LIMIT:
                raise ValueError("Only small metadata may be read during inventory construction: " + logical)
            blob = source.read_bytes()
            sha = hashlib.sha256(blob).hexdigest()
            self.add({"path": logical, "sha256": sha}, reason)
            self.blobs[logical] = blob
            self.json_cache[logical] = json.loads(blob)
        if expected is not None and self.items[logical]["sha256"] != expected:
            raise ValueError("Frozen metadata changed: " + logical)
        self.items[logical]["reasons"].add(reason)
        return self.json_cache[logical]

    def destination(self, logical):
        path = Path(logical)
        data = Path(self.common.DATA_ROOT)
        if path.is_relative_to("/data/zjr"):
            return path  # Existing official model archive already lives on data.
        if path.is_relative_to(HOME_HUB):
            return data / "cache/huggingface/hub" / path.relative_to(HOME_HUB)
        return data / "inputs/legacy" / str(path).lstrip("/")

    def contract(self):
        return [{"path": path, "sha256": row["sha256"], "physical": str(self.destination(path))}
                for path, row in sorted(self.items.items())]


def _add_inputs(inv, inputs):
    sample = inv.add(inputs["sample"], "bound image/DINO input sample")
    directory = Path(sample["path"]).parent
    receipt = inv.json(directory / "receipt.json", "input cache metadata")
    if receipt["sample"] != inputs["sample"]:
        raise ValueError("Input sample receipt disagrees with frozen SS input")
    images = inv.json(receipt["images"], "input image/camera metadata")
    inv.add(receipt["encoder"], "encoder identity metadata only; no encoder weights")
    for view in images["views"]:
        # All four cached files remain provenance; model selection is unchanged
        # (the stool still only reads views 0/11 through selected_indices).
        for row in view["files"].values():
            inv.add(row, "cached input crop visualization")
        for row in view["sources"].values():
            inv.add(row, "shared full-frame RGB/visible mask")


def _add_ss(inv, bound, reason):
    row = inv.json(bound, reason)
    if not row["success"]:
        raise ValueError("Unsuccessful archived SS record")
    inv.add(row["coords"], reason + " coordinates")
    _add_inputs(inv, row["inputs"])
    return row


def _add_mesh(inv, bound, reason):
    row = inv.json(bound, reason)
    if not row["success"]:
        raise ValueError("Unsuccessful archived mesh record")
    for key in ("mesh", "world_mesh", "gt"):
        inv.add(row[key], reason + " " + key)
    if row.get("sampled_slat_receipt"):
        receipt = inv.json(row["sampled_slat_receipt"], reason + " sampled latent receipt")
        inv.add(receipt["latent"], reason + " sampled latent payload")
    return row


def inventory(common=None):
    """Read metadata and construct a bounded list; never hash large files here."""
    common = common or _common()
    inv = Inventory(common)
    old = Path(common.OLD)
    plan_wrapper = inv.json(old / "plan.json", "previous frozen plan")
    p = plan_wrapper["plan"]
    if common.digest(p) != plan_wrapper["identity"]:
        raise ValueError("Previous plan identity changed")
    summary = inv.json(old / "summary.json", "previous completed comparison")
    visual = inv.json(old / "visual_audit.json", "previous visual audit")
    smoke = inv.json(old / "smoke.json", "previous short checks")
    inv.json(old / "evaluation_identity.json", "previous runtime identity")
    for key in ("previous_evaluation_identity", "previous_summary", "base_summary", "base_plan",
                "shared_plan", "runtime_amendment"):
        inv.add(p[key], "frozen provenance metadata: " + key)
    inv.add(p["parent_checkpoint"], "frozen complete appearance parent")
    parent_run = inv.json(p["parent_training_record"], "frozen appearance parent training contract")
    inv.add(parent_run["contract"]["appearance_base"], "parent appearance adapter checkpoint")
    inv.add(parent_run["contract"]["manifest"], "parent training manifest metadata only")
    for obj in p["objects"]:
        for key in ("ss", "slat", "mesh"):
            inv.add(obj[key], "four-object fixed target " + key)
    for row in p["evaluation_gt"]:
        inv.add(row, "fixed four-frame GT for oracle extraction/evaluation")
    for rows in p["evidence_policy"]["full_visible_masks"].values():
        for row in rows:
            inv.add(row, "fixed 2072 visible instance mask")

    references = [row for row in summary["records"] if row["arm"] in ("appearance_control", "geometry_observed")]
    if len(references) != 24:
        raise ValueError("Require exactly 24 completed matched SharedSS references")
    ss_example = None
    for row in references:
        ss = _add_ss(inv, row["ss_record"], "matched SharedSS reference")
        _add_mesh(inv, row["mesh_record"], "matched SharedSS reference")
        _add_ss(inv, row["condition_record"], "original frozen image condition")
        ss_example = ss_example or ss
    for arm in ("appearance_control", "geometry_observed"):
        source = summary["training_fairness"]["sources"][arm]
        for key in ("run", "history"):
            inv.add(source[key], "matched training fairness " + arm + " " + key)
        for key in ("run", "completion", "checkpoint"):
            inv.add(source["training"][key], "matched training " + arm + " " + key)

    for key in ("initial_check", "training_check", "replay"):
        inv.json(smoke[key], "previous smoke " + key)
    # The smoke metadata binds histories/checkpoints needed for exact matching.
    # It does not recursively traverse the old source_files/provenance graph.
    for row in smoke["artifacts"]:
        inv.add(row, "previous smoke bound artifact")
    replay = inv.json(smoke["replay"], "frozen parent replay")
    for key in ("reference_ss", "ss_record"):
        _add_ss(inv, replay[key], "frozen parent replay " + key)
    for key in ("reference_mesh", "mesh_record"):
        _add_mesh(inv, replay[key], "frozen parent replay " + key)

    render_rows = [row for row in visual["render_records"]
                   if row["arm"] in ("mesh_gt", "appearance_control", "geometry_observed")]
    if len(render_rows) != 28:
        raise ValueError("Require four GT + 24 matched reference render records")
    for row in render_rows:
        render = inv.json(row["record"], "reusable fixed-camera render record")
        inv.add(render["mesh"], "rendered canonical mesh")
        for view in render["views"]:
            inv.add({"path": view["png"], "sha256": view["png_sha256"]}, "reusable rendered view")

    # Four stock TRELLIS models, each its config + safetensors, at the pinned
    # revision. No 5 GB VGGT predictor and no blanket Hugging Face cache copy.
    parent_bindings = ss_example["model_binding"]["parent_bindings"]
    inv.add(parent_bindings["parent"], "official frozen SS parent already on data")
    slat = p["slat_binding"]
    inv.add({"path": slat["checkpoint"], "sha256": slat["sha256"]}, "official frozen SLat parent already on data")
    models = [parent_bindings["stock"], ss_example["decoder_binding"], slat["flow"], slat["decoder"]]
    if len({row["model_name"] for row in models}) != 4:
        raise ValueError("Expected exactly four distinct stock TRELLIS models")
    for model in models:
        for suffix, key in (("json", "config_sha256"), ("safetensors", "weights_sha256")):
            inv.add({"path": str(HOME_HUB / HF_RELATIVE / "ckpts" / (model["model_name"] + "." + suffix)),
                     "sha256": model[key]}, "pinned stock TRELLIS model " + model["model_name"])
    inv.context = {"previous_plan_identity": plan_wrapper["identity"], "previous_runtime_identity": summary["identity"],
                   "reference_cases": 24, "reference_render_records": 28, "stock_models": 4,
                   "excluded": ["raw scene assets", "source_files graph", "VGGT predictor weights", "unrelated active jobs"]}
    return inv


def _cache_directories(common):
    data = Path(common.DATA_ROOT)
    mapping = {"TMPDIR": data / "tmp", "TORCH_HOME": data / "cache/torch",
               "HF_HOME": data / "cache/huggingface", "HF_HUB_CACHE": data / "cache/huggingface/hub",
               "TORCH_EXTENSIONS_DIR": data / "cache/torch_extensions",
               "MPLCONFIGDIR": data / "cache/matplotlib", "XDG_CACHE_HOME": data / "cache/xdg",
               "NUMBA_CACHE_DIR": data / "cache/numba"}
    for name, path in mapping.items():
        os.environ[name] = str(path)
        path.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    return {key: str(value) for key, value in mapping.items()}


def _low_priority():
    result = {}
    try:
        result["nice"] = os.nice(10)
    except OSError as exc:
        result["nice_error"] = str(exc)
    try:
        completed = subprocess.run(["ionice", "-c", "2", "-n", "7", "-p", str(os.getpid())],
                                   check=False, capture_output=True, text=True)
        result["ionice_exit"] = completed.returncode
        if completed.returncode:
            result["ionice_error"] = completed.stderr.strip()
    except OSError as exc:
        result["ionice_error"] = str(exc)
    return result


def _copy_one(inv, row):
    logical, expected = row["path"], row["sha256"]
    destination = inv.destination(logical)
    if destination.exists():
        sha, size = _hash_file(destination)
        if sha != expected:
            raise ValueError("Existing data file fails frozen checksum; preserved without overwrite: " + str(destination))
        return {"physical": str(destination), "sha256": sha, "size": size,
                "stat": _stat(destination), "verification": "existing_data_sha256", "home_removed": False}
    if not str(destination).startswith(str(inv.common.DATA_ROOT) + "/"):
        raise ValueError("A missing external data archive cannot be created by this migration: " + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".copy-{os.getpid()}.tmp")
    source = inv.source(logical)
    before = _stat(source)
    try:
        with temporary.open("xb") as output:
            if logical in inv.blobs:
                handle = io.BytesIO(inv.blobs[logical])
            else:
                handle = source.open("rb")
            with handle:
                sha, size = _sha_stream(handle, output)
            output.flush()
            os.fsync(output.fileno())
        if sha != expected or size != before["size"] or _stat(source) != before:
            raise ValueError("Source hash/size/stat changed during migration: " + logical)
        copied_sha, copied_size = _hash_file(temporary)
        if (copied_sha, copied_size) != (sha, size):
            raise ValueError("Destination verification failed: " + str(destination))
        temporary.replace(destination)
        return {"physical": str(destination), "sha256": sha, "size": size,
                "stat": _stat(destination), "verification": "source_stream_sha256_and_destination_sha256",
                "home_removed": False}
    except BaseException:
        if temporary.exists():
            temporary.unlink()  # Only this process's unpublished temporary file.
        raise


def run():
    common = _common()
    priority = _low_priority()
    caches = _cache_directories(common)
    inv = inventory(common)
    contract = inv.contract()
    identity = common.digest(contract)
    target = Path(common.OUT) / "storage.json"
    if inv.old_manifest:
        old = inv.old_manifest
        if (old.get("schema") != SCHEMA or old.get("inventory_identity") != identity
                or old.get("success") is not True or old.get("complete") is not True):
            raise ValueError("Immutable storage manifest does not match this inventory")
        if set(old["mapping"]) != set(inv.items):
            raise ValueError("Immutable storage inventory set changed")
        for row in contract:
            bound = old["mapping"][row["path"]]
            if bound["physical"] != row["physical"] or bound["sha256"] != row["sha256"]:
                raise ValueError("Immutable storage binding changed")
            if _stat(bound["physical"]) != bound["stat"]:
                if _hash_file(bound["physical"]) != (bound["sha256"], bound["size"]):
                    raise ValueError("Previously migrated data changed: " + bound["physical"])
        print(json.dumps({"time": common.now(), "stage": "storage_verified", "files": len(contract),
                          "inventory_identity": identity, "home_removed": False}), flush=True)
        return old
    mapping = {}
    total_bytes = 0
    for i, row in enumerate(contract, 1):
        result = _copy_one(inv, row)
        mapping[row["path"]] = result
        total_bytes += result["size"]
        progress = {"time": common.now(), "stage": "storage_copy", "completed": i, "total": len(contract),
                    "logical": row["path"], "physical": result["physical"], "completed_bytes": total_bytes}
        common.write(Path(common.OUT) / "storage_progress.json", progress)
        print(json.dumps(progress), flush=True)
    value = {"schema": SCHEMA, "success": True, "complete": True, "time": common.now(), "inventory_identity": identity,
             "mapping": mapping, "file_count": len(mapping), "total_bytes": total_bytes,
             "context": inv.context, "cache_environment": caches, "priority": priority,
             "home_removed": False, "frozen_logical_bindings_unchanged": True,
             "scope": "bounded dependencies for visible-surface oracle; copy and verify only, no source deletion"}
    if target.exists():
        raise ValueError("Storage manifest appeared concurrently; refusing overwrite")
    common.write(target, value)
    print(json.dumps({"time": common.now(), "stage": "storage_complete", "files": len(mapping),
                      "bytes": total_bytes, "inventory_identity": identity}), flush=True)
    return value


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("storage.py takes no arguments; copy the explicit frozen inventory")
    run()
