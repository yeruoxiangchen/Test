"""Audit source independence and locally cached official TRELLIS.2 contracts."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from safetensors import safe_open

from .constants import (
    DINO_MODEL,
    DINO_FEATURE_DIM,
    FLOW_BLOCKS,
    FLOW_MODEL_CHANNELS,
    SHAPE_DECODER_MODEL,
    SHAPE_FLOW_MODEL,
    SHAPE_LATENT_CHANNELS,
    SHAPE_MAX_TOKENS,
    SHAPE_LATENT_RESOLUTION,
    SHAPE_LATENT_MEAN,
    SHAPE_LATENT_STD,
    SHAPE_SAMPLER_DEFAULTS,
    SS_DECODER_MODEL,
    SS_FLOW_MODEL,
    SS_LATENT_CHANNELS,
    SS_LATENT_RESOLUTION,
    SS_SAMPLER_DEFAULTS,
    TRELLIS2_REPOSITORY,
)
from .prepare_proobjaverse_cache import (
    DEFAULT_PROTOCOL,
    DEFAULT_SELECTION,
    load_frozen_rows,
)
from .trellis2_toolkit_compat.datasets import ObjaverseXL


FORBIDDEN_IMPORT_ROOTS = {
    "ar_ss_flow",
    "genrecon",
    "pose_aligned_reconstruction",
    "pose_point_depth_mv",
    "reconvggt_ar_adapter_a",
    "trellis",
    "trellis_point_prior_mv",
}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-root", default=str(Path(__file__).resolve().parent)
    )
    return parser


def _identifier_parts(identifier: str) -> tuple[str, str]:
    parts = str(identifier).split("/")
    if len(parts) < 3:
        raise ValueError(f"model identifier has no repository-relative file: {identifier}")
    return "/".join(parts[:2]), "/".join(parts[2:])


def _local_model_file(identifier: str, suffix: str) -> Path:
    repo, filename = _identifier_parts(identifier)
    filename = str(Path(filename).with_suffix(suffix))
    return Path(
        hf_hub_download(repo, filename=filename, local_files_only=True)
    ).resolve()


def audit_imports(package_root: Path) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    files = sorted(package_root.rglob("*.py"))
    for source in files:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                root = name.split(".", 1)[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    violations.append(
                        {
                            "file": str(source),
                            "line": int(node.lineno),
                            "import": name,
                        }
                    )
    if violations:
        raise RuntimeError(f"forbidden runtime imports={violations}")
    return {
        "python_files": len(files),
        "forbidden_roots": sorted(FORBIDDEN_IMPORT_ROOTS),
        "violations": [],
    }


def _audit_flow(
    identifier: str,
    *,
    name: str,
    resolution: int,
    channels: int,
) -> dict[str, Any]:
    config_path = _local_model_file(identifier, ".json")
    weights_path = _local_model_file(identifier, ".safetensors")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    arguments = config.get("args", {})
    expected = {
        "resolution": int(resolution),
        "in_channels": int(channels),
        "out_channels": int(channels),
        "model_channels": FLOW_MODEL_CHANNELS,
        "cond_channels": DINO_FEATURE_DIM,
        "num_blocks": FLOW_BLOCKS,
        "pe_mode": "rope",
        "share_mod": True,
        "dtype": "bfloat16",
    }
    mismatch = {
        key: (arguments.get(key), value)
        for key, value in expected.items()
        if arguments.get(key) != value
    }
    if config.get("name") != name or mismatch:
        raise RuntimeError(
            f"official flow config differs: name={config.get('name')} mismatch={mismatch}"
        )
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
    missing_attention: list[str] = []
    for block in range(FLOW_BLOCKS):
        required = (
            f"blocks.{block}.self_attn.to_qkv.weight",
            f"blocks.{block}.self_attn.to_out.weight",
            f"blocks.{block}.cross_attn.to_q.weight",
            f"blocks.{block}.cross_attn.to_kv.weight",
            f"blocks.{block}.cross_attn.to_out.weight",
        )
        missing_attention.extend(key for key in required if key not in keys)
    if missing_attention:
        raise RuntimeError(f"LoRA attention surface missing={missing_attention[:8]}")
    return {
        "identifier": identifier,
        "config": str(config_path),
        "weights": str(weights_path),
        "model": config["name"],
        "schema": expected,
        "weight_tensor_count": len(keys),
        "lora_attention_blocks_covered": FLOW_BLOCKS,
    }


def audit_official_models() -> dict[str, Any]:
    ss = _audit_flow(
        SS_FLOW_MODEL,
        name="SparseStructureFlowModel",
        resolution=SS_LATENT_RESOLUTION,
        channels=SS_LATENT_CHANNELS,
    )
    shape = _audit_flow(
        SHAPE_FLOW_MODEL,
        name="SLatFlowModel",
        resolution=SHAPE_LATENT_RESOLUTION,
        channels=SHAPE_LATENT_CHANNELS,
    )
    ss_decoder_path = _local_model_file(SS_DECODER_MODEL, ".json")
    ss_decoder = json.loads(ss_decoder_path.read_text(encoding="utf-8"))
    if (
        ss_decoder.get("name") != "SparseStructureDecoder"
        or ss_decoder.get("args", {}).get("latent_channels") != SS_LATENT_CHANNELS
        or ss_decoder.get("args", {}).get("out_channels") != 1
    ):
        raise RuntimeError("official sparse-structure decoder schema differs")
    shape_decoder_path = _local_model_file(SHAPE_DECODER_MODEL, ".json")
    shape_decoder = json.loads(shape_decoder_path.read_text(encoding="utf-8"))
    if (
        shape_decoder.get("name") != "FlexiDualGridVaeDecoder"
        or shape_decoder.get("args", {}).get("latent_channels")
        != SHAPE_LATENT_CHANNELS
    ):
        raise RuntimeError("official Flexible Dual Grid decoder schema differs")
    pipeline_path = Path(
        hf_hub_download(
            TRELLIS2_REPOSITORY,
            filename="pipeline.json",
            local_files_only=True,
        )
    ).resolve()
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))["args"]
    if pipeline["image_cond_model"]["args"]["model_name"] != DINO_MODEL:
        raise RuntimeError("official pipeline DINOv3 identity differs")
    if pipeline["sparse_structure_sampler"]["params"] != SS_SAMPLER_DEFAULTS:
        raise RuntimeError("official SS sampler defaults differ")
    if pipeline["shape_slat_sampler"]["params"] != SHAPE_SAMPLER_DEFAULTS:
        raise RuntimeError("official Shape sampler defaults differ")
    normalization = pipeline["shape_slat_normalization"]
    if (
        tuple(normalization["mean"]) != SHAPE_LATENT_MEAN
        or tuple(normalization["std"]) != SHAPE_LATENT_STD
    ):
        raise RuntimeError("official Shape-512 normalization differs")
    shape_training_config = (
        Path(__file__).resolve().parents[1]
        / "TRELLIS2"
        / "configs"
        / "gen"
        / "slat_flow_img2shape_dit_1_3B_512_bf16.json"
    )
    shape_training = json.loads(shape_training_config.read_text(encoding="utf-8"))
    if shape_training["dataset"]["args"].get("max_tokens") != SHAPE_MAX_TOKENS:
        raise RuntimeError("official Shape-512 active-token limit differs")
    return {
        "ss_flow": ss,
        "shape_flow_512": shape,
        "ss_decoder_config": str(ss_decoder_path),
        "shape_decoder_config": str(shape_decoder_path),
        "pipeline_config": str(pipeline_path),
        "shape_training_config": str(shape_training_config),
        "shape_max_active_tokens": SHAPE_MAX_TOKENS,
        "dino_model": DINO_MODEL,
        "texture_models_checked_or_loaded": False,
    }


def audit_local_frozen_source() -> dict[str, Any]:
    selection = Path(DEFAULT_SELECTION)
    protocol = Path(DEFAULT_PROTOCOL)
    if not selection.is_file() or not protocol.is_file():
        return {"available": False}
    counts: dict[str, int] = {}
    source_root: Path | None = None
    identities: dict[str, str] = {}
    for split in (*("decoder_audit", "predicted_support_bridge", "dev", "train"), "all"):
        rows, source_root, identity = load_frozen_rows(
            selection,
            protocol_path=protocol,
            data_root=None,
            split=split,
            max_samples=0,
        )
        counts[split] = len(rows)
        identities[split] = str(identity["selected_uid_sha256"])
    assert source_root is not None
    native_target_candidates = [
        source_root / "ss_latents" / "ss_enc_conv3d_16l8_fp16_64",
        source_root
        / "shape_latents"
        / "shape_enc_next_dc_f16c32_fp16_512",
    ]
    return {
        "available": True,
        "source_root": str(source_root),
        "split_counts": counts,
        "split_uid_sha256": identities,
        "posed_render_root_present": (source_root / "renders_random_env").is_dir(),
        "old_trellis1_lh_slat_root_present": (source_root / "lh-slats").is_dir(),
        "native_trellis2_target_roots_present": [
            path.is_dir() for path in native_target_candidates
        ],
        "native_targets_ready": all(
            path.is_dir() for path in native_target_candidates
        ),
    }


def audit_trellis2_toolkit_bridge(package_root: Path) -> dict[str, Any]:
    trellis2_toolkit = package_root.parent / "TRELLIS2" / "data_toolkit"
    if not trellis2_toolkit.is_dir():
        raise FileNotFoundError(trellis2_toolkit)
    scripts = ("build_metadata.py", "download.py", "dump_mesh.py", "dual_grid.py")
    missing_scripts = [name for name in scripts if not (trellis2_toolkit / name).is_file()]
    if missing_scripts:
        raise RuntimeError(f"TRELLIS.2 toolkit scripts missing={missing_scripts}")
    foreach_parameters = inspect.signature(ObjaverseXL.foreach_instance).parameters
    download_parameters = inspect.signature(ObjaverseXL.download).parameters
    if "no_file" not in foreach_parameters:
        raise RuntimeError("ObjaverseXL bridge lacks TRELLIS.2 no_file callback")
    if not {"root", "download_root"}.issubset(download_parameters):
        raise RuntimeError("ObjaverseXL bridge lacks TRELLIS.2 download roots")
    bridge = Path(ObjaverseXL.__file__).resolve()
    if package_root not in bridge.parents:
        raise RuntimeError("ObjaverseXL bridge is not self-contained in this package")
    return {
        "official_toolkit": str(trellis2_toolkit),
        "official_objaversexl_adapter_present": (
            trellis2_toolkit / "datasets" / "ObjaverseXL.py"
        ).is_file(),
        "local_compatibility_adapter": str(bridge),
        "foreach_supports_no_file": True,
        "download_supports_toolkit_roots": True,
    }


def run(package_root: str | Path) -> dict[str, Any]:
    root = Path(package_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    return {
        "format": "pose_constrained_trellis2.verification.v1",
        "passed": True,
        "package_root": str(root),
        "independence": audit_imports(root),
        "official_models": audit_official_models(),
        "trellis2_data_toolkit": audit_trellis2_toolkit_bridge(root),
        "local_frozen_30k_source": audit_local_frozen_source(),
        "migration": {
            "base_initialization": "official_trellis2_checkpoints",
            "old_30k_checkpoint_loaded": False,
            "distillation": False,
            "geometry_only": True,
        },
    }


def main() -> None:
    args = make_parser().parse_args()
    print(json.dumps(run(args.package_root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
