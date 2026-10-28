"""Strict, content-identified loading of repository-level TRELLIS assets."""

from __future__ import annotations

import json
from pathlib import Path

from ._bootstrap import bootstrap_external_paths
from .paired_dataset import sha256_file


DEFAULT_PRETRAINED = "Stable-X/trellis-vggt-v0-2"
DEFAULT_REVISION = "647659a5ad5fbf67e22793e7b5e2cee4b30c5d13"


def load_trellis_model(pretrained: str, model_name: str):
    """Load all stock tensors strictly; pin the known SS30K base revision."""
    bootstrap_external_paths()
    from safetensors.torch import load_file
    from trellis import models

    local = Path(pretrained).expanduser() / "ckpts" / model_name
    if local.with_suffix(".json").is_file() and local.with_suffix(".safetensors").is_file():
        config_path = local.with_suffix(".json")
        weights_path = local.with_suffix(".safetensors")
    else:
        from huggingface_hub import hf_hub_download

        revision = DEFAULT_REVISION if pretrained == DEFAULT_PRETRAINED else None
        config_path = Path(hf_hub_download(pretrained, f"ckpts/{model_name}.json", revision=revision))
        weights_path = Path(hf_hub_download(pretrained, f"ckpts/{model_name}.safetensors", revision=revision))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = getattr(models, config["name"])(**config["args"])
    model.load_state_dict(load_file(str(weights_path)), strict=True)
    binding = {
        "model_name": model_name,
        "config_sha256": sha256_file(config_path),
        "weights_sha256": sha256_file(weights_path),
    }
    return model, binding


__all__ = ["DEFAULT_PRETRAINED", "DEFAULT_REVISION", "load_trellis_model"]
