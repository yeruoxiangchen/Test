"""Encode synchronized raw and masked-target views with the same frozen DINO."""

from __future__ import annotations

from typing import Any

from PIL import Image
import torch

from ._bootstrap import bootstrap_external_paths
from .dino_context import build_masked_dino_contexts


def build_frozen_dino_encoder(
    *, dino_model: str = "dinov2_vitl14_reg", device: torch.device
) -> Any:
    bootstrap_external_paths()
    from trellis.pipelines import TrellisImageTo3DPipeline

    pipeline = TrellisImageTo3DPipeline()
    pipeline.models = {}
    pipeline._device = device
    pipeline.low_vram = False
    pipeline._init_image_cond_model(str(dino_model))
    encoder = pipeline.models.get("image_cond_model")
    if encoder is None:
        raise RuntimeError("TRELLIS did not construct image_cond_model")
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return pipeline


@torch.no_grad()
def _encode_patch_tokens(pipeline: Any, images: list[Image.Image]) -> torch.Tensor:
    encoded = pipeline.encode_image(images)
    if encoded.ndim == 3:
        encoded = encoded[None]
    if (
        encoded.ndim != 4
        or int(encoded.shape[0]) != 1
        or int(encoded.shape[1]) != len(images)
        or int(encoded.shape[-1]) != 1024
    ):
        raise RuntimeError(
            "unexpected TRELLIS DINO layout: "
            f"{tuple(encoded.shape)}, views={len(images)}"
        )
    # TRELLIS DINOv2 has one CLS and four register tokens before 37x37 patches.
    patches = encoded[0, :, 5:]
    if patches.ndim != 3 or int(patches.shape[1]) != 37 * 37:
        raise RuntimeError(f"unexpected DINO patch layout={tuple(patches.shape)}")
    return patches.float().contiguous()


@torch.no_grad()
def encode_paired_views(
    pipeline: Any,
    *,
    raw_images: list[Image.Image],
    masked_images: list[Image.Image],
    ss_context_tokens: int = 4096,
) -> dict[str, Any]:
    if len(raw_images) != len(masked_images) or not raw_images:
        raise ValueError("raw/masked image lists must be non-empty and aligned")
    raw = _encode_patch_tokens(pipeline, raw_images)
    masked = _encode_patch_tokens(pipeline, masked_images)
    if raw.shape != masked.shape:
        raise RuntimeError("raw/masked DINO encodings are not aligned")
    contexts = build_masked_dino_contexts(
        masked, ss_context_tokens=int(ss_context_tokens)
    )
    return {
        **contexts,
        "raw_visual_patch_features": raw,
        "paired_encoding_contract": {
            "same_frozen_dino_encoder": True,
            "same_crop_and_raster": True,
            "raw_branch_role": "target_conditioned_layered_geometry_only",
            "masked_branch_role": "trellis_appearance_context_only",
        },
    }


__all__ = ["build_frozen_dino_encoder", "encode_paired_views"]
