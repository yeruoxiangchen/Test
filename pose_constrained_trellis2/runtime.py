"""Load only the TRELLIS.2 geometry components required by this project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .constants import (
    DINO_MODEL,
    SHAPE_DECODER_MODEL,
    SHAPE_FLOW_MODEL,
    SHAPE_LATENT_CHANNELS,
    SHAPE_MAX_TOKENS,
    SHAPE_LATENT_MEAN,
    SHAPE_LATENT_STD,
    SHAPE_OUTPUT_RESOLUTION,
    SHAPE_SAMPLER_DEFAULTS,
    SIGMA_MIN,
    SS_DECODED_RESOLUTION,
    SS_DECODER_MODEL,
    SS_FLOW_MODEL,
    SS_LATENT_CHANNELS,
    SS_LATENT_RESOLUTION,
    SS_SAMPLER_DEFAULTS,
)
from .dependency import ensure_trellis2_importable, import_sparse_module
from .models import (
    DensePoseConditionedFlow,
    SparsePoseConditionedFlow,
    build_pose_conditioned_flow,
)


def load_official_flow(stage: str, *, model_path: str | None = None) -> torch.nn.Module:
    ensure_trellis2_importable()
    from trellis2 import models

    if stage == "ss":
        source = model_path or SS_FLOW_MODEL
    elif stage == "shape":
        source = model_path or SHAPE_FLOW_MODEL
    else:
        raise ValueError(f"unsupported stage={stage!r}")
    return models.from_pretrained(source).eval()


def load_pose_conditioned_flow(
    stage: str,
    *,
    device: torch.device | str,
    model_path: str | None = None,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    gradient_checkpointing: bool = False,
) -> DensePoseConditionedFlow | SparsePoseConditionedFlow:
    base = load_official_flow(stage, model_path=model_path)
    model = build_pose_conditioned_flow(
        base,
        stage=stage,  # type: ignore[arg-type]
        lora_rank=int(lora_rank),
        lora_alpha=int(lora_alpha),
        gradient_checkpointing=bool(gradient_checkpointing),
    )
    return model.to(device)


def make_sampler(stage: str) -> tuple[Any, dict[str, Any]]:
    ensure_trellis2_importable()
    from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler

    if stage == "ss":
        defaults = dict(SS_SAMPLER_DEFAULTS)
    elif stage == "shape":
        defaults = dict(SHAPE_SAMPLER_DEFAULTS)
    else:
        raise ValueError(f"unsupported stage={stage!r}")
    return FlowEulerGuidanceIntervalSampler(SIGMA_MIN), defaults


def load_ss_decoder(*, device: torch.device | str) -> torch.nn.Module:
    ensure_trellis2_importable()
    from trellis2 import models

    decoder = models.from_pretrained(SS_DECODER_MODEL).eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    return decoder.to(device)


def load_shape_decoder(*, device: torch.device | str) -> torch.nn.Module:
    ensure_trellis2_importable()
    from trellis2 import models

    decoder = models.from_pretrained(SHAPE_DECODER_MODEL).eval()
    decoder.set_resolution(SHAPE_OUTPUT_RESOLUTION)
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    return decoder.to(device)


def load_dinov3(*, device: torch.device | str, model_name: str = DINO_MODEL) -> Any:
    ensure_trellis2_importable()
    from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor

    extractor = DinoV3FeatureExtractor(model_name, image_size=512)
    extractor.to(device)
    return extractor


@torch.no_grad()
def ss_latent_to_shape_coords(
    latent: torch.Tensor,
    decoder: torch.nn.Module,
    *,
    shape_resolution: int = 32,
) -> torch.Tensor:
    expected = (1, SS_LATENT_CHANNELS) + (SS_LATENT_RESOLUTION,) * 3
    if tuple(latent.shape) != expected:
        raise ValueError(f"SS latent must be {expected}, got {tuple(latent.shape)}")
    occupancy = decoder(latent) > 0
    if tuple(occupancy.shape[-3:]) != (SS_DECODED_RESOLUTION,) * 3:
        raise RuntimeError(
            f"SS decoder produced unexpected grid={tuple(occupancy.shape)}"
        )
    if int(shape_resolution) != SS_DECODED_RESOLUTION:
        if SS_DECODED_RESOLUTION % int(shape_resolution) != 0:
            raise ValueError("shape resolution must divide the decoded SS resolution")
        ratio = SS_DECODED_RESOLUTION // int(shape_resolution)
        occupancy = torch.nn.functional.max_pool3d(
            occupancy.float(), ratio, ratio, 0
        ) > 0.5
    coords = torch.argwhere(occupancy)[:, [0, 2, 3, 4]].int()
    if int(coords.shape[0]) == 0:
        raise RuntimeError("sampled sparse structure decoded to empty support")
    if int(coords.shape[0]) > SHAPE_MAX_TOKENS:
        raise RuntimeError(
            f"sampled sparse structure has {int(coords.shape[0])} Shape tokens; "
            f"the Shape-512 contract permits at most {SHAPE_MAX_TOKENS}"
        )
    return coords


def make_shape_noise(coords: torch.Tensor, *, device: torch.device | str) -> Any:
    sparse = import_sparse_module()
    coords = coords.to(device=device, dtype=torch.int32)
    return sparse.SparseTensor(
        feats=torch.randn(
            (int(coords.shape[0]), SHAPE_LATENT_CHANNELS), device=device
        ),
        coords=coords,
    )


def normalize_shape_feats(features: torch.Tensor) -> torch.Tensor:
    mean = features.new_tensor(SHAPE_LATENT_MEAN)[None]
    std = features.new_tensor(SHAPE_LATENT_STD)[None]
    return (features - mean) / std


def denormalize_shape_latent(latent: Any) -> Any:
    mean = latent.feats.new_tensor(SHAPE_LATENT_MEAN)[None]
    std = latent.feats.new_tensor(SHAPE_LATENT_STD)[None]
    return latent.replace(latent.feats * std + mean)


@torch.no_grad()
def decode_shape_latent(latent: Any, decoder: torch.nn.Module) -> Any:
    decoder.set_resolution(SHAPE_OUTPUT_RESOLUTION)
    meshes = decoder(denormalize_shape_latent(latent))
    if not isinstance(meshes, (list, tuple)) or not meshes:
        raise RuntimeError("TRELLIS.2 Shape decoder returned no meshes")
    return meshes[0]


def export_geometry_mesh(mesh: Any, output: str | Path) -> None:
    """Export an untextured geometry mesh as PLY or OBJ without AR dependencies."""

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = mesh.vertices.detach().float().cpu().numpy()
    faces = mesh.faces.detach().long().cpu().numpy()
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError(f"decoder produced invalid vertices={vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError(f"decoder produced invalid triangle faces={faces.shape}")
    if not np.isfinite(vertices).all():
        raise RuntimeError("decoder produced non-finite mesh vertices")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise RuntimeError("decoder produced out-of-range mesh face indices")
    suffix = path.suffix.lower()
    if suffix == ".obj":
        with path.open("w", encoding="utf-8") as stream:
            for x, y, z in vertices:
                stream.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
            for a, b, c in faces:
                stream.write(f"f {int(a)+1} {int(b)+1} {int(c)+1}\n")
        return
    if suffix != ".ply":
        raise ValueError("geometry output must end in .ply or .obj")
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(vertices)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write(f"element face {len(faces)}\n")
        stream.write("property list uchar int vertex_indices\nend_header\n")
        np.savetxt(stream, vertices, fmt="%.9g %.9g %.9g")
        for a, b, c in faces:
            stream.write(f"3 {int(a)} {int(b)} {int(c)}\n")


__all__ = [
    "decode_shape_latent",
    "denormalize_shape_latent",
    "export_geometry_mesh",
    "load_dinov3",
    "load_official_flow",
    "load_pose_conditioned_flow",
    "load_shape_decoder",
    "load_ss_decoder",
    "make_sampler",
    "make_shape_noise",
    "ss_latent_to_shape_coords",
]
