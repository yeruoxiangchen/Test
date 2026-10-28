"""Self-contained camera projection for dense SS queries."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def canonical_query_points(
    side: int,
    grid_transform: str,
    *,
    device: torch.device,
) -> torch.Tensor:
    side = int(side)
    if side <= 0:
        raise ValueError("query-grid side must be positive")
    axis = np.arange(side, dtype=np.float32)
    xyz = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    points = (xyz.reshape(-1, 3) + 0.5) / float(side) - 0.5
    if grid_transform == "pixal3d_rotation":
        rotation = np.asarray(
            ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
            dtype=np.float32,
        )
        points = points @ rotation.T
    elif grid_transform != "identity":
        raise ValueError(f"unsupported grid_transform={grid_transform!r}")
    return torch.from_numpy(points.astype(np.float32)).to(device=device)


def project_queries(
    points: torch.Tensor,
    *,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    extrinsics_type: str,
    camera_forward_sign: float,
    image_height: int,
    image_width: int,
    patch_grid_side: int,
) -> dict[str, torch.Tensor]:
    """Project object-frame points and retain both image and DINO grids."""

    if points.ndim != 2 or int(points.shape[1]) != 3:
        raise ValueError("query points must be [N,3]")
    if intrinsics.ndim != 3 or tuple(intrinsics.shape[1:]) != (3, 3):
        raise ValueError("intrinsics must be [V,3,3]")
    if extrinsics.shape != (int(intrinsics.shape[0]), 4, 4):
        raise ValueError("extrinsics must be [V,4,4]")
    if extrinsics_type not in {"c2w", "w2c"}:
        raise ValueError("extrinsics_type must be c2w or w2c")
    if float(camera_forward_sign) not in {-1.0, 1.0}:
        raise ValueError("camera_forward_sign must be -1 or +1")
    if min(int(image_height), int(image_width), int(patch_grid_side)) <= 0:
        raise ValueError("raster and patch sizes must be positive")

    device = intrinsics.device
    points = points.to(device=device, dtype=torch.float32)
    points_h = torch.cat(
        (points, torch.ones((len(points), 1), device=device)), dim=1
    )
    w2c = (
        torch.linalg.inv(extrinsics.float())
        if extrinsics_type == "c2w"
        else extrinsics.float()
    )
    camera = torch.einsum("vij,nj->vni", w2c, points_h)[..., :3]
    depth = camera[..., 2] * float(camera_forward_sign)
    safe_depth = depth.clamp_min(1.0e-6)
    u = (
        intrinsics[:, None, 0, 0].float() * camera[..., 0] / safe_depth
        + intrinsics[:, None, 0, 2].float()
    )
    v = (
        intrinsics[:, None, 1, 1].float() * camera[..., 1] / safe_depth
        + intrinsics[:, None, 1, 2].float()
    )
    valid = (
        (depth > 1.0e-5)
        & (u >= 0.0)
        & (u <= float(image_width - 1))
        & (v >= 0.0)
        & (v <= float(image_height - 1))
    )
    image_grid = torch.stack(
        (
            2.0 * u / max(float(image_width - 1), 1.0) - 1.0,
            2.0 * v / max(float(image_height - 1), 1.0) - 1.0,
        ),
        dim=-1,
    )
    patch_u = (u + 0.5) / (float(image_width) / float(patch_grid_side)) - 0.5
    patch_v = (v + 0.5) / (float(image_height) / float(patch_grid_side)) - 0.5
    patch_grid = torch.stack(
        (
            2.0 * patch_u / max(float(patch_grid_side - 1), 1.0) - 1.0,
            2.0 * patch_v / max(float(patch_grid_side - 1), 1.0) - 1.0,
        ),
        dim=-1,
    )
    return {
        "image_grid": image_grid,
        "patch_grid": patch_grid,
        "camera_depth": depth,
        "valid": valid,
        "camera_points": camera,
    }


def build_ss_projection(
    sample: dict[str, Any],
    *,
    device: torch.device,
    view_indices: torch.Tensor | None = None,
    projection_mode: str = "correct",
    for_layer_branch: bool = False,
) -> dict[str, torch.Tensor]:
    if projection_mode not in {"correct", "pose_cyclic1"}:
        raise ValueError(f"unsupported projection_mode={projection_mode!r}")
    intrinsics = sample["intrinsics"].to(device=device, dtype=torch.float32)
    if for_layer_branch:
        y_sign = float(sample.get("layer_camera_y_sign", 1.0))
        if y_sign not in (-1.0, 1.0):
            raise ValueError("layer_camera_y_sign must be -1 or +1")
        intrinsics = intrinsics.clone()
        intrinsics[:, 1, 1] *= y_sign
    extrinsics = sample["extrinsics"].to(device=device, dtype=torch.float32)
    if view_indices is not None:
        indices = view_indices.to(device=device, dtype=torch.long)
        intrinsics = intrinsics.index_select(0, indices)
        extrinsics = extrinsics.index_select(0, indices)
    if projection_mode == "pose_cyclic1" and len(extrinsics) > 1:
        extrinsics = torch.roll(extrinsics, shifts=1, dims=0)
    image_height, image_width = map(int, sample["image_size"])
    patches = int(sample["masked_visual_patch_features"].shape[1])
    patch_side = int(round(math.sqrt(patches)))
    if patch_side * patch_side != patches:
        raise ValueError("DINO patch count must be square")
    points = canonical_query_points(
        16, str(sample["grid_transform"]), device=device
    )
    return project_queries(
        points,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        extrinsics_type=str(sample["extrinsics_type"]),
        camera_forward_sign=float(sample["camera_forward_sign"]),
        image_height=image_height,
        image_width=image_width,
        patch_grid_side=patch_side,
    )


def resize_sampling_grid(
    image_grid: torch.Tensor,
    *,
    image_size: tuple[int, int],
    map_size: tuple[int, int],
) -> torch.Tensor:
    """Map image-pixel coordinates onto a resized raster's pixel centers.

    Inputs/outputs use grid_sample's align_corners=True convention, while the
    raster resize uses the half-pixel convention.  Reusing image_grid directly
    on a 37x37 prediction would shift samples relative to DINO patch centers.
    """
    if tuple(image_size) == tuple(map_size):
        return image_grid
    height, width = map(int, image_size)
    map_height, map_width = map(int, map_size)
    u = (image_grid[..., 0] + 1) * max(width - 1, 1) / 2
    v = (image_grid[..., 1] + 1) * max(height - 1, 1) / 2
    map_u = (u + 0.5) * map_width / width - 0.5
    map_v = (v + 0.5) * map_height / height - 0.5
    return torch.stack(
        (
            2 * map_u / max(map_width - 1, 1) - 1,
            2 * map_v / max(map_height - 1, 1) - 1,
        ),
        dim=-1,
    )


def sample_maps(values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample [V,C,H,W] maps at a [V,N,2] normalized grid."""

    if values.ndim != 4 or grid.ndim != 3 or int(grid.shape[-1]) != 2:
        raise ValueError("map/grid shapes must be [V,C,H,W] and [V,N,2]")
    if int(values.shape[0]) != int(grid.shape[0]):
        raise ValueError("map/grid view counts differ")
    sampled = F.grid_sample(
        values,
        grid[:, :, None, :],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[..., 0]


def project_masked_dino(
    sample: dict[str, Any],
    geometry: dict[str, torch.Tensor],
    *,
    device: torch.device,
    view_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    features = sample["masked_visual_patch_features"].to(
        device=device, dtype=torch.float32
    )
    if view_indices is not None:
        features = features.index_select(
            0, view_indices.to(device=device, dtype=torch.long)
        )
    views, patches, channels = map(int, features.shape)
    patch_side = int(round(math.sqrt(patches)))
    maps = features.permute(0, 2, 1).reshape(
        views, channels, patch_side, patch_side
    )
    projected = sample_maps(maps, geometry["patch_grid"].float())
    projected = projected.permute(0, 2, 1).contiguous()
    return projected * geometry["valid"][..., None].to(projected.dtype)


__all__ = [
    "build_ss_projection",
    "canonical_query_points",
    "project_masked_dino",
    "project_queries",
    "resize_sampling_grid",
    "sample_maps",
]
