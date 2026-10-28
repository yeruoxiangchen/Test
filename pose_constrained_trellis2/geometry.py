"""Camera-aware projection from TRELLIS.2 voxel queries to DINOv3 patches."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .constants import (
    DINO_FEATURE_DIM,
    DINO_FULL_TOKENS,
    DINO_PATCH_SIDE,
    DINO_PATCH_TOKENS,
    DINO_PREFIX_TOKENS,
)


def dense_coords(
    resolution: int,
    *,
    device: torch.device | str | None = None,
    with_batch: bool = True,
) -> torch.Tensor:
    """Return x/y/z voxel coordinates in the same order as a flattened volume."""

    if int(resolution) <= 0:
        raise ValueError("resolution must be positive")
    axis = torch.arange(int(resolution), dtype=torch.int32, device=device)
    xyz = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    xyz = xyz.reshape(-1, 3)
    if not with_batch:
        return xyz
    return torch.cat((torch.zeros_like(xyz[:, :1]), xyz), dim=1)


def canonical_points(
    coords: torch.Tensor,
    *,
    resolution: int,
    grid_transform: str = "identity",
) -> torch.Tensor:
    """Map integer voxel centres to the canonical TRELLIS.2 AABB [-.5,.5]^3."""

    if coords.ndim != 2 or int(coords.shape[1]) not in (3, 4):
        raise ValueError(f"coords must be [N,3] or [N,4], got {tuple(coords.shape)}")
    xyz = coords[:, -3:].float()
    if xyz.numel() and bool(((xyz < 0) | (xyz >= int(resolution))).any().item()):
        raise ValueError(f"coordinates fall outside resolution {resolution}")
    points = (xyz + 0.5) / float(resolution) - 0.5
    if grid_transform == "identity":
        return points
    if grid_transform == "pixal3d_rotation":
        rotation = points.new_tensor(
            ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))
        )
        return points @ rotation.transpose(0, 1)
    raise ValueError(f"unsupported grid_transform={grid_transform!r}")


def split_dinov3_tokens(
    features: torch.Tensor,
    *,
    allow_patch_only: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split DINOv3's five global tokens from its 32x32 spatial tokens.

    ``features`` is ``[views,tokens,1024]``.  New caches retain all 1029
    tokens.  Patch-only tensors remain accepted for explicit compatibility
    tests, but cannot reproduce the stock cross-attention context exactly.
    """

    if features.ndim != 3 or int(features.shape[-1]) != DINO_FEATURE_DIM:
        raise ValueError(
            "DINOv3 features must be [views,tokens,1024], got "
            f"{tuple(features.shape)}"
        )
    tokens = int(features.shape[1])
    if tokens == DINO_FULL_TOKENS:
        return (
            features[:, :DINO_PREFIX_TOKENS].contiguous(),
            features[:, DINO_PREFIX_TOKENS:].contiguous(),
        )
    if allow_patch_only and tokens == DINO_PATCH_TOKENS:
        return features[:, :0].contiguous(), features.contiguous()
    raise ValueError(
        f"expected {DINO_FULL_TOKENS} full DINOv3 tokens"
        + (f" or {DINO_PATCH_TOKENS} patch-only tokens" if allow_patch_only else "")
        + f", got {tokens}"
    )


def projection_geometry(
    *,
    coords: torch.Tensor,
    resolution: int,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
    patch_grid_side: int = DINO_PATCH_SIDE,
    grid_transform: str = "identity",
    extrinsics_type: str = "w2c",
    camera_forward_sign: float = 1.0,
    camera_image_y_sign: float = 1.0,
    object_to_world: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Project canonical query points with pixel-space camera intrinsics.

    ``camera_image_y_sign`` binds the camera's vertical axis to image rows:
    use ``+1`` for OpenCV-style camera coordinates (+Y points down) and ``-1``
    for OpenGL/Blender-style coordinates (+Y points up).  Keeping this
    separate from the forward-axis sign prevents an implicit dataset-specific
    image flip.
    """

    if intrinsics.ndim != 3 or tuple(intrinsics.shape[1:]) != (3, 3):
        raise ValueError(f"intrinsics must be [V,3,3], got {tuple(intrinsics.shape)}")
    views = int(intrinsics.shape[0])
    if tuple(extrinsics.shape) != (views, 4, 4):
        raise ValueError(f"extrinsics must be [{views},4,4]")
    if extrinsics_type not in ("c2w", "w2c"):
        raise ValueError("extrinsics_type must be c2w or w2c")
    if float(camera_forward_sign) not in (-1.0, 1.0):
        raise ValueError("camera_forward_sign must be -1 or +1")
    if float(camera_image_y_sign) not in (-1.0, 1.0):
        raise ValueError("camera_image_y_sign must be -1 or +1")
    if min(int(image_height), int(image_width), int(patch_grid_side)) <= 0:
        raise ValueError("image and patch-grid dimensions must be positive")

    device = intrinsics.device
    points = canonical_points(
        coords.to(device=device),
        resolution=int(resolution),
        grid_transform=str(grid_transform),
    ).float()
    if object_to_world is not None:
        transform = object_to_world.to(device=device, dtype=torch.float32)
        if tuple(transform.shape) != (4, 4) or not bool(torch.isfinite(transform).all()):
            raise ValueError("object_to_world must be a finite [4,4] tensor")
        points = points @ transform[:3, :3].transpose(0, 1) + transform[:3, 3]

    points_h = torch.cat(
        (points, torch.ones((len(points), 1), device=device, dtype=torch.float32)),
        dim=1,
    )
    camera_matrices = (
        torch.linalg.inv(extrinsics.float())
        if extrinsics_type == "c2w"
        else extrinsics.float()
    )
    camera = torch.einsum("vij,nj->vni", camera_matrices, points_h)[..., :3]
    depth = camera[..., 2] * float(camera_forward_sign)
    safe_depth = depth.clamp_min(1.0e-6)
    k = intrinsics.float()
    u = k[:, None, 0, 0] * camera[..., 0] / safe_depth + k[:, None, 0, 2]
    v = (
        float(camera_image_y_sign)
        * k[:, None, 1, 1]
        * camera[..., 1]
        / safe_depth
        + k[:, None, 1, 2]
    )
    valid = (
        (depth > 1.0e-5)
        & (u >= 0.0)
        & (u <= float(image_width - 1))
        & (v >= 0.0)
        & (v <= float(image_height - 1))
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
        "points": points,
        "camera": camera,
        "camera_depth": depth,
        "patch_grid": patch_grid,
        "valid": valid,
    }


def _selected(value: torch.Tensor, indices: torch.Tensor | None) -> torch.Tensor:
    if indices is None:
        return value
    return value.index_select(0, indices.to(device=value.device, dtype=torch.long))


def project_pose_features(
    sample: dict[str, Any],
    coords: torch.Tensor,
    *,
    resolution: int,
    device: torch.device | str,
    view_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Project all selected views' DINOv3 patch tokens onto 3D query points."""

    features = sample["dino_features"].to(device=device, dtype=torch.float32)
    _, patches = split_dinov3_tokens(features)
    intrinsics = sample["intrinsics"].to(device=device, dtype=torch.float32)
    extrinsics = sample["extrinsics"].to(device=device, dtype=torch.float32)
    patches = _selected(patches, view_indices)
    intrinsics = _selected(intrinsics, view_indices)
    extrinsics = _selected(extrinsics, view_indices)
    if int(patches.shape[0]) <= 0:
        raise ValueError("projection selected no views")

    image_size = sample.get("image_size", (512, 512))
    if torch.is_tensor(image_size):
        image_size = image_size.tolist()
    if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
        raise ValueError("image_size must be [height,width]")
    image_height, image_width = map(int, image_size)
    object_to_world = sample.get("object_to_world")
    geometry = projection_geometry(
        coords=coords,
        resolution=int(resolution),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        image_height=image_height,
        image_width=image_width,
        patch_grid_side=DINO_PATCH_SIDE,
        grid_transform=str(sample.get("grid_transform", "identity")),
        extrinsics_type=str(sample.get("extrinsics_type", "w2c")),
        camera_forward_sign=float(sample.get("camera_forward_sign", 1.0)),
        camera_image_y_sign=float(sample.get("camera_image_y_sign", 1.0)),
        object_to_world=object_to_world,
    )
    views = int(patches.shape[0])
    maps = patches.permute(0, 2, 1).reshape(
        views, DINO_FEATURE_DIM, DINO_PATCH_SIDE, DINO_PATCH_SIDE
    )
    projected = F.grid_sample(
        maps,
        geometry["patch_grid"][:, :, None, :].float(),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[..., 0]
    projected = projected.permute(0, 2, 1).contiguous()
    valid = geometry["valid"].bool()
    projected = projected * valid[..., None].to(projected.dtype)
    if tuple(projected.shape) != (views, int(coords.shape[0]), DINO_FEATURE_DIM):
        raise RuntimeError(f"unexpected projected shape={tuple(projected.shape)}")
    if not bool(torch.isfinite(projected).all().item()):
        raise RuntimeError("projection produced non-finite features")
    stats = {
        "view_count": projected.new_tensor(float(views)),
        "visible_fraction": valid.float().mean(),
        "supported_fraction": valid.any(dim=0).float().mean(),
        "query_count": projected.new_tensor(float(coords.shape[0])),
    }
    return projected, valid, stats
