from __future__ import annotations

import torch

from pose_aligned_layered_reconstruction.contracts import LAYERED_SAMPLE_FORMAT


def synthetic_sample(
    *,
    channels: int = 8,
    patch_side: int = 4,
    image_side: int = 64,
    layers: int = 5,
    include_target: bool = True,
) -> dict:
    views = 2
    c2w = torch.eye(4).repeat(views, 1, 1)
    c2w[:, 2, 3] = -2.0
    c2w[1, 0, 3] = 0.1
    intrinsics = torch.tensor(
        [
            [32.0, 0.0, (image_side - 1) / 2.0],
            [0.0, 32.0, (image_side - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ]
    ).repeat(views, 1, 1)
    visible = torch.zeros((views, image_side, image_side), dtype=torch.float32)
    visible[:, image_side // 4 : 3 * image_side // 4, image_side // 4 : 3 * image_side // 4] = 1.0
    patches = patch_side * patch_side
    masked = torch.randn((views, patches, channels), generator=torch.Generator().manual_seed(1))
    raw = torch.randn((views, patches, channels), generator=torch.Generator().manual_seed(2))
    depth_values = torch.linspace(1.0, 3.0, layers)
    depths = depth_values[None, :, None, None].expand(
        views, layers, patch_side, patch_side
    ).clone()
    stopping = torch.full((views, patch_side, patch_side), layers, dtype=torch.long)
    target = torch.zeros_like(depths)
    target[:, min(2, layers - 1)] = 1.0
    sample = {
        "format": LAYERED_SAMPLE_FORMAT,
        "uid": "synthetic-0",
        "object_uid": "synthetic-object",
        "masked_visual_patch_features": masked,
        "raw_visual_patch_features": raw,
        "visible_target_mask": visible,
        "intrinsics": intrinsics,
        "extrinsics": c2w,
        "image_size": (image_side, image_side),
        "grid_transform": "identity",
        "extrinsics_type": "c2w",
        "camera_forward_sign": 1.0,
        "layer_depths_gt": depths,
        "ray_stop_index_gt": stopping,
        "layer_target_gt": target,
    }
    if include_target:
        sample["target"] = torch.randn(
            (8, 16, 16, 16), generator=torch.Generator().manual_seed(3)
        )
    return sample
