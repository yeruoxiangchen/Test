"""Construct target-aware layered labels from separate depth-peeling passes."""

from __future__ import annotations

import torch


def merge_target_and_occluder_layers(
    target_depths: torch.Tensor,
    non_target_depths: torch.Tensor,
    *,
    output_layers: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Depth-sort two intersection stacks and retain object ownership.

    Inputs use [V,L,H,W] positive camera-forward depth and zero for invalid
    entries.  The function is renderer-agnostic, so official LaRI depth-peeling
    output or another rasterizer can feed the same paired-cache builder.
    """

    if target_depths.ndim != 4 or non_target_depths.ndim != 4:
        raise ValueError("depth-peeling stacks must be [V,L,H,W]")
    if (
        int(target_depths.shape[0]) != int(non_target_depths.shape[0])
        or target_depths.shape[-2:] != non_target_depths.shape[-2:]
    ):
        raise ValueError("target/non-target depth stacks are not raster-aligned")
    if int(output_layers) <= 0:
        raise ValueError("output_layers must be positive")
    for name, value in (
        ("target", target_depths),
        ("non-target", non_target_depths),
    ):
        if not bool(torch.isfinite(value.float()).all().item()):
            raise ValueError(f"{name} depth stack contains non-finite values")
        if bool((value < 0).any().item()):
            raise ValueError(f"{name} depth stack contains negative depths")

    target = target_depths.float()
    other = non_target_depths.float()
    combined = torch.cat((target, other), dim=1)
    ownership = torch.cat(
        (torch.ones_like(target, dtype=torch.bool), torch.zeros_like(other, dtype=torch.bool)),
        dim=1,
    )
    # The cache contract has a fixed layer dimension.  A renderer may expose
    # fewer intersections than requested, so append invalid candidates before
    # sorting instead of silently returning a shorter tensor.
    missing = int(output_layers) - int(combined.shape[1])
    if missing > 0:
        padding_shape = (
            int(combined.shape[0]),
            missing,
            int(combined.shape[2]),
            int(combined.shape[3]),
        )
        combined = torch.cat((combined, combined.new_zeros(padding_shape)), dim=1)
        ownership = torch.cat(
            (
                ownership,
                torch.zeros(padding_shape, dtype=torch.bool, device=ownership.device),
            ),
            dim=1,
        )
    valid = combined > 0
    infinity = torch.full_like(combined, torch.inf)
    sortable = torch.where(valid, combined, infinity)
    sorted_depths, order = torch.sort(sortable, dim=1)
    sorted_ownership = torch.gather(ownership, 1, order)
    retained_depths = sorted_depths[:, : int(output_layers)]
    retained_target = sorted_ownership[:, : int(output_layers)]
    retained_valid = torch.isfinite(retained_depths)
    retained_depths = torch.where(
        retained_valid, retained_depths, torch.zeros_like(retained_depths)
    )
    retained_target = retained_target & retained_valid
    stopping_index = retained_valid.sum(dim=1).long()
    return retained_depths, stopping_index, retained_target.float()


__all__ = ["merge_target_and_occluder_layers"]
