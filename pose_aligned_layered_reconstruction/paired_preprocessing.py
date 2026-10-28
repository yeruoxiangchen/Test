"""Generate geometrically aligned raw and masked versions of each view."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PairedPreparedViews:
    raw_images: list[Image.Image]
    masked_images: list[Image.Image]
    visible_masks: np.ndarray
    source_to_feature_affines: np.ndarray
    crop_boxes_xyxy: list[tuple[int, int, int, int]]
    source_sizes_wh: list[tuple[int, int]]

    def transform_intrinsics(self, intrinsics: np.ndarray) -> np.ndarray:
        value = np.asarray(intrinsics, dtype=np.float32)
        if value.shape != self.source_to_feature_affines.shape:
            raise ValueError(
                "intrinsics/paired affine shape mismatch: "
                f"{value.shape} != {self.source_to_feature_affines.shape}"
            )
        result = self.source_to_feature_affines @ value
        result[:, 2, :] = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
        return result.astype(np.float32)

    def transform_view_layer_stack(
        self,
        values: np.ndarray,
        *,
        output_resolution: int = 37,
    ) -> np.ndarray:
        """Apply the exact paired crop to a source-aligned [V,L,H,W] stack.

        Nearest-neighbor resampling is intentional: depth peeling layers have
        discontinuities and must not be blended across target/occluder edges.
        """

        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 4 or int(array.shape[0]) != len(self.crop_boxes_xyxy):
            raise ValueError("aligned layer stack must be [V,L,H,W]")
        if int(output_resolution) <= 0:
            raise ValueError("layer output resolution must be positive")
        if not np.isfinite(array).all() or bool((array < 0).any()):
            raise ValueError("layer depths must be finite and non-negative")
        output: list[np.ndarray] = []
        for view_index, box in enumerate(self.crop_boxes_xyxy):
            source_width, source_height = self.source_sizes_wh[view_index]
            if tuple(array[view_index].shape[-2:]) != (
                source_height,
                source_width,
            ):
                raise ValueError(
                    f"view {view_index} layer/source raster mismatch: "
                    f"{tuple(array[view_index].shape[-2:])} != "
                    f"{(source_height, source_width)}"
                )
            layers: list[np.ndarray] = []
            for layer in array[view_index]:
                crop = _crop_with_padding(layer, box)
                resized = np.asarray(
                    Image.fromarray(crop.astype(np.float32), mode="F").resize(
                        (int(output_resolution), int(output_resolution)),
                        Image.Resampling.NEAREST,
                    ),
                    dtype=np.float32,
                )
                layers.append(resized)
            output.append(np.stack(layers))
        return np.stack(output).astype(np.float32)


def make_masked_rgb(
    raw_rgb: np.ndarray,
    visible_mask: np.ndarray,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    raw = np.asarray(raw_rgb)
    mask = np.asarray(visible_mask)
    if raw.ndim != 3 or raw.shape[-1] != 3:
        raise ValueError("raw RGB must be [H,W,3]")
    if mask.ndim == 3 and int(mask.shape[-1]) == 1:
        mask = mask[..., 0]
    if mask.shape != raw.shape[:2]:
        raise ValueError("visible mask does not match raw RGB")
    if raw.dtype != np.uint8:
        if not np.issubdtype(raw.dtype, np.number) or not np.isfinite(raw).all():
            raise ValueError("raw RGB must contain finite numeric values")
        raw = np.clip(raw, 0, 255).astype(np.uint8)
    mask_float = mask.astype(np.float32)
    if mask_float.size and float(mask_float.max()) > 1.0:
        mask_float /= 255.0
    if not np.isfinite(mask_float).all():
        raise ValueError("visible mask contains non-finite values")
    keep = mask_float > float(threshold)
    return (raw * keep[..., None].astype(np.uint8)).astype(np.uint8)


def _square_crop_from_mask(
    mask: np.ndarray,
    *,
    margin: float,
    source_name: str,
) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask > 0.5)
    if not len(xs):
        raise ValueError(f"visible target mask is empty: {source_name}")
    width = int(xs.max()) - int(xs.min()) + 1
    height = int(ys.max()) - int(ys.min()) + 1
    side = max(1, int(math.ceil(max(width, height) * float(margin))))
    center_x = (int(xs.min()) + int(xs.max()) + 1) / 2.0
    center_y = (int(ys.min()) + int(ys.max()) + 1) / 2.0
    left = int(math.floor(center_x - side / 2.0))
    top = int(math.floor(center_y - side / 2.0))
    return left, top, left + side, top + side


def _normalize_box(
    box: Sequence[int], *, margin: float
) -> tuple[int, int, int, int]:
    if len(box) != 4:
        raise ValueError("crop boxes must contain x0,y0,x1,y1")
    x0, y0, x1, y1 = map(int, box)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"invalid crop box={tuple(box)}")
    width = x1 - x0
    height = y1 - y0
    side = max(1, int(math.ceil(max(width, height) * float(margin))))
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    left = int(math.floor(center_x - side / 2.0))
    top = int(math.floor(center_y - side / 2.0))
    return left, top, left + side, top + side


def _crop_with_padding(array: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = box
    side = right - left
    if side <= 0 or bottom - top != side:
        raise ValueError("paired crop must be a positive square")
    channels = () if array.ndim == 2 else (int(array.shape[2]),)
    canvas = np.zeros((side, side, *channels), dtype=array.dtype)
    height, width = map(int, array.shape[:2])
    source_left = max(0, left)
    source_top = max(0, top)
    source_right = min(width, right)
    source_bottom = min(height, bottom)
    if source_right <= source_left or source_bottom <= source_top:
        raise ValueError(f"crop {box} misses source raster {(width, height)}")
    destination_left = source_left - left
    destination_top = source_top - top
    destination_right = destination_left + source_right - source_left
    destination_bottom = destination_top + source_bottom - source_top
    canvas[destination_top:destination_bottom, destination_left:destination_right] = (
        array[source_top:source_bottom, source_left:source_right]
    )
    return canvas


def prepare_paired_arrays(
    raw_images: Sequence[np.ndarray],
    visible_masks: Sequence[np.ndarray],
    *,
    resolution: int = 518,
    crop_margin: float = 1.35,
    alpha_threshold: float = 0.5,
    tracking_boxes_xyxy: Sequence[Sequence[int]] | None = None,
) -> PairedPreparedViews:
    """Apply one crop/resize transform to both synchronized image branches.

    A tracker/detector box is preferred for severe occlusion because the
    visible fragment alone can yield a crop that is too tight.  The box is
    used only to define the crop; it is never supplied as hidden RGB evidence.
    """

    if len(raw_images) != len(visible_masks) or not raw_images:
        raise ValueError("raw image and visible-mask lists must be aligned")
    if int(resolution) <= 0:
        raise ValueError("resolution must be positive")
    if not math.isfinite(float(crop_margin)) or float(crop_margin) < 1.0:
        raise ValueError("crop_margin must be finite and >= 1")
    if not 0.0 < float(alpha_threshold) < 1.0:
        raise ValueError("alpha_threshold must be in (0,1)")
    if tracking_boxes_xyxy is not None and len(tracking_boxes_xyxy) != len(raw_images):
        raise ValueError("tracking boxes must match view count")

    output_raw: list[Image.Image] = []
    output_masked: list[Image.Image] = []
    output_masks: list[np.ndarray] = []
    affines: list[np.ndarray] = []
    boxes: list[tuple[int, int, int, int]] = []
    sizes: list[tuple[int, int]] = []
    for index, (raw_value, mask_value) in enumerate(zip(raw_images, visible_masks)):
        raw = np.asarray(raw_value)
        mask = np.asarray(mask_value)
        if raw.ndim != 3 or raw.shape[-1] != 3:
            raise ValueError(f"view {index} raw RGB must be [H,W,3]")
        if mask.ndim == 3 and int(mask.shape[-1]) == 1:
            mask = mask[..., 0]
        if mask.shape != raw.shape[:2]:
            raise ValueError(f"view {index} raw/mask shape mismatch")
        if raw.dtype != np.uint8:
            raw = np.clip(raw, 0, 255).astype(np.uint8)
        mask_float = mask.astype(np.float32)
        if mask_float.size and float(mask_float.max()) > 1.0:
            mask_float /= 255.0
        if not np.isfinite(mask_float).all():
            raise ValueError(f"view {index} mask contains non-finite values")

        box = (
            _square_crop_from_mask(
                mask_float,
                margin=float(crop_margin),
                source_name=f"view_{index}",
            )
            if tracking_boxes_xyxy is None
            else _normalize_box(
                tracking_boxes_xyxy[index], margin=float(crop_margin)
            )
        )
        raw_crop = _crop_with_padding(raw, box)
        mask_crop = _crop_with_padding(mask_float, box)
        raw_resized = np.asarray(
            Image.fromarray(raw_crop, mode="RGB").resize(
                (int(resolution), int(resolution)), Image.Resampling.BILINEAR
            ),
            dtype=np.uint8,
        )
        mask_resized = np.asarray(
            Image.fromarray(mask_crop.astype(np.float32), mode="F").resize(
                (int(resolution), int(resolution)), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        ).clip(0.0, 1.0)
        masked_resized = make_masked_rgb(
            raw_resized, mask_resized, threshold=float(alpha_threshold)
        )
        side = int(box[2] - box[0])
        scale = float(resolution) / float(side)
        affine = np.asarray(
            (
                (scale, 0.0, -scale * float(box[0])),
                (0.0, scale, -scale * float(box[1])),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float32,
        )
        output_raw.append(Image.fromarray(raw_resized, mode="RGB"))
        output_masked.append(Image.fromarray(masked_resized, mode="RGB"))
        output_masks.append(mask_resized)
        affines.append(affine)
        boxes.append(box)
        sizes.append((int(raw.shape[1]), int(raw.shape[0])))

    return PairedPreparedViews(
        raw_images=output_raw,
        masked_images=output_masked,
        visible_masks=np.stack(output_masks).astype(np.float32),
        source_to_feature_affines=np.stack(affines),
        crop_boxes_xyxy=boxes,
        source_sizes_wh=sizes,
    )


def prepare_paired_paths(
    raw_image_paths: Sequence[str | Path],
    visible_mask_paths: Sequence[str | Path],
    **kwargs,
) -> PairedPreparedViews:
    if len(raw_image_paths) != len(visible_mask_paths) or not raw_image_paths:
        raise ValueError("raw image and mask paths must be aligned")
    raw_images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for raw_path, mask_path in zip(raw_image_paths, visible_mask_paths):
        with Image.open(raw_path) as handle:
            raw_images.append(np.asarray(handle.convert("RGB"), dtype=np.uint8))
        with Image.open(mask_path) as handle:
            masks.append(np.asarray(handle.convert("L"), dtype=np.uint8))
    return prepare_paired_arrays(raw_images, masks, **kwargs)


__all__ = [
    "PairedPreparedViews",
    "make_masked_rgb",
    "prepare_paired_arrays",
    "prepare_paired_paths",
]
