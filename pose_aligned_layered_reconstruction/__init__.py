"""Pose-aligned target-aware layered reconstruction research package."""

from .contracts import (
    LAYERED_SAMPLE_FORMAT,
    LAYERED_SAMPLE_MANIFEST_FORMAT,
    LayeredPrediction,
    validate_paired_sample,
)


PACKAGE_VERSION = "pose_aligned_layered_reconstruction.v1"

__all__ = [
    "LAYERED_SAMPLE_FORMAT",
    "LAYERED_SAMPLE_MANIFEST_FORMAT",
    "LayeredPrediction",
    "PACKAGE_VERSION",
    "validate_paired_sample",
]
