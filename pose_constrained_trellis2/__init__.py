"""Pose-constrained multiview reconstruction on the TRELLIS.2 geometry stack.

The package is intentionally independent from the historical
``pose_point_depth_mv`` and ``pose_aligned_reconstruction`` packages.  Its only
repository-local runtime dependency is the sibling upstream ``TRELLIS2``
checkout.
"""

from .constants import (
    DINO_FEATURE_DIM,
    DINO_IMAGE_SIZE,
    DINO_PATCH_SIDE,
    SHAPE_LATENT_CHANNELS,
    SHAPE_LATENT_RESOLUTION,
    SS_LATENT_CHANNELS,
    SS_LATENT_RESOLUTION,
)

__version__ = "0.1.0"

__all__ = [
    "DINO_FEATURE_DIM",
    "DINO_IMAGE_SIZE",
    "DINO_PATCH_SIDE",
    "SHAPE_LATENT_CHANNELS",
    "SHAPE_LATENT_RESOLUTION",
    "SS_LATENT_CHANNELS",
    "SS_LATENT_RESOLUTION",
]
