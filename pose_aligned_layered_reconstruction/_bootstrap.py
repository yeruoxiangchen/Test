"""Resolve repository-local third-party runtime dependencies.

The research package does not import the maintained
``pose_aligned_reconstruction`` package.  TRELLIS itself remains an external
repository-level dependency and is discovered relative to Tracker so command
line entry points also work outside the repository root.
"""

from __future__ import annotations

from pathlib import Path
import sys


TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"


def bootstrap_external_paths() -> None:
    for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


__all__ = [
    "RECONVIAGEN_ROOT",
    "TRACKER_ROOT",
    "VGGT_ROOT",
    "bootstrap_external_paths",
]
