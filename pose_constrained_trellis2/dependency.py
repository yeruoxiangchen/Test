"""Lazy access to the sibling upstream TRELLIS.2 checkout."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from types import ModuleType


def trellis2_root() -> Path:
    configured = os.environ.get("TRELLIS2_ROOT", "").strip()
    root = (
        Path(configured).expanduser()
        if configured
        else Path(__file__).resolve().parents[1] / "TRELLIS2"
    )
    root = root.resolve()
    if not (root / "trellis2" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"TRELLIS.2 checkout not found at {root}; set TRELLIS2_ROOT"
        )
    return root


def ensure_trellis2_importable() -> Path:
    root = trellis2_root()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


def import_trellis2() -> ModuleType:
    ensure_trellis2_importable()
    import trellis2

    return trellis2


def import_sparse_module() -> ModuleType:
    ensure_trellis2_importable()
    from trellis2.modules import sparse

    return sparse
