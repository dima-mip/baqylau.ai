"""Resource paths that work both in dev and in PyInstaller frozen EXE.

In a onefile build everything lives under ``sys._MEIPASS`` (read-only);
data files are bundled there via --add-data. Always resolve bundled files
through :func:`resource`, never through bare relative paths.
"""
from __future__ import annotations

import os
import sys


def base_dir() -> str:
    """Project root: baqylau/ in dev, _MEIPASS in frozen EXE."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS  # type: ignore[attr-defined]
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resource(*parts: str) -> str:
    """Absolute path to a bundled resource (models, qss, svg)."""
    return os.path.join(base_dir(), *parts)


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))
