"""Baqylau brand assets for the desktop app (Qt side).

Renders ``ui/brand.svg`` (same mark as the web panel) into QIcon/QPixmap.
Never raises — falls back to an empty icon when QtSvg is missing.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def brand_svg_path() -> str:
    try:
        from core.res import resource

        p = resource("ui", "brand.svg")
        if os.path.exists(p):
            return p
    except Exception:
        pass
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "brand.svg")


def logo_pixmap(size: int = 256):
    """Render brand mark to QPixmap. Returns null pixmap on failure."""
    try:
        from PySide6.QtCore import QByteArray
        from PySide6.QtGui import QPixmap, QImage, QPainter
        from PySide6.QtSvg import QSvgRenderer

        with open(brand_svg_path(), "rb") as f:
            data = QByteArray(f.read())
        rnd = QSvgRenderer(data)
        if not rnd.isValid():
            return QPixmap()
        img = QImage(size, size, QImage.Format_ARGB32)
        img.fill(0)
        p = QPainter(img)
        try:
            rnd.render(p)
        finally:
            p.end()
        return QPixmap.fromImage(img)
    except Exception as exc:
        logger.debug("brand render failed: %s", exc)
        try:
            from PySide6.QtGui import QPixmap

            return QPixmap()
        except Exception:
            return None


def app_icon(size: int = 256):
    """QIcon for windows/taskbar. Never raises."""
    try:
        from PySide6.QtGui import QIcon

        pm = logo_pixmap(size)
        return QIcon(pm) if pm is not None and not pm.isNull() else QIcon()
    except Exception:
        try:
            from PySide6.QtGui import QIcon

            return QIcon()
        except Exception:
            return None
