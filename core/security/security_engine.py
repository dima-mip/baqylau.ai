"""Unified workstation-protection engine (case §2.3).

Owns :class:`HotkeyGuard` + :class:`FocusGuard`, exposes one drain for the
fusion loop. Each guard can be toggled; missing optional deps degrade to
"disabled" instead of crashing the proctor.
"""
from __future__ import annotations

import logging

from .focus import FocusGuard
from .hotkeys import HotkeyGuard, SecurityEvent

logger = logging.getLogger(__name__)


class SecurityEngine:
    """Hotkeys + window-focus protection behind one interface."""

    def __init__(self, enable_hotkeys: bool = True, enable_focus: bool = True,
                 suppress: bool = True, grace_sec: float = 2.0,
                 allowed_titles: tuple = ("proctor", "экзамен", "exam"),
                 screenshot_dir: str = "data/logs/snapshots",
                 do_screenshot: bool = True) -> None:
        self.hotkeys = HotkeyGuard(suppress=suppress) if enable_hotkeys else None
        self.focus = (FocusGuard(grace_sec=grace_sec, allowed_substrings=allowed_titles,
                                 screenshot_dir=screenshot_dir,
                                 do_screenshot=do_screenshot)
                      if enable_focus else None)

    def start(self) -> dict:
        """Start guards, return status dict."""
        hk = self.hotkeys.start() if self.hotkeys else False
        fc = self.focus.start() if self.focus else False
        st = self.status()
        logger.info("Security: %s", st)
        return st

    def stop(self) -> None:
        try:
            if self.hotkeys:
                self.hotkeys.stop()
        finally:
            if self.focus:
                self.focus.stop()

    def drain_events(self) -> list[SecurityEvent]:
        """Collect pending violations from all guards. Never raises."""
        out: list[SecurityEvent] = []
        try:
            if self.hotkeys:
                out.extend(self.hotkeys.drain_events())
            if self.focus:
                out.extend(self.focus.drain_events())
        except Exception:
            pass
        return out

    def status(self) -> dict:
        return {
            "hotkeys": "on" if (self.hotkeys and self.hotkeys.running) else "off",
            "focus": "on" if (self.focus and self.focus.running) else "off",
            "baseline": self.focus.baseline if self.focus else "",
        }

    @property
    def active(self) -> bool:
        s = self.status()
        return s["hotkeys"] == "on" or s["focus"] == "on"
