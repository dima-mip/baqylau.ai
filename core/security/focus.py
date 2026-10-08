"""Exam-window focus guard (case §2.3: no tab/app switching).

Learns the foreground window at ``start()`` as the exam window and flags any
other window that stays focused longer than ``grace_sec``. Optionally grabs
a screenshot of the intruding window for evidence. Polls in its own daemon
thread; never raises.
"""
from __future__ import annotations

import logging
import os
import threading
import time

from .hotkeys import SecurityEvent

logger = logging.getLogger(__name__)


class FocusGuard:
    """Detect leaving the exam window."""

    def __init__(self, grace_sec: float = 2.0, poll_sec: float = 1.0,
                 allowed_substrings: tuple = (),
                 screenshot_dir: str = "", do_screenshot: bool = True) -> None:
        self.grace_sec = max(0.5, float(grace_sec))
        self.poll_sec = max(0.25, float(poll_sec))
        self.allowed = tuple(s.lower() for s in allowed_substrings)
        self.screenshot_dir = screenshot_dir
        self.do_screenshot = do_screenshot
        self._baseline: str | None = None
        self._events: list[SecurityEvent] = []
        self._lock = threading.Lock()
        self._on = False
        self._thr: threading.Thread | None = None
        self._away_since: float | None = None
        self._fired_for: str | None = None

    @property
    def available(self) -> bool:
        try:
            import pygetwindow  # noqa: F401

            return True
        except Exception:
            return False

    @property
    def running(self) -> bool:
        return self._on

    @property
    def baseline(self) -> str:
        return self._baseline or ""

    def start(self) -> bool:
        if self._on:
            return True
        if not self.available:
            logger.warning("FocusGuard unavailable (pygetwindow missing)")
            return False
        try:
            self._baseline = self._active_title()
            self._on = True
            self._thr = threading.Thread(target=self._loop, daemon=True)
            self._thr.start()
            logger.info("FocusGuard on (baseline=%r)", self._baseline)
            return True
        except Exception as exc:
            logger.warning("FocusGuard start failed: %s", exc)
            self._on = False
            return False

    def stop(self) -> None:
        self._on = False

    def drain_events(self) -> list[SecurityEvent]:
        try:
            with self._lock:
                out, self._events = self._events, []
            return out
        except Exception:
            return []

    # -- internals ------------------------------------------------------
    def _active_title(self) -> str:
        try:
            import pygetwindow as gw

            w = gw.getActiveWindow()
            return str(getattr(w, "title", "") or "")
        except Exception:
            return ""

    def _allowed(self, title: str) -> bool:
        if not title:
            return True  # desktop/lock screen: ignore, don't spam
        if self._baseline and title == self._baseline:
            return True
        low = title.lower()
        return any(s in low for s in self.allowed)

    def _loop(self) -> None:
        while self._on:
            try:
                self._check(time.time())
            except Exception as exc:
                logger.debug("focus check failed: %s", exc)
            time.sleep(self.poll_sec)

    def _check(self, now: float) -> None:
        title = self._active_title()
        if self._allowed(title):
            self._away_since = None
            self._fired_for = None
            return
        if self._away_since is None:
            self._away_since = now
            return
        if now - self._away_since >= self.grace_sec and self._fired_for != title:
            self._fired_for = title
            shot = self._screenshot(title, now) if self.do_screenshot else ""
            detail = f"Left exam window -> {title!r}" + (f" [{shot}]" if shot else "")
            with self._lock:
                self._events.append(SecurityEvent(now, "focus_lost", detail))

    def _screenshot(self, title: str, now: float) -> str:
        """Best-effort screen capture, returns path or ''."""
        try:
            if not self.screenshot_dir:
                return ""
            os.makedirs(self.screenshot_dir, exist_ok=True)
            name = "SCREEN_{}.png".format(time.strftime("%Y%m%d_%H%M%S", time.localtime(now)))
            path = os.path.join(self.screenshot_dir, name)
            try:
                import pyautogui

                pyautogui.screenshot(path)
                return path
            except Exception:
                from PIL import ImageGrab  # fallback without pyautogui

                ImageGrab.grab().save(path)
                return path
        except Exception as exc:
            logger.debug("screenshot failed: %s", exc)
            return ""
