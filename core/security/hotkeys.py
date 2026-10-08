"""Blocked-hotkey guard for exam lockdown (case §2.3).

Intercepts banned combos (Alt+Tab, Ctrl+C/V, Win, PrtScn, ...) via the
``keyboard`` library with ``suppress=True`` so the OS never sees them.
Every blocked press is recorded as a violation event drained by the fusion
loop. Never raises; fully disabled when ``keyboard`` is unavailable.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Combos banned during an exam (``keyboard`` hotkey syntax, Windows).
BANNED_COMBOS: tuple[str, ...] = (
    "alt+tab",
    "alt+shift+tab",
    "alt+esc",
    "alt+f4",
    "ctrl+esc",
    "ctrl+c",
    "ctrl+v",
    "ctrl+x",
    "ctrl+a",
    "ctrl+t",
    "ctrl+w",
    "ctrl+n",
    "ctrl+p",
    "ctrl+tab",
    "ctrl+shift+tab",
    "windows",
    "print screen",
)


@dataclass
class SecurityEvent:
    """One workstation-protection violation."""

    t: float
    kind: str  # "shortcut" | "focus_lost"
    detail: str


class HotkeyGuard:
    """Suppress banned combos and record each blocked press."""

    def __init__(self, combos: tuple | None = None, suppress: bool = True) -> None:
        self.combos = tuple(combos) if combos else BANNED_COMBOS
        self.suppress = suppress
        self._events: list[SecurityEvent] = []
        self._lock = threading.Lock()
        self._on = False

    @property
    def available(self) -> bool:
        try:
            import keyboard  # noqa: F401

            return True
        except Exception:
            return False

    @property
    def running(self) -> bool:
        return self._on

    def start(self) -> bool:
        """Install hooks. Returns False when unavailable (guard disabled)."""
        if self._on:
            return True
        try:
            import keyboard

            for combo in self.combos:
                keyboard.add_hotkey(combo, lambda c=combo: self._fire(c),
                                    suppress=self.suppress)
            self._on = True
            logger.info("HotkeyGuard on (%d combos, suppress=%s)", len(self.combos), self.suppress)
            return True
        except Exception as exc:
            logger.warning("HotkeyGuard unavailable: %s", exc)
            self._on = False
            return False

    def stop(self) -> None:
        try:
            if self._on:
                import keyboard

                keyboard.unhook_all_hotkeys()
        except Exception:
            pass
        self._on = False

    def _fire(self, combo: str) -> None:
        try:
            with self._lock:
                self._events.append(SecurityEvent(time.time(), "shortcut",
                                                  f"Blocked hotkey: {combo}"))
        except Exception:
            pass

    def drain_events(self) -> list[SecurityEvent]:
        """Take pending events (clears buffer). Never raises."""
        try:
            with self._lock:
                out, self._events = self._events, []
            return out
        except Exception:
            return []
