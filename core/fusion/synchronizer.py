"""Temporal alignment of CV frames (~30 Hz) and EEG windows (~4 Hz).

Keeps the latest vision + EEG states; ``fused()`` returns the pair whose
timestamps both fall inside a W=1.0 s window, else None.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class FusedSample:
    t: float
    vision: Any
    eeg: Any


class Synchronizer:
    """Simple latest-within-window synchronizer."""

    def __init__(self, window_sec: float = 1.0, maxlen: int = 128) -> None:
        self.window = window_sec
        self._vision: deque = deque(maxlen=maxlen)
        self._eeg: deque = deque(maxlen=maxlen)

    def push_vision(self, state) -> None:
        self._vision.append(state)

    def push_eeg(self, state) -> None:
        self._eeg.append(state)

    def fused(self) -> FusedSample | None:
        if not self._vision or not self._eeg:
            return None
        v, e = self._vision[-1], self._eeg[-1]
        vt = getattr(v, "timestamp", 0)
        et = getattr(getattr(e, "metrics", None), "timestamp", getattr(e, "timestamp", 0))
        now = time.time()
        if (now - vt) > self.window or (now - et) > self.window:
            return None
        if abs(vt - et) > self.window:
            return None
        return FusedSample(t=max(vt, et), vision=v, eeg=e)

    def __len__(self) -> int:
        return min(len(self._vision), len(self._eeg))
