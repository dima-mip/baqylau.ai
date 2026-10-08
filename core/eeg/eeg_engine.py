"""Unified EEG processing loop: stream -> filter -> PSD -> metrics."""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from . import signal_proc
from .metrics import EEGMetrics, compute_metrics
from .stream import BaseStream, make_stream

logger = logging.getLogger(__name__)


@dataclass
class EEGState:
    timestamp: float
    metrics: EEGMetrics | None
    bandpower: dict | None
    ok: bool  # False when no fresh window (disconnected / warming up)


class EEGEngine:
    """Pulls sliding windows and emits cognitive state.

    Args:
        fs: sampling rate. simulate: force synthetic stream.
        focus_drop_delta: drop vs baseline that flags ``focus_drop``.
    """

    def __init__(self, fs: int = 256, window_sec: float = 2.0,
                 low: float = 1.0, high: float = 40.0, order: int = 4,
                 notch: float = 50.0, simulate: bool = False,
                 focus_drop_delta: float = 0.4,
                 blink_thresh: float = 75.0, gamma_thresh: float = 15.0) -> None:
        self.fs = fs
        self.window_sec = window_sec
        self.low, self.high, self.order = low, high, order
        self.notch = notch
        self.blink_thresh = blink_thresh
        self.gamma_thresh = gamma_thresh
        self.focus_drop_delta = focus_drop_delta
        self.stream: BaseStream = make_stream(simulate=simulate, fs=fs)
        self._baseline: deque = deque(maxlen=40)  # ~10 s at 4 Hz
        self._last_state: EEGState | None = None

    def start(self) -> bool:
        ok = self.stream.connect()
        if not ok:
            logger.warning("EEG stream connect failed; will retry in loop")
        return ok

    def stop(self) -> None:
        try:
            self.stream.disconnect()
        except Exception:
            pass

    @property
    def connected(self) -> bool:
        try:
            return self.stream.connected
        except Exception:
            return False

    def tick(self) -> EEGState:
        """Process one window. Call at ~4 Hz (every step_sec). Never raises."""
        ts = time.time()
        try:
            raw = self.stream.get_window(self.window_sec)
            if raw is None:
                return EEGState(ts, None, None, ok=False)
            filt = signal_proc.apply_filters(
                raw, self.stream.sampling_rate, self.low, self.high, self.order, self.notch
            )
            psd = signal_proc.welch_psd(filt, self.stream.sampling_rate)
            m = compute_metrics(raw, psd.bandpower, self.blink_thresh, self.gamma_thresh)
            # focus-drop vs rolling median baseline
            if len(self._baseline) >= 8:
                base = float(np.median(list(self._baseline)))
                m.focus_drop = (base - m.focus) > self.focus_drop_delta
            self._baseline.append(m.focus)
            self._last_state = EEGState(m.timestamp, m, psd.bandpower, ok=True)
            return self._last_state
        except Exception as exc:
            logger.exception("EEG tick failed: %s", exc)
            return EEGState(ts, None, None, ok=False)

    def channel_quality(self, raw_hint: np.ndarray | None = None) -> dict[str, str]:
        """Per-channel quality stub: Good/Check based on variance."""
        names = ("TP9", "AF7", "AF8", "TP10")
        try:
            if self._last_state is None or not self._last_state.ok:
                return {c: "NoSignal" for c in names}
            return {c: "Good" for c in names}
        except Exception:
            return {c: "Unknown" for c in names}
