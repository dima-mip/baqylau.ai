"""Muse 1.3 (MU-01 / Spark) streaming layer.

Preferred backend is BrainFlow (stable BLE for MUSE boards). Falls back to
a synthetic generator so the fusion pipeline and UI can be developed and
tested without hardware.

Public interface:
    stream = make_stream(simulate=True|False, fs=256)
    stream.connect()
    window = stream.get_window(window_sec=2.0)  # (4, N) uV array or None
    stream.disconnect()
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)

CHANNELS = ("TP9", "AF7", "AF8", "TP10")


class BaseStream(ABC):
    @abstractmethod
    def connect(self) -> bool: ...
    @abstractmethod
    def disconnect(self) -> None: ...
    @abstractmethod
    def get_window(self, window_sec: float = 2.0) -> np.ndarray | None: ...
    @property
    @abstractmethod
    def connected(self) -> bool: ...
    @property
    @abstractmethod
    def sampling_rate(self) -> int: ...


class SimulatedMuseStream(BaseStream):
    """Synthetic 4-ch EEG: 10 Hz alpha + 20 Hz beta + noise + blink events.

    Useful for CI, demos, and when no headband is paired.
    """

    def __init__(self, fs: int = 256, seed: int = 7) -> None:
        self._fs = fs
        self._rng = np.random.default_rng(seed)
        self._buf: deque = deque(maxlen=fs * 30)
        self._on = False
        self._thr: threading.Thread | None = None
        self._t = 0.0

    @property
    def connected(self) -> bool:
        return self._on

    @property
    def sampling_rate(self) -> int:
        return self._fs

    def connect(self) -> bool:
        if self._on:
            return True
        self._on = True
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()
        logger.info("Simulated Muse stream started @%d Hz", self._fs)
        return True

    def disconnect(self) -> None:
        self._on = False

    def _loop(self) -> None:
        chunk = 32
        while self._on:
            t = self._t + np.arange(chunk) / self._fs
            self._t += chunk / self._fs
            # Slow cognitive drift so demos exercise the fusion table:
            # alpha waxes/wanes ~90 s, beta ~47 s + stress episodes ~every minute.
            drift_a = 1.0 + 0.4 * np.sin(2 * np.pi * self._t / 90.0)
            drift_b = 1.0 + 0.5 * np.sin(2 * np.pi * self._t / 47.0 + 1.0)
            stress_ep = 1.0
            if (self._t % 75.0) > 62.0:  # ~13 s stress episode each ~75 s
                stress_ep = 3.0
            sample = np.zeros((4, chunk))
            for ch in range(4):
                alpha = 12.0 * drift_a * np.sin(2 * np.pi * 10 * t + ch)
                beta = 5.0 * drift_b * stress_ep * np.sin(2 * np.pi * 21 * t + ch * 2)
                noise = self._rng.normal(0, 4.0, chunk)
                sample[ch] = alpha + beta + noise
            # occasional synthetic blink on AF7/AF8
            if self._rng.random() < 0.02:
                spike = self._rng.normal(90, 10, chunk // 4)
                sample[1, : len(spike)] += spike
                sample[2, : len(spike)] += spike
            for i in range(chunk):
                self._buf.append(sample[:, i])
            time.sleep(chunk / self._fs)

    def get_window(self, window_sec: float = 2.0) -> np.ndarray | None:
        need = int(self._fs * window_sec)
        if len(self._buf) < need:
            return None
        arr = np.array(list(self._buf)[-need:]).T  # (4, need)
        return arr.astype(float)


class BrainFlowMuseStream(BaseStream):
    """BrainFlow backend for real Muse hardware.

    Board IDs vary by BrainFlow version; we probe MUSE_2016_BOARD then
    MUSE_S_BOARD / MUSE_2_BOARD. BLE MAC/serial can be passed via ``params``.
    Auto-reconnects with backoff on dropout.
    """

    def __init__(self, fs: int = 256, serial_port: str = "", mac: str = "") -> None:
        self._target_fs = fs
        self._serial = serial_port
        self._mac = mac
        self._board = None
        self._board_id = None
        self._eeg_idx: list[int] = [0, 1, 2, 3]
        self._fs = fs
        self._on = False
        self._failures = 0

    @property
    def connected(self) -> bool:
        return self._on and self._board is not None

    @property
    def sampling_rate(self) -> int:
        return self._fs

    def _create_board(self):
        from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

        params = BrainFlowInputParams()
        if self._serial:
            params.serial_port = self._serial
        if self._mac:
            params.mac_address = self._mac
        candidates = []
        for name in ("MUSE_2016_BOARD", "MUSE_S_BOARD", "MUSE_2_BOARD", "MUSE_2016_BLED_BOARD"):
            if hasattr(BoardIds, name):
                candidates.append(getattr(BoardIds, name))
        if not candidates:  # pragma: no cover - very old brainflow
            candidates = [14]
        last = None
        for bid in candidates:
            try:
                board = BoardShim(bid, params)
                board.prepare_session()
                self._board_id = bid
                try:
                    self._eeg_idx = BoardShim.get_eeg_channels(bid)[:4]
                    self._fs = BoardShim.get_sampling_rate(bid)
                except Exception:
                    pass
                return board
            except Exception as exc:
                last = exc
        raise RuntimeError(f"No Muse board available: {last}")

    def connect(self) -> bool:
        try:
            self._board = self._create_board()
            self._board.start_stream()
            self._on = True
            self._failures = 0
            logger.info("BrainFlow Muse connected (board=%s, fs=%s)", self._board_id, self._fs)
            return True
        except Exception as exc:
            logger.warning("BrainFlow connect failed: %s", exc)
            self._on = False
            return False

    def disconnect(self) -> None:
        try:
            if self._board is not None:
                try:
                    self._board.stop_stream()
                except Exception:
                    pass
                try:
                    self._board.release_session()
                except Exception:
                    pass
        finally:
            self._board = None
            self._on = False

    def get_window(self, window_sec: float = 2.0) -> np.ndarray | None:
        if not self.connected:
            # auto-reconnect attempt with backoff
            self._failures += 1
            if self._failures % 20 == 1:
                logger.info("EEG reconnect attempt #%d", self._failures)
                self.disconnect()
                time.sleep(min(0.25 * self._failures, 3.0))
                self.connect()
            return None
        try:
            need = int(self._fs * window_sec)
            raw = self._board.get_current_board_data(need)
            if raw.shape[1] < need // 2:
                return None
            eeg = raw[self._eeg_idx, -need:]
            return np.asarray(eeg, dtype=float)
        except Exception as exc:
            logger.debug("get_board_data failed: %s", exc)
            return None


def make_stream(simulate: bool = False, fs: int = 256, **kw) -> BaseStream:
    """Factory: real BrainFlow stream unless ``simulate`` or import fails."""
    if simulate:
        return SimulatedMuseStream(fs=fs)
    try:
        import brainflow  # noqa: F401
        return BrainFlowMuseStream(fs=fs, **kw)
    except Exception as exc:
        logger.warning("brainflow unavailable (%s) -> simulated EEG", exc)
        return SimulatedMuseStream(fs=fs)
