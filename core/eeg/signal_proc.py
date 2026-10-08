"""EEG signal preprocessing: bandpass + notch + Welch PSD.

Channels: TP9, AF7, AF8, TP10 (Muse 1.3). All functions are pure numpy/scipy
so they are unit-testable without hardware.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
HIGH_BETA = (20.0, 30.0)


@dataclass
class PSDResult:
    freqs: np.ndarray
    psd: np.ndarray  # shape (n_channels, n_freqs), units uV^2/Hz
    bandpower: dict  # band -> np.ndarray (n_channels,)
    fs: float


def design_filters(fs: float, low: float = 1.0, high: float = 40.0,
                   order: int = 4, notch: float = 50.0, q: float = 30.0):
    """Return (bandpass_sos, notch_b, notch_a)."""
    from scipy import signal

    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    b_notch, a_notch = signal.iirnotch(notch, q, fs)
    return sos, b_notch, a_notch


def apply_filters(data: np.ndarray, fs: float, low: float = 1.0, high: float = 40.0,
                  order: int = 4, notch: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Zero-phase bandpass + notch. ``data`` shape (n_channels, n_samples).

    Falls back to unfiltered data if scipy is missing or the window is too
    short for filtfilt padding.
    """
    try:
        from scipy import signal

        sos, b_n, a_n = design_filters(fs, low, high, order, notch, q)
        # filtfilt needs len > 3 * padlen; guard for tiny windows
        out = data.astype(float)
        if out.shape[1] > 64:
            out = signal.sosfiltfilt(sos, out, axis=1)
            out = signal.filtfilt(b_n, a_n, out, axis=1)
        return out
    except Exception as exc:
        logger.debug("filter fallback: %s", exc)
        return np.asarray(data, dtype=float)


def welch_psd(data: np.ndarray, fs: float, nperseg: int | None = None) -> PSDResult:
    """Welch PSD + mean bandpower per channel.

    Args:
        data: (n_channels, n_samples) already filtered, in microvolts.
        fs: sampling rate Hz.
    """
    from scipy import signal

    n = data.shape[1]
    if nperseg is None:
        nperseg = min(n, int(fs * 2))
    nperseg = max(32, min(nperseg, n))
    freqs, psd = None, []
    for ch in data:
        f, p = signal.welch(ch, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
        if freqs is None:
            freqs = f
        psd.append(p)
    psd = np.array(psd)
    bandpower: dict[str, np.ndarray] = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            # integrate PSD over band
            bp = np.trapezoid(psd[:, mask], freqs[mask], axis=1)
        else:
            bp = np.zeros(data.shape[0])
        bandpower[name] = bp
    # high-beta sub-band for stress index
    lo, hi = HIGH_BETA
    mask = (freqs >= lo) & (freqs < hi)
    bandpower["high_beta"] = (
        np.trapezoid(psd[:, mask], freqs[mask], axis=1) if mask.any() else np.zeros(data.shape[0])
    )
    return PSDResult(freqs=freqs, psd=psd, bandpower=bandpower, fs=fs)
