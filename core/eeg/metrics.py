"""Cognitive & artifact metrics from EEG bandpower.

Indices (frontal/temporal average unless noted):
    focus   = beta / (theta + alpha)
    stress  = high_beta / alpha          (cognitive load)
    fatigue = (theta + alpha) / beta     (drowsiness)

Artifacts:
    blink        : |AF7| or |AF8| peak > 75 uV in raw window
    jaw_clench   : gamma power spike on TP9/TP10 (muscle >30 Hz)
    signal_loss  : flat line (std < 0.5 uV) or NaN
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-9


@dataclass
class EEGMetrics:
    timestamp: float
    focus: float
    stress: float
    fatigue: float
    alpha: float
    beta: float
    theta: float
    gamma: float
    blink: bool = False
    blink_count: int = 0
    jaw_clench: bool = False
    signal_loss: bool = False
    focus_drop: bool = False  # set by EEGEngine vs rolling baseline


def band_means(bandpower: dict) -> dict[str, float]:
    """Average each band across the 4 channels."""
    out = {}
    for k, v in bandpower.items():
        arr = np.asarray(v, dtype=float)
        out[k] = float(np.nanmean(arr)) if arr.size else 0.0
    return out


def cognitive_indices(means: dict[str, float]) -> tuple[float, float, float]:
    theta = means.get("theta", 0.0)
    alpha = means.get("alpha", 0.0)
    beta = means.get("beta", 0.0)
    high_beta = means.get("high_beta", beta)
    focus = beta / (theta + alpha + EPS)
    stress = high_beta / (alpha + EPS)
    fatigue = (theta + alpha) / (beta + EPS)
    # clamp to sane display range
    focus = float(min(max(focus, 0.0), 10.0))
    stress = float(min(max(stress, 0.0), 10.0))
    fatigue = float(min(max(fatigue, 0.0), 10.0))
    return focus, stress, fatigue


def detect_blinks(raw: np.ndarray, thresh_uv: float = 75.0) -> tuple[bool, int]:
    """Count threshold crossings on AF7/AF8 (channels 1,2)."""
    try:
        frontal = raw[[1, 2], :] if raw.shape[0] >= 3 else raw
        peaks = np.abs(frontal) > thresh_uv
        # count rising edges as individual blinks
        count = 0
        for ch in peaks:
            edges = np.diff(ch.astype(int)) == 1
            count += int(edges.sum())
        return (count > 0), count
    except Exception:
        return False, 0


def detect_jaw_clench(bandpower: dict, thresh: float = 15.0) -> bool:
    """High gamma on temporals (TP9 idx0, TP10 idx3) => muscle activity."""
    try:
        gamma = np.asarray(bandpower.get("gamma", [0, 0, 0, 0]), dtype=float)
        if gamma.size >= 4:
            temporal = float(max(gamma[0], gamma[3]))
        else:
            temporal = float(np.max(gamma)) if gamma.size else 0.0
        return bool(temporal > thresh)
    except Exception:
        return False


def detect_signal_loss(raw: np.ndarray) -> bool:
    try:
        if raw is None or raw.size == 0 or np.isnan(raw).any():
            return True
        return bool(np.std(raw) < 0.5)
    except Exception:
        return True


def compute_metrics(raw_window: np.ndarray, bandpower: dict,
                    thresh_uv: float = 75.0, gamma_thresh: float = 15.0) -> EEGMetrics:
    """Fuse bandpower + raw artifacts into one metrics object."""
    ts = time.time()
    means = band_means(bandpower)
    focus, stress, fatigue = cognitive_indices(means)
    blink, n_blink = detect_blinks(raw_window, thresh_uv)
    jaw = detect_jaw_clench(bandpower, gamma_thresh)
    loss = detect_signal_loss(raw_window)
    return EEGMetrics(
        timestamp=ts,
        focus=focus, stress=stress, fatigue=fatigue,
        alpha=means.get("alpha", 0.0), beta=means.get("beta", 0.0),
        theta=means.get("theta", 0.0), gamma=means.get("gamma", 0.0),
        blink=blink, blink_count=n_blink, jaw_clench=jaw, signal_loss=loss,
    )
