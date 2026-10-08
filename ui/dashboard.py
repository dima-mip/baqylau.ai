"""Real-time proctor HUD (OpenCV overlay).

Layout (single window ``Proctor HUD``):
  * left: annotated video (YOLO boxes, pose text, gaze arrow)
  * right panel: EEG bars (alpha/beta/theta), Focus/Stress/Fatigue gauges,
    Muse connection + per-channel quality, risk score + level.

Also exposes ``run_streamlit()`` (optional) for a browser dashboard when
``streamlit`` is installed.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

PANEL_W = 360


def _bar(img, x, y, w, h, frac, color, label=""):
    import cv2

    frac = max(0.0, min(1.0, float(frac)))
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + int(w * frac), y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1)
    if label:
        cv2.putText(img, label, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)


def draw_hud(frame: np.ndarray, vision=None, eeg_state=None, risk=None,
             eeg_connected: bool = False, channel_quality: dict | None = None) -> np.ndarray:
    """Compose HUD side-by-side. Never raises; returns new array."""
    try:
        import cv2

        h, w = frame.shape[:2]
        panel = np.zeros((h, PANEL_W, 3), dtype=np.uint8)
        y = 24
        cv2.putText(panel, "EEG  Muse-1.3", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 22
        conn = "CONNECTED" if eeg_connected else "NO SIGNAL"
        cv2.putText(panel, conn, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if eeg_connected else (0, 0, 255), 2)
        y += 20
        if channel_quality:
            txt = " ".join(f"{k}:{v[:4]}" for k, v in channel_quality.items())
            cv2.putText(panel, txt[:44], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            y += 16

        m = getattr(eeg_state, "metrics", None) if eeg_state is not None else None
        if m is not None and getattr(eeg_state, "ok", False):
            y += 8
            mx = max(m.alpha, m.beta, m.theta, 1e-9)
            _bar(panel, 12, y, 200, 14, m.alpha / mx, (255, 200, 0), f"alpha {m.alpha:.1f}")
            y += 34
            _bar(panel, 12, y, 200, 14, m.beta / mx, (0, 200, 255), f"beta  {m.beta:.1f}")
            y += 34
            _bar(panel, 12, y, 200, 14, m.theta / mx, (255, 0, 200), f"theta {m.theta:.1f}")
            y += 34
            _bar(panel, 12, y, 200, 16, min(m.focus / 3, 1), (0, 255, 0), f"Focus {m.focus:.2f}")
            y += 36
            _bar(panel, 12, y, 200, 16, min(m.stress / 3, 1), (0, 0, 255), f"Stress {m.stress:.2f}")
            y += 36
            _bar(panel, 12, y, 200, 16, min(m.fatigue / 4, 1), (255, 255, 0), f"Fatigue {m.fatigue:.2f}")
            y += 26
            flags = []
            if m.blink:
                flags.append(f"blinkx{m.blink_count}")
            if m.jaw_clench:
                flags.append("JAW")
            if m.focus_drop:
                flags.append("FOCUS-DROP")
            if m.signal_loss:
                flags.append("SIG-LOSS")
            cv2.putText(panel, " ".join(flags) or "artifacts: none", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
            y += 20
        else:
            cv2.putText(panel, "EEG warming up...", (12, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)
            y += 60

        # Vision summary
        y += 10
        cv2.putText(panel, "VISION", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 22
        if vision is not None:
            det = vision.detector
            lines = [
                f"pose: {vision.pose.direction} yaw {vision.pose.yaw:+.0f}",
                f"gaze: {'OFF' if vision.gaze.off_screen else 'on'} "
                f"({vision.gaze.yaw_deg:+.0f},{vision.gaze.pitch_deg:+.0f})",
                f"persons: {det.person_count} phone: {'YES' if det.phone_detected else 'no'}",
            ]
            for ln in lines:
                cv2.putText(panel, ln[:42], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
                y += 18
        # Risk footer
        y = max(y + 8, h - 70)
        score = float(getattr(risk, "score", 0.0)) if risk is not None else 0.0
        level = str(getattr(risk, "level", "OK")) if risk is not None else "OK"
        lcol = {"OK": (0, 255, 0), "LOW": (0, 255, 255), "MEDIUM": (0, 165, 255),
                "HIGH": (0, 0, 255), "CRITICAL": (0, 0, 255)}.get(level, (255, 255, 255))
        _bar(panel, 12, y, PANEL_W - 24, 20, score / 100.0, lcol, "")
        cv2.putText(panel, f"RISK {score:.0f}  {level}", (12, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, lcol, 2)
        hud = np.hstack([frame, panel])
        return hud
    except Exception as exc:
        logger.debug("draw_hud failed: %s", exc)
        return frame


def run_streamlit():  # pragma: no cover - optional UI
    """Optional browser dashboard (requires ``pip install streamlit``)."""
    import pandas as pd
    import streamlit as st

    from core.fusion.alert_manager import AlertManager

    st.title("Proctoring — Post-Exam Violation Report")
    am = AlertManager()
    evs = am.read_events()
    if not evs:
        st.info("No events logged yet.")
        return
    df = pd.DataFrame(evs)
    st.dataframe(df)
    st.bar_chart(df.set_index("time")[["score"]] if "score" in df else df)
