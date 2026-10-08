"""Event logger + snapshot exporter (JSON/CSV + frame captures)."""
from __future__ import annotations

import csv
import json
import logging
import os
import time
from dataclasses import asdict

import numpy as np

logger = logging.getLogger(__name__)


class AlertManager:
    """Append-only violation log with auto snapshot on CRITICAL."""

    def __init__(self, log_dir: str = "data/logs", snapshot_dir: str = "data/logs/snapshots") -> None:
        self.log_dir = log_dir
        self.snapshot_dir = snapshot_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(snapshot_dir, exist_ok=True)
        self.jsonl_path = os.path.join(log_dir, "events.jsonl")
        self.csv_path = os.path.join(log_dir, "events.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["time", "level", "kind", "points", "score", "detail"])

    def log(self, risk_state, frame: np.ndarray | None = None) -> None:
        """Persist each RiskEvent; snapshot on any CRITICAL event or state."""
        try:
            for ev in risk_state.events:
                rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ev.t)),
                       "ts": ev.t, "level": ev.level, "kind": ev.kind,
                       "points": ev.points, "score": round(risk_state.score, 1),
                       "detail": ev.detail}
                with open(self.jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([rec["time"], rec["level"], rec["kind"],
                                            rec["points"], rec["score"], rec["detail"]])
            critical = risk_state.level == "CRITICAL" or any(
                e.level == "CRITICAL" for e in risk_state.events
            )
            if critical and frame is not None:
                self.snapshot(frame, risk_state)
        except Exception as exc:
            logger.warning("alert log failed: %s", exc)

    def snapshot(self, frame: np.ndarray, risk_state) -> str:
        """Save frame JPEG, return path."""
        try:
            import cv2

            name = f"CRITICAL_{time.strftime('%Y%m%d_%H%M%S')}_{int(risk_state.score)}.jpg"
            path = os.path.join(self.snapshot_dir, name)
            cv2.imwrite(path, frame)
            return path
        except Exception as exc:
            logger.warning("snapshot failed: %s", exc)
            return ""

    def read_events(self) -> list[dict]:
        out = []
        try:
            with open(self.jsonl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
        except FileNotFoundError:
            pass
        return out

    def timeline_figure(self, save_path: str = ""):
        """Matplotlib timeline of risk events. Returns fig or None."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            evs = self.read_events()
            if not evs:
                return None
            xs = [e["ts"] for e in evs]
            ys = [e["score"] for e in evs]
            colors = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red", "CRITICAL": "darkred"}
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.scatter(xs, [e["level"] for e in evs],
                       c=[colors.get(e["level"], "gray") for e in evs], s=[y + 10 for y in ys])
            ax.set_title("Violation timeline (level sized by score)")
            ax.set_xlabel("time")
            fig.tight_layout()
            if save_path:
                fig.savefig(save_path)
            return fig
        except Exception as exc:
            logger.debug("timeline failed: %s", exc)
            return None

    @staticmethod
    def _safe(o):
        try:
            return asdict(o)
        except Exception:
            return str(o)
