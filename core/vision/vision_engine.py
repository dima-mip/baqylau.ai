"""Unified computer-vision pipeline.

Combines :class:`ObjectDetector` and :class:`PoseGazeEstimator` into one
per-frame call returning a :class:`VisionState`. Thread-safe for use from
the main capture loop.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from .detector import DetectorResult, ObjectDetector
from .pose_gaze import Gaze, HeadPose, PoseGazeEstimator

logger = logging.getLogger(__name__)


@dataclass
class VisionState:
    timestamp: float
    detector: DetectorResult
    pose: HeadPose
    gaze: Gaze
    calibrating: bool = False  # идёт калибровка: риск/события глушатся

    @property
    def head_turned(self) -> bool:
        return (not self.calibrating) and self.pose.face_found and self.pose.direction != "Forward"

    @property
    def eyes_off(self) -> bool:
        return (not self.calibrating) and self.gaze.off_screen

    @property
    def face_missing(self) -> bool:
        return (not self.calibrating) and (not self.pose.face_found) and self.detector.face_missing


class VisionEngine:
    """Owns detector + pose/gaze instances and camera handling."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.45,
        prohibited_labels: tuple | None = None,
        yaw_left: float = -28.0,
        yaw_right: float = 28.0,
        pitch_down: float = -20.0,
        pitch_up: float = 22.0,
        gaze_thresh: float = 20.0,
        smooth_alpha: float = 0.35,
        persist_frames: int = 10,
        calib_frames: int = 60,
        eye_yaw_gain: float = 40.0,
        eye_pitch_gain: float = 32.0,
        head_mix: float = 0.0,
        raise_y_frac: float = 0.45,
        raised_min_h_frac: float = 0.08,
    ) -> None:
        self.detector = ObjectDetector(model_path, conf, prohibited_labels,
                                       raise_y_frac, raised_min_h_frac)
        self.pose_gaze = PoseGazeEstimator(
            yaw_left, yaw_right, pitch_down, pitch_up, gaze_thresh,
            smooth_alpha, persist_frames, calib_frames,
            eye_yaw_gain, eye_pitch_gain, head_mix,
        )

    def process(self, frame: np.ndarray) -> VisionState:
        """Run full CV stack on one BGR frame."""
        ts = time.time()
        det = self.detector.predict(frame)
        pose, gaze = self.pose_gaze.estimate(frame)
        return VisionState(timestamp=ts, detector=det, pose=pose, gaze=gaze,
                           calibrating=self.pose_gaze.calibrating)

    def annotate(self, frame: np.ndarray, state: VisionState) -> np.ndarray:
        """Draw all overlays (boxes + pose + gaze)."""
        try:
            frame = self.detector.draw(frame, state.detector)
            frame = self.pose_gaze.draw(frame, state.pose, state.gaze)
        except Exception as exc:
            logger.debug("annotate failed: %s", exc)
        return frame

    def close(self) -> None:
        self.pose_gaze.close()
