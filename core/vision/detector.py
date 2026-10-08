"""YOLOv8 object detection wrapper.

Detects prohibited objects (phones, tablets, books, laptops) and counts
persons in frame. Refactored from ``utils.electronicDevicesDetection`` and
``utils.MTOP_Detection`` into a stateless, testable class.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# COCO ids of interest (ultralytics yolov8n). Names are resolved via model.names.
_PROHIBITED = {"cell phone", "mobile phone", "laptop", "tablet", "book", "remote"}


@dataclass
class Detection:
    """Single bounding box."""

    label: str
    conf: float
    xyxy: tuple  # (x1, y1, x2, y2)


@dataclass
class DetectorResult:
    """Per-frame detection summary."""

    timestamp: float
    detections: list = field(default_factory=list)
    person_count: int = 0
    prohibited: list = field(default_factory=list)
    raised: list = field(default_factory=list)  # phones held up (photo scenario)
    frame_wh: tuple = (0, 0)
    face_missing: bool = False  # True when no person detected at all

    @property
    def phone_detected(self) -> bool:
        return len(self.prohibited) > 0

    @property
    def phone_raised(self) -> bool:
        """Phone held high/close — possible monitor photographing (§2.1)."""
        return len(self.raised) > 0

    @property
    def multi_person(self) -> bool:
        return self.person_count > 1


def flag_raised(phone_xyxy: list, frame_w: int, frame_h: int,
                y_frac: float = 0.45, min_h_frac: float = 0.08) -> list:
    """Pure heuristic: phone in upper frame part and big enough.

    Returns indices into ``phone_xyxy`` that look raised toward the monitor.
    Never raises.
    """
    try:
        out = []
        if frame_w <= 0 or frame_h <= 0:
            return out
        for i, (x1, y1, x2, y2) in enumerate(phone_xyxy):
            cy = (y1 + y2) / 2.0
            h = y2 - y1
            if cy < y_frac * frame_h and h > min_h_frac * frame_h:
                out.append(i)
        return out
    except Exception:
        return []


class ObjectDetector:
    """Thin wrapper around ``ultralytics.YOLO`` with graceful fallback.

    Args:
        model_path: Path to ``.pt`` weights (e.g. ``yolov8n.pt``).
        conf: Confidence threshold.
        prohibited_labels: Extra labels treated as violations.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.45,
        prohibited_labels: tuple | None = None,
        raise_y_frac: float = 0.45,
        raised_min_h_frac: float = 0.08,
    ) -> None:
        self.model_path = model_path
        self.conf = conf
        self.prohibited = set(prohibited_labels) if prohibited_labels else set(_PROHIBITED)
        self.raise_y_frac = raise_y_frac
        self.raised_min_h_frac = raised_min_h_frac
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO

            from core.res import resource

            candidates = [self.model_path]
            if not os.path.isabs(self.model_path):
                candidates.append(resource(self.model_path))
                candidates.append(os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "..", self.model_path))
            for cand in candidates:
                if cand and os.path.exists(cand):
                    self._model = YOLO(cand)
                    break
            else:
                self._model = YOLO(self.model_path)  # may auto-download
            logger.info("YOLOv8 loaded from %s", self.model_path)
        except Exception as exc:  # missing weights / no ultralytics
            logger.warning("YOLO load failed (%s). Detector runs in stub mode.", exc)
            self._model = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def predict(self, frame: np.ndarray) -> DetectorResult:
        """Run detection on a BGR frame.

        Never raises: on failure returns an empty result so the fusion
        loop keeps running.
        """
        ts = time.time()
        if self._model is None or frame is None:
            return DetectorResult(timestamp=ts)
        try:
            results = self._model.predict(source=[frame], conf=self.conf, verbose=False)
            dets: list[Detection] = []
            persons = 0
            prohibited: list[Detection] = []
            phones: list[Detection] = []
            for r in results:
                names = r.names
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = str(names.get(cls_id, cls_id))
                    conf = float(box.conf[0])
                    xyxy = tuple(int(v) for v in box.xyxy[0].tolist())
                    d = Detection(label=label, conf=conf, xyxy=xyxy)
                    dets.append(d)
                    low = label.lower()
                    if low == "person":
                        persons += 1
                    if low in self.prohibited or any(p in low for p in self.prohibited):
                        prohibited.append(d)
                    if "phone" in low:
                        phones.append(d)
            h, w = frame.shape[:2]
            raised_idx = flag_raised([p.xyxy for p in phones], w, h,
                                     self.raise_y_frac, self.raised_min_h_frac)
            raised = [phones[i] for i in raised_idx]
            return DetectorResult(
                timestamp=ts,
                detections=dets,
                person_count=persons,
                prohibited=prohibited,
                raised=raised,
                frame_wh=(w, h),
                face_missing=(persons == 0),
            )
        except Exception as exc:
            logger.exception("YOLO predict failed: %s", exc)
            return DetectorResult(timestamp=ts)

    def draw(self, frame: np.ndarray, result: DetectorResult) -> np.ndarray:
        """Draw bounding boxes in-place (returns same array)."""
        try:
            import cv2

            raised_ids = {id(d) for d in result.raised}
            for d in result.detections:
                x1, y1, x2, y2 = d.xyxy
                color = (0, 0, 255) if d in result.prohibited else (0, 255, 0)
                if d.label.lower() == "person":
                    color = (255, 0, 255)
                label = f"{d.label} {d.conf:.2f}"
                if id(d) in raised_ids:
                    color = (0, 165, 255)  # orange: held up to the screen
                    label = "PHOTO? " + label
                    cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), color, 3)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                )
        except Exception:
            pass
        return frame
