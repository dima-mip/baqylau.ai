"""Head-pose estimation + gaze tracking via MediaPipe FaceMesh / FaceLandmarker.

Supports three backends (auto-selected):
  1. Legacy Solutions API (mediapipe < 0.10.14):
     ``mediapipe.solutions.face_mesh`` or ``mediapipe.python.solutions.face_mesh``
  2. Tasks API (mediapipe >= 0.10.14, Python 3.12+ where ``solutions`` was removed):
     ``mediapipe.tasks.python.vision.FaceLandmarker`` (+ auto-downloaded .task model)
  3. Haar-cascade fallback (OpenCV only): face present/absent + pose Unknown.

Public API is unchanged: ``estimate(frame_bgr) -> (HeadPose, Gaze)``.
Never raises; returns ``face_found=False`` when no backend/face available.
"""
from __future__ import annotations

import logging
import math
import os
import time
import urllib.request
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Landmark indices used for solvePnP (must match FaceMesh topology)
_PNP_IDS = (33, 263, 1, 61, 291, 199)
# Approximate iris / eye-corner indices (MediaPipe FaceMesh w/ refine_landmarks)
_LEFT_IRIS = (468, 469, 470, 471, 472)
_RIGHT_IRIS = (473, 474, 475, 476, 477)
_LEFT_CORNERS = (33, 133)
_RIGHT_CORNERS = (263, 362)
# Eyelids for eye-size-normalized gaze (vertical sensitivity)
LEFT_LID_TOP, LEFT_LID_BOT = 159, 145
RIGHT_LID_TOP, RIGHT_LID_BOT = 386, 374
_CHIN_IDS = (152, 199)  # prefer 152 (chin bottom), fallback 199
# Scale: relative nose displacement -> degrees (delta-calibrated, offset irrelevant)
GEO_PITCH_SCALE = 500.0
EYE_YAW_GAIN = 40.0    # iris-to-corner (in half-widths) -> deg, eye-in-head only
EYE_PITCH_GAIN = 32.0  # vertical excursion physically smaller -> gain higher per unit
EYE_MIN_ASPECT = 0.06  # eye h/w below -> closed (blink); narrow eyes must still pass
# Guided calibration points: (key, instruction shown to the student)
GUIDED_POINTS: tuple[tuple[str, str], ...] = (
    ("center", "Смотрите ПРЯМО в камеру"),
    ("left", "Смотрите ВЛЕВО"),
    ("right", "Смотрите ВПРАВО"),
    ("up", "Смотрите ВВЕРХ"),
    ("down", "Смотрите ВНИЗ"),
)


@dataclass
class HeadPose:
    timestamp: float
    pitch: float  # deg, + = looking up
    yaw: float  # deg, + = looking right
    roll: float = 0.0
    direction: str = "Forward"
    face_found: bool = False


@dataclass
class Gaze:
    timestamp: float
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    off_screen: bool = False
    gaze_vector: tuple = (0.0, 0.0)
    eyes_found: bool = False


LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
LANDMARKER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "models", "face_landmarker.task",
)


def _try_import_legacy_facemesh():
    """Return (module, flavour) or (None, ''). Never raises."""
    # Newer wheels (>=0.10.x) dropped top-level `solutions`; older ones keep it.
    for target in ("mediapipe.solutions.face_mesh", "mediapipe.python.solutions.face_mesh"):
        try:
            import importlib

            mod = importlib.import_module(target)
            return mod, target
        except Exception:
            continue
    return None, ""


def _ensure_landmarker_model(path: str = LANDMARKER_PATH) -> str | None:
    """Download FaceLandmarker .task bundle on first use. Returns path or None."""
    try:
        if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
            return path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.info("Downloading FaceLandmarker model (~15 MB) ...")
        urllib.request.urlretrieve(LANDMARKER_URL, path)
        return path
    except Exception as exc:
        logger.warning("FaceLandmarker model download failed: %s", exc)
        return None


class PoseGazeEstimator:
    """Stateful wrapper (holds one FaceMesh / Landmarker / Haar instance).

    Attributes:
        backend: 'facemesh' | 'landmarker' | 'haar' | 'none'

    Gaze logic: взгляд — глаза (iris vs уголки, нормировано на размер глаза),
    голова НЕ подмешивается (head_mix=0.0): измерение радужки по вебке само
    содержит артефакт поворота головы, и полный подмес (1.0) двойно считает
    голову — детектор орёт постоянно. Поворот головы ловит отдельный канал
    head_turned/head_sustained. Связка осталась опцией (head_mix).
    Закрытые глаза (моргание, aspect < EYE_MIN_ASPECT) исключаются.
    Дальше: персональный ноль (автoкалибровка), EMA, гистерезис persist_frames.
    """

    def __init__(
        self,
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
    ) -> None:
        self.yaw_left = yaw_left
        self.yaw_right = yaw_right
        self.pitch_down = pitch_down
        self.pitch_up = pitch_up
        self.gaze_thresh = gaze_thresh
        self.smooth_alpha = float(smooth_alpha)
        self.persist_frames = max(1, int(persist_frames))
        self.calib_frames = max(0, int(calib_frames))
        self.eye_yaw_gain = float(eye_yaw_gain)
        self.eye_pitch_gain = float(eye_pitch_gain)
        self.head_mix = float(head_mix)
        self.yaw_left = yaw_left
        self.yaw_right = yaw_right
        self.pitch_down = pitch_down
        self.pitch_up = pitch_up
        self.gaze_thresh = gaze_thresh
        self.smooth_alpha = float(smooth_alpha)
        self.persist_frames = max(1, int(persist_frames))
        self.calib_frames = max(0, int(calib_frames))
        self._mesh = None
        self._landmarker = None
        self._haar = None
        self.backend: str = "none"
        # smoothing / calibration state
        self._yaw_sm: float | None = None
        self._pitch_sm: float | None = None
        self._gx_sm: float | None = None
        self._gy_sm: float | None = None
        self._yaw0 = 0.0
        self._pitch0 = 0.0
        self._gx0 = 0.0
        self._gy0 = 0.0
        self._calib_buf: list[tuple[float, float, float, float]] = []
        self.calibrated = self.calib_frames == 0
        self._off_count = 0
        self._on_count = 0
        self._off_state = False
        # guided multi-point calibration state
        self.guided = False
        self._guided_idx = 0
        self._guided_per_point = 45
        self._guided_pose: dict[str, list] = {}
        self._guided_gaze: dict[str, list] = {}
        self._guided_last_face = 0.0
        self._init_backend()

    @property
    def calibrating(self) -> bool:
        """True while frontal or guided calibration is in progress."""
        return (not self.calibrated) or self.guided

    @property
    def calib_prompt(self) -> str:
        """Current instruction for the student ('' when idle)."""
        if self.guided and 0 <= self._guided_idx < len(GUIDED_POINTS):
            key, label = GUIDED_POINTS[self._guided_idx]
            return f"{label}  ({self._guided_idx + 1}/{len(GUIDED_POINTS)})"
        if not self.calibrated:
            n = len(self._calib_buf)
            return f"Смотрите прямо в камеру  ({n}/{self.calib_frames})"
        return ""

    def start_guided(self, per_point_frames: int = 45) -> None:
        """Запустить гидовую калибровку по 5 точкам (центр/лево/право/верх/низ).

        По итогам выставляются персональный ноль И пороги yaw/pitch/gaze.
        """
        self.guided = True
        self._guided_idx = 0
        self._guided_per_point = max(10, int(per_point_frames))
        self._guided_pose = {k: [] for k, _ in GUIDED_POINTS}
        self._guided_gaze = {k: [] for k, _ in GUIDED_POINTS}
        self._guided_last_face = time.time()
        self._off_count = self._on_count = 0
        self._off_state = False
        logger.info("Guided calibration started (%d points)", len(GUIDED_POINTS))

    def reset_calibration(self, calib_frames: int | None = None) -> None:
        """Перезапустить автокалибровку нуля (смотреть прямо в экран)."""
        if calib_frames is not None:
            self.calib_frames = max(0, int(calib_frames))
        self._calib_buf.clear()
        self._yaw0 = self._pitch0 = self._gx0 = self._gy0 = 0.0
        self.calibrated = self.calib_frames == 0
        self.guided = False
        self._guided_idx = 0
        self._guided_pose = {}
        self._guided_gaze = {}
        self._off_count = self._on_count = 0
        self._off_state = False

    def _init_backend(self) -> None:
        # 1) Legacy Solutions FaceMesh (fast path, no model download)
        mod, flavour = _try_import_legacy_facemesh()
        if mod is not None:
            try:
                self._mesh = mod.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.backend = "facemesh"
                logger.info("Face backend: legacy FaceMesh (%s)", flavour)
                return
            except Exception as exc:
                logger.warning("Legacy FaceMesh init failed: %s", exc)
                self._mesh = None
        # 2) Tasks API FaceLandmarker (mediapipe >= 0.10.14, no `solutions`)
        try:
            from mediapipe.tasks.python import vision as mp_vision
            from mediapipe.tasks.python import BaseOptions

            if hasattr(mp_vision, "FaceLandmarker"):
                model = _ensure_landmarker_model()
                if model:
                    opts = mp_vision.FaceLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=model),
                        running_mode=mp_vision.RunningMode.IMAGE,
                        num_faces=1,
                    )
                    self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
                    self.backend = "landmarker"
                    logger.info("Face backend: Tasks FaceLandmarker")
                    return
        except Exception as exc:
            logger.debug("FaceLandmarker unavailable: %s", exc)
        # 3) Haar fallback — face present/absent only, pose via solvePnP skipped
        try:
            import cv2

            path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            haar = cv2.CascadeClassifier(path)
            if not haar.empty():
                self._haar = haar
                self.backend = "haar"
                logger.info("Face backend: Haar fallback (no landmarks)")
                return
        except Exception as exc:
            logger.debug("Haar fallback unavailable: %s", exc)
        self.backend = "none"
        logger.warning("No face backend available; pose/gaze disabled.")

    def close(self) -> None:
        try:
            if self._mesh is not None:
                self._mesh.close()
        except Exception:
            pass
        try:
            if self._landmarker is not None:
                self._landmarker.close()
        except Exception:
            pass
        self._mesh = None
        self._landmarker = None

    # -- public ---------------------------------------------------------
    def estimate(self, frame_bgr: np.ndarray) -> tuple[HeadPose, Gaze]:
        """Estimate pose + gaze for a BGR frame. Never raises."""
        ts = time.time()
        if frame_bgr is None:
            return HeadPose(ts, 0, 0, 0, "Unknown", False), Gaze(ts)
        try:
            if self.backend == "facemesh" and self._mesh is not None:
                raw_pose, raw_gaze = self._estimate_facemesh(frame_bgr, ts)
            elif self.backend == "landmarker" and self._landmarker is not None:
                raw_pose, raw_gaze = self._estimate_landmarker(frame_bgr, ts)
            elif self.backend == "haar" and self._haar is not None:
                return self._estimate_haar(frame_bgr, ts)
            else:
                return HeadPose(ts, 0, 0, 0, "Unknown", False), Gaze(ts)
            if not raw_pose.face_found:
                self._off_count = 0
                self._on_count += 1
                if self._on_count >= self.persist_frames:
                    self._off_state = False
                return raw_pose, raw_gaze
            return self._postprocess(raw_pose, raw_gaze, ts)
        except Exception as exc:
            logger.exception("pose/gaze failed: %s", exc)
            return HeadPose(ts, 0, 0, 0, "Unknown", False), Gaze(ts)

    def _postprocess(self, raw_pose: HeadPose, raw_gaze: Gaze, ts: float) -> tuple[HeadPose, Gaze]:
        """Калибровка нуля + EMA + гистерезис off_screen."""
        if self.guided:
            return self._guided_step(raw_pose, raw_gaze, ts)
        ry, rp = raw_pose.yaw, raw_pose.pitch
        rgx, rgy = raw_gaze.yaw_deg, raw_gaze.pitch_deg

        # 1) autocalibration: collect frontal baseline (only frames with eyes)
        if not self.calibrated:
            if raw_gaze.eyes_found:
                self._calib_buf.append((ry, rp, rgx, rgy))
            if len(self._calib_buf) >= self.calib_frames:
                import statistics

                ys = [b[0] for b in self._calib_buf]
                ps = [b[1] for b in self._calib_buf]
                gxs = [b[2] for b in self._calib_buf]
                gys = [b[3] for b in self._calib_buf]
                self._yaw0 = statistics.median(ys)
                self._pitch0 = statistics.median(ps)
                self._gx0 = statistics.median(gxs)
                self._gy0 = statistics.median(gys)
                self.calibrated = True
                self._yaw_sm = self._pitch_sm = self._gx_sm = self._gy_sm = None
                logger.info("Gaze calibrated: yaw0=%+.1f pitch0=%+.1f gx0=%+.1f gy0=%+.1f",
                            self._yaw0, self._pitch0, self._gx0, self._gy0)
            n = len(self._calib_buf)
            pose = HeadPose(ts, rp - self._pitch0, ry - self._yaw0, 0.0,
                            f"Калибровка… {n}/{self.calib_frames}", True)
            gaze = Gaze(ts, 0.0, 0.0, False, (0.0, 0.0), raw_gaze.eyes_found)
            return pose, gaze

        # 2) subtract personal zero
        cy, cp = ry - self._yaw0, rp - self._pitch0
        cgx, cgy = rgx - self._gx0, rgy - self._gy0

        # 3) EMA smoothing (gaze freezes while eyes not found: blink/profile)
        a = self.smooth_alpha
        self._yaw_sm = cy if self._yaw_sm is None else a * cy + (1 - a) * self._yaw_sm
        self._pitch_sm = cp if self._pitch_sm is None else a * cp + (1 - a) * self._pitch_sm
        if raw_gaze.eyes_found:
            self._gx_sm = cgx if self._gx_sm is None else a * cgx + (1 - a) * self._gx_sm
            self._gy_sm = cgy if self._gy_sm is None else a * cgy + (1 - a) * self._gy_sm
        sy, sp = self._yaw_sm, self._pitch_sm
        sgx = self._gx_sm if self._gx_sm is not None else 0.0
        sgy = self._gy_sm if self._gy_sm is not None else 0.0

        # 4) direction from smoothed calibrated pose
        direction = "Forward"
        if sy < self.yaw_left:
            direction = "Looking Left"
        elif sy > self.yaw_right:
            direction = "Looking Right"
        elif sp < self.pitch_down:
            direction = "Looking Down"
        elif sp > self.pitch_up:
            direction = "Looking Up"

        # 5) gaze off_screen with hysteresis (N frames подряд)
        mag = math.hypot(sgx, sgy)
        raw_off = mag > self.gaze_thresh
        if raw_off:
            self._off_count += 1
            self._on_count = 0
            if self._off_count >= self.persist_frames:
                self._off_state = True
        else:
            self._on_count += 1
            self._off_count = 0
            if self._on_count >= self.persist_frames:
                self._off_state = False
        pose = HeadPose(ts, float(sp), float(sy), 0.0, direction, True)
        gaze = Gaze(ts, float(sgx), float(sgy), bool(self._off_state),
                    (float(sgx), float(sgy)), raw_gaze.eyes_found)
        return pose, gaze

    def _guided_step(self, raw_pose: HeadPose, raw_gaze: Gaze, ts: float) -> tuple[HeadPose, Gaze]:
        """Один кадр гидовой калибровки: копим точку, двигаемся дальше."""
        key = GUIDED_POINTS[self._guided_idx][0]
        if raw_pose.face_found:
            self._guided_last_face = ts
            self._guided_pose[key].append((raw_pose.yaw, raw_pose.pitch))
            if raw_gaze.eyes_found:
                self._guided_gaze[key].append((raw_gaze.yaw_deg, raw_gaze.pitch_deg))
        elif ts - self._guided_last_face > 15.0:
            # лицо пропало надолго — выходим, оставляя старую калибровку
            logger.warning("Guided calibration aborted (no face 15s)")
            self.guided = False
            return HeadPose(ts, 0, 0, 0, "Forward", raw_pose.face_found), \
                Gaze(ts, 0, 0, False, (0.0, 0.0), False)
        if len(self._guided_pose[key]) >= self._guided_per_point:
            self._guided_idx += 1
            if self._guided_idx >= len(GUIDED_POINTS):
                return self._guided_finalize(ts)
        label = GUIDED_POINTS[self._guided_idx][1]
        n = len(self._guided_pose[GUIDED_POINTS[self._guided_idx][0]])
        pose = HeadPose(ts, 0, 0, 0,
                        f"◉ {label} ({self._guided_idx + 1}/{len(GUIDED_POINTS)})", True)
        return pose, Gaze(ts, 0, 0, False, (0.0, 0.0), raw_gaze.eyes_found)

    def _guided_finalize(self, ts: float) -> tuple[HeadPose, Gaze]:
        """Считаем ноль и персональные пороги по 5 точкам."""
        import statistics

        med = statistics.median

        def pose_med(k: str):
            b = self._guided_pose.get(k) or [(0.0, 0.0)]
            return med([p[0] for p in b]), med([p[1] for p in b])

        cy, cp = pose_med("center")
        cg = self._guided_gaze.get("center") or [(0.0, 0.0)]
        cgx, cgy = med([g[0] for g in cg]), med([g[1] for g in cg])
        self._yaw0, self._pitch0, self._gx0, self._gy0 = cy, cp, cgx, cgy

        ly, _ = pose_med("left")
        ry, _ = pose_med("right")
        _, up = pose_med("up")
        _, dn = pose_med("down")
        self.yaw_left = -max(15.0, abs(ly - cy) * 0.55)
        self.yaw_right = max(15.0, abs(ry - cy) * 0.55)
        self.pitch_up = max(15.0, abs(up - cp) * 0.55)
        self.pitch_down = -max(15.0, abs(dn - cp) * 0.55)

        extreme = 0.0
        for k in ("left", "right", "up", "down"):
            for gx, gy in self._guided_gaze.get(k, []):
                m = math.hypot(gx - cgx, gy - cgy)
                if m > extreme:
                    extreme = m
        # медиана была бы точнее, но max устойчивее к ленивым точкам;
        # берём половину экстремума как порог
        if extreme > 1e-9:
            self.gaze_thresh = min(35.0, max(18.0, extreme * 0.5))

        self._yaw_sm = self._pitch_sm = self._gx_sm = self._gy_sm = None
        self._off_count = self._on_count = 0
        self._off_state = False
        self.calibrated = True
        self.guided = False
        logger.info("Guided calibration done: yaw0=%+.1f pitch0=%+.1f | "
                    "yaw[%.0f,%.0f] pitch[%.0f,%.0f] gaze>%.0f",
                    cy, cp, self.yaw_left, self.yaw_right,
                    self.pitch_down, self.pitch_up, self.gaze_thresh)
        return HeadPose(ts, 0, 0, 0, "Forward", True), \
            Gaze(ts, 0, 0, False, (0.0, 0.0), True)

    def _estimate_facemesh(self, frame_bgr: np.ndarray, ts: float) -> tuple[HeadPose, Gaze]:
        import cv2

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._mesh.process(rgb)
        if not res.multi_face_landmarks:
            return HeadPose(ts, 0, 0, 0, "NoFace", False), Gaze(ts)
        lm = res.multi_face_landmarks[0].landmark
        pose = self._solve_pnp(lm, w, h, ts)
        gaze = self._estimate_gaze(lm, pose, ts)
        return pose, gaze

    def _estimate_landmarker(self, frame_bgr: np.ndarray, ts: float) -> tuple[HeadPose, Gaze]:
        """Tasks API path: mp.Image -> FaceLandmarkerResult."""
        import cv2
        import mediapipe as mp

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._landmarker.detect(mp_img)
        if not getattr(res, "face_landmarks", None):
            return HeadPose(ts, 0, 0, 0, "NoFace", False), Gaze(ts)
        lm = res.face_landmarks[0]  # NormalizedLandmark list, same 478 topology
        pose = self._solve_pnp(lm, w, h, ts)
        gaze = self._estimate_gaze(lm, pose, ts)
        return pose, gaze

    def _estimate_haar(self, frame_bgr: np.ndarray, ts: float) -> tuple[HeadPose, Gaze]:
        import cv2

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(gray, 1.2, 5)
        if len(faces) == 0:
            return HeadPose(ts, 0, 0, 0, "NoFace", False), Gaze(ts)
        # Face present but no landmarks -> direction unknown, gaze proxies yaw 0
        return HeadPose(ts, 0, 0, 0, "Forward(haar)", True), Gaze(ts)

    # -- internals ------------------------------------------------------
    def _solve_pnp(self, lm, w: int, h: int, ts: float) -> HeadPose:
        """Raw (uncalibrated, unsmoothed) pose.

        yaw: stable euler from solvePnP (degrees).
        pitch: blend of solvePnP euler + geometric nose/eye proxy — чистый
        solvePnP почти не видит кивки вверх/вниз на плоской 3D-модели,
        поэтому вертикаль берём из относительного смещения носа.
        """
        import cv2

        face_2d, face_3d = [], []
        for idx in _PNP_IDS:
            p = lm[idx]
            x, y = int(p.x * w), int(p.y * h)
            face_2d.append([x, y])
            face_3d.append([x, y, p.z * 8000 if hasattr(p, "z") else 0])
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)
        focal = 1.0 * w
        cam = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])
        dist = np.zeros((4, 1), dtype=np.float64)
        try:
            _, rvec, _ = cv2.solvePnP(face_3d, face_2d, cam, dist)
            rmat, _ = cv2.Rodrigues(rvec)
            # Stable euler (deg): yaw around Y, pitch around X
            yaw = float(math.degrees(math.atan2(rmat[1, 0], rmat[0, 0])))
            pnp_pitch = float(math.degrees(math.atan2(-rmat[2, 0], math.hypot(rmat[2, 1], rmat[2, 2]))))
            roll = float(math.degrees(math.atan2(rmat[2, 1], rmat[2, 2])))
        except Exception:
            pnp_pitch, yaw, roll = 0.0, 0.0, 0.0
        geo_pitch = self._geo_pitch(lm)
        # yaw — эйлер solvePnP (горизонталь он видит корректно);
        # pitch — ТОЛЬКО геометрия: канал эйлера solvePnP на плоской 3D-модели
        # несёт не ту ось и кивки вверх/вниз не видит вообще.
        pitch = geo_pitch if geo_pitch is not None else 0.0
        # Direction решит _postprocess (после калибровки); здесь заглушка.
        return HeadPose(ts, float(pitch), float(yaw), float(roll), "Forward", True)

    @staticmethod
    def _geo_pitch(lm) -> float | None:
        """Нос относительно линии глаз, нормированный на высоту лица.

        Кивок вниз -> нос уезжает вниз от линии глаз (nose_rel растёт),
        по конвенции pitch вниз = отрицательный. Абсолютный оффсет не важен
        (калибровка вычтет медиану), важна только чувствительность дельты.
        """
        try:
            n = len(lm)
            chin_idx = next((i for i in _CHIN_IDS if i < n), None)
            if chin_idx is None:
                return None
            eye_y = sum(lm[i].y for i in (33, 133, 263, 362)) / 4.0
            chin_y = lm[chin_idx].y
            nose_y = lm[1].y
            face_h = chin_y - eye_y
            if face_h < 1e-6:  # лицо в профиль / мусор
                return None
            nose_rel = (nose_y - eye_y) / face_h  # ~0.35 фронтально
            return (0.35 - nose_rel) * GEO_PITCH_SCALE
        except Exception:
            return None

    def _estimate_gaze(self, lm, pose: HeadPose, ts: float) -> Gaze:
        """Raw eye-in-head gaze (uncalibrated). mag/threshold решит _postprocess.

        Каждый глаз считается отдельно и отбрасывается, если закрыт/скошен
        (aspect h/w < EYE_MIN_ASPECT: моргание, профиль) — иначе мусор радужки
        даёт ложный off_screen. Итог — чисто глазной сигнал; голова сюда
        подмешивается только через head_mix (дефолт 0 = развязано).
        """
        try:
            n = len(lm)

            def _pt(i):
                p = lm[i]
                return np.array([p.x, p.y])

            # Guard against builds without iris landmarks
            if max(_RIGHT_IRIS) >= n:
                return Gaze(ts, yaw_deg=0.0, pitch_deg=0.0,
                            off_screen=False, gaze_vector=(0.0, 0.0), eyes_found=False)
            li = np.mean([_pt(i) for i in _LEFT_IRIS], axis=0)
            ri = np.mean([_pt(i) for i in _RIGHT_IRIS], axis=0)
            sides = []  # (dx_norm, dy_norm) per usable eye
            if n > max(RIGHT_LID_TOP, RIGHT_LID_BOT, LEFT_LID_TOP, LEFT_LID_BOT):
                for iris, c1, c2, lt, lb in (
                    (li, 33, 133, LEFT_LID_TOP, LEFT_LID_BOT),
                    (ri, 263, 362, RIGHT_LID_TOP, RIGHT_LID_BOT),
                ):
                    w = float(np.linalg.norm(_pt(c2) - _pt(c1)))
                    h = float(np.linalg.norm(_pt(lt) - _pt(lb)))
                    if w < 1e-9 or h < 1e-9:
                        continue
                    if h / w < EYE_MIN_ASPECT:  # closed / foreshortened
                        continue
                    mid = (_pt(c1) + _pt(c2)) / 2.0
                    sides.append(((iris[0] - mid[0]) / (w / 2.0),
                                  (iris[1] - mid[1]) / (h / 2.0)))
            else:
                # Legacy fallback (нет век): старый масштаб, оба глаза сразу
                lc = np.mean([_pt(i) for i in _LEFT_CORNERS], axis=0)
                rc = np.mean([_pt(i) for i in _RIGHT_CORNERS], axis=0)
                dx = float(((li[0] - lc[0]) + (ri[0] - rc[0])) / 2.0 * 200.0)
                dy = float(((li[1] - lc[1]) + (ri[1] - rc[1])) / 2.0 * 200.0)
                yaw = max(-60.0, min(60.0, dx + pose.yaw * self.head_mix))
                pitch = max(-60.0, min(60.0, dy + pose.pitch * self.head_mix))
                return Gaze(ts, yaw_deg=yaw, pitch_deg=pitch,
                            off_screen=False, gaze_vector=(yaw, pitch), eyes_found=True)
            if not sides:
                return Gaze(ts, yaw_deg=0.0, pitch_deg=0.0,
                            off_screen=False, gaze_vector=(0.0, 0.0), eyes_found=False)
            dx = max(-1.5, min(1.5, sum(s[0] for s in sides) / len(sides)))
            dy = max(-1.5, min(1.5, sum(s[1] for s in sides) / len(sides)))
            yaw = max(-60.0, min(60.0, dx * self.eye_yaw_gain + pose.yaw * self.head_mix))
            # image Y растёт вниз: взгляд вниз (dy>0) -> pitch отрицательный
            pitch = max(-60.0, min(60.0, -dy * self.eye_pitch_gain + pose.pitch * self.head_mix))
            return Gaze(ts, yaw_deg=yaw, pitch_deg=pitch,
                        off_screen=False, gaze_vector=(yaw, pitch), eyes_found=True)
        except Exception:
            return Gaze(ts, yaw_deg=0.0, pitch_deg=0.0,
                        off_screen=False, gaze_vector=(0.0, 0.0), eyes_found=False)

    def draw(self, frame, pose: HeadPose, gaze: Gaze):
        """Overlay pose text + gaze arrow. Best-effort, never raises."""
        try:
            import cv2

            h, w = frame.shape[:2]
            cv2.putText(frame, pose.direction, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            if gaze.eyes_found:
                cx, cy = w // 2, h // 2
                gx = int(cx + gaze.gaze_vector[0] * 4)
                gy = int(cy + gaze.gaze_vector[1] * 4)
                color = (0, 0, 255) if gaze.off_screen else (0, 255, 0)
                cv2.arrowedLine(frame, (cx, cy), (gx, gy), color, 2)
                cv2.putText(frame, f"gaze {gaze.yaw_deg:+.0f},{gaze.pitch_deg:+.0f}",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except Exception:
            pass
        return frame
