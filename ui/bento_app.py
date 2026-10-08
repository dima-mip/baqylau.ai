"""Soft Bento Grid desktop HUD (PySide6).

Layout:
  Header: avatar + student name + status badge + session timer + EEG/cam dots
  Grid:  Video (dark) | EEG attention (purple) / Fatigue (green) | Alerts (yellow)
  Controls: Snapshot / Pause / Quit pastel buttons.

Video + EEG run in separate QThreads so the UI never freezes.
Console logs are redirected into the alert feed / status widgets.
"""
from __future__ import annotations

import logging
import time

from PySide6.QtCore import QObject, QThread, Qt, Signal, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

LEVEL_COLORS = {
    "LOW": "#2E7D32",
    "MEDIUM": "#EF6C00",
    "HIGH": "#C62828",
    "CRITICAL": "#7B0000",
}


class FatigueRing(QWidget):
    """Circular fatigue indicator (0-10 scale). Repaints on set_value()."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._value = 0.0
        self.setMinimumSize(130, 130)
        self.setMaximumSize(160, 160)

    def set_value(self, v: float) -> None:
        self._value = max(0.0, min(10.0, float(v)))
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt naming)
        try:
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            side = min(self.width(), self.height()) - 16
            x, y = (self.width() - side) // 2, (self.height() - side) // 2
            p.setPen(QPen(QColor("#FFFFFF"), 12, Qt.SolidLine, Qt.RoundCap))
            p.drawArc(x, y, side, side, 0, 360 * 16)
            frac = self._value / 10.0
            color = "#2E7D32" if frac < 0.4 else ("#EF6C00" if frac < 0.65 else "#C62828")
            p.setPen(QPen(QColor(color), 12, Qt.SolidLine, Qt.RoundCap))
            p.drawArc(x, y, side, side, 90 * 16, -int(360 * 16 * frac))
            p.setPen(Qt.black)
            p.drawText(x, y, side, side, Qt.AlignCenter, f"{self._value:.2f}")
            p.end()
        except Exception:
            pass


class VideoWorker(QObject):
    """Camera + VisionEngine loop. Emits annotated QImage + VisionState + BGR frame."""

    frame_ready = Signal(object, object, object)  # (QImage, VisionState, BGR np.ndarray)
    finished = Signal()

    def __init__(self, vision, camera: int = 0, mirror: bool = False) -> None:
        super().__init__()
        self._vision = vision
        self._camera = camera
        self._mirror = mirror
        self._running = True

    def stop(self) -> None:
        self._running = False

    def arm(self) -> None:
        self._running = True

    def run(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self._camera)
        if not cap.isOpened():
            logger.error("Cannot open camera %s", self._camera)
            self.finished.emit()
            return
        while self._running:
            ok, frame = cap.read()
            if not ok or frame is None:
                QThread.msleep(50)
                continue
            if self._mirror:
                frame = cv2.flip(frame, 1)
            try:
                state = self._vision.process(frame)
                annotated = self._vision.annotate(frame.copy(), state)
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
                self.frame_ready.emit(img, state, frame.copy())
            except Exception as exc:  # never kill the thread
                logger.debug("video tick failed: %s", exc)
            QThread.msleep(33)  # ~30 fps
        cap.release()
        self.finished.emit()


class EEGWorker(QObject):
    """EEGEngine loop at ~4 Hz. Emits EEGState."""

    eeg_ready = Signal(object)
    finished = Signal()

    def __init__(self, eeg) -> None:
        super().__init__()
        self._eeg = eeg
        self._running = True
        self._step = 0.25

    def stop(self) -> None:
        self._running = False

    def arm(self) -> None:
        self._running = True

    def run(self) -> None:
        if self._eeg is not None:
            try:
                self._eeg.start()
            except Exception as exc:
                logger.warning("EEG start failed: %s", exc)
        while self._running:
            try:
                if self._eeg is not None:
                    self.eeg_ready.emit(self._eeg.tick())
            except Exception as exc:
                logger.debug("eeg tick failed: %s", exc)
            # QThread.msleep takes int ms; accumulate precisely enough
            QThread.msleep(int(self._step * 1000))
        try:
            if self._eeg is not None:
                self._eeg.stop()
        except Exception:
            pass
        self.finished.emit()


class BentoWindow(QMainWindow):
    """Main Soft-Bento window. Owns fusion (RiskEngine + Synchronizer)."""

    def __init__(self, cfg, args, vision, eeg, risk, sync, alerts,
                 student: str = "Student", security=None,
                 client=None, standalone: bool = True) -> None:
        super().__init__()
        self.cfg, self.args = cfg, args
        self.vision, self.eeg = vision, eeg
        self.risk, self.sync, self.alerts = risk, sync, alerts
        self.security = security
        self.client = client
        self.standalone = standalone
        self._paused = False
        self._t0 = time.time()
        self._last_frame = None
        self._last_bgr = None
        self._risk_state = None
        self._calib_was = False
        self._last_video_fuse = 0.0
        self._video_fuse_min_dt = 0.5  # vision-only fuse не чаще 2 Гц
        self._last_sec_t = 0.0
        # session gating (server mode): camera OFF until teacher starts exam
        self._session_id = None
        self._session_active = standalone
        self._vthread = None
        self._vworker = None
        self._ethread = None
        self._eworker = None
        # evidence ring buffer: ~9s @10fps, 320px (for RISK100 video clips)
        from collections import deque as _dq
        self._ring = _dq(maxlen=90)
        self._ring_tick = 0
        self._last_evidence_t = 0.0
        self._last_thumb_t = 0.0
        self._thumb_interval = float(getattr(args, "thumb_interval", 4.0) or 4.0)
        # exam widget state
        self._exam_deadline = 0.0
        self._exam_groups = []
        self._build_ui(student)
        self._build_workers()
        if standalone:
            self._start_capture()
        else:
            self._sess_timer = QTimer(self)
            self._sess_timer.timeout.connect(self._poll_session)
            self._sess_timer.start(5000)
            self.badge.setText("⏳ Ожидание экзамена…")
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick_clock)
        self._timer.start(1000)
        self._hb_timer = QTimer(self)
        self._hb_timer.timeout.connect(self._heartbeat)
        self._hb_timer.start(int(self._thumb_interval * 1000))

    # -- UI ------------------------------------------------------------
    def _card(self, name: str) -> QFrame:
        f = QFrame()
        f.setObjectName(name)
        f.setProperty("card", True)
        f.setLayout(QVBoxLayout())
        f.layout().setContentsMargins(18, 16, 18, 16)
        f.layout().setSpacing(10)
        return f

    def _build_ui(self, student: str) -> None:
        self.setWindowTitle("Baqylau — Student")
        self.resize(1240, 780)
        root = QWidget()
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setSpacing(14)
        lay.setContentsMargins(18, 18, 18, 18)

        # Header
        header = QFrame()
        header.setObjectName("HeaderCard")
        header.setProperty("card", True)
        hl = QHBoxLayout(header)
        avatar = QLabel(student[:1].upper() or "S")
        avatar.setObjectName("Avatar")
        avatar.setAlignment(Qt.AlignCenter)
        title_box = QVBoxLayout()
        title_box.addWidget(self._mk("AppTitle", student))
        title_box.addWidget(self._mk("AppSub", "Baqylau  •  YOLO + MediaPipe + защита окружения"))
        hl.addWidget(avatar)
        hl.addLayout(title_box)
        hl.addStretch(1)
        self.badge = QLabel("● Экзамен активен")
        self.badge.setObjectName("Badge")
        hl.addWidget(self.badge)
        self.clock = QLabel("00:00")
        self.clock.setObjectName("CardTitle")
        hl.addWidget(self.clock)
        self.eeg_dot = QLabel("● EEG")
        self.eeg_dot.setObjectName("Dot")
        self.cam_dot = QLabel("● CAM")
        self.cam_dot.setObjectName("Dot")
        self.sec_dot = QLabel("● SEC")
        self.sec_dot.setObjectName("Dot")
        self.sec_dot.setStyleSheet("color:#999999;")
        hl.addWidget(self.eeg_dot)
        hl.addWidget(self.cam_dot)
        hl.addWidget(self.sec_dot)
        lay.addWidget(header)

        # Grid
        grid = QGridLayout()
        grid.setSpacing(14)

        # Video (dark, spans 2 rows)
        self.video_card = self._card("VideoCard")
        self.video_card.layout().addWidget(self._mk_dark_title("Камера  •  YOLOv8 + gaze"))
        self.calib_label = QLabel("")
        self.calib_label.setStyleSheet(
            "background:#FFF4A3; color:#1A1A1A; font-weight:800; font-size:15px;"
            "border-radius:14px; padding:10px;")
        self.calib_label.setAlignment(Qt.AlignCenter)
        self.calib_label.setVisible(False)
        self.video_card.layout().addWidget(self.calib_label)
        self.video = QLabel("Ожидание камеры…")
        self.video.setObjectName("VideoLabel")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(560, 420)
        self.video_card.layout().addWidget(self.video, 1)
        self.vision_line = QLabel("pose: —   gaze: —   persons: —")
        self.vision_line.setStyleSheet("color:#BBBBBB;")
        self.video_card.layout().addWidget(self.vision_line)
        grid.addWidget(self.video_card, 0, 0, 2, 1)

        # EEG (purple)
        self.eeg_card = self._card("EegCard")
        self.eeg_card.layout().addWidget(self._mk("CardTitle", "EEG  •  Внимание (Muse 1.3)"))
        self.focus_wrap, self.focus_bar = self._bar("FocusBar", "Фокус")
        self.stress_wrap, self.stress_bar = self._bar("StressBar", "Стресс")
        self.alpha_lbl = QLabel("α/β/θ: —")
        self.eeg_card.layout().addWidget(self.focus_wrap)
        self.eeg_card.layout().addWidget(self.stress_wrap)
        self.eeg_card.layout().addWidget(self.alpha_lbl)
        self.artifact_lbl = QLabel("артефакты: —")
        self.eeg_card.layout().addWidget(self.artifact_lbl)
        grid.addWidget(self.eeg_card, 0, 1)

        # Fatigue (green)
        self.fat_card = self._card("FatigueCard")
        self.fat_card.layout().addWidget(self._mk("CardTitle", "Усталость / Fatigue"))
        row = QHBoxLayout()
        self.ring = FatigueRing()
        row.addWidget(self.ring)
        right = QVBoxLayout()
        self.fatigue_wrap, self.fatigue_bar = self._bar("FatigueBar", "Fatigue")
        self.risk_big = QLabel("RISK 0")
        self.risk_big.setObjectName("RiskBig")
        self.risk_level = QLabel("OK")
        self.risk_level.setObjectName("CardTitle")
        right.addWidget(self.fatigue_wrap)
        right.addWidget(self.risk_big)
        right.addWidget(self.risk_level)
        row.addLayout(right, 1)
        self.fat_card.layout().addLayout(row)
        grid.addWidget(self.fat_card, 0, 2)

        # Alerts (yellow)
        self.alerts_card = self._card("AlertsCard")
        self.alerts_card.layout().addWidget(self._mk("CardTitle", "Нарушения / Алерты"))
        self.feed = QListWidget()
        self.feed.setMinimumHeight(200)
        self.alerts_card.layout().addWidget(self.feed, 1)
        grid.addWidget(self.alerts_card, 1, 1, 1, 2)

        # Exam card (visible only while a server session is active)
        self.exam_card = self._card("ExamCard")
        self.exam_card.layout().addWidget(self._mk("CardTitle", "Экзамен"))
        self.exam_title = QLabel("—")
        self.exam_card.layout().addWidget(self.exam_title)
        self.exam_timer = QLabel("")
        self.exam_timer.setObjectName("CardTitle")
        self.exam_card.layout().addWidget(self.exam_timer)
        from PySide6.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(180)
        self.exam_body = QWidget()
        self.exam_body.setLayout(QVBoxLayout())
        scroll.setWidget(self.exam_body)
        self.exam_card.layout().addWidget(scroll, 1)
        self.btn_submit = QPushButton("✔  Отправить ответы")
        self.btn_submit.setObjectName("BtnSnap")
        self.btn_submit.clicked.connect(self._submit_exam)
        self.exam_card.layout().addWidget(self.btn_submit)
        self.exam_card.setVisible(False)
        grid.addWidget(self.exam_card, 2, 0, 1, 3)

        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(2, 2)
        lay.addLayout(grid, 1)

        # Controls
        ctrl = QFrame()
        ctrl.setObjectName("ControlCard")
        ctrl.setProperty("card", True)
        cl = QHBoxLayout(ctrl)
        self.btn_snap = QPushButton("📸  Сделать снимок")
        self.btn_snap.setObjectName("BtnSnap")
        self.btn_calib = QPushButton("🎯  Калибровка")
        self.btn_calib.setObjectName("BtnPause")
        self.btn_pause = QPushButton("⏸  Пауза")
        self.btn_pause.setObjectName("BtnPause")
        self.btn_quit = QPushButton("■  Завершить")
        self.btn_quit.setObjectName("BtnQuit")
        self.btn_snap.clicked.connect(self._snapshot)
        self.btn_calib.clicked.connect(self._recalibrate)
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_quit.clicked.connect(self.close)
        cl.addWidget(self.btn_snap)
        cl.addWidget(self.btn_calib)
        cl.addWidget(self.btn_pause)
        cl.addWidget(self.btn_quit)
        lay.addWidget(ctrl)
        self._apply_role_view()

    def _apply_role_view(self) -> None:
        """Exam-only view for students (server mode): test UI only.

        Camera, EEG, fatigue, risk and violation feed are hidden — full
        telemetry goes to the teacher/admin web panel instead.
        """
        if self.standalone:
            return
        for w in (self.video_card, self.eeg_card, self.fat_card, self.alerts_card,
                  self.btn_snap, self.btn_pause, self.btn_calib):
            try:
                w.setVisible(False)
            except Exception:
                pass

    def _mk(self, obj: str, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName(obj)
        return lbl

    def _mk_dark_title(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#FFFFFF; font-weight:700; font-size:13px;")
        return lbl

    def _bar(self, obj: str, title: str):
        """Create labeled progress bar. Returns (wrap, bar); keep both refs alive."""
        wrap = QWidget()
        box = QVBoxLayout(wrap)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(QLabel(title))
        bar = QProgressBar(wrap)
        bar.setObjectName(obj)
        bar.setRange(0, 100)
        box.addWidget(bar)
        return wrap, bar

    # -- threads --------------------------------------------------------
    def _build_workers(self) -> None:
        """Create worker objects (threads start in _start_capture)."""
        self._vworker = VideoWorker(self.vision, self.args.camera, self.args.mirror)
        self._eworker = EEGWorker(self.eeg) if self.eeg is not None else None

    def _start_capture(self) -> None:
        """Open camera + start CV/EEG threads. No-op if already running."""
        if self._vthread is not None:
            return
        self._vworker.arm()
        self._vthread = QThread(self)
        self._vworker.moveToThread(self._vthread)
        self._vthread.started.connect(self._vworker.run)
        self._vworker.frame_ready.connect(self._on_frame)
        self._vthread.start()
        if self._eworker is not None and self._ethread is None:
            self._eworker.arm()
            self._ethread = QThread(self)
            self._eworker.moveToThread(self._ethread)
            self._ethread.started.connect(self._eworker.run)
            self._eworker.eeg_ready.connect(self._on_eeg)
            self._ethread.start()
        logger.info("Capture started (camera ON)")

    def _stop_capture(self) -> None:
        """Stop threads + release camera (exam finished / idle)."""
        try:
            if self._vworker is not None:
                self._vworker.stop()
            if self._vthread is not None:
                self._vthread.quit()
                self._vthread.wait(2000)
        except Exception:
            pass
        self._vthread = None
        try:
            if self._eworker is not None:
                self._eworker.stop()
            if self._ethread is not None:
                self._ethread.quit()
                self._ethread.wait(2000)
        except Exception:
            pass
        self._ethread = None
        self.video.setText("Камера выключена — ожидайте начала экзамена")
        self.video.setPixmap(QPixmap())
        try:
            self.cam_dot.setStyleSheet("color:#999999;")
        except Exception:
            pass
        logger.info("Capture stopped (camera OFF)")

    # -- slots ----------------------------------------------------------
    def _on_frame(self, img: QImage, vstate, bgr=None) -> None:
        if self._paused:
            return
        self._last_frame = (img, vstate)
        self._last_bgr = bgr.copy() if bgr is not None else None
        # evidence ring: every 3rd frame @320px (~10fps x 9s)
        try:
            self._ring_tick += 1
            if self._ring_tick % 3 == 0 and bgr is not None:
                import cv2

                h, w = bgr.shape[:2]
                sc = 320.0 / max(1, w)
                self._ring.append(cv2.resize(bgr, (320, max(1, int(h * sc)))))
        except Exception:
            pass
        self.sync.push_vision(vstate)
        pix = QPixmap.fromImage(img).scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video.setPixmap(pix)
        self.cam_dot.setStyleSheet("color:#2E7D32;")
        # подсказка калибровки (читается из estimator, benign race)
        try:
            prompt = self.vision.pose_gaze.calib_prompt
        except Exception:
            prompt = ""
        self.calib_label.setVisible(bool(prompt))
        if prompt:
            self.calib_label.setText(f"🎯 {prompt}")
            self._calib_was = True
        elif getattr(self, "_calib_was", False):
            self._calib_was = False
            pg = self.vision.pose_gaze
            self._push_feed("LOW", "calibration",
                            f"Готово: yaw[{pg.yaw_left:.0f},{pg.yaw_right:.0f}] "
                            f"pitch[{pg.pitch_down:.0f},{pg.pitch_up:.0f}] "
                            f"gaze>{pg.gaze_thresh:.0f}")
            self.badge.setText("● Экзамен активен")
        det = vstate.detector
        self.vision_line.setText(
            f"pose: {vstate.pose.direction} yaw {vstate.pose.yaw:+.0f}° pitch {vstate.pose.pitch:+.0f}°  "
            f"gaze: {'OFF' if vstate.gaze.off_screen else 'on'} "
            f"({vstate.gaze.yaw_deg:+.0f},{vstate.gaze.pitch_deg:+.0f})  "
            f"persons: {det.person_count}  phone: {'YES' if det.phone_detected else 'no'}"
        )
        now = time.time()
        if now - self._last_video_fuse >= self._video_fuse_min_dt:
            self._last_video_fuse = now
            self._fuse(vstate, None)

    def _on_eeg(self, estate) -> None:
        if self._paused:
            return
        self.sync.push_eeg(estate)
        m = getattr(estate, "metrics", None)
        ok = bool(getattr(estate, "ok", False) and m is not None)
        self.eeg_dot.setStyleSheet("color:#2E7D32;" if ok else "color:#C62828;")
        if ok:
            self.focus_bar.setValue(int(min(m.focus / 3, 1) * 100))
            self.stress_bar.setValue(int(min(m.stress / 3, 1) * 100))
            self.fatigue_bar.setValue(int(min(m.fatigue / 6, 1) * 100))
            self.ring.set_value(m.fatigue)
            self.alpha_lbl.setText(f"α {m.alpha:.1f}  β {m.beta:.1f}  θ {m.theta:.1f}")
            flags = []
            if m.blink:
                flags.append(f"blink×{m.blink_count}")
            if m.jaw_clench:
                flags.append("JAW")
            if m.focus_drop:
                flags.append("FOCUS-DROP")
            if m.signal_loss:
                flags.append("SIG-LOSS")
            self.artifact_lbl.setText("артефакты: " + (" ".join(flags) if flags else "нет"))
        fused = self.sync.fused()
        if fused is not None:
            self._fuse(fused.vision, fused.eeg)

    def _fuse(self, vision, eeg_state) -> None:
        try:
            sec = self.security.drain_events() if self.security is not None else []
            if sec:
                self._last_sec_t = time.time()
            rs = self.risk.update(vision, eeg_state if eeg_state is not None else self._empty_eeg(), sec)
        except Exception:
            return
        self._risk_state = rs
        # SEC-индикатор: серый выкл / зелёный ок / красный недавнее нарушение
        try:
            if self.security is None or not self.security.active:
                self.sec_dot.setStyleSheet("color:#999999;")
            elif time.time() - self._last_sec_t < 10.0:
                self.sec_dot.setStyleSheet("color:#C62828;")
            else:
                self.sec_dot.setStyleSheet("color:#2E7D32;")
        except Exception:
            pass
        # На калибровке риск заморожен: показываем CAL, в лог/ленту ничего.
        if bool(getattr(vision, "calibrating", False)):
            self.risk_big.setText("RISK —")
            self.risk_level.setText("Калибровка…")
            self.risk_level.setStyleSheet("color:#6B6B6B; font-size:16px; font-weight:800;")
            self.badge.setText("🎯 Калибровка…")
            return
        self.risk_big.setText(f"RISK {rs.score:.0f}")
        self.risk_level.setText(rs.level)
        color = LEVEL_COLORS.get(rs.level, "#1A1A1A")
        self.risk_level.setStyleSheet(f"color:{color}; font-size:16px; font-weight:800;")
        if rs.events:
            # Use stored BGR frame directly (no QImage->numpy conversion).
            frame = getattr(self, "_last_bgr", None)
            try:
                self.alerts.log(rs, frame)
            except Exception:
                pass
            for ev in rs.events:
                item = QListWidgetItem(f"[{ev.level}] {ev.kind}: {ev.detail}")
                item.setForeground(Qt.white if ev.level == "CRITICAL" else Qt.black)
                item.setBackground(Qt.transparent)
                bg = {"LOW": "#E8F5E9", "MEDIUM": "#FFF3E0", "HIGH": "#FFCDD2",
                      "CRITICAL": "#7B0000"}.get(ev.level, "#FFFFFF")
                try:
                    from PySide6.QtGui import QColor

                    item.setBackground(QColor(bg))
                except Exception:
                    pass
                self.feed.addItem(item)
                self.feed.scrollToBottom()
                while self.feed.count() > 200:
                    self.feed.takeItem(0)
        self._maybe_evidence(rs)

    def _empty_eeg(self):
        class _E:
            ok = False
            metrics = None
        return _E()

    def _snapshot(self) -> None:
        try:
            frame = getattr(self, "_last_bgr", None)
            if self._risk_state is not None and frame is not None:
                path = self.alerts.snapshot(frame, self._risk_state)
                self._push_feed("LOW", "snapshot", f"Снимок сохранён: {path}")
        except Exception as exc:
            logger.warning("snapshot failed: %s", exc)

    def _push_feed(self, level: str, kind: str, detail: str) -> None:
        item = QListWidgetItem(f"[{level}] {kind}: {detail}")
        self.feed.addItem(item)
        self.feed.scrollToBottom()

    def _recalibrate(self) -> None:
        """Кнопка: заново прогнать гидовую калибровку по 5 точкам."""
        try:
            self.vision.pose_gaze.start_guided()
            self._push_feed("LOW", "calibration",
                            "Калибровка запущена: следуйте указаниям на видео")
        except Exception as exc:
            logger.warning("recalibrate failed: %s", exc)

    # -- exam session (server mode: camera OFF until teacher starts) ------
    def _poll_session(self) -> None:
        if self.standalone or self.client is None:
            return
        try:
            s = self.client.session()
        except Exception:
            return
        try:
            status, sid = s.get("status", "idle"), s.get("session_id")
            for wtxt in s.get("warnings", []) or []:
                try:
                    from PySide6.QtWidgets import QMessageBox

                    QMessageBox.warning(self, "Baqylau — предупреждение",
                                        str(wtxt) + "\nВернись к экзамену.")
                except Exception:
                    pass
                self._push_feed("HIGH", "warn", str(wtxt))
            if status == "active" and sid != self._session_id:
                self._session_id = sid
                self._session_active = True
                self._start_capture()
                self.badge.setText("● Экзамен идёт")
                self._push_feed("LOW", "session", f"Экзамен запущен: {s.get('title', '')}")
                self._load_exam()
            elif self._session_active and self._session_id is not None and status != "active":
                self._session_active = False
                self._stop_capture()
                self.exam_card.setVisible(False)
                self._exam_deadline = 0.0
                self.badge.setText("■ Экзамен завершён")
                self._push_feed("LOW", "session", "Экзамен завершён преподавателем")
        except Exception as exc:
            logger.debug("session poll failed: %s", exc)

    def _load_exam(self) -> None:
        try:
            ex = self.client.exam()
            qs = ex.get("questions", [])
            self.exam_title.setText(ex.get("title", "Экзамен"))
            # clear old
            while self.exam_body.layout().count():
                it = self.exam_body.layout().takeAt(0)
                if it.widget():
                    it.widget().deleteLater()
            from PySide6.QtWidgets import QButtonGroup, QLabel, QRadioButton

            self._exam_groups = []
            for qi, q in enumerate(qs):
                qlbl = QLabel(f"{qi + 1}. {q.get('q', '')}")
                qlbl.setObjectName("ExamQ")
                qlbl.setWordWrap(True)
                self.exam_body.layout().addWidget(qlbl)
                grp = QButtonGroup(self)
                for oi, opt in enumerate(q.get("options", [])):
                    rb = QRadioButton(str(opt))
                    rb.setObjectName("ExamOpt")
                    grp.addButton(rb, oi)
                    rb.toggled.connect(lambda _c, g=grp: self._paint_opts(g))
                    self.exam_body.layout().addWidget(rb)
                self._exam_groups.append(grp)
            self._exam_deadline = (time.time() + int(ex.get("duration_min", 30)) * 60) if qs else 0.0
            self.btn_submit.setEnabled(True)
            self.exam_card.setVisible(True)
        except Exception as exc:
            logger.warning("exam load failed: %s", exc)

    @staticmethod
    def _paint_opts(grp) -> None:
        """Highlight the selected answer: filled pill + bold, rest plain."""
        try:
            checked = grp.checkedButton()
            for btn in grp.buttons():
                btn.setProperty("picked", btn is checked)
                btn.style().unpolish(btn)
                btn.style().polish(btn)
        except Exception:
            pass

    def _submit_exam(self) -> None:
        try:
            if not self._exam_groups:
                return
            ans = []
            for grp in self._exam_groups:
                ans.append(grp.checkedId())
            r = self.client.submit(ans)
            self._exam_deadline = 0.0
            self.exam_timer.setText(f"Результат: {r.get('score', '?')}/{r.get('total', '?')}")
            self.btn_submit.setEnabled(False)
            self._push_feed("LOW", "exam", f"Ответы отправлены: {r.get('score')}/{r.get('total')}")
        except Exception as exc:
            logger.warning("submit failed: %s", exc)

    def _heartbeat(self) -> None:
        """Periodic thumbnail + risk summary to teacher panel (low load)."""
        try:
            if self.standalone or self.client is None or not self._session_active:
                return
            if self._last_bgr is None:
                return
            import cv2

            h, w = self._last_bgr.shape[:2]
            scale = 320.0 / max(1, w)
            small = cv2.resize(self._last_bgr, (320, max(1, int(h * scale))))
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if not ok:
                return
            rs = self._risk_state
            viol = [{"kind": e.kind, "detail": e.detail}
                    for e in (rs.events if rs else [])][:5]
            self.client.heartbeat(float(rs.score) if rs else 0.0,
                                  str(rs.level) if rs else "OK",
                                  viol, bytes(buf.tobytes()))
        except Exception as exc:
            logger.debug("heartbeat failed: %s", exc)

    def _maybe_evidence(self, rs) -> None:
        """RISK 100 -> photo + short video clip to teacher/admin + local save."""
        try:
            if self.standalone or self.client is None or not self._session_active:
                return
            if float(getattr(rs, "score", 0)) < 100.0:
                return
            now = time.time()
            if now - self._last_evidence_t < 60.0:
                return
            self._last_evidence_t = now
            frames = [f.copy() for f in self._ring]
            bgr = self._last_bgr.copy() if self._last_bgr is not None else None
            import threading

            threading.Thread(target=self._evidence_job,
                             args=(float(rs.score), frames, bgr,
                                   int(self._session_id or 0)),
                             daemon=True).start()
        except Exception:
            pass

    def _evidence_job(self, score: float, frames: list, bgr, session_id: int) -> None:
        if not frames and bgr is None:
            logger.warning("evidence skipped: ring buffer empty (no frames recorded)")
            return
        try:
            import cv2
            import os

            snapdir = getattr(self.alerts, "snapshot_dir", "data/logs/snapshots")
            os.makedirs(snapdir, exist_ok=True)
            tag = time.strftime("%Y%m%d_%H%M%S")
            jobs = []
            if bgr is not None:
                p = os.path.join(snapdir, f"EVIDENCE_{tag}_{int(score)}.jpg")
                cv2.imwrite(p, bgr)
                jobs.append((p, "photo"))
            if frames:
                h, w = frames[0].shape[:2]
                vp = os.path.join(snapdir, f"EVIDENCE_{tag}_{int(score)}.mp4")
                # avc1 = H.264 in MP4: plays in Chrome/Edge/Firefox.
                # (mp4v = MPEG-4 Part 2: black player in browsers.)
                wr = None
                for fourcc in ("avc1", "mp4v"):
                    try:
                        cand = cv2.VideoWriter(
                            vp, cv2.VideoWriter_fourcc(*fourcc), 10, (w, h))
                        if cand.isOpened():
                            wr = cand
                            break
                        cand.release()
                    except Exception:
                        pass
                if wr is not None:
                    for f in frames:
                        try:
                            wr.write(f)
                        except Exception:
                            pass
                    wr.release()
                    if _playable(vp):
                        jobs.append((vp, "video"))
                    else:
                        try:
                            os.remove(vp)
                        except Exception:
                            pass
                        logger.warning("evidence clip unreadable, dropped")
                else:
                    logger.warning("no video encoder available")
            for path, kind in jobs:
                try:
                    with open(path, "rb") as fh:
                        self.client.evidence(fh.read(), os.path.basename(path),
                                             kind, score, session_id)
                except Exception as exc:
                    logger.debug("evidence upload %s failed: %s", kind, exc)
            logger.info("Evidence saved+sent: %d files (R=%.0f)", len(jobs), score)
        except Exception as exc:
            logger.warning("evidence job failed: %s", exc)

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        self.badge.setText("⏸ Пауза" if self._paused else "● Экзамен активен")
        self.badge.setProperty("class", "paused" if self._paused else "")
        self.btn_pause.setText("▶  Продолжить" if self._paused else "⏸  Пауза")

    def _tick_clock(self) -> None:
        s = int(time.time() - self._t0)
        self.clock.setText(f"{s // 60:02d}:{s % 60:02d}")
        # exam countdown + autosubmit
        try:
            if self._exam_deadline and self._session_active:
                left = int(self._exam_deadline - time.time())
                if left <= 0:
                    self.exam_timer.setText("Время вышло — отправляю…")
                    self._submit_exam()
                else:
                    self.exam_timer.setText(f"Осталось: {left // 60:02d}:{left % 60:02d}")
        except Exception:
            pass

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        try:
            if hasattr(self, "_sess_timer"):
                self._sess_timer.stop()
            if hasattr(self, "_hb_timer"):
                self._hb_timer.stop()
        except Exception:
            pass
        try:
            self._stop_capture()
        except Exception:
            pass
        try:
            if self.security is not None:
                self.security.stop()
        except Exception:
            pass
        try:
            self.vision.close()
        except Exception:
            pass
        super().closeEvent(event)


def run_bento(cfg, args, vision, eeg, risk, sync, alerts, student: str = "Дмитрий Горбунов",
              security=None, client=None, standalone: bool = True) -> int:
    """Create QApplication, apply QSS, show BentoWindow. Returns exit code."""
    import os

    app = QApplication.instance() or QApplication([])
    try:
        from core.res import resource

        qss_path = resource("ui", "bento_style.qss")
    except Exception:
        qss_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bento_style.qss")
    try:
        with open(qss_path, encoding="utf-8") as f:
            app.setStyleSheet(f.read())
    except Exception as exc:
        logger.warning("QSS load failed: %s", exc)
    try:
        from .branding import app_icon

        _icon = app_icon()
        if _icon is not None and not _icon.isNull():
            app.setWindowIcon(_icon)
    except Exception:
        pass
    win = BentoWindow(cfg, args, vision, eeg, risk, sync, alerts, student, security,
                    client, standalone)
    try:
        from .branding import app_icon

        _wi = app_icon()
        if _wi is not None and not _wi.isNull():
            win.setWindowIcon(_wi)
    except Exception:
        pass
    win.show()
    return app.exec()


def _playable(path: str) -> bool:
    """True when the clip re-opens with >=1 decodable frame. Never raises."""
    try:
        import cv2
        import os

        if not os.path.exists(path) or os.path.getsize(path) < 20_000:
            return False
        cap = cv2.VideoCapture(path)
        try:
            ok, _ = cap.read()
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return bool(ok) and n > 0
        finally:
            cap.release()
    except Exception:
        return False
