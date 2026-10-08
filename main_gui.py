"""Desktop launcher: Soft Bento Grid GUI (PySide6).

Usage:
    python main_gui.py --simulate-eeg --student "Dmitry Gorbunov"
    python main_gui.py --no-eeg --camera 0
    python main_gui.py --notch 60

Requires: pip install PySide6
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("gui")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Proctor Bento GUI")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.45)
    p.add_argument("--simulate-eeg", action="store_true")
    p.add_argument("--eeg", action="store_true", help="real Muse headband via BrainFlow")
    p.add_argument("--no-eeg", action="store_true", help="disable EEG branch (default)")
    p.add_argument("--notch", type=float, default=50.0)
    p.add_argument("--log-dir", default="data/logs")
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--student", default="Дмитрий Горбунов")
    p.add_argument("--gaze-thresh", type=float, default=None, help="порог off-screen, град (def 20)")
    p.add_argument("--yaw", type=float, default=None, help="симм. порог yaw, град (def 28)")
    p.add_argument("--calib-sec", type=float, default=None, help="секунд автокалибровки (def ~2)")
    p.add_argument("--no-calib", action="store_true", help="без калибровки (по умолчанию)")
    p.add_argument("--quick-calib", action="store_true",
                   help="быстрая фронтальная калибровка ~2с при старте")
    p.add_argument("--guided-calib", action="store_true",
                   help="гидовая калибровка по 5 точкам при старте (~8с)")
    p.add_argument("--no-security", action="store_true", help="disable workstation protection")
    p.add_argument("--server", default="", help="Baqylau Web URL (student mode), e.g. http://192.168.1.10:5050")
    p.add_argument("--login", default="", help="student login (server mode)")
    p.add_argument("--password", default="", help="student password (server mode)")
    p.add_argument("--thumb-interval", type=float, default=4.0, help="thumbnail seconds (server mode)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        import PySide6  # noqa: F401
    except ImportError:
        print("PySide6 is not installed. Run: pip install PySide6")
        return 2

    from config import AppConfig
    from core.eeg.eeg_engine import EEGEngine
    from core.fusion.alert_manager import AlertManager
    from core.fusion.risk_engine import RiskEngine
    from core.fusion.synchronizer import Synchronizer
    from core.vision.vision_engine import VisionEngine
    from ui.bento_app import run_bento

    cfg = AppConfig()
    gaze_th = args.gaze_thresh if args.gaze_thresh is not None else cfg.vision.gaze_off_thresh_deg
    yaw = args.yaw if args.yaw is not None else 28.0
    # По умолчанию — без калибровки; opt-in: --quick-calib / --guided-calib / кнопка 🎯.
    if args.quick_calib:
        calib = int(args.calib_sec * 30) if args.calib_sec is not None else cfg.vision.calib_frames
    else:
        calib = 0
    vision = VisionEngine(
        model_path=args.model, conf=args.conf,
        prohibited_labels=cfg.vision.prohibited_labels,
        yaw_left=-abs(yaw), yaw_right=abs(yaw),
        pitch_down=cfg.vision.pitch_down_thresh, pitch_up=cfg.vision.pitch_up_thresh,
        gaze_thresh=gaze_th,
        smooth_alpha=cfg.vision.smooth_alpha,
        persist_frames=cfg.vision.persist_frames,
        calib_frames=calib,
        eye_yaw_gain=cfg.vision.eye_yaw_gain,
        eye_pitch_gain=cfg.vision.eye_pitch_gain,
        head_mix=cfg.vision.head_mix,
        raise_y_frac=cfg.vision.raise_y_frac,
        raised_min_h_frac=cfg.vision.raised_min_h_frac,
    )
    eeg = None
    if args.eeg or args.simulate_eeg:
        eeg = EEGEngine(fs=cfg.eeg.sampling_rate, window_sec=cfg.eeg.window_sec,
                        low=cfg.eeg.bandpass_low, high=cfg.eeg.bandpass_high,
                        order=cfg.eeg.bandpass_order, notch=args.notch,
                        simulate=args.simulate_eeg,
                        focus_drop_delta=cfg.eeg.focus_drop_delta,
                        blink_thresh=cfg.eeg.blink_thresh_uv,
                        gamma_thresh=cfg.eeg.emg_gamma_thresh_uv2)
    risk = RiskEngine(focus_high=cfg.eeg.focus_high_thresh, stress_high=cfg.eeg.stress_high_thresh,
                      fatigue_high=cfg.eeg.fatigue_high_thresh, decay_lambda=cfg.fusion.decay_lambda,
                      points=dict(cfg.fusion.event_points), thresholds=dict(cfg.fusion.risk_thresholds),
                      cooldown_sec=cfg.fusion.cooldown_sec,
                      head_sustain_sec=cfg.fusion.head_sustain_sec,
                      gaze_sustain_sec=cfg.fusion.gaze_sustain_sec)
    sync = Synchronizer(window_sec=cfg.fusion.fusion_window_sec)
    alerts = AlertManager(log_dir=args.log_dir,
                          snapshot_dir=os.path.join(args.log_dir, "snapshots"))
    security = None
    if cfg.security.enabled and not args.no_security:
        from core.security.security_engine import SecurityEngine

        security = SecurityEngine(
            enable_hotkeys=cfg.security.enable_hotkeys,
            enable_focus=cfg.security.enable_focus,
            suppress=cfg.security.suppress_hotkeys,
            grace_sec=cfg.security.focus_grace_sec,
            allowed_titles=cfg.security.allowed_titles,
            screenshot_dir=os.path.join(args.log_dir, "snapshots"),
            do_screenshot=cfg.security.screenshot_on_violation,
        )
        logger.info("Security: %s", security.start())
    logger.info("Face backend: %s | EEG: %s", vision.pose_gaze.backend,
                "off" if eeg is None else type(eeg.stream).__name__)
    if args.guided_calib:
        vision.pose_gaze.start_guided()
        logger.info("Guided calibration started (5 points, follow prompts)")
    # student auth page: CLI creds or interactive dialog (server mode default)
    client, standalone, student_name = None, True, args.student
    if args.server:
        from core.net.client import BaqylauClient

        standalone = False
        client = BaqylauClient(args.server)
        if not args.login or not client.login(args.login, args.password):
            print("Baqylau: server login failed (need --login/--password).")
            return 2
        student_name = client.name or args.student
        logger.info("Logged in as %s (camera OFF until exam starts)", student_name)
    else:
        from PySide6.QtWidgets import QApplication, QDialog

        from ui.login_dialog import LoginDialog

        _qa = QApplication.instance() or QApplication([])
        dlg = LoginDialog(server="", username=args.login or "")
        res = dlg.exec()
        if dlg.local_mode:
            standalone = True
            logger.info("Local mode (no server, camera works immediately)")
        elif res == QDialog.Accepted and dlg.client is not None:
            standalone = False
            client = dlg.client
            student_name = client.name or args.student
            logger.info("Logged in as %s (camera OFF until exam starts)", student_name)
        else:
            return 0
    return run_bento(cfg, args, vision, eeg, risk, sync, alerts,
                     student=student_name, security=security,
                     client=client, standalone=standalone)


if __name__ == "__main__":
    raise SystemExit(main())
