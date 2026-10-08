"""Entry point: dual-modal proctoring (YOLOv8 vision + Muse 1.3 EEG).

Usage:
    python main.py --simulate-eeg --camera 0 --model yolov8n.pt
    python main.py --no-eeg            # vision-only (original behaviour)
    python main.py --notch 60          # 60 Hz mains countries

Keys in HUD window:  q -> quit,  s -> manual snapshot.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2

from config import AppConfig
from core.eeg.eeg_engine import EEGEngine
from core.fusion.alert_manager import AlertManager
from core.fusion.risk_engine import RiskEngine
from core.fusion.synchronizer import Synchronizer
from core.vision.vision_engine import VisionEngine
from ui.dashboard import draw_hud

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Multi-modal exam proctor")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.45)
    p.add_argument("--simulate-eeg", action="store_true", help="synthetic EEG, no headband needed")
    p.add_argument("--eeg", action="store_true", help="real Muse headband via BrainFlow")
    p.add_argument("--no-eeg", action="store_true", help="disable EEG branch (default)")
    p.add_argument("--notch", type=float, default=50.0)
    p.add_argument("--log-dir", default="data/logs")
    p.add_argument("--mirror", action="store_true", help="selfie-view flip to match original repo")
    p.add_argument("--gaze-thresh", type=float, default=None, help="off-screen deg (def 20)")
    p.add_argument("--yaw", type=float, default=None, help="symmetric yaw deg (def 28)")
    p.add_argument("--no-calib", action="store_true", help="без калибровки (по умолчанию)")
    p.add_argument("--quick-calib", action="store_true",
                   help="быстрая фронтальная калибровка ~2с при старте")
    p.add_argument("--guided-calib", action="store_true",
                   help="гидовая калибровка по 5 точкам при старте (~8с)")
    p.add_argument("--no-security", action="store_true", help="disable workstation protection")
    return p.parse_args(argv)


def build(cfg: AppConfig, args) -> tuple:
    gaze_th = getattr(args, "gaze_thresh", None) or cfg.vision.gaze_off_thresh_deg
    yaw = getattr(args, "yaw", None) or 28.0
    # По умолчанию — без калибровки (сразу дефолтные пороги); opt-in флагами.
    calib = cfg.vision.calib_frames if getattr(args, "quick_calib", False) else 0
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
    if cfg.security.enabled and not getattr(args, "no_security", False):
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
    return vision, eeg, risk, sync, alerts, security


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = AppConfig()
    vision, eeg, risk, sync, alerts, security = build(cfg, args)
    if getattr(args, "guided_calib", False):
        vision.pose_gaze.start_guided()
        logger.info("Guided calibration: follow console prompts")
    elif getattr(args, "quick_calib", False):
        logger.info("Quick frontal calibration (~2s): look straight at the screen")

    if eeg is not None:
        eeg.start()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error("Cannot open camera %s", args.camera)
        return 2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.vision.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.vision.frame_height)

    eeg_mode = "off" if eeg is None else type(eeg.stream).__name__
    sec_mode = "off"
    if security is not None:
        sec_mode = str(security.start())
    logger.info("Proctor running. q=quit s=snapshot | EEG=%s | SEC=%s", eeg_mode, sec_mode)
    last_eeg = 0.0
    last_fuse = 0.0
    last_prompt = ""
    risk_state = None
    eeg_state = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                logger.warning("Empty frame; retrying")
                time.sleep(0.05)
                continue
            if args.mirror:
                frame = cv2.flip(frame, 1)

            v_state = vision.process(frame)
            sync.push_vision(v_state)
            prompt = vision.pose_gaze.calib_prompt
            if prompt != last_prompt:
                last_prompt = prompt
                if prompt:
                    logger.info("🎯 Калибровка: %s", prompt)
                else:
                    pg = vision.pose_gaze
                    logger.info("Калибровка готова: yaw[%.0f,%.0f] pitch[%.0f,%.0f] gaze>%.0f",
                                pg.yaw_left, pg.yaw_right, pg.pitch_down, pg.pitch_up,
                                pg.gaze_thresh)

            # EEG at ~4 Hz (every 0.25 s)
            now = time.time()
            if eeg is not None and (now - last_eeg) >= cfg.eeg.step_sec:
                last_eeg = now
                eeg_state = eeg.tick()
                sync.push_eeg(eeg_state)

            fused = sync.fused()
            if fused is not None and (now - last_fuse) >= 0.5:
                last_fuse = now
                sec = security.drain_events() if security is not None else []
                risk_state = risk.update(fused.vision, fused.eeg, sec)
                alerts.log(risk_state, frame)
                for ev in risk_state.events:
                    logger.info("[%s] %s: %s (R=%.0f)", ev.level, ev.kind, ev.detail, risk_state.score)

            annotated = vision.annotate(frame.copy(), v_state)
            hud = draw_hud(annotated, v_state, eeg_state, risk_state,
                           eeg_connected=(eeg.connected if eeg else False))
            cv2.imshow("Proctor HUD (q=quit, s=snapshot)", hud)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                path = alerts.snapshot(frame, risk_state) if risk_state else alerts.snapshot(frame, type(
                    "R", (), {"score": 0})())
                logger.info("Manual snapshot: %s", path)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        vision.close()
        if eeg is not None:
            eeg.stop()
        if security is not None:
            security.stop()
        logger.info("Final risk score: %.1f", risk.score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
