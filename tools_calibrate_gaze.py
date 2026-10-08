"""Gaze calibration helper: смотри прямо в экран 10 секунд.

Показывает живые yaw/pitch/gaze и в конце советует пороги.

Usage:
    python tools_calibrate_gaze.py --camera 0 --mirror
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2

from core.vision.pose_gaze import PoseGazeEstimator


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--sec", type=float, default=10.0)
    a = p.parse_args()

    est = PoseGazeEstimator(calib_frames=0, persist_frames=1)  # сырые значения без нуля
    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {a.camera}")
        return 2
    print("Смотри ПРЯМО в экран. Сбор 10 с... (q — выход)")
    yaws, pitchs, mags = [], [], []
    t0 = time.time()
    while time.time() - t0 < a.sec:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if a.mirror:
            frame = cv2.flip(frame, 1)
        pose, gaze = est.estimate(frame)
        import math

        mag = math.hypot(gaze.yaw_deg, gaze.pitch_deg)
        if pose.face_found:
            yaws.append(pose.yaw)
            pitchs.append(pose.pitch)
            mags.append(mag)
        print(f"\ryaw={pose.yaw:+6.1f} pitch={pose.pitch:+6.1f} "
              f"gx={gaze.yaw_deg:+6.1f} gy={gaze.pitch_deg:+6.1f} mag={mag:5.1f} "
              f"backend={est.backend}   ", end="")
        cv2.imshow("calib (q=quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    print()
    cap.release()
    cv2.destroyAllWindows()
    if not yaws:
        print("Лицо не найдено. Проверь свет/камеру.")
        return 1

    def med(v):
        return statistics.median(v)

    print(f"Фронтальный ноль: yaw0={med(yaws):+.1f} pitch0={med(pitchs):+.1f} "
          f"gaze_mag~{med(mags):.1f}")
    print(f"Разброс yaw: p5={sorted(yaws)[len(yaws)//20]:+.1f} "
          f"p95={sorted(yaws)[-len(yaws)//20]:+.1f}")
    rec_yaw = max(18.0, abs(med(yaws)) + (sorted(yaws)[-len(yaws)//20] - sorted(yaws)[len(yaws)//20]) / 2 + 12)
    rec_gaze = max(20.0, med(mags) + 15)
    print(f"Рекомендую: --yaw {rec_yaw:.0f} --gaze-thresh {rec_gaze:.0f}")
    print("Если всё равно ложит — подними ещё на 5–10.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
