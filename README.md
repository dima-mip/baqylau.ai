# Baqylau — локальный AI-прокторинг (YOLO + MediaPipe + защита окружения)

Dual-modal refactor of `aungkhantmyat/The-Online-Exam-Proctor`.

## Structure

```
proctoring_system/
├── config.py
├── main.py          # OpenCV HUD (console)
├── main_gui.py      # Soft Bento Grid desktop HUD (PySide6)
├── tools_calibrate_gaze.py
├── requirements.txt
├── core/vision/   detector.py, pose_gaze.py, vision_engine.py
├── core/eeg/      stream.py, signal_proc.py, metrics.py, eeg_engine.py
├── core/fusion/   synchronizer.py, risk_engine.py, alert_manager.py
├── ui/dashboard.py
├── ui/bento_app.py
└── data/logs/
```

## Quickstart

```bash
pip install -r requirements.txt
pip install PySide6
# Bento GUI, simulated EEG (dev without headband):
python main_gui.py --simulate-eeg
# vision-only (no headband):
python main.py --no-eeg
# real Muse 1.3 via BrainFlow BLE:
python main.py --notch 50
```

## Student app (Baqylau Web + desktop)

Teacher PC (light web panel): `cd baqylau` then `python -m server.app`
→ `http://<lan-ip>:5050` (admin/admin123 first run).

Student PCs (heavy work stays here): the app opens a **login page**
(server + login + password, remembered). Camera turns on only after the
teacher starts the exam; thumbnails + risk fly to the panel every few
seconds; RISK 100 saves photo + video evidence locally and uploads it.

Keys (console): `q` quit, `s` snapshot. GUI: buttons Snapshot / 🎯 Калибровка / Pause / Quit.

## Workstation protection (case §2.3)

- Hotkey lockdown: Alt+Tab, Ctrl+C/V/X/A/T/W, Win, PrtScn, Alt+F4 — suppressed
  via `keyboard` + counted as HIGH violations (`core/security/hotkeys.py`).
- Focus guard: leaving the exam window > grace seconds → HIGH + screen capture
  (`core/security/focus.py`). Baseline = foreground window at start.
- Off: `--no-security`. Deps install: `pip install keyboard pygetwindow pyautogui`.

## Post-exam report

```bash
python tools_report.py --student "Dmitry Gorbunov"
# -> data/logs/report_*.html : timeline + snapshot gallery + verdict
```

## Calibration (gaze) — opt-in

By default the app starts instantly with fixed thresholds (no waiting).
If gaze feels off for a particular student/room:

- GUI: 🎯 button — guided 5-point calibration (~8s: center/left/right/up/down),
  sets personal zero AND thresholds. Quick frontal: `--quick-calib`.
- Console: `--guided-calib` / `--quick-calib`.
- During calibration risk is frozen (`CAL`), no events are logged.
- Diagnose raw numbers: `python tools_calibrate_gaze.py --mirror`.

## EEG indices

- Focus `β/(θ+α)`, Stress `highβ/α`, Fatigue `(θ+α)/β`
- Blink: `|AF7/AF8| > 75 uV`; Jaw: temporal gamma spike; 1–40 Hz bandpass + 50/60 Hz notch, Welch PSD.

## Fusion rules

| Vision | EEG | Flag |
|---|---|---|
| head-turn | high stress / focus-drop | CRITICAL |
| gaze-off | relaxed | LOW daydream |
| gaze-off | high focus | HIGH cheat-sheet |
| phone/tablet | any | CRITICAL |
| face-missing | signal loss | HIGH tamper |
| jaw EMG | focus swing | MEDIUM talking |

Risk `R(t)∈[0,100]` with exponential decay `exp(-λ·dt)`.
