"""Multi-modal rule engine + cumulative risk score R(t) in [0, 100].

Fusion table (W = 1.0 s):
  head-turn + high stress/focus-drop      -> CRITICAL
  gaze-off  + relaxed (high alpha)       -> LOW (daydreaming)
  gaze-off  + high focus                 -> HIGH (hidden cheat sheet)
  phone/object                           -> CRITICAL (vision-only)
  face-missing + eeg noise/loss          -> HIGH (tampering / left frame)
  jaw-clench + focus fluctuation         -> MEDIUM (talking/signalling)
  drowsy (high fatigue)                  -> MEDIUM

R(t) = clip(R(t-1) * exp(-lambda*dt) + sum(points), 0, 100).
Level thresholds come from FusionConfig.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

LEVELS = ("OK", "LOW", "MEDIUM", "HIGH", "CRITICAL")


@dataclass
class RiskEvent:
    t: float
    kind: str
    level: str
    points: float
    detail: str


@dataclass
class RiskState:
    t: float
    score: float
    level: str
    events: list = field(default_factory=list)


class RiskEngine:
    """Накопительный риск с кулдауном: один и тот же kind не чаще раза в cooldown.

    Без кулдауна update() дёргается 30 раз/сек и drowsy +12/кадр мгновенно
    даёт R=100 «ничего не делая». С кулдауном 2-3с очки капают дозированно,
    а decay успевает гасить фон.
    """

    def __init__(self,
                 focus_high: float = 1.2,
                 stress_high: float = 1.5,
                 fatigue_high: float = 2.5,
                 decay_lambda: float = 0.18,
                 points: dict | None = None,
                 thresholds: dict | None = None,
                 cooldown_sec: float = 2.0,
                 head_sustain_sec: float = 3.0,
                 gaze_sustain_sec: float = 4.0) -> None:
        self.focus_high = focus_high
        self.stress_high = stress_high
        self.fatigue_high = fatigue_high
        self.decay = decay_lambda
        self.cooldown = max(0.0, float(cooldown_sec))
        self.head_sustain = max(0.0, float(head_sustain_sec))
        self.gaze_sustain = max(0.0, float(gaze_sustain_sec))
        # Пер-kind кулдауны: фон (drowsy/daydream) капает редко, критичное — быстро.
        self._cooldowns = {
            "phone": 1.5, "phone_raised": 1.5, "reading_device": 2.0,
            "multi_person": 2.0, "head_stress": 2.0, "head_turn": 3.0,
            "head_sustained": 2.0,
            "gaze_focus": 3.0, "gaze_relaxed": 8.0, "gaze_off": 4.0, "gaze_sustained": 4.0,
            "face_missing": 2.0, "jaw": 3.0, "drowsy": 8.0,
            "shortcut": 2.0, "focus_lost": 2.0,
        }
        self.points = points or {
            "phone_critical": 65.0, "phone_raised_critical": 65.0,
            "reading_device_critical": 40.0,
            "head_stress_critical": 30.0,
            "head_turn_medium": 5.0, "head_sustained_high": 25.0,
            "gaze_focus_high": 12.0, "gaze_relaxed_low": 2.0,
            "gaze_off_medium": 3.0, "gaze_sustained_high": 10.0,
            "face_missing_high": 20.0, "jaw_medium": 10.0, "drowsy_medium": 3.0,
            "shortcut_high": 20.0, "focus_lost_high": 25.0,
        }
        self.thresholds = thresholds or {"LOW": 20.0, "MEDIUM": 45.0, "HIGH": 65.0, "CRITICAL": 85.0}
        self._score = 0.0
        self._last_t: float | None = None
        self._last_fire: dict[str, float] = {}
        self._head_since: float | None = None  # непрерывный отворот головы с…
        self._gaze_since: float | None = None  # непрерывный off-screen с…
        self.history: list[RiskEvent] = []

    @property
    def score(self) -> float:
        return self._score

    def _level(self, score: float) -> str:
        if score >= self.thresholds["CRITICAL"]:
            return "CRITICAL"
        if score >= self.thresholds["HIGH"]:
            return "HIGH"
        if score >= self.thresholds["MEDIUM"]:
            return "MEDIUM"
        if score >= self.thresholds["LOW"]:
            return "LOW"
        return "OK"

    def update(self, vision, eeg_state, security=None) -> RiskState:
        """Fuse one synchronized pair + workstation-security events.

        Args:
            vision: VisionState (has .calibrating guard).
            eeg_state: EEGState or None-safe stub.
            security: iterable of SecurityEvent (kind/detail) drained from
                SecurityEngine, or None.
        """
        now = time.time()
        dt = 0.25 if self._last_t is None else max(0.0, now - self._last_t)
        self._last_t = now
        # exponential decay so brief distractions fade
        self._score *= math.exp(-self.decay * dt)
        # На калибровке риск не растёт и события не пишутся — только затухание.
        if bool(getattr(vision, "calibrating", False)):
            return RiskState(t=now, score=self._score,
                             level=self._level(self._score), events=[])

        events: list[RiskEvent] = []
        try:
            det = getattr(vision, "detector", None)
            pose = getattr(vision, "pose", None)
            gaze = getattr(vision, "gaze", None)
            m = getattr(eeg_state, "metrics", None)
            eeg_ok = bool(getattr(eeg_state, "ok", False) and m is not None)

            head_turned = bool(getattr(vision, "head_turned", False))
            eyes_off = bool(getattr(vision, "eyes_off", False))
            face_missing = bool(getattr(vision, "face_missing", False))
            phone = bool(det.phone_detected) if det is not None else False
            multi = bool(det.multi_person) if det is not None else False

            focus = float(m.focus) if eeg_ok else 0.0
            stress = float(m.stress) if eeg_ok else 0.0
            fatigue = float(m.fatigue) if eeg_ok else 0.0
            focus_drop = bool(getattr(m, "focus_drop", False)) if eeg_ok else False
            jaw = bool(getattr(m, "jaw_clench", False)) if eeg_ok else False
            blink = bool(getattr(m, "blink", False)) if eeg_ok else False
            loss = bool(getattr(m, "signal_loss", False)) if eeg_ok else False
            _ = blink

            def add(kind, level, pts, detail):
                cd = self._cooldowns.get(kind, self.cooldown)
                last = self._last_fire.get(kind, 0.0)
                if now - last < cd:
                    return
                self._last_fire[kind] = now
                events.append(RiskEvent(now, kind, level, pts, detail))
                self._score += pts
                self.history.append(events[-1])

            P = self.points
            # непрерывность состояний (для эскалации залипаний)
            if head_turned:
                if self._head_since is None:
                    self._head_since = now
            else:
                self._head_since = None
            if eyes_off:
                if self._gaze_since is None:
                    self._gaze_since = now
            else:
                self._gaze_since = None
            # 1. prohibited device -> CRITICAL regardless of EEG
            if phone:
                labels = ",".join(d.label for d in det.prohibited[:3])
                add("phone", "CRITICAL", P["phone_critical"], f"Prohibited device: {labels}")
            # 1b. phone held up to the screen -> photographing (§2.1)
            if bool(getattr(det, "phone_raised", False)):
                add("phone_raised", "CRITICAL", P.get("phone_raised_critical", 65.0),
                    "Phone raised to screen — possible photographing")
            # multi-person is at least HIGH
            if multi:
                add("multi_person", "HIGH", P["face_missing_high"], f"Persons={det.person_count}")
            # 2. head turned + stress/focus-drop -> CRITICAL
            if head_turned and eeg_ok and (stress > self.stress_high or focus_drop):
                add("head_stress", "CRITICAL", P["head_stress_critical"],
                    f"{pose.direction} stress={stress:.2f} focus={focus:.2f}")
            elif head_turned:
                add("head_turn", "MEDIUM", P.get("head_turn_medium", 5.0),
                    f"{getattr(pose, 'direction', '?')} (no EEG confirm)")
            # 2b. sustained look-away escalates even without EEG: staring
            # off-screen for seconds is cheating-like whatever the headband says.
            if (head_turned and self._head_since is not None
                    and now - self._head_since >= self.head_sustain):
                add("head_sustained", "HIGH", P.get("head_sustained_high", 15.0),
                    f"Looking away {now - self._head_since:.0f}s: {getattr(pose, 'direction', '?')}")
            # 3/4. gaze-off fused with cognitive state
            if eyes_off and eeg_ok:
                if focus > self.focus_high:
                    add("gaze_focus", "HIGH", P["gaze_focus_high"],
                        f"Gaze off + focus {focus:.2f} (possible cheat sheet)")
                else:
                    add("gaze_relaxed", "LOW", P["gaze_relaxed_low"],
                        f"Gaze off + low focus {focus:.2f} (daydreaming)")
            elif eyes_off:
                add("gaze_off", "MEDIUM", P.get("gaze_off_medium", 5.0), "Gaze off-screen")
            # 4b. sustained eyes-off (reading with eyes only, head still)
            if (eyes_off and self._gaze_since is not None
                    and now - self._gaze_since >= self.gaze_sustain):
                add("gaze_sustained", "HIGH", P.get("gaze_sustained_high", 12.0),
                    f"Eyes off-screen {now - self._gaze_since:.0f}s")
            # 5. face missing + signal loss/noise -> HIGH tamper
            if face_missing and (loss or not eeg_ok):
                add("face_missing", "HIGH", P["face_missing_high"],
                    "Face missing + EEG loss (tampering/left frame)")
            elif face_missing:
                add("face_missing", "MEDIUM", P["drowsy_medium"], "Face missing in frame")
            # 6. jaw clench + focus fluctuation -> MEDIUM
            if jaw and (focus_drop or (eeg_ok and stress > self.stress_high)):
                add("jaw", "MEDIUM", P["jaw_medium"],
                    f"Jaw EMG + focus {focus:.2f} (possible talking)")
            # 7. drowsiness
            if eeg_ok and fatigue > self.fatigue_high:
                add("drowsy", "MEDIUM", P["drowsy_medium"], f"Fatigue {fatigue:.2f}")
            # 8. down-gaze + phone in frame -> reading from device (§2.2)
            looking_down = (
                (pose is not None and getattr(pose, "direction", "") == "Looking Down")
                or (gaze is not None and float(getattr(gaze, "pitch_deg", 0.0)) < -15.0)
            )
            if phone and looking_down:
                add("reading_device", "CRITICAL", P.get("reading_device_critical", 40.0),
                    "Looking down at phone — possible reading")
            # 9. workstation security (§2.3)
            if security:
                for sev in security:
                    kind = getattr(sev, "kind", "")
                    detail = getattr(sev, "detail", str(sev))
                    if kind == "shortcut":
                        add("shortcut", "HIGH", P.get("shortcut_high", 20.0), detail)
                    elif kind == "focus_lost":
                        add("focus_lost", "HIGH", P.get("focus_lost_high", 25.0), detail)
        except Exception as exc:
            logger.exception("risk update failed: %s", exc)

        self._score = float(min(100.0, max(0.0, self._score)))
        level = self._level(self._score)
        return RiskState(t=now, score=self._score, level=level, events=events)
