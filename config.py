"""Central configuration for the multi-modal proctoring system."""
from dataclasses import dataclass, field


@dataclass
class EEGConfig:
    """Muse 1.3 (MU-01 / Spark) streaming configuration."""

    sampling_rate: int = 256  # Muse via BrainFlow resampled rate
    n_channels: int = 4
    channel_names: tuple = ("TP9", "AF7", "AF8", "TP10")
    window_sec: float = 2.0
    step_sec: float = 0.25
    bandpass_low: float = 1.0
    bandpass_high: float = 40.0
    bandpass_order: int = 4
    notch_freq: float = 50.0  # set 60.0 for US/JP mains
    notch_q: float = 30.0
    # Metric thresholds (tune per cohort during calibration)
    focus_high_thresh: float = 1.2
    focus_drop_delta: float = 0.4
    stress_high_thresh: float = 1.5
    fatigue_high_thresh: float = 2.5
    blink_thresh_uv: float = 75.0
    emg_gamma_thresh_uv2: float = 15.0
    simulate: bool = False  # True forces synthetic EEG (no hardware)


@dataclass
class VisionConfig:
    """YOLOv8 + head-pose/gaze configuration."""

    model_path: str = "yolov8n.pt"
    conf_thresh: float = 0.45
    prohibited_labels: tuple = (
        "cell phone",
        "mobile phone",
        "laptop",
        "tablet",
        "book",
        "remote",
    )
    yaw_left_thresh: float = -28.0
    yaw_right_thresh: float = 28.0
    pitch_down_thresh: float = -20.0
    pitch_up_thresh: float = 22.0
    gaze_off_thresh_deg: float = 20.0
    eye_yaw_gain: float = 40.0
    eye_pitch_gain: float = 32.0
    head_mix: float = 0.0  # 0 = взгляд только по глазам (стабильно); 1.0 = +голова (шумит)
    raise_y_frac: float = 0.45  # телефон выше этой доли кадра = поднят
    raised_min_h_frac: float = 0.08  # ...и выше этой доли высоты = близко/крупно
    smooth_alpha: float = 0.35
    persist_frames: int = 10
    calib_frames: int = 60  # ~2 c при 30 fps; 0 = без автокалибровки
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480


@dataclass
class FusionConfig:
    """Temporal fusion + risk scoring configuration."""

    fusion_window_sec: float = 1.0
    decay_lambda: float = 0.18  # мягче: фон гаснет быстро
    cooldown_sec: float = 2.0  # один kind не чаще раза в N сек
    head_sustain_sec: float = 3.0  # отворот головы дольше -> HIGH сам по себе
    gaze_sustain_sec: float = 4.0  # взгляд мимо дольше -> HIGH сам по себе
    critical_snapshot: bool = True
    risk_thresholds: dict = field(
        default_factory=lambda: {"LOW": 20.0, "MEDIUM": 45.0, "HIGH": 65.0, "CRITICAL": 85.0}
    )
    # Points added per fused event (before decay)
    event_points: dict = field(
        default_factory=lambda: {
            "phone_critical": 65.0,
            "phone_raised_critical": 65.0,
            "reading_device_critical": 40.0,
            "head_stress_critical": 30.0,
            "head_turn_medium": 5.0,
            "head_sustained_high": 25.0,
            "gaze_focus_high": 12.0,
            "gaze_relaxed_low": 2.0,
            "gaze_off_medium": 3.0,
            "gaze_sustained_high": 10.0,
            "face_missing_high": 20.0,
            "jaw_medium": 10.0,
            "drowsy_medium": 3.0,
            "shortcut_high": 20.0,
            "focus_lost_high": 25.0,
        }
    )


@dataclass
class SecurityConfig:
    """Workstation protection, case §2.3 (hotkeys + exam-window focus)."""

    enabled: bool = True
    enable_hotkeys: bool = True
    enable_focus: bool = True
    suppress_hotkeys: bool = True  # False = только фиксировать, не блокировать
    focus_grace_sec: float = 2.0
    allowed_titles: tuple = ("proctor", "экзамен", "exam")
    screenshot_on_violation: bool = True


@dataclass
class AppConfig:
    eeg: EEGConfig = field(default_factory=EEGConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    log_dir: str = "data/logs"
    snapshot_dir: str = "data/logs/snapshots"
