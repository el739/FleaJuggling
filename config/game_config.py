#!/usr/bin/env python3
"""
Configuration Management - Centralized configuration for all modules
"""

from dataclasses import dataclass, field
from typing import Dict

@dataclass
class DetectionConfig:
    """Detection module configuration"""
    # YOLO Detection
    confidence_threshold: float = 0.5
    model_path: str = "runs/detect/train/weights/best.pt"

    # Class IDs
    player_class_id: int = 0  # hero
    ball_class_id: int = 1    # ordinary

    # Performance
    max_detection_history: int = 30
    detection_timeout_ms: float = 100.0

@dataclass
class ScreenConfig:
    """Screen and display configuration"""
    # Screen dimensions
    width: int = 1920
    height: int = 1080

    # Display settings
    display_width: int = 960
    display_height: int = 540
    window_name: str = "Juggling AI"

    # Frame rate
    fps: int = 15
    frame_time: float = field(init=False)

    def __post_init__(self):
        self.frame_time = 1.0 / self.fps

@dataclass
class JuggleZoneConfig:
    """Juggle zone boundaries"""
    min_y: int = 465  # 最高位置（y坐标较小）
    max_y: int = 750  # 最低位置（y坐标较大）

    # Visual colors (BGR format)
    zone_line_color: tuple = (0, 255, 255)  # Cyan
    zone_line_thickness: int = 2

@dataclass
class PlayerConfig:
    """Player movement configuration"""
    # Movement speeds (pixels/second)
    move_speed: float = field(init=False)  # Calculated from base values
    base_pixels: int = 610
    base_frames: int = 68
    base_fps: int = 15

    # Dash settings
    dash_distance: int = 264  # pixels
    dash_duration: float = 0.2  # seconds

    # Position tolerance
    position_tolerance: int = 30  # pixels
    juggle_position_tolerance: int = 60  # pixels

    def __post_init__(self):
        self.move_speed = (self.base_pixels / self.base_frames) * self.base_fps

@dataclass
class ControlConfig:
    """Game control configuration"""
    # Key mappings (Virtual Key Codes)
    key_mapping: Dict[str, int] = field(default_factory=lambda: {
        'a': 0x41,       # A key
        'd': 0x44,       # D key
        'b': 0x42,       # B key
        'w': 0x57,       # W key
        'ctrl': 0xA2,    # Left Ctrl key
    })

    # Action durations
    juggle_duration: float = 0.1
    dash_duration: float = 0.2

    # Control settings
    action_cooldown: float = 0.05
    reaction_time: float = 0.1

@dataclass
class TrajectoryConfig:
    """Ball trajectory prediction configuration"""
    # Physics constants
    gravity_coefficient: float = 1.438131e-02  # Fixed gravity coefficient

    # Tracking parameters
    max_ball_speed: float = 50.0  # pixels/frame
    min_stable_frames: int = 15
    max_missing_frames: int = 5
    max_history_frames: int = 50
    recent_points_count: int = 8

    # Direction detection
    direction_threshold: float = 0.5
    min_movement_threshold: float = 1.0  # pixels
    direction_history_count: int = 10

    # Scoring weights
    stability_weight: float = 0.7
    recency_weight: float = 0.3

@dataclass
class AnalysisConfig:
    """Game analysis configuration"""
    # Confidence thresholds
    min_prediction_confidence: float = 0.7
    min_action_confidence: float = 0.5

    # Confidence levels
    confidence_no_data: float = 0.0
    confidence_waiting: float = 0.1
    confidence_no_prediction: float = 0.1
    confidence_insufficient_time: float = 0.2
    confidence_moving: float = 0.8
    confidence_moving_risky: float = 0.6
    confidence_tracking: float = 0.7
    confidence_ready: float = 0.9

    # Timing
    juggle_timing_window: float = 0.15  # seconds

@dataclass
class VisualizationConfig:
    """Visualization configuration"""
    # Colors (BGR format)
    colors: Dict[str, tuple] = field(default_factory=lambda: {
        'player': (0, 255, 0),      # Green
        'ball': (255, 0, 0),        # Blue
        'trajectory': (255, 255, 0), # Yellow
        'landing': (0, 0, 255),     # Red
        'juggle_zone': (0, 255, 255), # Cyan
        'text': (255, 255, 255),    # White
        'ball_colors': [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    })

    # Drawing settings
    player_radius: int = 15
    ball_radius: int = 10
    trajectory_thickness: int = 3
    landing_point_radius: int = 10

    # Text settings
    font: int = 0  # cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.6
    text_thickness: int = 2
    text_line_height: int = 25
    text_y_offset: int = 60

@dataclass
class MonitoringConfig:
    """Performance monitoring configuration"""
    # Reporting
    report_interval: float = 5.0  # seconds
    max_frame_history: int = 60

    # Timeouts and limits
    thread_join_timeout: float = 2.0
    capture_retry_delay: float = 0.1

@dataclass
class RecordingConfig:
    """Video recording configuration"""
    # Video settings
    fourcc: str = 'mp4v'
    default_fps: int = 15
    frame_size: tuple = (960, 540)

    # File naming
    timestamp_format: str = "%Y%m%d_%H%M%S"
    recording_prefix: str = "juggling_ai_recording"
    screenshot_prefix: str = "debug_screenshot"

@dataclass
class GameConfig:
    """Complete game configuration - combines all sub-configs"""
    # Sub-configurations
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    juggle_zone: JuggleZoneConfig = field(default_factory=JuggleZoneConfig)
    player: PlayerConfig = field(default_factory=PlayerConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)

    # Legacy compatibility properties
    @property
    def SCREEN_WIDTH(self) -> int:
        return self.screen.width

    @property
    def SCREEN_HEIGHT(self) -> int:
        return self.screen.height

    @property
    def FPS(self) -> int:
        return self.screen.fps

    @property
    def FRAME_TIME(self) -> float:
        return self.screen.frame_time

    @property
    def JUGGLE_MIN_Y(self) -> int:
        return self.juggle_zone.min_y

    @property
    def JUGGLE_MAX_Y(self) -> int:
        return self.juggle_zone.max_y

    @property
    def PLAYER_SPEED(self) -> float:
        return self.player.move_speed

    @property
    def DASH_DISTANCE(self) -> int:
        return self.player.dash_distance

    @property
    def KEYS(self) -> Dict[str, str]:
        """Legacy key mapping format"""
        return {
            'left': 'a',
            'right': 'd',
            'left_dash': 'a+ctrl',
            'right_dash': 'd+ctrl',
            'juggle': 'b'
        }

# Global configuration instance
config = GameConfig()

def get_config() -> GameConfig:
    """Get the global configuration instance"""
    return config

def update_config(**kwargs):
    """Update configuration values"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            # Try to update sub-configurations
            for sub_config_name in ['detection', 'screen', 'juggle_zone', 'player',
                                  'control', 'trajectory', 'analysis', 'visualization',
                                  'monitoring', 'recording']:
                sub_config = getattr(config, sub_config_name)
                if hasattr(sub_config, key):
                    setattr(sub_config, key, value)
                    break
            else:
                print(f"Warning: Unknown configuration key: {key}")
