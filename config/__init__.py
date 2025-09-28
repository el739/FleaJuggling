#!/usr/bin/env python3
"""
Configuration module package
"""

from .game_config import (
    GameConfig,
    DetectionConfig,
    ScreenConfig,
    JuggleZoneConfig,
    PlayerConfig,
    ControlConfig,
    TrajectoryConfig,
    AnalysisConfig,
    VisualizationConfig,
    MonitoringConfig,
    RecordingConfig,
    get_config,
    update_config,
    config
)

__all__ = [
    'GameConfig',
    'DetectionConfig',
    'ScreenConfig',
    'JuggleZoneConfig',
    'PlayerConfig',
    'ControlConfig',
    'TrajectoryConfig',
    'AnalysisConfig',
    'VisualizationConfig',
    'MonitoringConfig',
    'RecordingConfig',
    'get_config',
    'update_config',
    'config'
]