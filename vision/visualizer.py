#!/usr/bin/env python3
"""
Visualization Module - Handles drawing and visualization of detection results
"""

import cv2
import time
import numpy as np
from typing import Optional, Tuple, List, Dict
from detection import DetectionResult
from trajectory_predictor import TrajectoryPredictor
from config import VisualizationConfig, JuggleZoneConfig, ScreenConfig

class Visualizer:
    """Handles visualization of game state and detection results"""

    def __init__(self, vis_config: VisualizationConfig = None,
                 zone_config: JuggleZoneConfig = None,
                 screen_config: ScreenConfig = None):
        """
        Initialize visualizer

        Args:
            vis_config: Visualization configuration
            zone_config: Juggle zone configuration
            screen_config: Screen configuration
        """
        self.vis_config = vis_config or VisualizationConfig()
        self.zone_config = zone_config or JuggleZoneConfig()
        self.screen_config = screen_config or ScreenConfig()

    def draw_detection_results(self, frame: np.ndarray, detection: DetectionResult) -> np.ndarray:
        """
        Draw detection results on frame

        Args:
            frame: Input frame
            detection: Detection results

        Returns:
            Frame with detection results drawn
        """
        vis_frame = frame.copy()

        # Draw player
        if detection.player_pos:
            x, y = detection.player_pos
            cv2.circle(vis_frame, (int(x), int(y)), self.vis_config.player_radius, self.vis_config.colors['player'], -1)
            cv2.putText(vis_frame, "PLAYER", (int(x) + 20, int(y)),
                       self.vis_config.font, self.vis_config.font_scale, self.vis_config.colors['player'], self.vis_config.text_thickness)

        # Draw balls
        for i, ball_pos in enumerate(detection.ball_positions):
            x, y = ball_pos
            cv2.circle(vis_frame, (int(x), int(y)), self.vis_config.ball_radius, self.vis_config.colors['ball'], -1)
            cv2.putText(vis_frame, f"BALL_{i}", (int(x) + 15, int(y) - 15),
                       self.vis_config.font, self.vis_config.font_scale, self.vis_config.colors['ball'], self.vis_config.text_thickness)

        return vis_frame

    def draw_juggle_zone(self, frame: np.ndarray) -> np.ndarray:
        """Draw the juggle zone boundaries"""
        vis_frame = frame.copy()

        # Draw juggle zone lines
        cv2.line(vis_frame, (0, self.zone_config.min_y),
                (self.screen_config.width, self.zone_config.min_y),
                self.zone_config.zone_line_color, self.zone_config.zone_line_thickness)
        cv2.line(vis_frame, (0, self.zone_config.max_y),
                (self.screen_config.width, self.zone_config.max_y),
                self.zone_config.zone_line_color, self.zone_config.zone_line_thickness)

        return vis_frame

    def draw_trajectory(self, frame: np.ndarray, predictor: TrajectoryPredictor,
                       landing_point: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Draw trajectory prediction

        Args:
            frame: Input frame
            predictor: Trajectory predictor
            landing_point: Predicted landing point

        Returns:
            Frame with trajectory drawn
        """
        vis_frame = frame.copy()

        # Use the predictor's visualization method
        vis_frame = predictor.visualize_trajectory(vis_frame, landing_point)

        return vis_frame

    def draw_status_info(self, frame: np.ndarray, status_info: Dict) -> np.ndarray:
        """
        Draw status information on frame

        Args:
            frame: Input frame
            status_info: Dictionary containing status information

        Returns:
            Frame with status info drawn
        """
        vis_frame = frame.copy()
        y_offset = self.vis_config.text_y_offset
        line_height = self.vis_config.text_line_height

        # Prepare info lines
        info_lines = []

        # Basic info
        if 'frame_count' in status_info:
            info_lines.append(f"Frame: {status_info['frame_count']}")
        if 'state' in status_info:
            info_lines.append(f"State: {status_info['state']}")
        if 'fps' in status_info:
            info_lines.append(f"FPS: {status_info['fps']:.1f}")
        if 'detection_time' in status_info:
            info_lines.append(f"Detection: {status_info['detection_time']:.1f}ms")

        # Tracking info
        if 'tracking_info' in status_info:
            tracking = status_info['tracking_info']
            info_lines.append(f"Active Ball: {tracking.get('active_ball_id', 'None')}")

        # Landing info
        if 'landing_point' in status_info:
            point = status_info['landing_point']
            info_lines.append(f"Landing: ({point[0]:.0f}, {point[1]:.0f})")

        if 'time_to_landing' in status_info:
            info_lines.append(f"Time: {status_info['time_to_landing']:.2f}s")

        # Draw info lines
        for i, line in enumerate(info_lines):
            y_pos = y_offset + i * line_height
            cv2.putText(vis_frame, line, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, self.vis_config.font_scale, self.vis_config.colors['text'], self.vis_config.text_thickness)

        return vis_frame

    def create_display_frame(self, frame: np.ndarray, detection: DetectionResult,
                           predictor: TrajectoryPredictor, status_info: Dict,
                           display_size: Tuple[int, int] = (960, 540)) -> np.ndarray:
        """
        Create a complete visualization frame

        Args:
            frame: Input frame
            detection: Detection results
            predictor: Trajectory predictor
            status_info: Status information
            display_size: Target display size

        Returns:
            Complete visualization frame
        """
        # Start with juggle zone
        vis_frame = self.draw_juggle_zone(frame)

        # Add detection results
        vis_frame = self.draw_detection_results(vis_frame, detection)

        # Add trajectory if available
        landing_point = status_info.get('landing_point')
        vis_frame = self.draw_trajectory(vis_frame, predictor, landing_point)

        # Add status information
        vis_frame = self.draw_status_info(vis_frame, status_info)

        # Resize for display
        display_frame = cv2.resize(vis_frame, display_size)

        return display_frame
