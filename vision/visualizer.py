#!/usr/bin/env python3
"""
Visualization Module - Handles drawing and visualization of detection results
"""

import cv2
import time
import numpy as np
from typing import Optional, Tuple, List, Dict
from detection import DetectionResult
from trajectory_predictor import TrajectoryPredictor, GameConfig

class Visualizer:
    """Handles visualization of game state and detection results"""

    def __init__(self, config: GameConfig):
        """
        Initialize visualizer

        Args:
            config: Game configuration
        """
        self.config = config
        self.colors = {
            'player': (0, 255, 0),      # Green
            'ball': (255, 0, 0),        # Blue
            'trajectory': (255, 255, 0), # Yellow
            'landing': (0, 0, 255),     # Red
            'juggle_zone': (0, 255, 255), # Cyan
            'text': (255, 255, 255)     # White
        }

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
            cv2.circle(vis_frame, (int(x), int(y)), 15, self.colors['player'], -1)
            cv2.putText(vis_frame, "PLAYER", (int(x) + 20, int(y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['player'], 2)

        # Draw balls
        for i, ball_pos in enumerate(detection.ball_positions):
            x, y = ball_pos
            cv2.circle(vis_frame, (int(x), int(y)), 10, self.colors['ball'], -1)
            cv2.putText(vis_frame, f"BALL_{i}", (int(x) + 15, int(y) - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['ball'], 2)

        return vis_frame

    def draw_juggle_zone(self, frame: np.ndarray) -> np.ndarray:
        """Draw the juggle zone boundaries"""
        vis_frame = frame.copy()

        # Draw juggle zone lines
        cv2.line(vis_frame, (0, self.config.JUGGLE_MIN_Y),
                (self.config.SCREEN_WIDTH, self.config.JUGGLE_MIN_Y),
                self.colors['juggle_zone'], 2)
        cv2.line(vis_frame, (0, self.config.JUGGLE_MAX_Y),
                (self.config.SCREEN_WIDTH, self.config.JUGGLE_MAX_Y),
                self.colors['juggle_zone'], 2)

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
        y_offset = 60
        line_height = 25

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
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)

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