#!/usr/bin/env python3
"""
Object Detection Module - Handles YOLO-based player and ball detection
"""

import time
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from ultralytics import YOLO

class DetectionResult:
    """Detection result container"""
    def __init__(self, player_pos: Optional[Tuple[float, float]] = None,
                 ball_positions: List[Tuple[float, float]] = None,
                 detection_time: float = 0.0, confidence_scores: Dict = None):
        self.player_pos = player_pos
        self.ball_positions = ball_positions or []
        self.detection_time = detection_time
        self.confidence_scores = confidence_scores or {}

class ObjectDetector:
    """YOLO-based object detector for players and balls"""

    def __init__(self, model_path: str = "runs/detect/train/weights/best.pt",
                 confidence_threshold: float = 0.5):
        """
        Initialize the object detector

        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

        # Performance tracking
        self.detection_times = []
        self.max_history_length = 30

        print(f"ObjectDetector initialized with model: {model_path}")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Detect objects in the frame

        Args:
            frame: Input image frame

        Returns:
            DetectionResult containing player and ball positions
        """
        start_time = time.time()

        # Run YOLO detection
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)

        player_pos = None
        ball_positions = []
        confidence_scores = {}

        if results[0].boxes is not None:
            boxes = results[0].boxes.data.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                cls = int(cls)

                # Calculate center point
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                if cls == 0:  # hero (player)
                    player_pos = (center_x, center_y)
                    confidence_scores['player'] = conf
                elif cls == 1:  # ordinary (ball)
                    ball_positions.append((center_x, center_y))
                    if 'balls' not in confidence_scores:
                        confidence_scores['balls'] = []
                    confidence_scores['balls'].append(conf)

        # Record detection time
        detection_time = time.time() - start_time
        self._update_performance_metrics(detection_time)

        return DetectionResult(player_pos, ball_positions, detection_time, confidence_scores)

    def _update_performance_metrics(self, detection_time: float):
        """Update performance tracking metrics"""
        self.detection_times.append(detection_time)
        if len(self.detection_times) > self.max_history_length:
            self.detection_times.pop(0)

    def get_average_detection_time(self) -> float:
        """Get average detection time in milliseconds"""
        if not self.detection_times:
            return 0.0
        return np.mean(self.detection_times) * 1000

    def get_statistics(self) -> Dict:
        """Get detector statistics"""
        return {
            'avg_detection_time_ms': self.get_average_detection_time(),
            'confidence_threshold': self.confidence_threshold,
            'model_loaded': self.model is not None,
            'recent_detections': len(self.detection_times)
        }