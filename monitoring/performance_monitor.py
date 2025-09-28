#!/usr/bin/env python3
"""
Performance Monitoring Module - Tracks and reports system performance
"""

import time
import threading
from typing import Dict, Optional
from dataclasses import dataclass, field

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    frame_count: int = 0
    start_time: float = field(default_factory=time.time)
    last_fps_report: float = field(default_factory=time.time)

    # FPS tracking
    current_fps: float = 0.0

    # Component timings
    detection_time: float = 0.0
    capture_time: float = 0.0
    analysis_time: float = 0.0
    visualization_time: float = 0.0

    # System status
    running_time: float = 0.0

class PerformanceMonitor:
    """Monitors and tracks system performance metrics"""

    def __init__(self, report_interval: float = 5.0):
        """
        Initialize performance monitor

        Args:
            report_interval: Interval between performance reports (seconds)
        """
        self.report_interval = report_interval
        self.metrics = PerformanceMetrics()
        self.frame_times = []
        self.max_frame_history = 60  # Keep last 60 frames for FPS calculation

        # Thread safety
        self.lock = threading.Lock()

        print("PerformanceMonitor initialized")

    def start_frame(self):
        """Mark the start of a new frame"""
        with self.lock:
            self.metrics.frame_count += 1
            self._calculate_fps()

    def update_detection_time(self, detection_time: float):
        """Update detection timing"""
        with self.lock:
            self.metrics.detection_time = detection_time * 1000  # Convert to ms

    def update_capture_time(self, capture_time: float):
        """Update capture timing"""
        with self.lock:
            self.metrics.capture_time = capture_time * 1000  # Convert to ms

    def update_analysis_time(self, analysis_time: float):
        """Update analysis timing"""
        with self.lock:
            self.metrics.analysis_time = analysis_time * 1000  # Convert to ms

    def update_visualization_time(self, viz_time: float):
        """Update visualization timing"""
        with self.lock:
            self.metrics.visualization_time = viz_time * 1000  # Convert to ms

    def _calculate_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        self.frame_times.append(current_time)

        # Keep only recent frames
        if len(self.frame_times) > self.max_frame_history:
            self.frame_times.pop(0)

        # Calculate FPS from recent frames
        if len(self.frame_times) >= 2:
            time_span = self.frame_times[-1] - self.frame_times[0]
            if time_span > 0:
                self.metrics.current_fps = (len(self.frame_times) - 1) / time_span

        # Update running time
        self.metrics.running_time = current_time - self.metrics.start_time

    def should_report(self) -> bool:
        """Check if it's time to report performance"""
        current_time = time.time()
        return current_time - self.metrics.last_fps_report >= self.report_interval

    def report_performance(self) -> str:
        """Generate performance report"""
        with self.lock:
            self.metrics.last_fps_report = time.time()

            report = f"""Performance Report:
  FPS: {self.metrics.current_fps:.1f}
  Frame Count: {self.metrics.frame_count}
  Running Time: {self.metrics.running_time:.1f}s
  Detection: {self.metrics.detection_time:.1f}ms
  Capture: {self.metrics.capture_time:.1f}ms
  Analysis: {self.metrics.analysis_time:.1f}ms
  Visualization: {self.metrics.visualization_time:.1f}ms"""

            return report

    def get_current_metrics(self) -> Dict:
        """Get current performance metrics as dictionary"""
        with self.lock:
            return {
                'frame_count': self.metrics.frame_count,
                'fps': self.metrics.current_fps,
                'running_time': self.metrics.running_time,
                'detection_time_ms': self.metrics.detection_time,
                'capture_time_ms': self.metrics.capture_time,
                'analysis_time_ms': self.metrics.analysis_time,
                'visualization_time_ms': self.metrics.visualization_time
            }

    def get_status_info(self) -> Dict:
        """Get status information for display"""
        with self.lock:
            return {
                'frame_count': self.metrics.frame_count,
                'fps': self.metrics.current_fps,
                'detection_time': self.metrics.detection_time,
                'running_time': self.metrics.running_time
            }