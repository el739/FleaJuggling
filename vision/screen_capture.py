#!/usr/bin/env python3
"""
Screen Capture Module - Handles screen capture functionality
"""

import cv2
import time
import pyautogui
import numpy as np
from typing import Optional

class ScreenCapture:
    """Screen capture handler"""

    def __init__(self):
        """Initialize screen capture"""
        self.capture_count = 0
        self.total_capture_time = 0.0
        print("ScreenCapture initialized")

    def capture(self) -> Optional[np.ndarray]:
        """
        Capture the current screen

        Returns:
            numpy array representing the captured frame, or None if capture fails
        """
        try:
            start_time = time.time()

            # Capture using pyautogui
            screenshot = pyautogui.screenshot()
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Update performance metrics
            capture_time = time.time() - start_time
            self.capture_count += 1
            self.total_capture_time += capture_time

            return frame

        except Exception as e:
            print(f"Error capturing screen: {e}")
            return None

    def get_average_capture_time(self) -> float:
        """Get average capture time in milliseconds"""
        if self.capture_count == 0:
            return 0.0
        return (self.total_capture_time / self.capture_count) * 1000

    def get_statistics(self) -> dict:
        """Get capture statistics"""
        return {
            'total_captures': self.capture_count,
            'avg_capture_time_ms': self.get_average_capture_time(),
            'total_capture_time': self.total_capture_time
        }