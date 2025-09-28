#!/usr/bin/env python3
"""
Video Recording Module - Handles video recording functionality
"""

import cv2
import time
from typing import Optional

class VideoRecorder:
    """Handles video recording of the AI gameplay"""

    def __init__(self, config=None):
        """Initialize video recorder"""
        self.config = config
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.is_recording = False
        self.output_path: Optional[str] = None
        # 从配置获取帧大小和FPS，如果没有配置则使用默认值
        self.frame_size = getattr(config, 'frame_size', (960, 540)) if config else (960, 540)  # Default display size
        self.fps = getattr(config, 'default_fps', 15) if config else 15

    def start_recording(self, output_path: Optional[str] = None) -> bool:
        """
        Start recording video

        Args:
            output_path: Path for output video file

        Returns:
            True if recording started successfully
        """
        if self.is_recording:
            print("Recording is already active")
            return False

        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"juggling_ai_recording_{timestamp}.mp4"

        self.output_path = output_path

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, self.frame_size)

        if not self.video_writer.isOpened():
            print(f"Failed to open video writer for {output_path}")
            return False

        self.is_recording = True
        print(f"Started recording to: {output_path}")
        return True

    def record_frame(self, frame) -> bool:
        """
        Record a frame to video

        Args:
            frame: Frame to record

        Returns:
            True if frame was recorded successfully
        """
        if not self.is_recording or self.video_writer is None:
            return False

        try:
            # Ensure frame is the correct size
            if frame.shape[:2] != (self.frame_size[1], self.frame_size[0]):
                frame = cv2.resize(frame, self.frame_size)

            self.video_writer.write(frame)
            return True
        except Exception as e:
            print(f"Error recording frame: {e}")
            return False

    def stop_recording(self) -> bool:
        """
        Stop recording video

        Returns:
            True if recording stopped successfully
        """
        if not self.is_recording:
            print("No active recording to stop")
            return False

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        self.is_recording = False
        print(f"Recording saved to: {self.output_path}")
        self.output_path = None
        return True

    def get_status(self) -> dict:
        """Get recording status"""
        return {
            'is_recording': self.is_recording,
            'output_path': self.output_path,
            'frame_size': self.frame_size,
            'fps': self.fps
        }
