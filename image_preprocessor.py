#!/usr/bin/env python3
"""
Image Preprocessing Module - Handles cropping and scaling for performance optimization
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from config import ScreenConfig

class ImagePreprocessor:
    """Handles image preprocessing for performance optimization"""

    def __init__(self, config: ScreenConfig):
        """
        Initialize the image preprocessor

        Args:
            config: Screen configuration containing preprocessing parameters
        """
        self.config = config
        self.original_size = (config.width, config.height)
        self.scale_factor = config.scale_factor
        self.crop_region = config.crop_region

        # Calculate effective dimensions after crop and scale
        if self.crop_region:
            self.crop_size = (self.crop_region[2], self.crop_region[3])
        else:
            self.crop_size = self.original_size

        self.processed_size = (
            int(self.crop_size[0] * self.scale_factor),
            int(self.crop_size[1] * self.scale_factor)
        )

        print(f"ImagePreprocessor initialized:")
        print(f"  Original size: {self.original_size}")
        print(f"  Crop region: {self.crop_region}")
        print(f"  Scale factor: {self.scale_factor}")
        print(f"  Final processed size: {self.processed_size}")

    def preprocess_for_detection(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for detection model (crop first, then scale)

        Args:
            frame: Original input frame

        Returns:
            Preprocessed frame optimized for detection
        """
        processed_frame = frame

        # Step 1: Crop if configured
        if self.crop_region:
            x, y, w, h = self.crop_region
            processed_frame = processed_frame[y:y+h, x:x+w]

        # Step 2: Scale down for performance
        if self.scale_factor != 1.0:
            processed_frame = cv2.resize(
                processed_frame,
                self.processed_size,
                interpolation=cv2.INTER_AREA  # Better for downscaling
            )

        return processed_frame

    def scale_coordinates_to_original(self, coordinates: Tuple[float, float]) -> Tuple[float, float]:
        """
        Scale coordinates from processed frame back to original frame coordinates

        Args:
            coordinates: (x, y) coordinates in processed frame

        Returns:
            (x, y) coordinates in original frame
        """
        x, y = coordinates

        # Scale up from processed frame
        if self.scale_factor != 1.0:
            x = x / self.scale_factor
            y = y / self.scale_factor

        # Adjust for crop offset
        if self.crop_region:
            crop_x, crop_y = self.crop_region[0], self.crop_region[1]
            x += crop_x
            y += crop_y

        return (x, y)

    def scale_coordinates_to_processed(self, coordinates: Tuple[float, float]) -> Tuple[float, float]:
        """
        Scale coordinates from original frame to processed frame coordinates

        Args:
            coordinates: (x, y) coordinates in original frame

        Returns:
            (x, y) coordinates in processed frame
        """
        x, y = coordinates

        # Adjust for crop offset
        if self.crop_region:
            crop_x, crop_y = self.crop_region[0], self.crop_region[1]
            x -= crop_x
            y -= crop_y

        # Scale down to processed frame
        if self.scale_factor != 1.0:
            x = x * self.scale_factor
            y = y * self.scale_factor

        return (x, y)

    def get_crop_region_for_display(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Get crop region coordinates for display overlay

        Returns:
            (x, y, width, height) of crop region or None if no cropping
        """
        return self.crop_region

    def get_performance_info(self) -> dict:
        """
        Get preprocessing performance information

        Returns:
            Dictionary with preprocessing configuration and performance metrics
        """
        return {
            'original_size': self.original_size,
            'processed_size': self.processed_size,
            'scale_factor': self.scale_factor,
            'crop_enabled': self.crop_region is not None,
            'crop_region': self.crop_region,
            'size_reduction_factor': (self.processed_size[0] * self.processed_size[1]) / (self.original_size[0] * self.original_size[1])
        }