#!/usr/bin/env python3
"""
Example configuration for image preprocessing optimization
Run this to see the performance optimization in action
"""

from config import GameConfig

# Create a custom configuration with image preprocessing
config = GameConfig()

# Example 1: Enable scaling only (reduce to 50% size for 4x performance improvement)
config.screen.scale_factor = 0.5

# Example 2: Enable cropping + scaling (crop game area only, then scale down)
# Crop to game area only (remove UI elements at top/bottom)
config.screen.crop_region = (0, 100, 1920, 880)  # (x, y, width, height)
config.screen.scale_factor = 0.6

# Example 3: Conservative optimization (minimal quality loss)
# config.screen.scale_factor = 0.8
# config.screen.crop_region = None  # No cropping

print("Configuration Examples:")
print("=======================")
print(f"Original screen size: {config.screen.width}x{config.screen.height}")
print(f"Scale factor: {config.screen.scale_factor}")
print(f"Crop region: {config.screen.crop_region}")
print(f"Final model input size: {config.screen.model_input_width}x{config.screen.model_input_height}")

# Calculate performance improvement
original_pixels = config.screen.width * config.screen.height
processed_pixels = config.screen.model_input_width * config.screen.model_input_height
improvement_factor = original_pixels / processed_pixels

print(f"Performance improvement: {improvement_factor:.2f}x faster")
print(f"Memory reduction: {100 * (1 - processed_pixels/original_pixels):.1f}%")