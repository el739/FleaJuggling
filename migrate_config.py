#!/usr/bin/env python3
"""
Configuration Migration Script - Updates all modules to use centralized configuration
"""

import os
import re
from pathlib import Path

def update_imports_in_file(file_path: str, old_imports: list, new_imports: list):
    """Update imports in a Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Replace old imports with new ones
        for old_import, new_import in zip(old_imports, new_imports):
            content = content.replace(old_import, new_import)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Updated {file_path}")
        return True
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def replace_hardcoded_values(file_path: str, replacements: dict):
    """Replace hardcoded values with config references"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Apply replacements
        for old_value, new_value in replacements.items():
            content = re.sub(old_value, new_value, content)

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Replaced hardcoded values in {file_path}")
            return True
        else:
            print(f"No changes needed in {file_path}")
            return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main migration function"""
    print("=== Configuration Migration Script ===")

    # Define replacement patterns for common hardcoded values
    common_replacements = {
        r'\b0\.5\b(?=.*conf)': 'self.config.confidence_threshold',  # confidence threshold
        r'\b30\b(?=.*position.*tolerance)': 'self.config.position_tolerance',  # position tolerance
        r'\b60\b(?=.*position.*tolerance)': 'self.config.juggle_position_tolerance',  # juggle tolerance
        r'\b0\.1\b(?=.*reaction.*time)': 'self.config.reaction_time',  # reaction time
        r'\b0\.15\b(?=.*timing.*window)': 'self.config.juggle_timing_window',  # juggle timing
        r'\b0\.05\b(?=.*cooldown)': 'self.config.action_cooldown',  # action cooldown
        r'\b1920\b': 'self.config.screen.width',  # screen width
        r'\b1080\b': 'self.config.screen.height',  # screen height
        r'\b465\b': 'self.config.juggle_zone.min_y',  # juggle min y
        r'\b750\b': 'self.config.juggle_zone.max_y',  # juggle max y
        r'\b264\b': 'self.config.player.dash_distance',  # dash distance
        r'\b15\b(?=.*fps)': 'self.config.screen.fps',  # FPS
        r'\b(960,\s*540)\b': '(self.config.screen.display_width, self.config.screen.display_height)',  # display size
    }

    # Files to process
    files_to_update = [
        'vision/visualizer.py',
        'vision/video_recorder.py',
        'monitoring/performance_monitor.py',
        'game_analyzer.py',
        'game_controller.py',
        'trajectory_predictor.py',
        'juggling_ai_modular.py'
    ]

    # Update files
    updated_count = 0
    for file_path in files_to_update:
        if Path(file_path).exists():
            if replace_hardcoded_values(file_path, common_replacements):
                updated_count += 1
        else:
            print(f"File not found: {file_path}")

    print(f"\nMigration completed. Updated {updated_count} files.")
    print("\nNOTE: Some manual adjustments may still be needed for complex cases.")
    print("Please test the system after migration.")

if __name__ == "__main__":
    main()