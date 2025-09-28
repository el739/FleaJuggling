# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FleaJuggling is a real-time AI-powered juggling game bot that uses computer vision and machine learning to automatically play a ball juggling game. The system captures the screen, detects player and ball positions using YOLO object detection, predicts ball trajectories, and automatically controls the game to keep the ball in the air.

## Tech Stack

- **Python 3** with PyTorch and Ultralytics YOLO (YOLOv8/v11)
- **Computer Vision**: OpenCV for image processing
- **Game Control**: Win32 API (pyautogui, win32api) for precise keyboard control
- **Data Science**: NumPy, Matplotlib, Pandas for analysis and visualization

## Development Commands

### Data Pipeline
```bash
# Prepare dataset for YOLO training (converts to proper format, 80/20 split)
python prepare_dataset.py

# Train YOLO model (100 epochs, batch size 8, YOLOv8n base)
python train.py
```

### Testing and Debugging
```bash
# Test detection on single image
python detect_image.py screenshots/screenshot_xxx.png

# Test detection on video with output
python detect_video.py input_video.mp4 -o output_detected.mp4

# Run main AI controller (real-time game automation)
python juggling_ai.py
```

### Runtime Controls
- Press 'q' to quit safely
- Press 'r' to start/stop recording
- Press 's' to save debug screenshots
- Ctrl+C for emergency stop

## Architecture Overview

The system follows a **real-time perception-action loop** with three main layers:

1. **Perception**: `juggling_ai.py` handles screen capture (15 FPS) and YOLO detection
2. **Cognition**: `trajectory_predictor.py` and `game_analyzer.py` handle physics prediction and decision making
3. **Action**: `game_controller.py` executes game controls via Win32 API

### Core State Machine
```
WAITING → TRACKING → MOVING → READY → JUGGLING
```

### Key Components
- **`juggling_ai.py`**: Main orchestrator with real-time loop
- **`trajectory_predictor.py`**: Physics-based ball trajectory prediction with multi-ball tracking
- **`game_analyzer.py`**: Game state analysis and movement strategy
- **`game_controller.py`**: Precise keyboard control using Win32 API
- **`prepare_dataset.py`**: Data preprocessing for YOLO training
- **`train.py`**: YOLO model training pipeline

## Game Configuration

- **Screen Resolution**: 1920x1080 fullscreen
- **Juggle Zone**: Y coordinates 567-750
- **Player Speed**: 134.6 pixels/second
- **Dash Distance**: 264 pixels
- **Ball Physics**: Parabolic trajectory with specific equation coefficients
- **Detection Classes**: "hero" (player), "ordinary" (ball)

## Dataset Structure

```
dataset/
├── images/train/    # Training images
├── images/val/      # Validation images
├── labels/train/    # YOLO format labels
├── labels/val/      # YOLO format labels
└── data.yaml       # YOLO dataset configuration
```

Training results stored in `runs/detect/train/weights/` with `best.pt` being the optimized model.

## Development Notes

- The system uses **modular architecture** with clear separation between perception, cognition, and action
- **Multi-ball tracking** with noise filtering for robust detection
- **Real-time performance monitoring** with FPS and latency metrics
- All game controls use **Win32 API** for precise timing and reliability
- **Emergency safety mechanisms** included for development safety
- Video recording capabilities for analysis and debugging