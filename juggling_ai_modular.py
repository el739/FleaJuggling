#!/usr/bin/env python3
"""
Refactored Juggling AI - Modular architecture with clear separation of concerns
"""

import cv2
import time
import threading
from pathlib import Path
from typing import Optional

# Import modular components
from detection import ObjectDetector
from vision import ScreenCapture, Visualizer, VideoRecorder
from monitoring import PerformanceMonitor
from config import GameConfig
from game_analyzer import GameStateAnalyzer, DecisionMaker
from game_controller import GameController

class JugglingAI:
    """Main AI orchestrator - coordinates all system components"""

    def __init__(self, model_path: str = "runs/detect/train/weights/best.pt"):
        """
        Initialize the Juggling AI system

        Args:
            model_path: Path to YOLO model weights
        """
        print("Initializing Modular Juggling AI...")

        # Initialize configuration
        self.config = GameConfig()

        # Initialize core components
        self._initialize_components(model_path)

        # Runtime control
        self.is_running = False
        self.control_thread: Optional[threading.Thread] = None

        # Display settings
        self.show_visualization = True

        print("Modular Juggling AI initialized successfully!")

    def _initialize_components(self, model_path: str):
        """Initialize all system components"""
        # Detection system
        self.detector = ObjectDetector(model_path, self.config.detection)

        # Vision system
        self.screen_capture = ScreenCapture()
        self.visualizer = Visualizer(
            self.config.visualization,
            self.config.juggle_zone,
            self.config.screen
        )
        self.video_recorder = VideoRecorder(self.config.recording)

        # Game logic
        self.analyzer = GameStateAnalyzer(self.config)
        self.controller = GameController()
        self.decision_maker = DecisionMaker(self.analyzer, self.controller)

        # Monitoring
        self.performance_monitor = PerformanceMonitor()

    def _control_loop(self):
        """Main control loop - coordinated execution of all components"""
        target_frame_time = 1.0 / self.config.FPS
        print(f"Starting control loop at {self.config.FPS} FPS")

        while self.is_running:
            loop_start_time = time.time()
            self.performance_monitor.start_frame()

            try:
                # 1. Capture screen
                capture_start = time.time()
                frame = self.screen_capture.capture()
                if frame is None:
                    time.sleep(0.1)
                    continue
                self.performance_monitor.update_capture_time(time.time() - capture_start)

                # 2. Detect objects
                detection_start = time.time()
                detection_result = self.detector.detect(frame)
                self.performance_monitor.update_detection_time(time.time() - detection_start)

                # 3. Update game analyzer
                analysis_start = time.time()
                current_time = time.time()
                frame_id = self.performance_monitor.metrics.frame_count

                # Update player position
                if detection_result.player_pos:
                    self.analyzer.update_player_position(
                        detection_result.player_pos[0],
                        detection_result.player_pos[1],
                        current_time
                    )

                # Update ball positions
                if detection_result.ball_positions:
                    self.analyzer.trajectory_predictor.add_ball_detections(
                        detection_result.ball_positions, frame_id
                    )

                # Analyze game state and make decisions
                analysis = self.analyzer.analyze_game_state()
                self.decision_maker.execute_decision(analysis)
                self.performance_monitor.update_analysis_time(time.time() - analysis_start)

                # 4. Visualization
                if self.show_visualization:
                    viz_start = time.time()
                    self._handle_visualization(frame, detection_result, analysis)
                    self.performance_monitor.update_visualization_time(time.time() - viz_start)

                # 5. Performance reporting
                if self.performance_monitor.should_report():
                    report = self.performance_monitor.report_performance()
                    print(report)

            except Exception as e:
                print(f"Error in control loop: {e}")
                continue

            # Frame rate control
            loop_time = time.time() - loop_start_time
            sleep_time = max(0, target_frame_time - loop_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _handle_visualization(self, frame, detection_result, analysis):
        """Handle visualization and user input"""
        # Prepare status information
        status_info = self._prepare_status_info(analysis)

        # Create visualization
        display_frame = self.visualizer.create_display_frame(
            frame, detection_result, self.analyzer.trajectory_predictor, status_info
        )

        # Show frame
        cv2.imshow('Juggling AI', display_frame)

        # Record frame if recording
        if self.video_recorder.is_recording:
            self.video_recorder.record_frame(display_frame)

        # Handle keyboard input
        self._handle_keyboard_input(display_frame)

    def _prepare_status_info(self, analysis) -> dict:
        """Prepare status information for visualization"""
        status_info = self.performance_monitor.get_status_info()
        status_info['state'] = self.analyzer.current_state.value

        # Add tracking information
        tracking_info = self.analyzer.trajectory_predictor.get_tracking_info()
        status_info['tracking_info'] = tracking_info

        # Add landing point if available
        landing_point = self.analyzer.trajectory_predictor.predict_landing_in_juggle_zone()
        if landing_point:
            status_info['landing_point'] = landing_point
            time_to_landing = self.analyzer.trajectory_predictor.estimate_time_to_landing(landing_point[0])
            if time_to_landing:
                status_info['time_to_landing'] = time_to_landing

        return status_info

    def _handle_keyboard_input(self, display_frame):
        """Handle keyboard input for control"""
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Quit requested")
            self.stop()
        elif key == ord('r'):
            self._toggle_recording()
        elif key == ord('s'):
            self._save_screenshot(display_frame)

    def _toggle_recording(self):
        """Toggle video recording"""
        if not self.video_recorder.is_recording:
            self.video_recorder.start_recording()
        else:
            self.video_recorder.stop_recording()

    def _save_screenshot(self, display_frame):
        """Save current frame as screenshot"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        screenshot_path = f"debug_screenshot_{timestamp}.png"
        cv2.imwrite(screenshot_path, display_frame)
        print(f"Screenshot saved: {screenshot_path}")

    def start(self):
        """Start the AI system"""
        if self.is_running:
            print("AI is already running")
            return

        print("Starting Modular Juggling AI...")
        self.is_running = True

        # Start control thread
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

        print("AI started! Press 'q' in visualization window to stop.")

    def stop(self):
        """Stop the AI system"""
        print("Stopping Modular Juggling AI...")
        self.is_running = False

        # Stop recording if active
        if self.video_recorder.is_recording:
            self.video_recorder.stop_recording()

        # Emergency stop controller
        self.controller.emergency_stop()

        # Wait for control thread
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)

        # Close visualization windows
        cv2.destroyAllWindows()

        print("AI stopped successfully")

    def get_statistics(self) -> dict:
        """Get comprehensive system statistics"""
        tracking_info = self.analyzer.trajectory_predictor.get_tracking_info()
        performance_metrics = self.performance_monitor.get_current_metrics()
        detection_stats = self.detector.get_statistics()
        capture_stats = self.screen_capture.get_statistics()
        recording_status = self.video_recorder.get_status()

        return {
            'performance': performance_metrics,
            'detection': detection_stats,
            'capture': capture_stats,
            'recording': recording_status,
            'game_state': self.analyzer.current_state.value,
            'tracking': tracking_info
        }

def main():
    """Main function"""
    print("=" * 60)
    print("Modular Juggling AI Controller")
    print("=" * 60)

    # Check model file
    model_path = "runs/detect/train/weights/best.pt"
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        print("Please train the model first using train.py")
        return

    try:
        # Create AI system
        ai = JugglingAI(model_path)

        print("\nControls:")
        print("- Press ENTER to start the AI")
        print("- Press 'q' in the visualization window to stop")
        print("- Press 'r' in the visualization window to start/stop recording")
        print("- Press 's' in the visualization window to save screenshot")
        print("- Press Ctrl+C for emergency stop")

        input("\nPress ENTER to start...")

        # Start AI
        ai.start()

        # Wait for user stop
        try:
            while ai.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nEmergency stop requested")

        # Show final statistics
        print("\nFinal Statistics:")
        stats = ai.get_statistics()
        for category, data in stats.items():
            print(f"\n{category.upper()}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {data}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("Program ended")

if __name__ == "__main__":
    main()
