#!/usr/bin/env python3
"""
AI运行录制工具 - 专门用于录制颠球AI的运行过程
"""

import cv2
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import pyautogui
from typing import Optional, Tuple
import argparse

from trajectory_predictor import TrajectoryPredictor, GameConfig
from game_analyzer import GameStateAnalyzer

class AIRecorder:
    """AI运行录制器"""

    def __init__(self, model_path: str = "runs/detect/train/weights/best.pt"):
        print("Initializing AI Recorder...")

        # 初始化配置
        self.config = GameConfig()

        # 初始化YOLO模型
        self.model = YOLO(model_path)
        print(f"Loaded YOLO model from: {model_path}")

        # 初始化分析器（仅用于可视化）
        self.analyzer = GameStateAnalyzer(self.config)

        # 录制控制
        self.is_recording = False
        self.frame_count = 0
        self.start_time = 0.0

        print("AI Recorder initialized successfully!")

    def capture_screen(self) -> Optional[np.ndarray]:
        """捕获屏幕画面"""
        try:
            screenshot = pyautogui.screenshot()
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        except Exception as e:
            print(f"Error capturing screen: {e}")
            return None

    def detect_objects(self, frame: np.ndarray) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """检测玩家和球的位置"""
        results = self.model(frame, conf=0.5, verbose=False)

        player_pos = None
        ball_pos = None

        if results[0].boxes is not None:
            boxes = results[0].boxes.data.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                cls = int(cls)

                # 计算中心点
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                if cls == 0:  # hero (玩家)
                    player_pos = (center_x, center_y)
                elif cls == 1:  # ordinary (球)
                    ball_pos = (center_x, center_y)

        return player_pos, ball_pos

    def visualize_frame(self, frame: np.ndarray, player_pos: Optional[Tuple[float, float]],
                       ball_pos: Optional[Tuple[float, float]]) -> np.ndarray:
        """可视化检测结果和预测信息"""
        vis_frame = frame.copy()

        # 绘制检测结果
        if player_pos:
            cv2.circle(vis_frame, (int(player_pos[0]), int(player_pos[1])), 15, (0, 255, 0), -1)
            cv2.putText(vis_frame, "PLAYER", (int(player_pos[0]) + 20, int(player_pos[1])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if ball_pos:
            cv2.circle(vis_frame, (int(ball_pos[0]), int(ball_pos[1])), 10, (255, 0, 0), -1)
            cv2.putText(vis_frame, "BALL", (int(ball_pos[0]) + 15, int(ball_pos[1]) - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 绘制可颠区间
        cv2.line(vis_frame, (0, self.config.JUGGLE_MIN_Y),
                (self.config.SCREEN_WIDTH, self.config.JUGGLE_MIN_Y), (0, 255, 255), 2)
        cv2.line(vis_frame, (0, self.config.JUGGLE_MAX_Y),
                (self.config.SCREEN_WIDTH, self.config.JUGGLE_MAX_Y), (0, 255, 255), 2)

        # 更新轨迹数据
        current_time = time.time()
        if player_pos:
            self.analyzer.update_player_position(player_pos[0], player_pos[1], current_time)
        if ball_pos:
            self.analyzer.update_ball_position(ball_pos[0], ball_pos[1], self.frame_count)

        # 绘制轨迹预测
        if self.analyzer.trajectory_predictor.current_trajectory:
            landing_point = self.analyzer.trajectory_predictor.predict_landing_in_juggle_zone()
            vis_frame = self.analyzer.trajectory_predictor.visualize_trajectory(vis_frame, landing_point)

        # 添加状态信息
        self._draw_status_info(vis_frame)

        return vis_frame

    def _draw_status_info(self, frame: np.ndarray):
        """绘制状态信息"""
        y_offset = 30
        line_height = 25

        info_lines = [
            f"Frame: {self.frame_count}",
            f"Recording: {'ON' if self.is_recording else 'OFF'}",
            f"FPS: {self._calculate_fps():.1f}"
        ]

        # 轨迹信息
        if self.analyzer.trajectory_predictor.current_trajectory:
            landing_point = self.analyzer.trajectory_predictor.predict_landing_in_juggle_zone()
            if landing_point:
                info_lines.append(f"Landing: ({landing_point[0]:.0f}, {landing_point[1]:.0f})")
                time_to_landing = self.analyzer.trajectory_predictor.estimate_time_to_landing(landing_point[0])
                if time_to_landing:
                    info_lines.append(f"Time: {time_to_landing:.2f}s")

        # 绘制信息
        for i, line in enumerate(info_lines):
            y_pos = y_offset + i * line_height
            cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _calculate_fps(self) -> float:
        """计算当前FPS"""
        if self.frame_count == 0 or self.start_time == 0:
            return 0.0
        elapsed_time = time.time() - self.start_time
        return self.frame_count / elapsed_time if elapsed_time > 0 else 0.0

    def record_session(self, output_path: str, duration: int = 60):
        """录制指定时长的会话"""
        print(f"Starting recording session...")
        print(f"Output: {output_path}")
        print(f"Duration: {duration} seconds")

        # 设置视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, self.config.FPS, (960, 540))

        self.is_recording = True
        self.start_time = time.time()
        self.frame_count = 0

        target_frame_time = 1.0 / self.config.FPS
        end_time = self.start_time + duration

        print("\\nControls during recording:")
        print("- Press 'q' to stop early")
        print("- Press 's' to save current frame screenshot")

        try:
            while time.time() < end_time and self.is_recording:
                loop_start_time = time.time()

                # 捕获屏幕
                frame = self.capture_screen()
                if frame is None:
                    time.sleep(0.1)
                    continue

                # 检测对象
                player_pos, ball_pos = self.detect_objects(frame)

                # 可视化
                vis_frame = self.visualize_frame(frame, player_pos, ball_pos)
                display_frame = cv2.resize(vis_frame, (960, 540))

                # 写入视频
                video_writer.write(display_frame)

                # 显示预览
                cv2.imshow('AI Recorder', display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\\nEarly stop requested")
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    screenshot_path = f"recorder_screenshot_{timestamp}.png"
                    cv2.imwrite(screenshot_path, display_frame)
                    print(f"Screenshot saved: {screenshot_path}")

                self.frame_count += 1

                # 控制帧率
                loop_time = time.time() - loop_start_time
                sleep_time = max(0, target_frame_time - loop_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # 进度报告
                if self.frame_count % (self.config.FPS * 5) == 0:  # 每5秒报告一次
                    elapsed = time.time() - self.start_time
                    remaining = duration - elapsed
                    print(f"Recording... {elapsed:.1f}s / {duration}s (remaining: {remaining:.1f}s)")

        except KeyboardInterrupt:
            print("\\nRecording interrupted by user")

        finally:
            video_writer.release()
            cv2.destroyAllWindows()
            self.is_recording = False

            total_time = time.time() - self.start_time
            fps = self.frame_count / total_time if total_time > 0 else 0
            print(f"\\nRecording completed!")
            print(f"Total frames: {self.frame_count}")
            print(f"Total time: {total_time:.1f}s")
            print(f"Average FPS: {fps:.1f}")
            print(f"Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='颠球AI录制工具')
    parser.add_argument('-o', '--output', default=None,
                       help='输出视频路径（默认：自动生成时间戳文件名）')
    parser.add_argument('-d', '--duration', type=int, default=60,
                       help='录制时长（秒，默认：60）')
    parser.add_argument('-m', '--model', default='runs/detect/train/weights/best.pt',
                       help='模型权重路径')

    args = parser.parse_args()

    # 检查模型文件
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return

    # 输出路径
    if args.output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"juggling_ai_recording_{timestamp}.mp4"

    print("=" * 60)
    print("颠球AI录制工具")
    print("=" * 60)

    try:
        recorder = AIRecorder(args.model)

        print(f"\\nReady to record for {args.duration} seconds")
        input("Press ENTER to start recording...")

        recorder.record_session(args.output, args.duration)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()