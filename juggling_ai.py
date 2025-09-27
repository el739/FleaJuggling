#!/usr/bin/env python3
"""
实时颠球AI控制器 - 集成检测、预测、决策和动作执行
"""

import cv2
import time
import threading
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import pyautogui
from typing import Optional, Tuple

from trajectory_predictor import TrajectoryPredictor, GameConfig
from game_controller import GameController
from game_analyzer import GameStateAnalyzer, DecisionMaker

class JugglingAI:
    """颠球AI主控制器"""

    def __init__(self, model_path: str = "runs/detect/train/weights/best.pt"):
        print("Initializing Juggling AI...")

        # 初始化配置
        self.config = GameConfig()

        # 初始化YOLO模型
        self.model = YOLO(model_path)
        print(f"Loaded YOLO model from: {model_path}")

        # 初始化各个组件
        self.analyzer = GameStateAnalyzer(self.config)
        self.controller = GameController()
        self.decision_maker = DecisionMaker(self.analyzer, self.controller)

        # 运行控制
        self.is_running = False
        self.capture_thread = None
        self.control_thread = None

        # 性能监控
        self.frame_count = 0
        self.start_time = 0.0
        self.detection_times = []
        self.last_fps_report = 0.0

        # 可视化
        self.show_visualization = True
        self.current_frame = None

        print("Juggling AI initialized successfully!")

    def capture_screen(self) -> Optional[np.ndarray]:
        """捕获屏幕画面"""
        try:
            # 使用pyautogui捕获全屏
            screenshot = pyautogui.screenshot()
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame

        except Exception as e:
            print(f"Error capturing screen: {e}")
            return None

    def detect_objects(self, frame: np.ndarray) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """检测玩家和球的位置"""
        start_time = time.time()

        # YOLO检测
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

        # 记录检测时间
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        if len(self.detection_times) > 30:  # 保持最近30次的记录
            self.detection_times.pop(0)

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

        # 绘制轨迹预测
        if self.analyzer.trajectory_predictor.current_trajectory:
            landing_point = self.analyzer.trajectory_predictor.predict_landing_in_juggle_zone()
            vis_frame = self.analyzer.trajectory_predictor.visualize_trajectory(vis_frame, landing_point)

        # 显示状态信息
        self._draw_status_info(vis_frame)

        return vis_frame

    def _draw_status_info(self, frame: np.ndarray):
        """绘制状态信息"""
        y_offset = 30
        line_height = 25

        # 基本信息
        info_lines = [
            f"Frame: {self.frame_count}",
            f"State: {self.analyzer.current_state.value}",
            f"FPS: {self._calculate_fps():.1f}"
        ]

        # 检测延迟
        if self.detection_times:
            avg_detection_time = np.mean(self.detection_times) * 1000
            info_lines.append(f"Detection: {avg_detection_time:.1f}ms")

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
        if self.frame_count == 0:
            return 0.0

        elapsed_time = time.time() - self.start_time
        return self.frame_count / elapsed_time if elapsed_time > 0 else 0.0

    def control_loop(self):
        """主控制循环"""
        self.start_time = time.time()
        target_frame_time = 1.0 / self.config.FPS

        print(f"Starting control loop at {self.config.FPS} FPS")

        while self.is_running:
            loop_start_time = time.time()

            # 捕获屏幕
            frame = self.capture_screen()
            if frame is None:
                time.sleep(0.1)
                continue

            self.current_frame = frame

            # 检测对象
            player_pos, ball_pos = self.detect_objects(frame)

            # 更新分析器
            current_time = time.time()
            if player_pos:
                self.analyzer.update_player_position(player_pos[0], player_pos[1], current_time)

            if ball_pos:
                self.analyzer.update_ball_position(ball_pos[0], ball_pos[1], self.frame_count)

            # 分析游戏状态并执行决策
            analysis = self.analyzer.analyze_game_state()
            self.decision_maker.execute_decision(analysis)

            # 可视化（如果启用）
            if self.show_visualization:
                vis_frame = self.visualize_frame(frame, player_pos, ball_pos)
                # 缩放显示
                display_frame = cv2.resize(vis_frame, (960, 540))
                cv2.imshow('Juggling AI', display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quit requested")
                    self.stop()
                    break

            self.frame_count += 1

            # 控制帧率
            loop_time = time.time() - loop_start_time
            sleep_time = max(0, target_frame_time - loop_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

            # 定期报告性能
            if time.time() - self.last_fps_report > 5.0:
                fps = self._calculate_fps()
                avg_detection = np.mean(self.detection_times) * 1000 if self.detection_times else 0
                print(f"Performance - FPS: {fps:.1f}, Detection: {avg_detection:.1f}ms")
                self.last_fps_report = time.time()

    def start(self):
        """启动AI控制器"""
        if self.is_running:
            print("AI is already running")
            return

        print("Starting Juggling AI...")
        self.is_running = True

        # 启动控制线程
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()

        print("Juggling AI started! Press 'q' in the visualization window to stop.")

    def stop(self):
        """停止AI控制器"""
        print("Stopping Juggling AI...")
        self.is_running = False

        # 停止所有动作
        self.controller.emergency_stop()

        # 等待线程结束
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)

        # 关闭显示窗口
        cv2.destroyAllWindows()

        print("Juggling AI stopped")

    def get_statistics(self) -> dict:
        """获取运行统计信息"""
        return {
            'frames_processed': self.frame_count,
            'fps': self._calculate_fps(),
            'avg_detection_time': np.mean(self.detection_times) * 1000 if self.detection_times else 0,
            'running_time': time.time() - self.start_time if self.start_time > 0 else 0,
            'current_state': self.analyzer.current_state.value,
            'ball_detections': len(self.analyzer.trajectory_predictor.ball_history)
        }

def main():
    """主函数"""
    print("=" * 60)
    print("颠球AI控制器")
    print("=" * 60)

    # 检查模型文件
    model_path = "runs/detect/train/weights/best.pt"
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        print("Please train the model first using train.py")
        return

    try:
        # 创建AI控制器
        ai = JugglingAI(model_path)

        print("\
Controls:")
        print("- Press ENTER to start the AI")
        print("- Press 'q' in the visualization window to stop")
        print("- Press Ctrl+C to emergency stop")

        input("\
Press ENTER to start...")

        # 启动AI
        ai.start()

        # 等待用户停止
        try:
            while ai.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\
Emergency stop requested")

        # 显示统计信息
        print("\
Final Statistics:")
        stats = ai.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("Program ended")

if __name__ == "__main__":
    main()