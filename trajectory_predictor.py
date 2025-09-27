#!/usr/bin/env python3
"""
球轨迹预测系统 - 基于物理学预测球的运动轨迹和落点
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2

@dataclass
class GameConfig:
    """游戏配置参数"""
    # 屏幕和采样
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080
    FPS = 15
    FRAME_TIME = 1.0 / FPS  # 每帧时间（秒）

    # 可颠球区间
    JUGGLE_MIN_Y = 567  # 最高位置（y坐标较小）
    JUGGLE_MAX_Y = 750  # 最低位置（y坐标较大）

    # 玩家参数
    PLAYER_SPEED = 610 / 68 * FPS  # 像素/秒 (610 pix per 68 frames at 15fps)
    DASH_DISTANCE = 264  # 冲刺距离（像素）

    # 控制按键
    KEYS = {
        'left': 'a',
        'right': 'd',
        'left_dash': 'a+ctrl',
        'right_dash': 'd+ctrl',
        'juggle': 'b'
    }

@dataclass
class BallState:
    """球的状态"""
    x: float
    y: float
    timestamp: float
    frame_id: int

class TrajectoryPredictor:
    """球轨迹预测器"""

    def __init__(self, config: GameConfig):
        self.config = config
        self.ball_history: List[BallState] = []
        self.current_trajectory = None

    def add_ball_detection(self, x: float, y: float, frame_id: int):
        """添加球的检测结果"""
        timestamp = frame_id * self.config.FRAME_TIME
        ball_state = BallState(x, y, timestamp, frame_id)
        self.ball_history.append(ball_state)

        # 保持历史记录不超过10帧
        if len(self.ball_history) > 10:
            self.ball_history.pop(0)

        # 如果有足够的数据点，拟合轨迹
        if len(self.ball_history) >= 2:  # 改为2个点即可
            self._fit_trajectory()

    def _fit_trajectory(self):
        """拟合球的抛物线轨迹"""
        if len(self.ball_history) < 2:  # 只需要2个点即可拟合线性部分
            return

        # 提取最近的数据点
        recent_points = self.ball_history[-5:]  # 使用最近5个点

        x_coords = np.array([point.x for point in recent_points])
        y_coords = np.array([point.y for point in recent_points])

        try:
            # 固定重力系数 a = 1.438131e-02，只拟合 b 和 c
            # y = ax² + bx + c 变为 y - ax² = bx + c
            # 这是一个线性方程，可以用最小二乘法求解

            fixed_a = 1.438131e-02

            # 计算 y - ax²
            y_adjusted = y_coords - fixed_a * x_coords * x_coords

            # 构建线性系统 [x, 1] * [b, c] = y_adjusted
            A = np.column_stack([x_coords, np.ones(len(x_coords))])

            # 最小二乘法求解 b 和 c
            coeffs_bc, residuals, rank, s = np.linalg.lstsq(A, y_adjusted, rcond=None)

            b, c = coeffs_bc

            self.current_trajectory = {
                'a': fixed_a,
                'b': b,
                'c': c,
                'fit_time': recent_points[-1].timestamp,
                'x_range': (min(x_coords), max(x_coords)),
                'residuals': residuals[0] if len(residuals) > 0 else 0
            }

            print(f"Trajectory fitted: y = {fixed_a:.6f}x² + {b:.6f}x + {c:.6f}")
            if len(residuals) > 0:
                print(f"Fit residual: {residuals[0]:.2f}")

        except np.linalg.LinAlgError:
            print("Failed to fit trajectory - insufficient data variation")
            self.current_trajectory = None

    def predict_trajectory_points(self, x_start: float, x_end: float, num_points: int = 100) -> List[Tuple[float, float]]:
        """预测轨迹上的点"""
        if not self.current_trajectory:
            return []

        x_points = np.linspace(x_start, x_end, num_points)
        y_points = []

        a, b, c = self.current_trajectory['a'], self.current_trajectory['b'], self.current_trajectory['c']

        for x in x_points:
            y = a * x * x + b * x + c
            y_points.append(y)

        return list(zip(x_points, y_points))

    def predict_landing_in_juggle_zone(self) -> Optional[Tuple[float, float]]:
        """预测球在可颠区间的落点"""
        if not self.current_trajectory:
            return None

        a, b, c = self.current_trajectory['a'], self.current_trajectory['b'], self.current_trajectory['c']

        # 找到y = JUGGLE_MIN_Y 和 y = JUGGLE_MAX_Y 时的x坐标
        landing_points = []

        for target_y in [self.config.JUGGLE_MIN_Y, self.config.JUGGLE_MAX_Y]:
            # 解方程 ax² + bx + c = target_y
            # 即 ax² + bx + (c - target_y) = 0
            discriminant = b * b - 4 * a * (c - target_y)

            if discriminant >= 0:
                sqrt_d = np.sqrt(discriminant)
                x1 = (-b + sqrt_d) / (2 * a)
                x2 = (-b - sqrt_d) / (2 * a)

                # 选择合理的解（通常是较大的x值，表示未来的位置）
                for x in [x1, x2]:
                    if 0 <= x <= self.config.SCREEN_WIDTH:
                        landing_points.append((x, target_y))

        # 返回最近的合理落点
        if landing_points:
            # 选择x坐标最大的点（最接近未来的位置）
            return max(landing_points, key=lambda p: p[0])

        return None

    def estimate_time_to_landing(self, landing_x: float) -> Optional[float]:
        """估计球到达落点的时间"""
        if not self.ball_history or not self.current_trajectory:
            return None

        current_ball = self.ball_history[-1]

        # 简单估计：假设球的水平速度大致恒定
        if len(self.ball_history) >= 2:
            prev_ball = self.ball_history[-2]
            dx = current_ball.x - prev_ball.x
            dt = current_ball.timestamp - prev_ball.timestamp

            if dt > 0 and abs(dx) > 0:
                horizontal_speed = dx / dt
                distance_to_landing = landing_x - current_ball.x

                if horizontal_speed != 0:
                    time_to_landing = distance_to_landing / horizontal_speed
                    return max(0, time_to_landing)  # 不能为负时间

        return None

    def visualize_trajectory(self, image: np.ndarray, landing_point: Optional[Tuple[float, float]] = None):
        """在图像上可视化轨迹"""
        if not self.current_trajectory or len(self.ball_history) == 0:
            return image

        img_copy = image.copy()

        # 绘制轨迹
        current_x = self.ball_history[-1].x
        trajectory_points = self.predict_trajectory_points(
            current_x, min(current_x + 500, self.config.SCREEN_WIDTH), 50
        )

        if trajectory_points:
            points = np.array(trajectory_points, dtype=np.int32)
            cv2.polylines(img_copy, [points], False, (255, 255, 0), 2)  # 黄色轨迹线

        # 绘制可颠区间
        cv2.line(img_copy, (0, self.config.JUGGLE_MIN_Y),
                (self.config.SCREEN_WIDTH, self.config.JUGGLE_MIN_Y), (0, 255, 255), 2)  # 青色线
        cv2.line(img_copy, (0, self.config.JUGGLE_MAX_Y),
                (self.config.SCREEN_WIDTH, self.config.JUGGLE_MAX_Y), (0, 255, 255), 2)

        # 绘制预测落点
        if landing_point:
            cv2.circle(img_copy, (int(landing_point[0]), int(landing_point[1])),
                      10, (0, 0, 255), -1)  # 红色圆点

            # 显示落点信息
            cv2.putText(img_copy, f"Landing: ({landing_point[0]:.1f}, {landing_point[1]:.1f})",
                       (int(landing_point[0]) + 15, int(landing_point[1]) - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return img_copy

def test_trajectory_prediction():
    """测试轨迹预测功能"""
    config = GameConfig()
    predictor = TrajectoryPredictor(config)

    # 使用提供的轨迹方程进行测试
    # y = 1.438131e-02 * x^2 + (-3.855779e+01) * x + 2.612136e+04
    a_test = 1.438131e-02
    b_test = -3.855779e+01
    c_test = 2.612136e+04

    print("Testing with provided trajectory equation:")
    print(f"y = {a_test:.6f}x² + {b_test:.6f}x + {c_test:.6f}")

    # 生成测试数据点
    test_x_values = np.linspace(800, 1200, 10)

    for i, x in enumerate(test_x_values):
        y = a_test * x * x + b_test * x + c_test
        predictor.add_ball_detection(x, y, i)
        print(f"Frame {i}: Ball at ({x:.1f}, {y:.1f})")

    # 预测落点
    landing_point = predictor.predict_landing_in_juggle_zone()
    if landing_point:
        print(f"\nPredicted landing point in juggle zone: ({landing_point[0]:.1f}, {landing_point[1]:.1f})")

        time_to_landing = predictor.estimate_time_to_landing(landing_point[0])
        if time_to_landing:
            print(f"Estimated time to landing: {time_to_landing:.2f} seconds")
    else:
        print("\nNo landing point predicted in juggle zone")

if __name__ == "__main__":
    test_trajectory_prediction()