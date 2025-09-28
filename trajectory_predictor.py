#!/usr/bin/env python3
"""
球轨迹预测系统 - 基于物理学预测球的运动轨迹和落点
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
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
    ball_id: int = -1  # 球的ID，用于跟踪

@dataclass
class BallTrack:
    """球的轨迹跟踪"""
    ball_id: int
    history: List[BallState]
    last_update_frame: int
    is_stable: bool = False  # 是否是稳定的球（出现超过30帧）

class TrajectoryPredictor:
    """球轨迹预测器"""

    def __init__(self, config: GameConfig):
        self.config = config
        self.ball_tracks: Dict[int, BallTrack] = {}  # 存储多个球的轨迹
        self.next_ball_id = 0
        self.current_frame = 0
        self.max_ball_speed = 50  # 增加最大球速度（像素/帧）
        self.min_stable_frames = 30  # 最少稳定帧数
        self.max_missing_frames = 10  # 增加最大丢失帧数
        self.current_trajectory = None
        self.active_ball_id = -1  # 当前跟踪的球ID
        self.last_landing_y = None  # 记录上一次落点的Y坐标
        self.last_landing_point = None  # 记录上一次的落点

    def add_ball_detections(self, detections: List[Tuple[float, float]], frame_id: int):
        """添加多个球的检测结果"""
        self.current_frame = frame_id

        # 为每个检测分配到对应的轨迹
        assigned_detections = set()

        # 尝试将检测结果分配给现有轨迹
        for ball_id, track in self.ball_tracks.items():
            if len(track.history) == 0:
                continue

            last_pos = track.history[-1]
            best_match = None
            best_distance = float('inf')

            for i, (x, y) in enumerate(detections):
                if i in assigned_detections:
                    continue

                distance = np.sqrt((x - last_pos.x)**2 + (y - last_pos.y)**2)

                # 检查距离是否在合理范围内
                if distance <= self.max_ball_speed and distance < best_distance:
                    best_distance = distance
                    best_match = i

            # 如果找到匹配，添加到轨迹
            if best_match is not None:
                x, y = detections[best_match]
                timestamp = frame_id * self.config.FRAME_TIME
                ball_state = BallState(x, y, timestamp, frame_id, ball_id)

                track.history.append(ball_state)
                track.last_update_frame = frame_id
                assigned_detections.add(best_match)

                # 检查是否达到稳定状态
                if len(track.history) >= self.min_stable_frames and not track.is_stable:
                    track.is_stable = True
                    print(f"Frame {frame_id}: Ball {ball_id} became STABLE (reached {len(track.history)} frames)")

                # 保持历史记录不超过50帧
                if len(track.history) > 50:
                    track.history.pop(0)

        # 为未分配的检测创建新轨迹
        for i, (x, y) in enumerate(detections):
            if i not in assigned_detections:
                self._create_new_track(x, y, frame_id)

        # 清理过期的轨迹
        self._cleanup_old_tracks()

        # 更新当前活跃的轨迹
        self._update_active_trajectory()

    def _create_new_track(self, x: float, y: float, frame_id: int):
        """创建新的球轨迹"""
        ball_id = self.next_ball_id
        self.next_ball_id += 1

        timestamp = frame_id * self.config.FRAME_TIME
        ball_state = BallState(x, y, timestamp, frame_id, ball_id)

        track = BallTrack(
            ball_id=ball_id,
            history=[ball_state],
            last_update_frame=frame_id,
            is_stable=False
        )

        self.ball_tracks[ball_id] = track
        print(f"Frame {frame_id}: Created new ball track {ball_id} at ({x:.1f}, {y:.1f})")

    def _cleanup_old_tracks(self):
        """清理过期的轨迹"""
        to_remove = []

        for ball_id, track in self.ball_tracks.items():
            # 如果球丢失超过最大帧数，删除轨迹
            if self.current_frame - track.last_update_frame > self.max_missing_frames:
                to_remove.append(ball_id)
                print(f"Frame {self.current_frame}: Removing old track {ball_id} (missing for {self.current_frame - track.last_update_frame} frames)")

        for ball_id in to_remove:
            del self.ball_tracks[ball_id]

    def _update_active_trajectory(self):
        """更新当前活跃的轨迹用于预测"""
        # 选择最稳定且最新的球进行预测
        best_track = None
        best_score = -1

        for ball_id, track in self.ball_tracks.items():
            if not track.is_stable or len(track.history) < 2:
                continue

            # 计算评分：稳定性 + 最新性
            stability_score = min(len(track.history), 50) / 50.0  # 归一化到0-1
            recency_score = 1.0 / (1 + self.current_frame - track.last_update_frame)
            total_score = stability_score * 0.7 + recency_score * 0.3

            if total_score > best_score:
                best_score = total_score
                best_track = track

        if best_track:
            # 只在切换到不同的球或状态变化时打印
            if self.active_ball_id != best_track.ball_id:
                print(f"Frame {self.current_frame}: Active ball switched to {best_track.ball_id} (frames: {len(best_track.history)}, stable: {best_track.is_stable})")

            self.active_ball_id = best_track.ball_id
            self.ball_history = best_track.history  # 兼容旧接口

            # 重新拟合轨迹
            if len(best_track.history) >= 2:
                self._fit_trajectory_for_track(best_track)

        else:
            if self.active_ball_id != -1:  # 只在状态改变时打印
                print(f"Frame {self.current_frame}: No active ball (no stable tracks available)")
            self.active_ball_id = -1
            self.ball_history = []
            self.current_trajectory = None

    def _fit_trajectory_for_track(self, track: BallTrack):
        """为指定轨迹拟合抛物线"""
        if len(track.history) < 2:
            return

        # 使用最近的数据点
        recent_points = track.history[-8:]  # 使用最近8个点

        x_coords = np.array([point.x for point in recent_points])
        y_coords = np.array([point.y for point in recent_points])

        try:
            # 固定重力系数 a = 1.438131e-02，只拟合 b 和 c
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
                'residuals': residuals[0] if len(residuals) > 0 else 0,
                'ball_id': track.ball_id
            }

            print(f"Frame {self.current_frame}: Trajectory fitted for ball {track.ball_id}: y = {fixed_a:.6f}x² + {b:.6f}x + {c:.6f}")
            if len(residuals) > 0:
                print(f"Frame {self.current_frame}: Fit residual: {residuals[0]:.2f}")

        except np.linalg.LinAlgError:
            print(f"Frame {self.current_frame}: Failed to fit trajectory for ball {track.ball_id}")
            self.current_trajectory = None

    def get_ball_direction_for_track(self, track: BallTrack) -> int:
        """获取指定轨迹的运动方向"""
        if len(track.history) < 2:
            return 0

        # 使用最近的几个点计算平均方向
        recent_points = track.history[-10:] if len(track.history) >= 3 else track.history[-2:]

        direction_sum = 0
        count = 0

        for i in range(1, len(recent_points)):
            dx = recent_points[i].x - recent_points[i-1].x
            if abs(dx) > 1:  # 忽略微小的变化
                direction_sum += 1 if dx > 0 else -1
                count += 1

        if count == 0:
            return 0

        # 如果方向一致性超过阈值，返回方向
        avg_direction = direction_sum / count
        if abs(avg_direction) > 0.5:
            return 1 if avg_direction > 0 else -1
        else:
            return 0

    def _get_ball_direction(self) -> int:
        """获取当前活跃球的运动方向（兼容旧接口）"""
        if self.active_ball_id == -1 or self.active_ball_id not in self.ball_tracks:
            return 0

        return self.get_ball_direction_for_track(self.ball_tracks[self.active_ball_id])

    def get_tracking_info(self) -> dict:
        """获取跟踪信息"""
        stable_balls = sum(1 for track in self.ball_tracks.values() if track.is_stable)
        total_balls = len(self.ball_tracks)

        return {
            'total_tracks': total_balls,
            'stable_tracks': stable_balls,
            'active_ball_id': self.active_ball_id,
            'current_frame': self.current_frame,
            'tracks_info': {
                ball_id: {
                    'frames': len(track.history),
                    'stable': track.is_stable,
                    'last_update': track.last_update_frame
                }
                for ball_id, track in self.ball_tracks.items()
            }
        }

    # 兼容旧接口的方法
    def add_ball_detection(self, x: float, y: float, frame_id: int):
        """添加单个球检测（兼容旧接口）"""
        self.add_ball_detections([(x, y)], frame_id)

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

        # 判断球的运动方向
        ball_direction = self._get_ball_direction()
        if ball_direction == 0:  # 无法确定方向
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

                # 根据球的运动方向选择合理的解
                current_x = self.ball_history[-1].x

                for x in [x1, x2]:
                    if 0 <= x <= self.config.SCREEN_WIDTH:
                        # 检查这个解是否符合运动方向
                        if ball_direction > 0 and x > current_x:  # 向右运动，选择右侧的解
                            landing_points.append((x, target_y))
                        elif ball_direction < 0 and x < current_x:  # 向左运动，选择左侧的解
                            landing_points.append((x, target_y))

        # 返回最近的合理落点，但优先考虑之前的落点类型
        if landing_points:
            # 如果有多个点，智能选择落点
            current_x = self.ball_history[-1].x

            # 将落点按Y坐标分组
            high_line_points = [p for p in landing_points if p[1] == self.config.JUGGLE_MIN_Y]  # 最高线
            low_line_points = [p for p in landing_points if p[1] == self.config.JUGGLE_MAX_Y]   # 最低线

            # 如果之前的落点在最高线，且现在也有最高线的落点，优先选择最高线
            if (self.last_landing_y == self.config.JUGGLE_MIN_Y and high_line_points):
                selected_point = min(high_line_points, key=lambda p: abs(p[0] - current_x))
            # 如果之前的落点在最低线，且现在也有最低线的落点，优先选择最低线
            elif (self.last_landing_y == self.config.JUGGLE_MAX_Y and low_line_points):
                selected_point = min(low_line_points, key=lambda p: abs(p[0] - current_x))
            # 其他情况，选择距离最近的点
            else:
                selected_point = min(landing_points, key=lambda p: abs(p[0] - current_x))

            # 更新记录的落点信息
            self.last_landing_y = selected_point[1]
            self.last_landing_point = selected_point

            return selected_point

        return None

    def _get_ball_direction(self) -> int:
        """获取球的运动方向
        返回: 1(向右), -1(向左), 0(无法确定)
        """
        if len(self.ball_history) < 2:
            return 0

        # 使用最近的几个点计算平均方向
        recent_points = self.ball_history[-3:] if len(self.ball_history) >= 3 else self.ball_history[-2:]

        direction_sum = 0
        count = 0

        for i in range(1, len(recent_points)):
            dx = recent_points[i].x - recent_points[i-1].x
            if abs(dx) > 1:  # 忽略微小的变化
                direction_sum += 1 if dx > 0 else -1
                count += 1

        if count == 0:
            return 0

        # 如果方向一致性超过阈值，返回方向
        avg_direction = direction_sum / count
        if abs(avg_direction) > 0.5:  # 阈值可调整
            return 1 if avg_direction > 0 else -1
        else:
            return 0

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
        img_copy = image.copy()

        # 绘制可颠区间
        cv2.line(img_copy, (0, self.config.JUGGLE_MIN_Y),
                (self.config.SCREEN_WIDTH, self.config.JUGGLE_MIN_Y), (0, 255, 255), 2)  # 青色线
        cv2.line(img_copy, (0, self.config.JUGGLE_MAX_Y),
                (self.config.SCREEN_WIDTH, self.config.JUGGLE_MAX_Y), (0, 255, 255), 2)

        # 绘制所有球的轨迹
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  # 不同颜色

        for i, (ball_id, track) in enumerate(self.ball_tracks.items()):
            if len(track.history) < 2:
                continue

            color = colors[i % len(colors)]

            # 绘制历史轨迹点
            for j, point in enumerate(track.history):
                alpha = 0.3 + 0.7 * j / len(track.history)  # 渐变透明度
                radius = 3 if track.is_stable else 2
                cv2.circle(img_copy, (int(point.x), int(point.y)), radius, color, -1)

            # 标注球ID和状态
            last_point = track.history[-1]
            status = "STABLE" if track.is_stable else f"UNSTABLE({len(track.history)})"
            cv2.putText(img_copy, f"Ball {ball_id}: {status}",
                       (int(last_point.x) + 10, int(last_point.y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 绘制活跃球的预测轨迹
        if self.active_ball_id != -1 and self.current_trajectory:
            active_track = self.ball_tracks.get(self.active_ball_id)
            if active_track:
                current_x = active_track.history[-1].x
                ball_direction = self.get_ball_direction_for_track(active_track)

                # 根据运动方向决定预测范围
                if ball_direction > 0:  # 向右运动
                    end_x = min(current_x + 500, self.config.SCREEN_WIDTH)
                elif ball_direction < 0:  # 向左运动
                    end_x = max(current_x - 500, 0)
                else:  # 方向不确定，两边都画
                    end_x = min(current_x + 500, self.config.SCREEN_WIDTH)

                trajectory_points = self.predict_trajectory_points(current_x, end_x, 50)

                if trajectory_points:
                    points = np.array(trajectory_points, dtype=np.int32)
                    cv2.polylines(img_copy, [points], False, (255, 255, 0), 3)  # 粗黄色轨迹线

                # 显示运动方向
                direction_text = "→" if ball_direction > 0 else "←" if ball_direction < 0 else "?"
                cv2.putText(img_copy, f"Active Ball {self.active_ball_id} Direction: {direction_text}",
                           (int(current_x) + 20, int(current_x) - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 绘制预测落点
        if landing_point:
            cv2.circle(img_copy, (int(landing_point[0]), int(landing_point[1])),
                      10, (0, 0, 255), -1)  # 红色圆点

            # 显示落点信息
            cv2.putText(img_copy, f"Landing: ({landing_point[0]:.1f}, {landing_point[1]:.1f})",
                       (int(landing_point[0]) + 15, int(landing_point[1]) - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 显示跟踪统计信息
        tracking_info = self.get_tracking_info()
        info_text = f"Tracks: {tracking_info['total_tracks']}, Stable: {tracking_info['stable_tracks']}"
        cv2.putText(img_copy, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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