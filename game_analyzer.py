#!/usr/bin/env python3
"""
游戏状态分析器 - 分析游戏状态并做出决策
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum

from trajectory_predictor import TrajectoryPredictor, GameConfig
from game_controller import GameController, ActionType, MovementCalculator

class GameState(Enum):
    """游戏状态"""
    WAITING = "waiting"           # 等待球出现
    TRACKING = "tracking"         # 跟踪球的轨迹
    MOVING_TO_POSITION = "moving" # 移动到位置
    READY_TO_JUGGLE = "ready"     # 准备颠球
    JUGGLING = "juggling"         # 正在颠球

@dataclass
class PlayerState:
    """玩家状态"""
    x: float
    y: float
    timestamp: float

@dataclass
class GameAnalysis:
    """游戏分析结果"""
    should_move: bool = False
    target_x: Optional[float] = None
    use_dash: bool = False
    should_juggle: bool = False
    time_to_action: Optional[float] = None
    confidence: float = 0.0

class GameStateAnalyzer:
    """游戏状态分析器"""

    def __init__(self, config: GameConfig):
        self.config = config
        self.trajectory_predictor = TrajectoryPredictor(config)
        self.movement_calculator = MovementCalculator()

        self.current_state = GameState.WAITING
        self.player_position: Optional[PlayerState] = None
        self.last_analysis_time = 0.0
        self.current_frame = 0  # 添加帧计数器

        # 从配置文件获取分析参数
        self.min_prediction_confidence = config.analysis.min_prediction_confidence
        self.reaction_time = config.control.reaction_time  # 反应时间（秒）
        self.position_tolerance = config.player.position_tolerance  # 位置容差（像素）

        print("Game State Analyzer initialized")

    def update_player_position(self, x: float, y: float, timestamp: float):
        """更新玩家位置"""
        self.player_position = PlayerState(x, y, timestamp)

    def update_ball_position(self, x: float, y: float, frame_id: int):
        """更新球位置"""
        self.current_frame = frame_id  # 更新帧计数器
        self.trajectory_predictor.add_ball_detection(x, y, frame_id)

    def analyze_game_state(self) -> GameAnalysis:
        """分析当前游戏状态并返回建议动作"""
        current_time = time.time()
        analysis = GameAnalysis()

        # 检查是否有足够的数据
        if not self.player_position:
            analysis.confidence = 0.0
            return analysis

        # 检查是否有活跃的稳定球
        tracking_info = self.trajectory_predictor.get_tracking_info()
        if tracking_info['active_ball_id'] == -1 or tracking_info['stable_tracks'] == 0:
            # 没有稳定的球，继续等待
            self.current_state = GameState.WAITING
            analysis.confidence = 0.1
            return analysis

        # 预测球的落点
        landing_point = self.trajectory_predictor.predict_landing_in_juggle_zone()

        if not landing_point:
            # 没有预测到落点，继续等待
            self.current_state = GameState.WAITING
            analysis.confidence = 0.1
            return analysis

        landing_x, landing_y = landing_point

        # 估计到达落点的时间
        time_to_landing = self.trajectory_predictor.estimate_time_to_landing(landing_x)

        if not time_to_landing or time_to_landing <= 0:
            analysis.confidence = 0.2
            return analysis

        # 计算玩家到落点的距离
        distance_to_landing = abs(self.player_position.x - landing_x)

        # 分析是否需要移动
        if distance_to_landing > self.position_tolerance:
            analysis.should_move = True
            analysis.target_x = landing_x

            # 计算移动策略
            action_type, move_time = self.movement_calculator.calculate_optimal_movement(
                self.player_position.x, landing_x, time_to_landing
            )

            analysis.use_dash = action_type in [ActionType.DASH_LEFT, ActionType.DASH_RIGHT]

            # 检查时间是否充足
            required_time = move_time + self.reaction_time
            if time_to_landing > required_time:
                analysis.confidence = 0.8
                self.current_state = GameState.MOVING_TO_POSITION
            else:
                # 时间不够，尝试冲刺
                analysis.use_dash = True
                analysis.confidence = 0.6

        else:
            # 已经在合适的位置
            analysis.should_move = False

            # 检查是否应该颠球
            if self._should_juggle_now(landing_x, landing_y, time_to_landing):
                analysis.should_juggle = True
                analysis.confidence = 0.9
                self.current_state = GameState.READY_TO_JUGGLE
            else:
                analysis.confidence = 0.7
                self.current_state = GameState.TRACKING

        analysis.time_to_action = time_to_landing
        self.last_analysis_time = current_time

        return analysis

    def _should_juggle_now(self, ball_x: float, ball_y: float, time_to_landing: float) -> bool:
        """判断是否应该现在颠球"""
        # 球必须在可颠区间内
        if not (self.config.JUGGLE_MIN_Y <= ball_y <= self.config.JUGGLE_MAX_Y):
            return False

        # 检查玩家和球的横坐标差值是否在允许范围内
        if self.player_position:
            x_diff = abs(self.player_position.x - ball_x)
            if x_diff > self.config.player.juggle_position_tolerance:  # 从配置获取横坐标差值限制
                return False

        # 时间判断：预留一定的反应时间
        juggle_timing_window = self.config.analysis.juggle_timing_window  # 从配置获取颠球时机窗口

        return time_to_landing <= juggle_timing_window

    def get_detailed_status(self) -> dict:
        """获取详细的状态信息"""
        # 获取轨迹跟踪信息
        tracking_info = self.trajectory_predictor.get_tracking_info()

        status = {
            'state': self.current_state.value,
            'player_position': None,
            'tracking_info': tracking_info,
            'has_trajectory': self.trajectory_predictor.current_trajectory is not None,
            'last_analysis': self.last_analysis_time
        }

        if self.player_position:
            status['player_position'] = {
                'x': self.player_position.x,
                'y': self.player_position.y,
                'timestamp': self.player_position.timestamp
            }

        if self.trajectory_predictor.current_trajectory:
            landing_point = self.trajectory_predictor.predict_landing_in_juggle_zone()
            if landing_point:
                status['predicted_landing'] = {
                    'x': landing_point[0],
                    'y': landing_point[1]
                }

                time_to_landing = self.trajectory_predictor.estimate_time_to_landing(landing_point[0])
                if time_to_landing:
                    status['time_to_landing'] = time_to_landing

        return status

class DecisionMaker:
    """决策制定器 - 基于分析结果制定具体行动"""

    def __init__(self, analyzer: GameStateAnalyzer, controller: GameController):
        self.analyzer = analyzer
        self.controller = controller
        self.last_action_time = 0.0
        self.action_cooldown = 0.05  # 动作冷却时间

    def execute_decision(self, analysis: GameAnalysis) -> bool:
        """执行决策"""
        current_time = time.time()

        # 检查冷却时间
        if current_time - self.last_action_time < self.action_cooldown:
            return False

        # 检查置信度
        if analysis.confidence < 0.5:
            return False

        action_executed = False

        # 执行颠球动作
        if analysis.should_juggle:
            print(f"Frame {self.analyzer.current_frame}: Decision: JUGGLE!")
            self.controller.juggle_ball()
            action_executed = True

        # 执行移动动作
        elif analysis.should_move and analysis.target_x is not None:
            current_x = self.analyzer.player_position.x if self.analyzer.player_position else 0
            distance = abs(analysis.target_x - current_x)

            print(f"Frame {self.analyzer.current_frame}: Decision: MOVE to {analysis.target_x:.1f} (distance: {distance:.1f}, dash: {analysis.use_dash})")

            self.controller.move_to_position(
                current_x,
                analysis.target_x,
                analysis.use_dash
            )
            action_executed = True

        if action_executed:
            self.last_action_time = current_time

        return action_executed

def test_analyzer():
    """测试分析器功能"""
    config = GameConfig()
    analyzer = GameStateAnalyzer(config)
    controller = GameController()
    decision_maker = DecisionMaker(analyzer, controller)

    print("Testing Game State Analyzer")
    print("=" * 50)

    # 模拟玩家位置
    analyzer.update_player_position(960, 800, time.time())  # 屏幕中央底部

    # 模拟球的轨迹（使用提供的方程）
    a_test = 1.438131e-02
    b_test = -3.855779e+01
    c_test = 2.612136e+04

    test_x_values = np.linspace(800, 1200, 20)

    for i, x in enumerate(test_x_values):
        y = a_test * x * x + b_test * x + c_test
        analyzer.update_ball_position(x, y, i)

        # 分析游戏状态
        analysis = analyzer.analyze_game_state()

        print(f"\nFrame {i}: Ball at ({x:.1f}, {y:.1f})")
        print(f"State: {analyzer.current_state.value}")
        print(f"Analysis: move={analysis.should_move}, juggle={analysis.should_juggle}, confidence={analysis.confidence:.2f}")

        if analysis.target_x:
            print(f"Target X: {analysis.target_x:.1f}, Use dash: {analysis.use_dash}")

        if analysis.time_to_action:
            print(f"Time to action: {analysis.time_to_action:.2f}s")

        # 执行决策（模拟）
        if decision_maker.execute_decision(analysis):
            print("Action executed!")

        time.sleep(0.1)  # 模拟帧率

    print("\nDetailed status:")
    status = analyzer.get_detailed_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_analyzer()
