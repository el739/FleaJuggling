#!/usr/bin/env python3
"""
动作执行系统 - 实现游戏控制动作
"""

import time
import threading
from enum import Enum
from dataclasses import dataclass
import win32api
import win32con

class ActionType(Enum):
    """动作类型"""
    IDLE = "idle"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    DASH_LEFT = "dash_left"
    DASH_RIGHT = "dash_right"
    JUGGLE = "juggle"

@dataclass
class ActionCommand:
    """动作命令"""
    action_type: ActionType
    duration: float = 0.0  # 动作持续时间（秒）
    timestamp: float = 0.0  # 命令时间戳

class GameController:
    """游戏控制器 - 负责执行游戏动作"""

    def __init__(self):
        self.current_action = ActionType.IDLE
        self.action_start_time = 0.0
        self.action_duration = 0.0
        self.is_executing = False
        self.executor_thread = None
        self.stop_flag = False

        # 按键映射
        self.key_mapping = {
            'a': 0x41,       # A键
            'd': 0x44,       # D键
            'b': 0x42,       # B键
            'ctrl': 0xA2,    # 左Ctrl键
        }

        print("Game Controller initialized")

    def press_key(self, key_code: int):
        """按下按键"""
        win32api.keybd_event(key_code, 0, 0, 0)

    def release_key(self, key_code: int):
        """释放按键"""
        win32api.keybd_event(key_code, 0, win32con.KEYEVENTF_KEYUP, 0)

    def execute_simple_action(self, action_type: ActionType, duration: float = 0.1):
        """执行简单动作（非持续性）"""
        if action_type == ActionType.JUGGLE:
            print("Executing JUGGLE action")
            self.press_key(self.key_mapping['b'])
            time.sleep(duration)
            self.release_key(self.key_mapping['b'])

        elif action_type == ActionType.DASH_LEFT:
            print("Executing DASH_LEFT action")
            # 按住Ctrl+A一段时间
            self.press_key(self.key_mapping['ctrl'])
            self.press_key(self.key_mapping['a'])
            time.sleep(duration)
            self.release_key(self.key_mapping['a'])
            self.release_key(self.key_mapping['ctrl'])

        elif action_type == ActionType.DASH_RIGHT:
            print("Executing DASH_RIGHT action")
            # 按住Ctrl+D一段时间
            self.press_key(self.key_mapping['ctrl'])
            self.press_key(self.key_mapping['d'])
            time.sleep(duration)
            self.release_key(self.key_mapping['d'])
            self.release_key(self.key_mapping['ctrl'])

    def start_continuous_action(self, action_type: ActionType):
        """开始持续性动作（如移动）"""
        if self.is_executing:
            self.stop_continuous_action()

        self.current_action = action_type
        self.is_executing = True
        self.stop_flag = False

        if action_type == ActionType.MOVE_LEFT:
            print("Starting MOVE_LEFT action")
            self.press_key(self.key_mapping['a'])

        elif action_type == ActionType.MOVE_RIGHT:
            print("Starting MOVE_RIGHT action")
            self.press_key(self.key_mapping['d'])

    def stop_continuous_action(self):
        """停止持续性动作"""
        if not self.is_executing:
            return

        print(f"Stopping {self.current_action.value} action")

        if self.current_action == ActionType.MOVE_LEFT:
            self.release_key(self.key_mapping['a'])

        elif self.current_action == ActionType.MOVE_RIGHT:
            self.release_key(self.key_mapping['d'])

        self.current_action = ActionType.IDLE
        self.is_executing = False
        self.stop_flag = True

    def execute_timed_action(self, action_type: ActionType, duration: float):
        """执行定时动作"""
        def timed_execution():
            self.start_continuous_action(action_type)
            time.sleep(duration)
            self.stop_continuous_action()

        if self.executor_thread and self.executor_thread.is_alive():
            self.stop_continuous_action()

        self.executor_thread = threading.Thread(target=timed_execution)
        self.executor_thread.daemon = True
        self.executor_thread.start()

    def calculate_movement_time(self, distance: float, action_type: ActionType) -> float:
        """计算移动到指定距离需要的时间"""
        # 基于提供的参数：610 pix per 68 frames at 15fps
        player_speed = (610 / 68) * 15  # 像素/秒

        if action_type in [ActionType.MOVE_LEFT, ActionType.MOVE_RIGHT]:
            return abs(distance) / player_speed

        elif action_type in [ActionType.DASH_LEFT, ActionType.DASH_RIGHT]:
            # 冲刺是固定距离264像素，时间估算
            dash_time = 0.3  # 假设冲刺需要0.3秒
            return dash_time

        return 0.0

    def move_to_position(self, current_x: float, target_x: float, use_dash: bool = False):
        """移动到指定位置"""
        distance = target_x - current_x

        if abs(distance) < 10:  # 距离太近，不需要移动
            print(f"Already at target position (distance: {distance:.1f})")
            return

        if use_dash:
            # 使用冲刺
            if distance > 0:
                action_type = ActionType.DASH_RIGHT
            else:
                action_type = ActionType.DASH_LEFT

            print(f"Using dash to move {distance:.1f} pixels")
            self.execute_simple_action(action_type, 0.2)

        else:
            # 使用普通移动
            if distance > 0:
                action_type = ActionType.MOVE_RIGHT
            else:
                action_type = ActionType.MOVE_LEFT

            move_time = self.calculate_movement_time(distance, action_type)
            print(f"Moving {distance:.1f} pixels for {move_time:.2f} seconds")
            self.execute_timed_action(action_type, move_time)

    def juggle_ball(self):
        """颠球动作"""
        print("Juggling ball!")
        self.execute_simple_action(ActionType.JUGGLE, 0.1)

    def emergency_stop(self):
        """紧急停止所有动作"""
        print("Emergency stop - releasing all keys")
        for key_code in self.key_mapping.values():
            self.release_key(key_code)

        self.stop_continuous_action()

class MovementCalculator:
    """移动计算器 - 计算最优移动策略"""

    def __init__(self, dash_distance: float = 264):
        self.dash_distance = dash_distance

    def calculate_optimal_movement(self, current_x: float, target_x: float,
                                 available_time: float) -> tuple[ActionType, float]:
        """计算最优移动策略"""
        distance = abs(target_x - current_x)
        direction = 1 if target_x > current_x else -1

        # 玩家速度：(610/68)*15 ≈ 134.6 pixels/second
        player_speed = (610 / 68) * 15

        # 计算普通移动需要的时间
        normal_move_time = distance / player_speed

        # 判断是否需要冲刺
        if distance > self.dash_distance * 0.8:  # 如果距离超过冲刺距离的80%
            if direction > 0:
                return ActionType.DASH_RIGHT, 0.2
            else:
                return ActionType.DASH_LEFT, 0.2

        elif available_time < normal_move_time * 1.2:  # 如果时间紧迫
            if direction > 0:
                return ActionType.DASH_RIGHT, 0.2
            else:
                return ActionType.DASH_LEFT, 0.2

        else:
            # 使用普通移动
            if direction > 0:
                return ActionType.MOVE_RIGHT, normal_move_time
            else:
                return ActionType.MOVE_LEFT, normal_move_time

def test_controller():
    """测试控制器功能"""
    controller = GameController()
    calculator = MovementCalculator()

    print("Testing Game Controller")
    print("Press 'q' to quit the test")

    try:
        while True:
            command = input("\nEnter command (left/right/dash_left/dash_right/juggle/move_to/quit): ").strip().lower()

            if command in ['quit', 'q']:
                break

            elif command == 'left':
                controller.execute_timed_action(ActionType.MOVE_LEFT, 1.0)

            elif command == 'right':
                controller.execute_timed_action(ActionType.MOVE_RIGHT, 1.0)

            elif command == 'dash_left':
                controller.execute_simple_action(ActionType.DASH_LEFT, 0.2)

            elif command == 'dash_right':
                controller.execute_simple_action(ActionType.DASH_RIGHT, 0.2)

            elif command == 'juggle':
                controller.juggle_ball()

            elif command == 'move_to':
                try:
                    current_x = float(input("Current X: "))
                    target_x = float(input("Target X: "))
                    use_dash = input("Use dash? (y/n): ").lower() == 'y'

                    controller.move_to_position(current_x, target_x, use_dash)

                except ValueError:
                    print("Invalid input")

            elif command == 'stop':
                controller.stop_continuous_action()

            else:
                print("Unknown command")

    except KeyboardInterrupt:
        print("\nTest interrupted")

    finally:
        controller.emergency_stop()
        print("Controller test ended")

if __name__ == "__main__":
    test_controller()