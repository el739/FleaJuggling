#!/usr/bin/env python3
"""
实时颠球AI控制器 - 使用模块化架构的简化接口
向后兼容原有API，内部使用新的模块化系统
"""

import time
from pathlib import Path

# Import the new modular system
from juggling_ai_modular import JugglingAI as ModularJugglingAI

class JugglingAI:
    """颠球AI主控制器 - 向后兼容的包装器"""

    def __init__(self, model_path: str = "runs/detect/train/weights/best.pt"):
        """初始化AI系统"""
        print("Initializing Juggling AI (using modular architecture)...")
        self._modular_ai = ModularJugglingAI(model_path)

        # Expose commonly used properties for backward compatibility
        self.config = self._modular_ai.config
        self.analyzer = self._modular_ai.analyzer
        self.controller = self._modular_ai.controller
        self.decision_maker = self._modular_ai.decision_maker

        # Runtime properties
        self.is_running = False
        self.show_visualization = True

    @property
    def frame_count(self) -> int:
        """获取当前帧数"""
        return self._modular_ai.performance_monitor.metrics.frame_count

    def start_recording(self, output_path: str = None):
        """开始录制"""
        return self._modular_ai.video_recorder.start_recording(output_path)

    def stop_recording(self):
        """停止录制"""
        return self._modular_ai.video_recorder.stop_recording()

    def start(self):
        """启动AI控制器"""
        self.is_running = True
        return self._modular_ai.start()

    def stop(self):
        """停止AI控制器"""
        self.is_running = False
        return self._modular_ai.stop()

    def get_statistics(self) -> dict:
        """获取运行统计信息"""
        return self._modular_ai.get_statistics()

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
        print("- Press 'r' in the visualization window to start/stop recording")
        print("- Press 's' in the visualization window to save screenshot")
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