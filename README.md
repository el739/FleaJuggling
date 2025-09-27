# 颠球AI项目

基于YOLO目标检测和轨迹预测的实时颠球AI系统。

## 项目结构

```
FleaJuggling/
├── screenshots/              # 原始截图数据
├── labelled/                # 标注数据
├── dataset/                 # 训练数据集
├── runs/                    # 训练结果
├── prepare_dataset.py       # 数据预处理脚本
├── train.py                 # YOLO模型训练脚本
├── detect_video.py          # 视频检测脚本
├── detect_image.py          # 图片检测脚本
├── trajectory_predictor.py  # 球轨迹预测系统
├── game_controller.py       # 游戏控制系统
├── game_analyzer.py         # 游戏状态分析器
├── juggling_ai.py          # 主AI控制器
└── README.md               # 本文件
```

## 使用步骤

### 1. 数据准备和模型训练

```bash
# 1. 预处理数据
python prepare_dataset.py

# 2. 训练YOLO模型
python train.py
```

### 2. 测试检测效果

```bash
# 测试单张图片
python detect_image.py screenshots/screenshot_xxx.png

# 测试视频
python detect_video.py input_video.mp4 -o output_detected.mp4
```

### 3. 运行AI控制器

```bash
# 启动实时AI控制器
python juggling_ai.py
```

## 系统配置

### 游戏参数
- 屏幕分辨率：1920x1080 (全屏)
- 采样率：15 FPS
- 可颠球区间：Y坐标 567-750

### 控制按键
- 左移：A键
- 右移：D键
- 左冲刺：A + 左Ctrl
- 右冲刺：D + 左Ctrl
- 颠球：B键

### 物理参数
- 玩家速度：610像素/68帧 (约134.6像素/秒)
- 冲刺距离：264像素
- 球轨迹方程：y = 0.014381x² - 38.557790x + 26121.360000

## 系统组件

### 1. 目标检测 (YOLO)
- 检测玩家和球的位置
- 实时置信度：>0.5
- 类别：hero(玩家), ordinary(球)

### 2. 轨迹预测
- 基于物理学的抛物线拟合
- 预测球在可颠区间的落点
- 估算到达时间

### 3. 决策系统
- 分析游戏状态
- 计算最优移动策略
- 判断颠球时机

### 4. 动作执行
- 键盘控制接口
- 普通移动/冲刺选择
- 精确时机控制

## 性能监控

AI系统提供实时性能监控：
- FPS显示
- 检测延迟
- 轨迹预测状态
- 决策置信度

## 调试功能

1. **可视化显示**：显示检测框、轨迹线、可颠区间
2. **状态信息**：实时显示系统状态和预测结果
3. **测试模式**：各组件独立测试功能

## 注意事项

1. **权限要求**：需要屏幕捕获和键盘控制权限
2. **游戏设置**：确保游戏全屏运行在1920x1080分辨率
3. **性能要求**：建议使用GPU加速推理
4. **安全停止**：按'q'键或Ctrl+C安全停止

## 故障排除

### 常见问题

1. **模型未找到**
   ```bash
   Error: Model file not found: runs/detect/train/weights/best.pt
   ```
   解决：先运行 `python train.py` 训练模型

2. **检测精度低**
   - 检查游戏画面是否清晰
   - 调整置信度阈值
   - 重新标注更多数据

3. **反应延迟高**
   - 检查系统性能
   - 降低检测频率
   - 使用GPU加速

4. **按键无响应**
   - 检查游戏窗口是否激活
   - 确认按键映射正确
   - 以管理员权限运行

## 参数调优

在 `trajectory_predictor.py` 中的 `GameConfig` 类中调整参数：

```python
class GameConfig:
    # 调整采样率
    FPS = 15

    # 调整可颠区间
    JUGGLE_MIN_Y = 567
    JUGGLE_MAX_Y = 750

    # 调整玩家速度
    PLAYER_SPEED = 610 / 68 * FPS
```

在 `game_analyzer.py` 中调整决策参数：

```python
# 调整反应时间
self.reaction_time = 0.1

# 调整位置容差
self.position_tolerance = 30

# 调整置信度阈值
self.min_prediction_confidence = 0.7
```