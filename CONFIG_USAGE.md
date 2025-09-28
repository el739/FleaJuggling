# 配置系统使用说明

## 📁 新的配置架构

### 🎯 核心改进
- **集中管理**: 所有硬编码变量现在集中在 `config/` 模块中
- **分类组织**: 按功能模块分组配置（检测、视觉、控制等）
- **向后兼容**: 保持原有API，内部使用新配置系统
- **易于修改**: 修改配置只需编辑一个地方

### 📂 配置文件结构
```
config/
├── __init__.py
└── game_config.py     # 主配置文件
```

### 🛠 使用方式

#### 1. 基本使用
```python
from config import GameConfig

# 创建配置实例
config = GameConfig()

# 访问配置值
print(f"Screen size: {config.screen.width}x{config.screen.height}")
print(f"FPS: {config.screen.fps}")
print(f"Detection confidence: {config.detection.confidence_threshold}")
```

#### 2. 向后兼容
```python
from config import GameConfig

config = GameConfig()

# 旧式访问方式仍然有效
print(config.SCREEN_WIDTH)  # 1920
print(config.FPS)           # 15
print(config.JUGGLE_MIN_Y)  # 465
```

#### 3. 配置自定义
```python
from config import GameConfig, get_config, update_config

# 全局配置更新
update_config(fps=30)  # 修改FPS

# 或直接修改配置对象
config = GameConfig()
config.detection.confidence_threshold = 0.7
config.screen.fps = 30
```

#### 4. 模块中使用
```python
# 在你的模块中
from config import DetectionConfig, ScreenConfig

class MyDetector:
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        # 使用 self.config.confidence_threshold 等
```

### 🎛 主要配置项

#### 检测配置 (DetectionConfig)
- `confidence_threshold`: 检测置信度阈值 (0.5)
- `player_class_id`: 玩家类别ID (0)
- `ball_class_id`: 球类别ID (1)

#### 屏幕配置 (ScreenConfig)
- `width/height`: 屏幕尺寸 (1920x1080)
- `display_width/height`: 显示尺寸 (960x540)
- `fps`: 帧率 (15)

#### 颠球区域配置 (JuggleZoneConfig)
- `min_y/max_y`: 颠球区域Y坐标 (465-750)

#### 玩家配置 (PlayerConfig)
- `move_speed`: 移动速度
- `dash_distance`: 冲刺距离 (264)
- `position_tolerance`: 位置容差 (30)

#### 控制配置 (ControlConfig)
- `key_mapping`: 按键映射
- `action_cooldown`: 动作冷却时间 (0.05s)
- `reaction_time`: 反应时间 (0.1s)

### 🔧 修改配置

要修改游戏参数，只需编辑 `config/game_config.py` 文件：

```python
@dataclass
class ScreenConfig:
    width: int = 1920      # 修改屏幕宽度
    height: int = 1080     # 修改屏幕高度
    fps: int = 20          # 修改帧率
```

### 🧪 测试配置

运行测试脚本验证配置系统：
```bash
python test_config.py
```

### 📋 迁移清单

已完成的模块：
- ✅ `detection/object_detector.py` - 使用DetectionConfig
- ✅ `config/` - 新配置系统
- ✅ `juggling_ai_modular.py` - 使用新配置
- ✅ `juggling_ai.py` - 向后兼容包装器

待完成的模块：
- ⏳ `vision/visualizer.py` - 部分更新
- ⏳ `game_analyzer.py` - 需要更新阈值配置
- ⏳ `game_controller.py` - 需要更新按键配置
- ⏳ `trajectory_predictor.py` - 需要更新轨迹配置

### 🚀 优势

1. **统一管理**: 所有配置参数集中在一个地方
2. **类型安全**: 使用dataclass提供类型提示
3. **模块化**: 不同模块的配置分离，便于维护
4. **扩展性**: 容易添加新的配置项
5. **向后兼容**: 不破坏现有代码
6. **文档化**: 配置项有清晰的注释和默认值

现在你可以通过修改 `config/game_config.py` 文件来调整游戏的任何参数，而不需要在各个文件中查找硬编码的数值！