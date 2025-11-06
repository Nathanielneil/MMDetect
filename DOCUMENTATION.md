# MMDect - 多模态无人机集群人机交互系统 完整文档

**版本**: v6.3
**最后更新**: 2025-11-06
**项目状态**: Phase 1 完成 ✅

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 快速开始](#2-快速开始)
- [3. 安装指南](#3-安装指南)
- [4. 使用手册](#4-使用手册)
- [5. 系统架构](#5-系统架构)
- [6. 检测器详解](#6-检测器详解)
- [7. 配置说明](#7-配置说明)
- [8. 故障排除](#8-故障排除)
- [9. 开发指南](#9-开发指南)
- [10. 性能指标](#10-性能指标)
- [11. 项目状态](#11-项目状态)

---

## 1. 项目概述

### 1.1 系统简介

MMDect（Multimodal Drone Detection）是一个基于ROS/Gazebo的**多模态指令识别与无人机集群控制系统**。系统支持四种输入模态，采用轻量化AI模型，实现低延迟实时响应。

**核心特性**：
- ✅ **四种输入模态**: 语音、手势、图像、鼠标/触屏
- ✅ **轻量化模型**: 总大小 < 50MB
- ✅ **实时性能**: 端到端延迟 < 200ms
- ✅ **统一架构**: 所有检测器继承统一基类
- ✅ **多模态融合**: 4种融合策略可选
- ✅ **即开即用**: 笔记本摄像头即可测试

### 1.2 支持的指令

| 指令类型 | 功能 | 参数 | 触发方式 |
|---------|------|------|---------|
| **TAKEOFF** | 起飞 | altitude(高度) | 语音"起飞" / 张开手掌(5指) / 符号T |
| **LAND** | 降落 | landing_position | 语音"降落" / 握拳(0指) / 符号H |
| **HOVER** | 悬停 | - | 语音"悬停" / 单指 / 符号L |
| **EXPLORE** | 探索 | - | 语音"探索" / V手势(2指) / 符号S |
| **FORMATION** | 编队 | formation_type | 语音"编队" / 三指 / 符号F |
| **WAYPOINT** | 航点飞行 | waypoints列表 | 鼠标绘制路径 |
| **EMERGENCY_STOP** | 紧急停止 | - | 语音"紧急停止" |

### 1.3 技术栈

**AI模型**：
- Vosk (~40MB) - 中文语音识别
- MediaPipe Hands (~2MB) - 21点手部跟踪
- YOLOv8-nano (~6MB) - 轻量级目标检测

**核心框架**：
- Python 3.8+
- OpenCV 4.8+
- PyTorch 2.0+

**仿真环境（待集成）**：
- ROS Noetic
- Gazebo 11
- MAVROS + PX4

---

## 2. 快速开始

### 2.1 最快测试（5分钟）

```bash
# 1. 克隆项目
cd ~/NGW/intern/MMDect

# 2. 安装依赖
pip install -r requirements.txt

# 3. 测试摄像头
python tools/test_camera.py

# 4. 测试手势识别（最简单）
python src/detectors/gesture_detector.py
# 对着摄像头做手势：张开手掌、握拳、V字、单指、三指

# 5. 测试高级可视化（带指尖轨迹）
python test_gesture_advanced.py
```

### 2.2 完整系统测试

```bash
# 启用所有检测器 + 统一查看器
python src/main.py --enable-all --unified-viewer

# 仅启用特定检测器组合
python src/main.py --gesture --image --mouse

# 调试模式
python src/main.py --enable-all --log-level DEBUG
```

### 2.3 命令行参数速查

```bash
# 检测器选择
--voice           # 启用语音识别
--gesture         # 启用手势识别
--image           # 启用图像/符号检测
--mouse           # 启用鼠标路径规划
--enable-all      # 启用所有检测器

# 可视化
--unified-viewer  # 使用统一查看器（推荐）

# 配置
--config <path>   # 指定配置文件
--log-level <level>  # DEBUG/INFO/WARNING/ERROR
```

---

## 3. 安装指南

### 3.1 Python环境

```bash
# 推荐使用conda创建虚拟环境
conda create -n mmdetect python=3.8
conda activate mmdetect

# 或使用venv
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

### 3.2 安装依赖

```bash
# 安装核心依赖
pip install -r requirements.txt

# 主要包含：
# opencv-python >= 4.8.0
# mediapipe >= 0.10.0
# ultralytics >= 8.0.0
# vosk >= 0.3.45
# numpy >= 1.24.0
# PyYAML >= 6.0
# pyaudio >= 0.2.13
```

### 3.3 下载预训练模型

```bash
# 自动下载脚本
bash scripts/download_models.sh

# 或手动下载：
cd models

# YOLOv8-nano (~6MB)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Vosk中文小模型 (~42MB)
wget https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip
unzip vosk-model-small-cn-0.22.zip
mv vosk-model-small-cn-0.22 vosk-model-small-cn
```

### 3.4 验证安装

```bash
# 测试摄像头
python tools/test_camera.py --quick

# 测试各检测器
python src/detectors/gesture_detector.py
python src/detectors/voice_detector.py
python src/detectors/image_detector.py
python src/detectors/mouse_detector.py
```

---

## 4. 使用手册

### 4.1 手势识别使用

**支持的手势**：

| 手势 | 手指数 | 对应指令 | 说明 |
|-----|-------|---------|------|
| 🖐️ 张开手掌 | 5指 | TAKEOFF | 五指完全伸展 |
| ✊ 握拳 | 0指 | LAND | 所有手指收起 |
| ☝️ 单指 | 1指 | HOVER | 仅食指伸展 |
| ✌️ V手势 | 2指 | EXPLORE | 食指+中指伸展 |
| 🤟 三指 | 3指 | FORMATION | 食指+中指+无名指 |

**最佳实践**：
- 距离摄像头 30-60cm
- 手放在画面中心
- 背景简单、光线充足
- 每个手势保持 1-2秒

**运行命令**：
```bash
# 简单版本
python src/detectors/gesture_detector.py

# 高级可视化（带指尖轨迹）
python test_gesture_advanced.py

# 集成到系统
python src/main.py --gesture --unified-viewer
```

### 4.2 语音识别使用

**支持的指令关键词**：

| 中文 | 英文 | 指令 |
|-----|------|-----|
| 起飞 | takeoff | TAKEOFF |
| 降落 | land | LAND |
| 悬停 | hover | HOVER |
| 探索 | explore | EXPLORE |
| 编队 | formation | FORMATION |
| 紧急停止 | emergency stop | EMERGENCY_STOP |

**参数提取示例**：
- "起飞到10米" → altitude=10
- "三角形编队" → formation=triangle
- "圆形编队" → formation=circle

**运行命令**：
```bash
python src/detectors/voice_detector.py
# 需要麦克风，说出指令即可
```

### 4.3 图像/符号检测使用

**符号标记映射**：
- **H** → LAND (Helipad/停机坪)
- **T** → TAKEOFF (起飞点)
- **F** → FORMATION (编队)
- **S** → EXPLORE (Search/搜索)
- **L** → HOVER (Loiter/悬停)

**通用目标映射**：
- landing_pad → LAND
- person → HOVER
- car → EXPLORE
- stop_sign → EMERGENCY_STOP

**运行命令**：
```bash
python src/detectors/image_detector.py
# 用手机显示符号图片对着摄像头
```

### 4.4 鼠标路径规划使用

**操作方式**：
1. 启动程序后会弹出画布窗口
2. 按住鼠标左键拖动绘制路径
3. 释放鼠标完成路径绘制
4. 按 `R` 清除路径重新绘制
5. 按 `Q` 退出

**运行命令**：
```bash
python src/detectors/mouse_detector.py
```

### 4.5 统一查看器使用

统一查看器可以在单个窗口中切换查看不同检测器的可视化效果。

**视图模式**：
- 按 `1` - 手势检测视图（带高级特效）
- 按 `2` - 图像检测视图（带边界框）
- 按 `3` - 鼠标路径规划视图
- 按 `4` - 四宫格分屏视图
- 按 `0` - 概览视图
- 按 `R` - 重置统计
- 按 `Q` - 退出

**运行命令**：
```bash
python src/main.py --gesture --image --mouse --unified-viewer
```

---

## 5. 系统架构

### 5.1 整体架构

```
┌─────────────────────────────────────────────────┐
│           输入模态（4种）                          │
├──────────┬──────────┬──────────┬────────────────┤
│  语音     │  手势     │  图像     │  鼠标/触屏      │
│  Vosk    │MediaPipe │ YOLOv8   │  OpenCV       │
└──────────┴──────────┴──────────┴────────────────┘
           ↓          ↓          ↓          ↓
┌─────────────────────────────────────────────────┐
│         BaseDetector（统一检测模板）              │
│  preprocess → detect → postprocess              │
└─────────────────────────────────────────────────┘
           ↓          ↓          ↓          ↓
┌─────────────────────────────────────────────────┐
│         CommandManager（多模态融合）              │
│  - 时间窗口融合（0.5s）                          │
│  - 4种融合策略（最新/置信度/投票/优先级）          │
│  - 置信度过滤（>0.5）                            │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│              融合后的指令                         │
│  {command, confidence, parameters, ...}         │
└─────────────────────────────────────────────────┘
                      ↓
┌──────────┬──────────────┬──────────────────────┐
│ ROS桥接  │  Web界面     │  统一查看器           │
│ (待实现) │  (待实现)    │  (已实现)            │
└──────────┴──────────────┴──────────────────────┘
```

### 5.2 核心模块

#### BaseDetector（统一检测器基类）
- 位置：`src/detectors/base_detector.py`
- 功能：所有检测器的抽象基类
- 流程：`initialize() → preprocess() → detect() → postprocess() → run()`

#### CommandManager（指令管理器）
- 位置：`src/core/command_manager.py`
- 功能：多模态指令融合与管理
- 特性：
  - 检测器注册管理
  - 时间窗口融合（默认0.5秒）
  - 4种融合策略
  - 回调通知机制

#### CameraManager（相机管理器）
- 位置：`src/core/camera_manager.py`
- 功能：单相机多订阅者共享
- 特性：
  - 统一帧捕获线程
  - 每个检测器独立帧队列
  - 帧率控制
  - 自动丢弃旧帧

### 5.3 命令字典格式

```python
{
    'command': CommandType.TAKEOFF,    # 指令类型（枚举）
    'confidence': 0.95,                # 置信度 (0-1)
    'timestamp': 1698765432.123,       # Unix时间戳
    'modality': ModalityType.GESTURE,  # 输入模态
    'parameters': {                    # 指令参数
        'altitude': 10.0,
        'formation': 'triangle',
        'gesture_type': 'open_palm',
        'handedness': 'Right'
    },
    'metadata': {                      # 元数据
        'latency_ms': 85.5,
        'detector': 'GestureDetector'
    }
}
```

---

## 6. 检测器详解

### 6.1 语音检测器（VoiceDetector）

**原理**：
- 使用Vosk离线语音识别
- 基于MFCC特征提取
- 关键词匹配识别指令

**技术细节**：
- 模型：vosk-model-small-cn (~40MB)
- 采样率：16000 Hz
- 延迟：< 100ms
- 准确率：~85%

**配置参数**：
```yaml
voice_detector:
  model_path: "models/vosk-model-small-cn"
  sample_rate: 16000
  confidence_threshold: 0.7
  language: "zh-cn"
```

### 6.2 手势检测器（GestureDetector）

**原理**：
- MediaPipe Hands 21点手部关键点检测
- 基于规则的手势分类
- 时序平滑减少抖动

**技术细节**：
- 手部关键点：21个（含指尖、关节、手掌）
- 帧率：30 FPS
- 延迟：~33ms
- 准确率：~95%

**高级可视化特效**：
- ✨ 渐变色骨架（根据深度）
- 💫 发光关键点（脉冲动画）
- 🌈 指尖轨迹（紫→粉渐隐）
- 💓 手掌脉冲（青蓝色呼吸）

**配置参数**：
```yaml
gesture_detector:
  min_detection_confidence: 0.7
  min_tracking_confidence: 0.5
  max_num_hands: 1
  enable_glow: true
  enable_trails: true
  trail_length: 20
```

### 6.3 图像检测器（ImageDetector）

**原理**：
- YOLOv8-nano轻量级目标检测
- 符号标记识别
- 置信度过滤

**技术细节**：
- 模型：yolov8n.pt (~6MB)
- 输入尺寸：640×640
- 帧率：25 FPS
- 延迟：~40ms
- 准确率：~75%（COCO数据集）

**支持的目标**：
- COCO 80类通用目标
- 自定义符号标记（H/T/F/S/L）

**配置参数**：
```yaml
image_detector:
  model_path: "models/yolov8n.pt"
  confidence_threshold: 0.6
  iou_threshold: 0.45
  target_classes: ["landing_pad", "person", "car", "stop_sign"]
```

### 6.4 鼠标检测器（MouseDetector）

**原理**：
- 鼠标拖动绘制路径
- Ramer-Douglas-Peucker算法简化路径
- 像素到世界坐标转换

**技术细节**：
- 路径简化：RDP算法
- 坐标转换：scale_factor = 0.01
- 延迟：< 10ms
- 准确率：100%（确定性算法）

**配置参数**：
```yaml
mouse_detector:
  window_width: 800
  window_height: 600
  min_path_length: 20
  simplify_epsilon: 10
  scale_factor: 0.01
```

---

## 7. 配置说明

### 7.1 配置文件位置

`config/detector_config.yaml`

### 7.2 完整配置示例

```yaml
# 语音检测器
voice_detector:
  model_path: "models/vosk-model-small-cn"
  sample_rate: 16000
  confidence_threshold: 0.7
  language: "zh-cn"

# 手势检测器
gesture_detector:
  camera_index: 0
  resolution: [640, 480]
  min_detection_confidence: 0.7
  min_tracking_confidence: 0.5
  max_num_hands: 1
  # 时序平滑
  history_window_size: 5
  min_consecutive_frames: 3
  command_timeout: 1.0
  # 可视化
  enable_glow: true
  enable_trails: true
  trail_length: 20

# 图像检测器
image_detector:
  model_path: "models/yolov8n.pt"
  confidence_threshold: 0.6
  iou_threshold: 0.45
  target_classes: ["landing_pad", "person", "car", "stop_sign"]
  # 符号检测
  enable_symbol_detection: true
  symbol_model_path: "runs/detect/train2/weights/best.pt"

# 鼠标检测器
mouse_detector:
  window_width: 800
  window_height: 600
  min_path_length: 20
  simplify_epsilon: 10
  scale_factor: 0.01

# 指令管理器
command_manager:
  fusion_strategy: "highest"  # latest/highest/voting/priority
  fusion_window: 0.5          # 秒
  confidence_threshold: 0.5
  modality_priority:
    - mouse
    - image
    - gesture
    - voice

# 相机管理器
camera_manager:
  camera_index: 0
  fps: 30
  buffer_size: 5
```

---

## 8. 故障排除

### 8.1 摄像头问题

**Q1: 摄像头打不开**
```bash
# 检测可用摄像头
python tools/test_camera.py --find

# 尝试不同的索引
python src/detectors/gesture_detector.py --camera 0
python src/detectors/gesture_detector.py --camera 1
```

**Q2: 画面卡顿/延迟高**
- 降低分辨率：修改config中的resolution
- 关闭不需要的检测器
- 检查CPU占用率

**Q3: 手势识别不准确**
- 保持简单背景
- 确保光线充足
- 调整置信度阈值
- 检查左右手设置

### 8.2 模型问题

**Q1: 模型文件找不到**
```bash
# 检查模型文件
ls -lh models/

# 重新下载
bash scripts/download_models.sh
```

**Q2: YOLOv8加载失败**
```bash
# 重新安装ultralytics
pip install --upgrade ultralytics

# 手动下载模型
cd models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### 8.3 依赖问题

**Q1: MediaPipe安装失败**
```bash
# Linux
pip install mediapipe --no-cache-dir

# Mac M1/M2
pip install mediapipe-silicon

# Windows
# 确保安装Visual C++ Redistributable
```

**Q2: PyAudio安装失败**
```bash
# Ubuntu
sudo apt-get install portaudio19-dev
pip install pyaudio

# Mac
brew install portaudio
pip install pyaudio

# Windows
pip install pipwin
pipwin install pyaudio
```

### 8.4 运行时错误

**Q1: 权限错误（摄像头/麦克风）**
- Linux: 添加用户到video/audio组
- Mac: 系统偏好设置→安全性与隐私→摄像头/麦克风
- Windows: 检查隐私设置

**Q2: 内存占用过高**
- 减少trail_length和buffer_size
- 关闭不需要的可视化特效
- 降低检测帧率

---

## 9. 开发指南

### 9.1 添加新的检测器

**步骤**：

1. 继承`BaseDetector`基类
```python
from src.detectors.base_detector import BaseDetector, CommandType, ModalityType

class MyDetector(BaseDetector):
    def __init__(self, config=None):
        super().__init__(ModalityType.CUSTOM, config)
        # 初始化你的成员变量

    def initialize(self) -> bool:
        # 初始化模型/资源
        pass

    def preprocess(self, raw_input):
        # 预处理输入数据
        pass

    def detect(self, preprocessed):
        # 检测逻辑
        pass

    def postprocess(self, detection_result):
        # 转换为命令字典
        return self.create_command_dict(
            command_type=CommandType.HOVER,
            confidence=0.9,
            parameters={'key': 'value'}
        )
```

2. 在主程序中注册
```python
# src/main.py
detector = MyDetector(config)
if detector.initialize():
    command = detector.run(input_data)
```

### 9.2 添加新的融合策略

在`src/core/command_manager.py`中添加：

```python
def _fuse_by_custom(self, commands: List[Dict]) -> Optional[Dict]:
    """自定义融合策略"""
    # 实现你的融合逻辑
    pass
```

### 9.3 训练自定义模型

**手势分类器**：
```bash
# 1. 采集数据
python tools/collect_gesture_data.py

# 2. 训练模型
python tools/train_gesture_classifier.py

# 3. 更新配置
# 修改 config/detector_config.yaml 中的模型路径
```

**YOLOv8符号检测**：
```bash
# 1. 准备数据集（YOLO格式）
# symbol_dataset/
#   ├── images/train/
#   ├── images/test/
#   ├── labels/train/
#   └── labels/test/

# 2. 训练
python train_symbol_model.py

# 3. 使用训练好的模型
# 模型位于: runs/detect/train/weights/best.pt
```

---

## 10. 性能指标

### 10.1 延迟测试

| 检测器 | 平均延迟 | 目标 | 状态 |
|-------|---------|------|------|
| 语音 | < 100ms | < 100ms | ✅ |
| 手势 | ~33ms | < 50ms | ✅ |
| 图像 | ~40ms | < 50ms | ✅ |
| 鼠标 | < 10ms | < 20ms | ✅ |
| **端到端** | **< 150ms** | **< 200ms** | **✅** |

### 10.2 准确率

| 检测器 | 准确率 | 备注 |
|-------|--------|------|
| 语音 | ~85% | Vosk小模型，中文指令 |
| 手势 | ~95% | 5种基础手势 |
| 图像 | ~75% | COCO 80类，通用场景 |
| 鼠标 | 100% | 确定性算法 |

### 10.3 资源占用

测试环境：Intel i5-8代, 8GB RAM, 笔记本前置摄像头

| 检测器 | CPU | 内存 | GPU |
|-------|-----|------|-----|
| 语音 | ~15% | ~200MB | 不需要 |
| 手势 | ~20% | ~300MB | 不需要 |
| 图像 | ~25% | ~400MB | 可选 |
| 鼠标 | ~5% | ~50MB | 不需要 |
| **总计** | **~65%** | **~950MB** | **可选** |

### 10.4 模型大小

| 模型 | 大小 | 类型 |
|-----|------|------|
| Vosk CN-Small | ~40MB | 语音识别 |
| MediaPipe Hands | ~2MB | 手势检测 |
| YOLOv8-nano | ~6MB | 目标检测 |
| **总计** | **~48MB** | **全部** |

---

## 11. 项目状态

### 11.1 当前版本: v1.0 Phase 1 ✅

**完成度**: 100%

**已完成模块**：
- ✅ 4个检测器全部实现并测试通过
- ✅ CommandManager多模态融合系统
- ✅ CameraManager相机共享机制
- ✅ 统一查看器可视化界面
- ✅ 配置系统（YAML）
- ✅ 测试工具（test_camera.py等）
- ✅ 完整文档（本文档）

**代码统计**：
- 源代码：12个文件，~2,400行
- 工具脚本：5个文件，~735行
- 文档：整合为本文档

### 11.2 Phase 2: ROS/Gazebo集成（待实现）

**计划功能**：
- ⏸️ ROS桥接（`src/core/ros_bridge.py`）
- ⏸️ Gazebo仿真环境配置
- ⏸️ MAVROS/PX4无人机接口
- ⏸️ 无人机控制器（`src/simulation/drone_controller.py`）
- ⏸️ RViz 3D可视化配置
- ⏸️ 多机编队算法

### 11.3 Phase 3: Web可视化（待实现）

**计划功能**：
- ⏸️ Flask后端服务器
- ⏸️ WebSocket实时通信
- ⏸️ Three.js 3D场景
- ⏸️ roslibjs ROS通信
- ⏸️ 交互式控制面板

### 11.4 已知问题

**轻微问题**：
1. 复杂背景下手势识别准确率下降
   - 解决方案：使用简单背景或调整阈值
2. 环境噪音影响语音识别
   - 解决方案：使用降噪麦克风或更大模型
3. COCO类别外物体无法识别
   - 解决方案：训练自定义YOLOv8模型

**无严重bug** ✅

---

## 附录

### A. 文件结构

```
MMDect/
├── src/
│   ├── detectors/              # 检测器模块
│   │   ├── base_detector.py    # 统一基类
│   │   ├── voice_detector.py   # 语音检测
│   │   ├── gesture_detector.py # 手势识别
│   │   ├── image_detector.py   # 图像检测
│   │   └── mouse_detector.py   # 鼠标路径
│   ├── core/                   # 核心功能
│   │   ├── command_manager.py  # 指令管理
│   │   └── camera_manager.py   # 相机管理
│   ├── visualization/          # 可视化
│   │   └── unified_viewer.py   # 统一查看器
│   └── main.py                 # 主程序
├── models/                     # 预训练模型
│   ├── yolov8n.pt
│   ├── yolo11n.pt
│   └── vosk-model-small-cn/
├── config/
│   └── detector_config.yaml    # 配置文件
├── tools/                      # 工具脚本
│   ├── test_camera.py
│   ├── collect_images.py
│   ├── collect_gesture_data.py
│   └── train_gesture_classifier.py
├── symbol_dataset/             # 符号检测数据集
├── runs/                       # YOLO训练输出
├── requirements.txt            # Python依赖
├── test_gesture_advanced.py    # 高级手势测试
└── DOCUMENTATION.md            # 本文档
```

### B. 参考资源

**相关项目**：
- [RACER (HKUST/SYSU-STAR)](https://github.com/SYSU-STAR/RACER)
- [EGO-Planner-Swarm (ZJU FAST-Lab)](https://github.com/ZJU-FAST-Lab/ego-planner-swarm)
- [Fast-Planner (HKUST)](https://github.com/HKUST-Aerial-Robotics/Fast-Planner)

**AI模型官网**：
- [Vosk Speech Recognition](https://alphacephei.com/vosk/)
- [MediaPipe](https://mediapipe.dev/)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)

### C. 快速命令参考卡

```bash
# 测试命令
python tools/test_camera.py              # 测试摄像头
python src/detectors/gesture_detector.py # 测试手势
python test_gesture_advanced.py          # 高级手势可视化
python src/detectors/voice_detector.py   # 测试语音
python src/detectors/image_detector.py   # 测试图像
python src/detectors/mouse_detector.py   # 测试鼠标

# 完整系统
python src/main.py --enable-all --unified-viewer  # 启动所有
python src/main.py --gesture --image              # 手势+图像
python src/main.py --voice --mouse                # 语音+鼠标

# 调试
python src/main.py --enable-all --log-level DEBUG
```

---

**版权**: MIT License
**作者**: Guowei Ni
**最后更新**: 2025-11-06

---

**快速链接**：
- 遇到问题？查看 [第8章 故障排除](#8-故障排除)
- 想要开发？查看 [第9章 开发指南](#9-开发指南)
- 查看性能？查看 [第10章 性能指标](#10-性能指标)
