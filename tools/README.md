# 训练工具使用指南

本目录包含用于训练自定义模型的工具脚本。

## 工具列表

| 工具 | 用途 | 模型 |
|------|------|------|
| `collect_gesture_data.py` | 采集手势训练数据 | 手势分类器 |
| `train_gesture_classifier.py` | 训练手势分类器 | 手势分类器 |
| `collect_images.py` | 采集图像数据 | YOLOv8 |
| `train_yolo.py` | 训练YOLO模型 | YOLOv8 |

---

## 一、手势分类器训练

### 1.1 采集手势数据

**交互式模式**(推荐):
```bash
# 自动采集5种预定义手势
python tools/collect_gesture_data.py
```

**命令行模式**:
```bash
# 采集单个手势
python tools/collect_gesture_data.py \
    --gesture open_palm \
    --samples 150 \
    --save-dir data/gestures

# 继续采集到已有数据集
python tools/collect_gesture_data.py \
    --gesture fist \
    --samples 150 \
    --load data/gestures/gesture_dataset_20251027_120000.json
```

**采集建议**:
- 每个手势至少 **150个样本**
- 尝试不同的**角度、距离、光照**
- 保持手势**稳定**,避免模糊

### 1.2 训练分类器

```bash
# 使用Random Forest(推荐,快速准确)
python tools/train_gesture_classifier.py \
    --data data/gestures/gesture_dataset.json \
    --algorithm random_forest \
    --output models/gesture_classifier.pkl

# 使用SVM
python tools/train_gesture_classifier.py \
    --data data/gestures/gesture_dataset.json \
    --algorithm svm \
    --output models/gesture_svm.pkl

# 使用神经网络
python tools/train_gesture_classifier.py \
    --data data/gestures/gesture_dataset.json \
    --algorithm mlp \
    --output models/gesture_mlp.pkl
```

**输出**:
- `models/gesture_classifier.pkl` - 训练好的模型
- `models/gesture_classifier.txt` - 模型信息
- `outputs/confusion_matrix.png` - 混淆矩阵

### 1.3 集成到检测器

修改 `src/detectors/gesture_detector.py`:

```python
import pickle

class GestureDetector(BaseDetector):
    def __init__(self, config):
        super().__init__(ModalityType.GESTURE, config)

        # 加载训练好的分类器
        model_path = config.get('classifier_path', 'models/gesture_classifier.pkl')
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.classifier = model_data['classifier']
            self.label_map = model_data['label_map']

    def _classify_gesture(self, landmarks):
        """使用训练好的分类器"""
        features = np.array(landmarks).flatten().reshape(1, -1)

        gesture_idx = self.classifier.predict(features)[0]
        proba = self.classifier.predict_proba(features)[0]
        confidence = np.max(proba)

        gesture = self.label_map[gesture_idx]

        return gesture, confidence
```

更新配置 `config/detector_config.yaml`:

```yaml
gesture_detector:
  use_trained_classifier: true
  classifier_path: "models/gesture_classifier.pkl"
```

---

## 二、YOLOv8目标检测训练

### 2.1 采集图像数据

**手动采集**:
```bash
# 采集降落靶标图像(手动按空格保存)
python tools/collect_images.py collect \
    --class landing_pad \
    --num 500 \
    --save-dir data/yolo/images

# 采集其他类别
python tools/collect_images.py collect \
    --class obstacle \
    --num 300 \
    --save-dir data/yolo/images
```

**自动采集**:
```bash
# 自动每1秒采集一张
python tools/collect_images.py collect \
    --class landing_pad \
    --num 500 \
    --auto \
    --interval 1.0
```

**预览数据集**:
```bash
python tools/collect_images.py preview \
    --class landing_pad \
    --save-dir data/yolo/images
```

### 2.2 标注数据

使用LabelImg标注:

```bash
# 安装LabelImg
pip install labelImg

# 启动标注工具
labelImg

# 操作:
# 1. Open Dir -> 选择 data/yolo/images/landing_pad
# 2. Change Save Dir -> 选择 data/yolo/labels/landing_pad
# 3. 按'w'创建框, 标注目标
# 4. 按'd'下一张
# 5. 保存为YOLO格式
```

**标注格式** (每个.jpg对应一个.txt):
```
# landing_pad_001.txt
0 0.5 0.5 0.3 0.3  # class_id x_center y_center width height
```

### 2.3 准备数据集

创建目录结构:
```
data/yolo/
├── images/
│   ├── train/
│   │   ├── landing_pad_001.jpg
│   │   └── ...
│   └── val/
│       └── ...
└── labels/
    ├── train/
    │   ├── landing_pad_001.txt
    │   └── ...
    └── val/
        └── ...
```

划分数据集(8:2):
```bash
python tools/split_dataset.py \
    --source data/yolo/images/landing_pad \
    --train-ratio 0.8
```

创建 `data/yolo/dataset.yaml`:
```yaml
path: /home/ubuntu/NGW/intern/MMDect/data/yolo
train: images/train
val: images/val

nc: 4  # 类别数量
names: ['landing_pad', 'drone', 'marker', 'obstacle']
```

### 2.4 训练YOLO

**命令行训练**:
```bash
# 训练YOLOv8-nano(推荐)
yolo task=detect mode=train \
    model=yolov8n.pt \
    data=data/yolo/dataset.yaml \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0 \
    name=landing_pad_detector

# CPU训练(无GPU)
yolo task=detect mode=train \
    model=yolov8n.pt \
    data=data/yolo/dataset.yaml \
    epochs=50 \
    imgsz=416 \
    batch=8 \
    device=cpu \
    name=landing_pad_cpu
```

**Python训练**:
```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 训练
results = model.train(
    data='data/yolo/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='landing_pad_detector',
    patience=20,    # 早停
    save=True,
    device=0        # GPU
)

# 验证
metrics = model.val()
print(f"mAP@50: {metrics.box.map50:.3f}")

# 导出
model.export(format='onnx')
```

### 2.5 监控训练

训练结果保存在:
```
runs/detect/landing_pad_detector/
├── weights/
│   ├── best.pt         # 最佳权重
│   └── last.pt         # 最新权重
├── results.png         # 训练曲线
├── confusion_matrix.png
└── val_batch0_pred.jpg
```

使用TensorBoard监控:
```bash
tensorboard --logdir runs/detect
# 访问 http://localhost:6006
```

### 2.6 测试模型

```bash
# 测试图像
yolo task=detect mode=predict \
    model=runs/detect/landing_pad_detector/weights/best.pt \
    source=test_images/

# 测试视频
yolo task=detect mode=predict \
    model=runs/detect/landing_pad_detector/weights/best.pt \
    source=test_video.mp4 \
    show=True

# 测试摄像头
yolo task=detect mode=predict \
    model=runs/detect/landing_pad_detector/weights/best.pt \
    source=0 \
    show=True
```

### 2.7 集成到检测器

修改配置文件:

```yaml
# config/detector_config.yaml
image_detector:
  model_path: "runs/detect/landing_pad_detector/weights/best.pt"
  confidence_threshold: 0.6
  target_classes:
    - landing_pad
    - obstacle
```

---

## 三、数据增强技巧

### 3.1 手势数据增强

在训练时自动应用:

```python
def augment_landmarks(landmarks):
    """数据增强"""
    landmarks = np.array(landmarks).reshape(21, 3)

    # 1. 旋转
    angle = np.random.uniform(-15, 15)
    # ... 应用旋转

    # 2. 缩放
    scale = np.random.uniform(0.9, 1.1)
    landmarks *= scale

    # 3. 平移
    shift = np.random.uniform(-0.1, 0.1, size=3)
    landmarks += shift

    # 4. 添加噪声
    noise = np.random.normal(0, 0.01, size=landmarks.shape)
    landmarks += noise

    return landmarks.flatten()
```

### 3.2 图像数据增强

YOLO自动应用,可调整:

```python
model.train(
    # 几何变换
    degrees=10.0,      # 旋转 ±10度
    translate=0.1,     # 平移 ±10%
    scale=0.5,         # 缩放 ±50%
    shear=0.0,         # 剪切
    flipud=0.0,        # 垂直翻转概率
    fliplr=0.5,        # 水平翻转概率

    # 颜色变换
    hsv_h=0.015,       # 色调
    hsv_s=0.7,         # 饱和度
    hsv_v=0.4,         # 明度

    # 高级增强
    mosaic=1.0,        # Mosaic
    mixup=0.5,         # Mixup
    copy_paste=0.0,    # Copy-Paste
)
```

---

## 四、常见问题

### Q1: 手势识别准确率低?

**解决方案**:
1. 增加训练样本(每个手势200+)
2. 在不同光照/背景下采集
3. 尝试不同算法(Random Forest vs SVM)
4. 检查手势定义是否清晰

### Q2: YOLO训练过拟合?

**解决方案**:
```python
model.train(
    dropout=0.3,         # 增加dropout
    weight_decay=0.0005, # 权重衰减
    patience=20,         # 早停
    augment=True         # 强数据增强
)
```

### Q3: 训练显存不足?

**解决方案**:
```python
model.train(
    batch=4,        # 减小batch size
    imgsz=416,      # 减小图像尺寸
    workers=2,      # 减少workers
    cache='disk'    # 使用磁盘缓存
)
```

### Q4: 训练太慢?

**解决方案**:
```python
model.train(
    amp=True,       # 混合精度训练
    device=0,       # 使用GPU
    batch=32,       # 增大batch(如果显存够)
    workers=8       # 增加数据加载线程
)
```

---

## 五、推荐训练流程

### 方案A: 仅训练YOLO(推荐)

适合:需要检测降落靶标等自定义目标

```bash
# 1. 采集图像
python tools/collect_images.py collect --class landing_pad --num 500

# 2. 标注数据
labelImg

# 3. 训练
yolo task=detect mode=train model=yolov8n.pt data=dataset.yaml epochs=100

# 4. 测试
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=0
```

估计时间: **1-2天**

### 方案B: 训练手势分类器

适合:需要识别特定手势

```bash
# 1. 采集手势数据
python tools/collect_gesture_data.py

# 2. 训练
python tools/train_gesture_classifier.py --data gesture_dataset.json

# 3. 测试
python src/detectors/gesture_detector.py
```

估计时间: **半天**

### 方案C: 全部使用预训练模型(最简单)

适合:快速原型,通用场景

```bash
# 直接使用
python src/main.py --enable-all
```

估计时间: **10分钟**

---

## 六、模型性能对比

| 模型 | 大小 | 速度 | 准确率 | 训练难度 |
|------|------|------|--------|---------|
| **预训练Vosk** | 40MB | 实时 | 85% | 无需训练 ⭐ |
| **规则手势** | 0 | 30fps | 90% | 无需训练 ⭐ |
| **训练手势分类** | <1MB | 30fps | 95% | 容易 ⭐⭐ |
| **预训练YOLO** | 6MB | 25fps | 70% | 无需训练 ⭐ |
| **微调YOLO** | 6MB | 25fps | 90% | 中等 ⭐⭐⭐ |

---

**需要帮助?**
- 查看主文档: `docs/MODEL_TRAINING_GUIDE.md`
- 提交Issue: (项目仓库)
