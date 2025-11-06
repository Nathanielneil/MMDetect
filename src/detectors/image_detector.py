"""
图像/目标检测器
使用YOLOv8-nano进行轻量级目标检测
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List
import logging

try:
    from ultralytics import YOLO
except ImportError:
    logging.warning("Ultralytics未安装,图像检测器将无法使用")

# 支持直接运行和模块导入两种方式
try:
    from .base_detector import BaseDetector, CommandType, ModalityType
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.detectors.base_detector import BaseDetector, CommandType, ModalityType


class ImageDetector(BaseDetector):
    """
    图像/目标检测器

    使用YOLOv8-nano检测场景中的目标(如降落靶标),触发对应指令
    """

    # 靶标符号到指令的映射（自定义靶标）
    SYMBOL_MARKER_MAP = {
        'H': CommandType.LAND,           # H形靶标 -> 降落 (Helipad)
        'T': CommandType.TAKEOFF,        # T形靶标 -> 起飞 (Takeoff)
        'F': CommandType.FORMATION,      # F形靶标 -> 编队 (Formation)
        'S': CommandType.EXPLORE,        # S形靶标 -> 搜索 (Search)
        'L': CommandType.HOVER           # L形靶标 -> 悬停 (Loiter)
    }

    # 目标类别到指令的映射（通用目标，备用）
    TARGET_COMMAND_MAP = {
        'landing_pad': CommandType.LAND,         # 降落靶标 -> 降落
        'person': CommandType.HOVER,             # 人 -> 悬停
        'car': CommandType.EXPLORE,              # 车 -> 探索
        'stop_sign': CommandType.EMERGENCY_STOP  # 停止标志 -> 紧急停止
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化图像检测器

        Args:
            config: 配置字典,包含:
                - model_path: YOLOv8模型路径(默认yolov8n.pt)
                - confidence_threshold: 置信度阈值(默认0.6)
                - iou_threshold: IoU阈值(默认0.45)
                - target_classes: 目标类别列表
        """
        super().__init__(ModalityType.IMAGE, config)

        self.model_path = self.config.get('model_path', 'models/yolov8n.pt')
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.iou_threshold = self.config.get('iou_threshold', 0.45)
        self.target_classes = self.config.get('target_classes', None)

        self.model = None
        self.class_names = None

    def initialize(self) -> bool:
        """初始化YOLOv8模型"""
        try:
            self.logger.info(f"加载YOLOv8模型: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.class_names = self.model.names

            self.is_initialized = True
            self.logger.info(f"图像检测器初始化成功,支持{len(self.class_names)}个类别")
            return True

        except Exception as e:
            self.logger.error(f"图像检测器初始化失败: {str(e)}")
            return False

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像

        Args:
            image: BGR图像

        Returns:
            np.ndarray: 预处理后的图像
        """
        # YOLOv8会自动处理,这里只做基本检查
        if image is None or image.size == 0:
            raise ValueError("输入图像无效")

        return image

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        执行目标检测

        Args:
            image: 预处理后的图像

        Returns:
            Dict: 检测结果
        """
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.target_classes,
            verbose=False
        )[0]

        detections = []

        for box in results.boxes:
            detection = {
                'class_id': int(box.cls[0]),
                'class_name': self.class_names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                'center': self._get_bbox_center(box.xyxy[0].cpu().numpy())
            }
            detections.append(detection)

        return {
            'detections': detections,
            'count': len(detections)
        }

    def postprocess(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        将检测结果转换为指令

        Args:
            detection_result: 检测结果

        Returns:
            Dict: 标准化指令字典
        """
        detections = detection_result['detections']

        if not detections:
            return self.create_command_dict(
                CommandType.HOVER,
                confidence=0.0,
                parameters={'reason': 'no_target_detected'}
            )

        # 选择置信度最高的检测
        best_detection = max(detections, key=lambda x: x['confidence'])

        # 映射到指令
        class_name = best_detection['class_name']

        # 优先使用符号靶标映射（对应训练的YOLOv8模型）
        command_type = self.SYMBOL_MARKER_MAP.get(class_name)

        # 如果不是符号靶标，使用通用目标映射
        if command_type is None:
            command_type = self.TARGET_COMMAND_MAP.get(class_name, CommandType.HOVER)

        parameters = {
            'detected_object': class_name,
            'object_center': best_detection['center'],
            'bbox': best_detection['bbox'],
            'num_detections': len(detections),
            'detection_type': 'symbol_marker' if class_name in self.SYMBOL_MARKER_MAP else 'generic_target'
        }

        # 如果检测到符号靶标，添加对应的类型
        if class_name in self.SYMBOL_MARKER_MAP:
            parameters['symbol'] = class_name
            parameters['target_center'] = best_detection['center']

        # 如果检测到降落靶标,添加降落参数
        if class_name == 'landing_pad':
            parameters['landing_position'] = best_detection['center']

        return self.create_command_dict(
            command_type,
            confidence=best_detection['confidence'],
            parameters=parameters
        )

    def _get_bbox_center(self, bbox: np.ndarray) -> List[float]:
        """
        计算边界框中心

        Args:
            bbox: [x1, y1, x2, y2]

        Returns:
            List[float]: [center_x, center_y]
        """
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2]

    def visualize(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        可视化检测结果

        Args:
            image: 原始图像
            detections: 检测结果列表

        Returns:
            np.ndarray: 带有标注的图像
        """
        vis_image = image.copy()

        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            class_name = det['class_name']
            confidence = det['confidence']

            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            # 绘制中心点
            center = tuple(map(int, det['center']))
            cv2.circle(vis_image, center, 5, (0, 0, 255), -1)

        return vis_image

    def detect_custom_target(
        self,
        image: np.ndarray,
        target_name: str,
        min_confidence: float = 0.5
    ) -> Optional[Dict[str, Any]]:
        """
        检测特定目标

        Args:
            image: 输入图像
            target_name: 目标类别名称
            min_confidence: 最小置信度

        Returns:
            Optional[Dict]: 检测到的目标信息,未检测到返回None
        """
        result = self.detect(image)

        for det in result['detections']:
            if det['class_name'] == target_name and det['confidence'] >= min_confidence:
                return det

        return None

    def detect_symbol(self, image: np.ndarray, symbol: str) -> Optional[Dict[str, Any]]:
        """
        检测特定符号靶标

        Args:
            image: 输入图像
            symbol: 符号类型 (H/T/F/S/L)

        Returns:
            Optional[Dict]: 检测到的符号信息,未检测到返回None
        """
        if symbol not in self.SYMBOL_MARKER_MAP:
            self.logger.warning(f"未知符号类型: {symbol}")
            return None

        return self.detect_custom_target(image, symbol, self.confidence_threshold)

    @staticmethod
    def create_symbol_detector_config(model_path: str, confidence: float = 0.6) -> Dict[str, Any]:
        """
        创建用于符号检测的配置

        Args:
            model_path: 训练好的YOLOv8模型路径
            confidence: 置信度阈值

        Returns:
            Dict: ImageDetector的配置字典
        """
        return {
            'model_path': model_path,
            'confidence_threshold': confidence,
            'iou_threshold': 0.45,
            'target_classes': None  # 检测所有类别
        }


# 使用示例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 方案1: 使用训练好的符号检测模型
    # config = ImageDetector.create_symbol_detector_config(
    #     model_path='/home/ubuntu/NGW/intern/MMDect/symbol_dataset/runs/detect/train2/weights/best.pt',
    #     confidence=0.6
    # )

    # 方案2: 使用通用的YOLOv8模型
    config = {
        'model_path': 'yolov8n.pt',
        'confidence_threshold': 0.5
    }

    detector = ImageDetector(config)

    if detector.initialize():
        cap = cv2.VideoCapture(0)

        print("按'q'退出")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # 检测目标
            command = detector.run(frame)

            if command and command['confidence'] > 0.5:
                # 可视化
                det_result = detector.detect(frame)
                vis_frame = detector.visualize(frame, det_result['detections'])

                # 显示指令
                cv2.putText(
                    vis_frame,
                    f"Command: {command['command']} ({command['confidence']:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2
                )

                cv2.imshow('Object Detection', vis_frame)

                print(f"检测到指令: {command}")
            else:
                cv2.imshow('Object Detection', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        detector.shutdown()
