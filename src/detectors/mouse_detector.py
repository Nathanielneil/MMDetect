"""
鼠标/触屏路径规划检测器
通过拖动绘制路径点生成航点指令
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging

# 支持直接运行和模块导入两种方式
try:
    from .base_detector import BaseDetector, CommandType, ModalityType
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.detectors.base_detector import BaseDetector, CommandType, ModalityType


class MouseDetector(BaseDetector):
    """
    鼠标/触屏检测器

    通过鼠标拖动或触屏绘制路径,生成航点指令
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化鼠标检测器

        Args:
            config: 配置字典,包含:
                - min_path_length: 最小路径长度(默认20像素)
                - simplify_epsilon: 路径简化参数(默认10)
                - scale_factor: 坐标缩放因子(默认0.01,像素->米)
        """
        super().__init__(ModalityType.MOUSE, config)

        self.min_path_length = self.config.get('min_path_length', 20)
        self.simplify_epsilon = self.config.get('simplify_epsilon', 10)
        self.scale_factor = self.config.get('scale_factor', 0.01)

        self.path_points = []
        self.is_drawing = False
        self.canvas = None

    def initialize(self) -> bool:
        """初始化检测器"""
        try:
            self.is_initialized = True
            self.logger.info("鼠标/触屏检测器初始化成功")
            return True

        except Exception as e:
            self.logger.error(f"鼠标检测器初始化失败: {str(e)}")
            return False

    def preprocess(self, mouse_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        预处理鼠标数据

        Args:
            mouse_data: 鼠标事件数据,包含:
                - event: 事件类型('down', 'move', 'up')
                - x, y: 坐标
                - canvas_size: 画布大小(可选)

        Returns:
            Dict: 预处理后的数据
        """
        return mouse_data

    def detect(self, mouse_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        收集路径点

        Args:
            mouse_data: 鼠标事件数据

        Returns:
            Dict: 路径数据
        """
        event = mouse_data['event']
        x, y = mouse_data['x'], mouse_data['y']

        if event == 'down':
            self.is_drawing = True
            self.path_points = [(x, y)]

        elif event == 'move' and self.is_drawing:
            self.path_points.append((x, y))

        elif event == 'up':
            self.is_drawing = False
            if len(self.path_points) > 1:
                self.path_points.append((x, y))

        return {
            'path_points': self.path_points.copy(),
            'is_drawing': self.is_drawing,
            'path_length': len(self.path_points)
        }

    def postprocess(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        将路径转换为航点指令

        Args:
            detection_result: 路径数据

        Returns:
            Dict: 标准化指令字典
        """
        path_points = detection_result['path_points']
        is_drawing = detection_result['is_drawing']

        # 如果正在绘制,返回悬停
        if is_drawing or len(path_points) < 2:
            return self.create_command_dict(
                CommandType.HOVER,
                confidence=0.0,
                parameters={'reason': 'path_incomplete'}
            )

        # 简化路径
        simplified_path = self._simplify_path(path_points)

        if len(simplified_path) < 2:
            return self.create_command_dict(
                CommandType.HOVER,
                confidence=0.0,
                parameters={'reason': 'path_too_short'}
            )

        # 计算路径总长度
        total_length = self._calculate_path_length(simplified_path)

        if total_length < self.min_path_length:
            return self.create_command_dict(
                CommandType.HOVER,
                confidence=0.0,
                parameters={'reason': 'path_too_short'}
            )

        # 转换为世界坐标(简单的缩放)
        waypoints = self._to_world_coordinates(simplified_path)

        # 计算置信度(基于路径平滑度)
        confidence = self._calculate_path_confidence(simplified_path)

        parameters = {
            'waypoints': waypoints,
            'num_waypoints': len(waypoints),
            'total_distance': total_length * self.scale_factor,
            'raw_points_count': len(path_points)
        }

        return self.create_command_dict(
            CommandType.WAYPOINT,
            confidence=confidence,
            parameters=parameters
        )

    def _simplify_path(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        使用Ramer-Douglas-Peucker算法简化路径

        Args:
            points: 原始路径点

        Returns:
            List: 简化后的路径点
        """
        if len(points) < 3:
            return points

        points_array = np.array(points, dtype=np.float32)
        simplified = cv2.approxPolyDP(
            points_array,
            epsilon=self.simplify_epsilon,
            closed=False
        )

        # 确保返回的是整数元组，而不是numpy数组
        return [(int(pt[0][0]), int(pt[0][1])) for pt in simplified]

    def _calculate_path_length(self, points: List[Tuple[int, int]]) -> float:
        """
        计算路径总长度

        Args:
            points: 路径点

        Returns:
            float: 路径长度(像素)
        """
        if len(points) < 2:
            return 0.0

        total_length = 0.0
        for i in range(len(points) - 1):
            dx = points[i + 1][0] - points[i][0]
            dy = points[i + 1][1] - points[i][1]
            total_length += np.sqrt(dx ** 2 + dy ** 2)

        return total_length

    def _to_world_coordinates(
        self,
        points: List[Tuple[int, int]]
    ) -> List[Dict[str, float]]:
        """
        将像素坐标转换为世界坐标

        Args:
            points: 像素坐标点

        Returns:
            List[Dict]: 世界坐标航点列表
        """
        waypoints = []

        for i, (x, y) in enumerate(points):
            waypoint = {
                'x': x * self.scale_factor,
                'y': y * self.scale_factor,
                'z': 10.0,  # 默认高度10米
                'index': i
            }
            waypoints.append(waypoint)

        return waypoints

    def _calculate_path_confidence(self, points: List[Tuple[int, int]]) -> float:
        """
        计算路径置信度(基于平滑度)

        Args:
            points: 路径点

        Returns:
            float: 置信度(0-1)
        """
        if len(points) < 3:
            return 0.5

        # 计算转角
        angles = []
        for i in range(1, len(points) - 1):
            v1 = np.array(points[i]) - np.array(points[i - 1])
            v2 = np.array(points[i + 1]) - np.array(points[i])

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 > 0 and norm2 > 0:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                angles.append(angle)

        if not angles:
            return 0.5

        # 平均转角越小,路径越平滑,置信度越高
        avg_angle = np.mean(angles)
        confidence = 1.0 - (avg_angle / np.pi) * 0.5

        return max(0.5, min(1.0, confidence))

    def visualize_path(
        self,
        canvas: np.ndarray,
        show_waypoints: bool = True
    ) -> np.ndarray:
        """
        可视化绘制的路径

        Args:
            canvas: 画布图像
            show_waypoints: 是否显示航点

        Returns:
            np.ndarray: 带有路径的图像
        """
        vis_canvas = canvas.copy()

        if len(self.path_points) < 2:
            return vis_canvas

        # 绘制原始路径
        for i in range(len(self.path_points) - 1):
            cv2.line(
                vis_canvas,
                self.path_points[i],
                self.path_points[i + 1],
                (0, 255, 0),
                2
            )

        if show_waypoints:
            # 简化路径并显示航点
            simplified = self._simplify_path(self.path_points)
            for i, pt in enumerate(simplified):
                cv2.circle(vis_canvas, pt, 5, (0, 0, 255), -1)
                cv2.putText(
                    vis_canvas,
                    str(i),
                    (pt[0] + 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1
                )

        return vis_canvas

    def reset(self):
        """重置路径"""
        self.path_points = []
        self.is_drawing = False

    def shutdown(self):
        """释放资源"""
        super().shutdown()
        self.reset()


# 使用示例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    detector = MouseDetector()

    if detector.initialize():
        # 创建画布
        canvas = np.ones((600, 800, 3), dtype=np.uint8) * 255

        def mouse_callback(event, x, y, flags, param):
            """OpenCV鼠标回调"""
            mouse_data = {'x': x, 'y': y}

            if event == cv2.EVENT_LBUTTONDOWN:
                mouse_data['event'] = 'down'
                detector.run(mouse_data)

            elif event == cv2.EVENT_MOUSEMOVE:
                mouse_data['event'] = 'move'
                detector.run(mouse_data)

            elif event == cv2.EVENT_LBUTTONUP:
                mouse_data['event'] = 'up'
                command = detector.run(mouse_data)

                if command and command['confidence'] > 0.5:
                    print(f"\n检测到航点指令:")
                    print(f"  航点数量: {command['parameters']['num_waypoints']}")
                    print(f"  总距离: {command['parameters']['total_distance']:.2f}米")
                    print(f"  置信度: {command['confidence']:.2f}")
                    print(f"  航点: {command['parameters']['waypoints']}")

        cv2.namedWindow('Path Planner')
        cv2.setMouseCallback('Path Planner', mouse_callback)

        print("使用鼠标拖动绘制路径,按'r'重置,按'q'退出")

        while True:
            vis_canvas = detector.visualize_path(canvas.copy())

            # 添加说明
            cv2.putText(
                vis_canvas,
                "Drag to draw path, 'r' to reset, 'q' to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1
            )

            cv2.imshow('Path Planner', vis_canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                detector.reset()
                canvas = np.ones((600, 800, 3), dtype=np.uint8) * 255

        cv2.destroyAllWindows()
        detector.shutdown()
