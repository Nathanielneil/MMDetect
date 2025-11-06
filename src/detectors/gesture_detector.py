"""
手势识别检测器
使用MediaPipe Hands + 简单分类器
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Deque
from collections import deque
import time
import logging

try:
    import mediapipe as mp
except ImportError:
    logging.warning("MediaPipe未安装,手势检测器将无法使用")

# 支持直接运行和模块导入两种方式
try:
    from .base_detector import BaseDetector, CommandType, ModalityType
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.detectors.base_detector import BaseDetector, CommandType, ModalityType


class GestureDetector(BaseDetector):
    """
    手势识别检测器

    使用MediaPipe Hands检测手部关键点,通过手势规则映射到指令
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化手势检测器

        Args:
            config: 配置字典,包含:
                - min_detection_confidence: 最小检测置信度(默认0.7)
                - min_tracking_confidence: 最小跟踪置信度(默认0.5)
                - max_num_hands: 最大手数(默认1)
        """
        super().__init__(ModalityType.GESTURE, config)

        self.min_detection_confidence = self.config.get('min_detection_confidence', 0.5)  # 降低检测阈值
        self.min_tracking_confidence = self.config.get('min_tracking_confidence', 0.5)
        self.max_num_hands = self.config.get('max_num_hands', 1)

        self.mp_hands = None
        self.hands = None
        self.mp_draw = None

        # 时序平滑参数
        self.history_window_size = self.config.get('history_window_size', 5)  # 历史帧窗口大小
        self.min_consecutive_frames = self.config.get('min_consecutive_frames', 3)  # 最小连续帧数
        self.command_timeout = self.config.get('command_timeout', 1.0)  # 指令超时时间(秒)

        # 历史记录
        self.command_history: Deque = deque(maxlen=self.history_window_size)
        self.last_stable_command = None  # 上一次稳定的指令
        self.last_stable_time = 0  # 上一次稳定指令的时间
        self.current_command_count = 0  # 当前指令的连续出现次数
        self.current_command_type = None  # 当前正在统计的指令类型

        # 可视化效果参数
        self.enable_glow = self.config.get('enable_glow', True)  # 发光效果
        self.enable_trails = self.config.get('enable_trails', True)  # 轨迹效果
        self.trail_length = self.config.get('trail_length', 20)  # 轨迹长度
        self.fingertip_trails = {i: deque(maxlen=self.trail_length) for i in [4, 8, 12, 16, 20]}  # 5个指尖的轨迹
        self.pulse_phase = 0  # 脉冲动画相位

    def initialize(self) -> bool:
        """初始化MediaPipe Hands"""
        try:
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils

            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )

            self.is_initialized = True
            self.logger.info("手势检测器初始化成功")
            return True

        except Exception as e:
            self.logger.error(f"手势检测器初始化失败: {str(e)}")
            return False

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像

        Args:
            image: BGR图像

        Returns:
            np.ndarray: RGB图像
        """
        # MediaPipe需要RGB格式
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def detect(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """
        检测手部关键点

        Args:
            image_rgb: RGB图像

        Returns:
            Dict: 检测结果,包含关键点坐标
        """
        results = self.hands.process(image_rgb)

        detection_result = {
            'detected': False,
            'landmarks': None,
            'handedness': None
        }

        if results.multi_hand_landmarks:
            detection_result['detected'] = True
            detection_result['landmarks'] = results.multi_hand_landmarks[0]
            if results.multi_handedness:
                detection_result['handedness'] = results.multi_handedness[0].classification[0].label

        # 保存最后一次检测结果供可视化使用
        self._last_detection_result = detection_result

        return detection_result

    def postprocess(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        将手势识别为指令（带时序平滑）

        Args:
            detection_result: 手部检测结果

        Returns:
            Dict: 标准化指令字典（经过平滑处理）
        """
        if not detection_result['detected']:
            raw_command = self.create_command_dict(
                CommandType.HOVER,
                confidence=0.0,
                parameters={'reason': 'no_hand_detected'}
            )
        else:
            landmarks = detection_result['landmarks']

            # 提取关键点
            landmarks_list = []
            for lm in landmarks.landmark:
                landmarks_list.append([lm.x, lm.y, lm.z])

            # 识别手势（传入左右手信息）
            handedness = detection_result.get('handedness', 'Right')
            gesture_type, confidence = self._classify_gesture(landmarks_list, handedness)

            # 映射到指令
            command_type = self._gesture_to_command(gesture_type)

            parameters = {
                'gesture_type': gesture_type,
                'handedness': handedness
            }

            raw_command = self.create_command_dict(command_type, confidence, parameters)

        # 应用时序平滑
        smoothed_command = self._apply_temporal_smoothing(raw_command)

        return smoothed_command

    def _classify_gesture(self, landmarks: List[List[float]], handedness: str = 'Right') -> tuple:
        """
        分类手势

        Args:
            landmarks: 21个手部关键点坐标
            handedness: 左手('Left')或右手('Right')

        Returns:
            tuple: (手势类型, 置信度)
        """
        # 优先检测握拳（使用专门的判断函数）
        if self._is_fist(landmarks):
            return 'fist', 0.95  # 高置信度

        # 计算手指伸展状态（考虑左右手）
        fingers_up = self._count_fingers_up(landmarks, handedness)

        # 规则分类
        if fingers_up == 5:  # 张开手掌
            return 'open_palm', 0.9
        elif fingers_up == 0:  # 手指数为0但不是握拳（可能是检测异常）
            return 'fist', 0.7  # 降低置信度
        elif fingers_up == 1:  # 竖起一根手指
            return 'one_finger', 0.85
        elif fingers_up == 2:  # V字手势
            if self._is_v_gesture(landmarks):
                return 'v_sign', 0.85
            return 'two_fingers', 0.8
        elif fingers_up == 3:
            return 'three_fingers', 0.8
        elif fingers_up == 4:
            return 'four_fingers', 0.8
        else:
            return 'unknown', 0.3

    def _count_fingers_up(self, landmarks: List[List[float]], handedness: str = 'Right') -> int:
        """
        计算伸展手指数量

        Args:
            landmarks: 手部关键点
            handedness: 左手('Left')或右手('Right')

        Returns:
            int: 伸展手指数量
        """
        # MediaPipe手部关键点索引
        # 0: 手腕, 4: 拇指尖, 8: 食指尖, 12: 中指尖, 16: 无名指尖, 20: 小指尖
        finger_tips = [4, 8, 12, 16, 20]
        finger_mcp = [2, 5, 9, 13, 17]  # MCP关节（掌指关节）

        count = 0

        # 拇指: 根据左右手判断
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]

        if handedness == 'Right':
            # 右手：拇指伸展时x坐标更小（向左）
            thumb_is_open = thumb_tip[0] < thumb_mcp[0] - 0.04
        else:  # Left
            # 左手：拇指伸展时x坐标更大（向右）
            thumb_is_open = thumb_tip[0] > thumb_mcp[0] + 0.04

        if thumb_is_open:
            count += 1

        # 其他四指: 使用y坐标判断（左右手相同）
        for i in range(1, 5):
            tip = landmarks[finger_tips[i]]
            mcp = landmarks[finger_mcp[i]]

            # 手指伸展：指尖明显高于MCP关节
            if tip[1] < mcp[1] - 0.03:  # y轴向下为正
                count += 1

        return count

    def _is_fist(self, landmarks: List[List[float]]) -> bool:
        """
        专门判断是否为握拳

        Args:
            landmarks: 手部关键点

        Returns:
            bool: 是否为握拳
        """
        # 策略：所有指尖都靠近手掌中心
        palm_center_y = (landmarks[0][1] + landmarks[9][1]) / 2  # 手腕和中指MCP的中点

        # 检查所有指尖是否都低于（或接近）手掌中心
        finger_tips = [4, 8, 12, 16, 20]
        tips_below_palm = 0

        for tip_idx in finger_tips:
            if landmarks[tip_idx][1] >= palm_center_y - 0.05:  # 指尖在手掌中心附近或下方
                tips_below_palm += 1

        # 至少4根手指的指尖都收起来，才算握拳
        return tips_below_palm >= 4

    def _is_v_gesture(self, landmarks: List[List[float]]) -> bool:
        """
        判断是否为V字手势

        Args:
            landmarks: 手部关键点

        Returns:
            bool: 是否为V字手势
        """
        # 检查食指和中指伸展,其他手指收起
        index_up = landmarks[8][1] < landmarks[6][1]
        middle_up = landmarks[12][1] < landmarks[10][1]
        ring_down = landmarks[16][1] > landmarks[14][1]
        pinky_down = landmarks[20][1] > landmarks[18][1]

        return index_up and middle_up and ring_down and pinky_down

    def _gesture_to_command(self, gesture_type: str) -> CommandType:
        """
        手势到指令的映射

        Args:
            gesture_type: 手势类型

        Returns:
            CommandType: 指令类型
        """
        gesture_map = {
            'open_palm': CommandType.TAKEOFF,      # 张开手掌=起飞
            'fist': CommandType.LAND,              # 握拳=降落
            'one_finger': CommandType.HOVER,       # 一根手指=悬停
            'v_sign': CommandType.EXPLORE,         # V字=探索
            'three_fingers': CommandType.FORMATION # 三根手指=编队
        }

        return gesture_map.get(gesture_type, CommandType.HOVER)

    def _apply_temporal_smoothing(self, raw_command: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用时序平滑，减少帧间抖动

        策略：
        1. 维护最近N帧的指令历史
        2. 只有连续M帧都是同一指令，才认为是稳定的
        3. 对于关键指令(LAND/TAKEOFF)，要求更高的稳定性

        Args:
            raw_command: 原始检测到的指令

        Returns:
            Dict: 平滑后的指令
        """
        current_time = time.time()
        command_type = raw_command['command']

        # 将当前指令加入历史
        self.command_history.append({
            'command': command_type,
            'confidence': raw_command['confidence'],
            'time': current_time
        })

        # 统计历史窗口中每个指令的出现次数
        command_counts = {}
        for cmd in self.command_history:
            cmd_type = cmd['command']
            command_counts[cmd_type] = command_counts.get(cmd_type, 0) + 1

        # 找出出现最多的指令
        if command_counts:
            most_common_command = max(command_counts.items(), key=lambda x: x[1])
            most_common_type, count = most_common_command
        else:
            # 没有历史，返回当前指令
            return raw_command

        # 判断是否稳定（连续出现）
        is_stable = False

        # 检查最近的帧是否连续都是这个指令
        recent_commands = list(self.command_history)[-self.min_consecutive_frames:]
        if len(recent_commands) >= self.min_consecutive_frames:
            if all(cmd['command'] == most_common_type for cmd in recent_commands):
                is_stable = True

        # 对于危险指令（LAND/TAKEOFF），要求更严格
        dangerous_commands = [CommandType.LAND, CommandType.TAKEOFF]
        if most_common_type in dangerous_commands:
            # 要求所有历史帧都是这个指令
            if count == len(self.command_history) and count >= self.min_consecutive_frames:
                is_stable = True
            else:
                is_stable = False

        # 决定返回哪个指令
        if is_stable:
            # 稳定的新指令
            self.last_stable_command = most_common_type
            self.last_stable_time = current_time

            # 更新原始指令的command字段
            smoothed_command = raw_command.copy()
            smoothed_command['command'] = most_common_type
            smoothed_command['parameters']['smoothed'] = True
            smoothed_command['parameters']['stability'] = count / len(self.command_history)
            return smoothed_command

        elif self.last_stable_command is not None:
            # 不稳定，返回上一次稳定的指令
            # 但检查是否超时
            if current_time - self.last_stable_time < self.command_timeout:
                smoothed_command = raw_command.copy()
                smoothed_command['command'] = self.last_stable_command
                smoothed_command['parameters']['smoothed'] = True
                smoothed_command['parameters']['holding'] = True
                smoothed_command['parameters']['stability'] = 0.5
                return smoothed_command
            else:
                # 超时了，清除历史
                self.last_stable_command = None
                self.command_history.clear()
                return raw_command
        else:
            # 没有稳定的历史指令，返回HOVER（安全）
            safe_command = raw_command.copy()
            safe_command['command'] = CommandType.HOVER
            safe_command['parameters']['smoothed'] = True
            safe_command['parameters']['unstable'] = True
            return safe_command

    def visualize(self, image: np.ndarray, landmarks) -> np.ndarray:
        """
        可视化手部关键点（简单版本）

        Args:
            image: BGR图像
            landmarks: 手部关键点

        Returns:
            np.ndarray: 带有标注的图像
        """
        if landmarks:
            self.mp_draw.draw_landmarks(
                image,
                landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )
        return image

    def visualize_advanced(self, image: np.ndarray, landmarks, h: int, w: int) -> np.ndarray:
        """
        高级可视化 - 科技感效果

        Args:
            image: BGR图像
            landmarks: 手部关键点
            h: 图像高度
            w: 图像宽度

        Returns:
            np.ndarray: 带有高级特效的图像
        """
        if not landmarks:
            return image

        # 创建叠加层（用于透明效果）
        overlay = image.copy()

        # 获取关键点坐标
        landmark_points = []
        for lm in landmarks.landmark:
            x, y, z = int(lm.x * w), int(lm.y * h), lm.z
            landmark_points.append((x, y, z))

        # 1. 绘制渐变色骨架
        self._draw_gradient_skeleton(overlay, landmark_points)

        # 2. 绘制发光关键点
        if self.enable_glow:
            self._draw_glowing_landmarks(overlay, landmark_points)

        # 3. 绘制指尖轨迹
        if self.enable_trails:
            self._draw_fingertip_trails(overlay, landmark_points)

        # 4. 绘制手掌中心脉冲
        self._draw_palm_pulse(overlay, landmark_points)

        # 混合叠加层和原图
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        return image

    def _draw_gradient_skeleton(self, image: np.ndarray, points: List[tuple]):
        """绘制渐变色骨架"""
        # MediaPipe手部连接关系
        connections = [
            # 拇指
            (0, 1), (1, 2), (2, 3), (3, 4),
            # 食指
            (0, 5), (5, 6), (6, 7), (7, 8),
            # 中指
            (0, 9), (9, 10), (10, 11), (11, 12),
            # 无名指
            (0, 13), (13, 14), (14, 15), (15, 16),
            # 小指
            (0, 17), (17, 18), (18, 19), (19, 20),
            # 手掌
            (5, 9), (9, 13), (13, 17)
        ]

        for start_idx, end_idx in connections:
            if start_idx < len(points) and end_idx < len(points):
                start = points[start_idx]
                end = points[end_idx]

                # 根据深度(z)计算颜色 - 越近越亮
                depth_start = max(0, min(1, (start[2] + 0.1) * 5))  # 归一化到0-1
                depth_end = max(0, min(1, (end[2] + 0.1) * 5))

                # 渐变色: 蓝色 -> 青色 -> 绿色
                color_start = self._get_depth_color(depth_start)
                color_end = self._get_depth_color(depth_end)

                # 绘制渐变线条
                self._draw_gradient_line(image, (start[0], start[1]), (end[0], end[1]),
                                        color_start, color_end, thickness=3)

    def _draw_glowing_landmarks(self, image: np.ndarray, points: List[tuple]):
        """绘制发光关键点"""
        # 指尖索引
        fingertips = [4, 8, 12, 16, 20]

        # 脉冲效果
        self.pulse_phase = (self.pulse_phase + 0.1) % (2 * np.pi)
        pulse = int(abs(np.sin(self.pulse_phase)) * 3) + 2

        for idx, (x, y, z) in enumerate(points):
            if idx in fingertips:
                # 指尖 - 更亮更大
                radius = 8 + pulse
                glow_radius = 15 + pulse
                color = (0, 255, 255)  # 黄色
            else:
                # 普通关键点
                radius = 4
                glow_radius = 8
                color = (255, 200, 100)  # 青色

            # 外圈光晕
            cv2.circle(image, (x, y), glow_radius, color, -1)
            # 内圈核心（更亮）
            cv2.circle(image, (x, y), radius, (255, 255, 255), -1)

    def _draw_fingertip_trails(self, image: np.ndarray, points: List[tuple]):
        """绘制指尖轨迹"""
        fingertips = [4, 8, 12, 16, 20]

        for tip_idx in fingertips:
            if tip_idx < len(points):
                x, y, z = points[tip_idx]

                # 添加当前位置到轨迹
                self.fingertip_trails[tip_idx].append((x, y))

                # 绘制轨迹（渐隐效果）
                trail = list(self.fingertip_trails[tip_idx])
                for i in range(len(trail) - 1):
                    # 透明度从旧到新递增
                    alpha = int(255 * (i + 1) / len(trail))
                    thickness = max(1, int(3 * (i + 1) / len(trail)))

                    # 颜色：紫色到粉色渐变
                    color = (alpha, 0, 255 - alpha // 2)

                    cv2.line(image, trail[i], trail[i + 1], color, thickness)

    def _draw_palm_pulse(self, image: np.ndarray, points: List[tuple]):
        """绘制手掌中心脉冲效果（低调版本）"""
        # 手掌中心 = 关键点0, 5, 9, 13, 17的平均位置
        palm_indices = [0, 5, 9, 13, 17]
        palm_points = [points[i] for i in palm_indices if i < len(points)]

        if palm_points:
            palm_x = int(np.mean([p[0] for p in palm_points]))
            palm_y = int(np.mean([p[1] for p in palm_points]))

            # 脉冲半径（减小幅度）
            pulse_radius = int(35 + 5 * abs(np.sin(self.pulse_phase)))

            # 绘制单层细线圆环，颜色与骨架协调（青蓝色系）
            # 使用半透明青色，与深度渐变骨架颜色协调
            alpha = int(60 + 40 * abs(np.sin(self.pulse_phase)))  # 60-100动态透明度
            color = (200, 180, alpha)  # 青蓝色，低饱和度
            cv2.circle(image, (palm_x, palm_y), pulse_radius, color, 1)  # 细线

    def _get_depth_color(self, depth: float) -> tuple:
        """根据深度获取渐变色"""
        # 深度 0-1 映射到颜色
        # 近: 黄色 (0, 255, 255)
        # 中: 青色 (255, 255, 0)
        # 远: 蓝色 (255, 0, 0)
        if depth < 0.5:
            # 蓝 -> 青
            ratio = depth * 2
            return (255, int(255 * ratio), 0)
        else:
            # 青 -> 黄
            ratio = (depth - 0.5) * 2
            return (int(255 * (1 - ratio)), 255, int(255 * ratio))

    def _draw_gradient_line(self, image: np.ndarray, pt1: tuple, pt2: tuple,
                           color1: tuple, color2: tuple, thickness: int = 2):
        """绘制渐变线条"""
        # 简化版: 绘制多段小线条来模拟渐变
        steps = 10
        for i in range(steps):
            t = i / steps
            x = int(pt1[0] + (pt2[0] - pt1[0]) * t)
            y = int(pt1[1] + (pt2[1] - pt1[1]) * t)
            x_next = int(pt1[0] + (pt2[0] - pt1[0]) * (t + 1/steps))
            y_next = int(pt1[1] + (pt2[1] - pt1[1]) * (t + 1/steps))

            # 插值颜色
            color = tuple([int(c1 + (c2 - c1) * t) for c1, c2 in zip(color1, color2)])

            cv2.line(image, (x, y), (x_next, y_next), color, thickness)

    def shutdown(self):
        """释放资源"""
        super().shutdown()
        if self.hands:
            self.hands.close()


# 使用示例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    detector = GestureDetector()

    if detector.initialize():
        cap = cv2.VideoCapture(0)

        print("\n" + "="*60)
        print("  手势检测器已启动")
        print("="*60)
        print("\n支持的手势:")
        print("  🖐️  张开手掌  (5指) -> TAKEOFF   (起飞)")
        print("  ✊  握拳      (0指) -> LAND      (降落)")
        print("  ☝️  单指      (1指) -> HOVER     (悬停)")
        print("  ✌️  V字手势   (2指) -> EXPLORE   (探索)")
        print("  🤟  三指      (3指) -> FORMATION (编队)")
        print("\n操作提示:")
        print("  - 距离摄像头 30-60cm")
        print("  - 手放在画面中心")
        print("  - 每个手势保持1-2秒")
        print("  - 按 'q' 退出")
        print("="*60 + "\n")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # 只检测一次，获取指令
            command = detector.run(frame)

            # 从检测器获取最后一次的检测结果用于可视化
            if hasattr(detector, '_last_detection_result') and detector._last_detection_result:
                landmarks = detector._last_detection_result.get('landmarks')
                if landmarks:
                    # 绘制手部关键点
                    detector.mp_draw.draw_landmarks(
                        frame,
                        landmarks,
                        detector.mp_hands.HAND_CONNECTIONS,
                        detector.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        detector.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )

                    # 获取调试信息
                    landmarks_list = []
                    for lm in landmarks.landmark:
                        landmarks_list.append([lm.x, lm.y, lm.z])

                    # 获取左右手信息
                    handedness = detector._last_detection_result.get('handedness', 'Right')
                    fingers_count = detector._count_fingers_up(landmarks_list, handedness)
                    is_fist = detector._is_fist(landmarks_list)

                    # 显示调试信息（包含左右手）
                    debug_text = f"{handedness} Hand | Fingers: {fingers_count}"
                    if is_fist:
                        debug_text += " [FIST]"

                    cv2.putText(
                        frame,
                        debug_text,
                        (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        1
                    )

            if command and command['confidence'] > 0.5:
                # 获取手势信息
                gesture_type = command['parameters'].get('gesture_type', 'unknown')
                cmd_name = command['command'].upper() if isinstance(command['command'], str) else command['command'].name

                # 检查是否经过平滑处理
                is_smoothed = command['parameters'].get('smoothed', False)
                stability = command['parameters'].get('stability', 0.0)
                is_holding = command['parameters'].get('holding', False)

                # 显示指令（更清晰的格式）
                display_text = f"{gesture_type} -> {cmd_name}"
                if is_holding:
                    display_text += " [HOLD]"
                elif is_smoothed:
                    display_text += f" [S:{stability:.1f}]"

                color = (0, 255, 0) if stability > 0.8 else (0, 255, 255)  # 高稳定性=绿色，否则=黄色

                cv2.putText(
                    frame,
                    display_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                # 显示置信度
                cv2.putText(
                    frame,
                    f"Confidence: {command['confidence']:.2f}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    1
                )

                # 终端打印（包含稳定性信息）
                status = ""
                if is_holding:
                    status = "[保持]"
                elif is_smoothed:
                    status = f"[稳定度:{stability:.2f}]"

                print(f"✓ {gesture_type:15s} -> {cmd_name:12s} (置信度: {command['confidence']:.2f}) {status}")
            elif command and command['confidence'] == 0.0:
                # 未检测到手部
                cv2.putText(
                    frame,
                    "No hand detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            cv2.imshow('Gesture Detection', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        detector.shutdown()
