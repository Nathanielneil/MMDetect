#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的多模态可视化界面
支持在一个窗口中切换查看四种模态的检测效果
"""

import cv2
import numpy as np
import threading
import time
import logging
from typing import Optional, Dict, Any, Tuple
from enum import Enum


class ViewMode(Enum):
    """视图模式"""
    GESTURE = "gesture"      # 手势识别视图
    IMAGE = "image"          # 图像检测视图
    MOUSE = "mouse"          # 鼠标路径规划视图
    QUAD = "quad"            # 四分屏视图
    OVERVIEW = "overview"    # 概览视图


class UnifiedViewer:
    """
    统一可视化查看器

    功能:
    1. 单窗口显示所有模态
    2. 键盘切换不同视图
    3. 实时显示检测结果
    4. 统计信息展示
    """

    def __init__(
        self,
        camera_manager=None,
        detectors: Dict[str, Any] = None,
        command_manager=None,
        window_size: Tuple[int, int] = (1280, 720)
    ):
        """
        初始化统一查看器

        Args:
            camera_manager: 摄像头管理器
            detectors: 检测器字典 {'gesture': detector, 'image': detector, ...}
            command_manager: 命令管理器
            window_size: 窗口大小 (width, height)
        """
        self.camera_manager = camera_manager
        self.detectors = detectors or {}
        self.command_manager = command_manager
        self.window_size = window_size

        self.window_name = "MMDect - Multi-Modal Viewer"
        self.current_mode = ViewMode.OVERVIEW

        self.is_running = False
        self.viewer_thread: Optional[threading.Thread] = None

        # 订阅摄像头
        self.camera_subscriber_id = "unified_viewer"

        # 鼠标路径画布
        self.mouse_canvas = np.ones((600, 800, 3), dtype=np.uint8) * 255

        # 最近的命令
        self.recent_commands = []
        self.max_recent_commands = 5

        # 统计信息
        self.fps = 0
        self.frame_count = 0
        self.start_time = None

        self.logger = logging.getLogger(self.__class__.__name__)

    def start(self):
        """启动查看器"""
        if self.is_running:
            self.logger.warning("查看器已在运行")
            return

        # 订阅摄像头
        if self.camera_manager:
            self.camera_manager.subscribe(self.camera_subscriber_id)

        self.is_running = True
        self.start_time = time.time()
        self.frame_count = 0

        self.viewer_thread = threading.Thread(
            target=self._viewer_loop,
            daemon=True
        )
        self.viewer_thread.start()
        self.logger.info("统一查看器已启动")

    def stop(self):
        """停止查看器"""
        if not self.is_running:
            return

        self.is_running = False

        if self.viewer_thread:
            self.viewer_thread.join(timeout=2.0)

        # 取消订阅
        if self.camera_manager:
            self.camera_manager.unsubscribe(self.camera_subscriber_id)

        cv2.destroyWindow(self.window_name)
        self.logger.info("查看器已停止")

    def add_command(self, command: Dict[str, Any]):
        """添加命令到最近命令列表"""
        self.recent_commands.append(command)
        if len(self.recent_commands) > self.max_recent_commands:
            self.recent_commands.pop(0)

    def _viewer_loop(self):
        """查看器主循环"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_size[0], self.window_size[1])

        while self.is_running:
            # 计算FPS
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed if elapsed > 0 else 0

            # 根据当前模式渲染画面
            if self.current_mode == ViewMode.GESTURE:
                frame = self._render_gesture_view()
            elif self.current_mode == ViewMode.IMAGE:
                frame = self._render_image_view()
            elif self.current_mode == ViewMode.MOUSE:
                frame = self._render_mouse_view()
            elif self.current_mode == ViewMode.QUAD:
                frame = self._render_quad_view()
            else:  # OVERVIEW
                frame = self._render_overview()

            # 添加顶部状态栏
            frame = self._add_status_bar(frame)

            # 添加底部控制提示
            frame = self._add_control_hints(frame)

            # 显示
            cv2.imshow(self.window_name, frame)

            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            self._handle_keyboard(key)

            time.sleep(0.001)

    def _render_gesture_view(self) -> np.ndarray:
        """渲染手势识别视图"""
        # 获取摄像头帧
        frame = self._get_camera_frame()
        if frame is None:
            frame = self._create_blank_frame("等待摄像头...")

        # 调整到窗口大小
        frame = cv2.resize(frame, self.window_size)

        # 添加手势检测信息
        gesture_detector = self.detectors.get('gesture')
        if gesture_detector:
            # TODO: 这里可以调用手势检测器的可视化方法
            cv2.putText(
                frame,
                "Gesture Detection Active",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3
            )

        # 添加标题
        cv2.putText(
            frame,
            "GESTURE RECOGNITION",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            2
        )

        return frame

    def _render_image_view(self) -> np.ndarray:
        """渲染图像检测视图"""
        frame = self._get_camera_frame()
        if frame is None:
            frame = self._create_blank_frame("等待摄像头...")

        frame = cv2.resize(frame, self.window_size)

        # 添加图像检测信息
        cv2.putText(
            frame,
            "IMAGE DETECTION (YOLOv8)",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Object Detection Active",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

        return frame

    def _render_mouse_view(self) -> np.ndarray:
        """渲染鼠标路径规划视图"""
        # 使用鼠标画布
        mouse_detector = self.detectors.get('mouse')
        if mouse_detector:
            canvas = mouse_detector.visualize_path(self.mouse_canvas.copy())
        else:
            canvas = self.mouse_canvas.copy()

        # 调整大小
        frame = cv2.resize(canvas, self.window_size)

        # 添加标题
        cv2.putText(
            frame,
            "MOUSE PATH PLANNING",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

        return frame

    def _render_quad_view(self) -> np.ndarray:
        """渲染四分屏视图"""
        # 获取原始帧
        camera_frame = self._get_camera_frame()
        if camera_frame is None:
            camera_frame = self._create_blank_frame("No Camera", (640, 480))

        # 计算每个子窗口的大小
        sub_width = self.window_size[0] // 2
        sub_height = self.window_size[1] // 2

        # 创建四个子视图
        gesture_frame = cv2.resize(camera_frame.copy(), (sub_width, sub_height))
        image_frame = cv2.resize(camera_frame.copy(), (sub_width, sub_height))

        mouse_detector = self.detectors.get('mouse')
        if mouse_detector:
            mouse_frame = mouse_detector.visualize_path(self.mouse_canvas.copy())
        else:
            mouse_frame = self.mouse_canvas.copy()
        mouse_frame = cv2.resize(mouse_frame, (sub_width, sub_height))

        stats_frame = self._create_stats_frame((sub_width, sub_height))

        # 添加标签
        self._add_label(gesture_frame, "GESTURE", (10, 30), (0, 255, 0))
        self._add_label(image_frame, "IMAGE", (10, 30), (255, 0, 0))
        self._add_label(mouse_frame, "MOUSE", (10, 30), (0, 0, 255))
        self._add_label(stats_frame, "STATS", (10, 30), (255, 255, 0))

        # 拼接四个视图
        top_row = np.hstack([gesture_frame, image_frame])
        bottom_row = np.hstack([mouse_frame, stats_frame])
        frame = np.vstack([top_row, bottom_row])

        return frame

    def _render_overview(self) -> np.ndarray:
        """渲染概览视图"""
        frame = self._create_blank_frame("", self.window_size)

        # 标题
        title = "MMDect Multi-Modal Drone Control System"
        cv2.putText(
            frame,
            title,
            (self.window_size[0]//2 - 400, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3
        )

        # 显示四个模态的状态
        y_start = 150
        spacing = 80

        modalities = [
            ("1. GESTURE", "gesture", (0, 255, 0)),
            ("2. IMAGE", "image", (255, 100, 0)),
            ("3. MOUSE", "mouse", (0, 100, 255)),
            ("4. VOICE", "voice", (255, 0, 255))
        ]

        for i, (name, key, color) in enumerate(modalities):
            y = y_start + i * spacing

            # 状态指示器
            status = "ACTIVE" if key in self.detectors else "INACTIVE"
            status_color = (0, 255, 0) if key in self.detectors else (0, 0, 255)

            cv2.putText(
                frame,
                name,
                (200, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2
            )

            cv2.putText(
                frame,
                status,
                (500, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2
            )

        # 最近命令
        y = y_start + len(modalities) * spacing + 60
        cv2.putText(
            frame,
            "Recent Commands:",
            (200, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        for i, cmd in enumerate(self.recent_commands[-5:]):
            y += 35
            cmd_text = f"{cmd.get('modality', '?')}: {cmd.get('command', '?')} (conf: {cmd.get('confidence', 0):.2f})"
            cv2.putText(
                frame,
                cmd_text,
                (220, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1
            )

        return frame

    def _create_stats_frame(self, size: Tuple[int, int]) -> np.ndarray:
        """创建统计信息帧"""
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)

        y = 50
        spacing = 40

        # 系统统计
        stats = [
            f"FPS: {self.fps:.1f}",
            f"Frame: {self.frame_count}",
            f"Active: {len(self.detectors)}",
            f"Commands: {len(self.recent_commands)}"
        ]

        for stat in stats:
            cv2.putText(
                frame,
                stat,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            y += spacing

        # 摄像头统计
        if self.camera_manager:
            cam_stats = self.camera_manager.get_statistics()
            y += 20
            cv2.putText(
                frame,
                "Camera:",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            y += spacing

            cv2.putText(
                frame,
                f"Cam FPS: {cam_stats['actual_fps']:.1f}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1
            )
            y += spacing - 5

            cv2.putText(
                frame,
                f"Dropped: {cam_stats['dropped_frames']}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1
            )

        return frame

    def _get_camera_frame(self) -> Optional[np.ndarray]:
        """获取摄像头帧"""
        if not self.camera_manager:
            return None
        return self.camera_manager.get_frame(self.camera_subscriber_id, timeout=0.05)

    def _create_blank_frame(self, text: str, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """创建空白帧"""
        if size is None:
            size = self.window_size

        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)

        if text:
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = (size[0] - text_size[0]) // 2
            text_y = (size[1] + text_size[1]) // 2

            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (100, 100, 100),
                2
            )

        return frame

    def _add_label(self, frame: np.ndarray, text: str, pos: Tuple[int, int], color: Tuple[int, int, int]):
        """添加标签"""
        cv2.putText(
            frame,
            text,
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    def _add_status_bar(self, frame: np.ndarray) -> np.ndarray:
        """添加顶部状态栏"""
        bar_height = 40
        bar = np.zeros((bar_height, self.window_size[0], 3), dtype=np.uint8)

        # 当前模式
        mode_text = f"Mode: {self.current_mode.value.upper()}"
        cv2.putText(
            bar,
            mode_text,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(
            bar,
            fps_text,
            (self.window_size[0] - 150, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        return np.vstack([bar, frame])

    def _add_control_hints(self, frame: np.ndarray) -> np.ndarray:
        """添加底部控制提示"""
        bar_height = 50
        bar = np.zeros((bar_height, self.window_size[0], 3), dtype=np.uint8)

        hints = "Keys: [1]Gesture [2]Image [3]Mouse [4]Quad [0]Overview [R]Reset [Q]Quit"
        cv2.putText(
            bar,
            hints,
            (10, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1
        )

        return np.vstack([frame, bar])

    def _handle_keyboard(self, key: int):
        """处理键盘输入"""
        if key == ord('q'):
            self.stop()
        elif key == ord('1'):
            self.current_mode = ViewMode.GESTURE
            self.logger.info("切换到手势识别视图")
        elif key == ord('2'):
            self.current_mode = ViewMode.IMAGE
            self.logger.info("切换到图像检测视图")
        elif key == ord('3'):
            self.current_mode = ViewMode.MOUSE
            self.logger.info("切换到鼠标路径视图")
        elif key == ord('4'):
            self.current_mode = ViewMode.QUAD
            self.logger.info("切换到四分屏视图")
        elif key == ord('0'):
            self.current_mode = ViewMode.OVERVIEW
            self.logger.info("切换到概览视图")
        elif key == ord('r') or key == ord('R'):
            # 重置
            self.recent_commands.clear()
            if 'mouse' in self.detectors:
                self.detectors['mouse'].reset()
            self.mouse_canvas = np.ones((600, 800, 3), dtype=np.uint8) * 255
            self.logger.info("已重置")


# 使用示例
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

    from src.core.camera_manager import CameraManager

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建摄像头管理器
    cam_manager = CameraManager(camera_index=0, fps=30)

    if cam_manager.open():
        cam_manager.start()

        # 创建统一查看器
        viewer = UnifiedViewer(
            camera_manager=cam_manager,
            detectors={},
            window_size=(1280, 720)
        )
        viewer.start()

        print("\n" + "="*60)
        print("统一查看器运行中")
        print("="*60)
        print("键盘控制:")
        print("  [1] 手势识别视图")
        print("  [2] 图像检测视图")
        print("  [3] 鼠标路径视图")
        print("  [4] 四分屏视图")
        print("  [0] 概览视图")
        print("  [R] 重置")
        print("  [Q] 退出")
        print("="*60)

        try:
            while viewer.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n收到中断信号")

        viewer.stop()
        cam_manager.close()
