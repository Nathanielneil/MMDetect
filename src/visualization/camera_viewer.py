#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
摄像头可视化查看器
显示从CameraManager订阅的摄像头流
"""

import cv2
import numpy as np
import threading
import time
import logging
from typing import Optional


class CameraViewer:
    """
    摄像头查看器

    从CameraManager订阅帧并显示可视化窗口
    """

    def __init__(self, camera_manager, window_name: str = "Camera View"):
        """
        初始化查看器

        Args:
            camera_manager: CameraManager实例
            window_name: 窗口名称
        """
        self.camera_manager = camera_manager
        self.window_name = window_name
        self.subscriber_id = f"viewer_{window_name}"

        self.is_running = False
        self.viewer_thread: Optional[threading.Thread] = None

        self.logger = logging.getLogger(self.__class__.__name__)

    def start(self):
        """启动查看器线程"""
        if self.is_running:
            self.logger.warning("查看器已在运行")
            return

        # 订阅摄像头
        if not self.camera_manager.subscribe(self.subscriber_id):
            self.logger.error("订阅摄像头失败")
            return

        self.is_running = True
        self.viewer_thread = threading.Thread(
            target=self._viewer_loop,
            daemon=True
        )
        self.viewer_thread.start()
        self.logger.info(f"查看器已启动: {self.window_name}")

    def stop(self):
        """停止查看器"""
        if not self.is_running:
            return

        self.is_running = False

        if self.viewer_thread:
            self.viewer_thread.join(timeout=2.0)

        # 取消订阅
        self.camera_manager.unsubscribe(self.subscriber_id)

        cv2.destroyWindow(self.window_name)
        self.logger.info("查看器已停止")

    def _viewer_loop(self):
        """查看器循环"""
        cv2.namedWindow(self.window_name)

        while self.is_running:
            # 获取帧
            frame = self.camera_manager.get_frame(self.subscriber_id, timeout=0.1)

            if frame is None:
                continue

            # 添加信息文本
            display_frame = frame.copy()

            # 添加标题
            cv2.putText(
                display_frame,
                self.window_name,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            # 添加FPS
            stats = self.camera_manager.get_statistics()
            fps_text = f"FPS: {stats['actual_fps']:.1f}"
            cv2.putText(
                display_frame,
                fps_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # 显示
            cv2.imshow(self.window_name, display_frame)

            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break


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

        # 创建查看器
        viewer = CameraViewer(cam_manager, "Camera View")
        viewer.start()

        print("摄像头查看器运行中...")
        print("按 'q' 键退出")

        try:
            while viewer.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n收到中断信号")

        viewer.stop()
        cam_manager.close()
