#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
摄像头管理器
实现单个摄像头数据流的多检测器共享
"""

import cv2
import numpy as np
import threading
import time
import logging
from typing import Optional, Callable, Dict, List
from queue import Queue, Full


class CameraManager:
    """
    摄像头管理器

    功能：
    1. 统一管理摄像头资源
    2. 一个摄像头支持多个检测器订阅
    3. 独立线程读取帧，避免阻塞
    4. 为每个订阅者维护独立的帧队列
    """

    def __init__(self, camera_index: int = 0, fps: int = 30, buffer_size: int = 2):
        """
        初始化摄像头管理器

        Args:
            camera_index: 摄像头索引
            fps: 目标帧率
            buffer_size: 每个订阅者的缓冲区大小
        """
        self.camera_index = camera_index
        self.fps = fps
        self.buffer_size = buffer_size

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False
        self.is_running = False

        # 订阅者管理
        self.subscribers: Dict[str, Queue] = {}
        self.subscriber_lock = threading.Lock()

        # 捕获线程
        self.capture_thread: Optional[threading.Thread] = None

        # 统计信息
        self.frame_count = 0
        self.dropped_frames = 0
        self.start_time = None

        self.logger = logging.getLogger(self.__class__.__name__)

    def open(self) -> bool:
        """
        打开摄像头

        Returns:
            bool: 是否成功打开
        """
        try:
            self.logger.info(f"正在打开摄像头 {self.camera_index}...")

            self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                self.logger.error(f"无法打开摄像头 {self.camera_index}")
                return False

            # 测试读取一帧
            success, frame = self.cap.read()
            if not success:
                self.logger.error("无法读取摄像头帧")
                self.cap.release()
                return False

            self.is_opened = True
            self.logger.info(
                f"摄像头 {self.camera_index} 已打开 "
                f"(分辨率: {frame.shape[1]}x{frame.shape[0]})"
            )

            return True

        except Exception as e:
            self.logger.error(f"打开摄像头失败: {str(e)}")
            return False

    def start(self) -> bool:
        """
        启动捕获线程

        Returns:
            bool: 是否成功启动
        """
        if not self.is_opened:
            self.logger.error("摄像头未打开，请先调用 open()")
            return False

        if self.is_running:
            self.logger.warning("捕获线程已在运行")
            return True

        self.is_running = True
        self.start_time = time.time()
        self.frame_count = 0
        self.dropped_frames = 0

        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )
        self.capture_thread.start()

        self.logger.info("摄像头捕获线程已启动")
        return True

    def stop(self):
        """停止捕获线程"""
        if not self.is_running:
            return

        self.logger.info("正在停止摄像头捕获...")
        self.is_running = False

        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)

        self.logger.info("摄像头捕获已停止")

    def close(self):
        """关闭摄像头"""
        self.stop()

        if self.cap:
            self.cap.release()
            self.is_opened = False
            self.logger.info(f"摄像头 {self.camera_index} 已关闭")

        # 清空所有订阅者
        with self.subscriber_lock:
            self.subscribers.clear()

    def subscribe(self, subscriber_id: str) -> bool:
        """
        订阅摄像头帧

        Args:
            subscriber_id: 订阅者唯一标识

        Returns:
            bool: 是否成功订阅
        """
        with self.subscriber_lock:
            if subscriber_id in self.subscribers:
                self.logger.warning(f"订阅者 {subscriber_id} 已存在")
                return False

            # 为订阅者创建帧队列
            self.subscribers[subscriber_id] = Queue(maxsize=self.buffer_size)
            self.logger.info(f"订阅者 {subscriber_id} 已注册")
            return True

    def unsubscribe(self, subscriber_id: str):
        """
        取消订阅

        Args:
            subscriber_id: 订阅者标识
        """
        with self.subscriber_lock:
            if subscriber_id in self.subscribers:
                del self.subscribers[subscriber_id]
                self.logger.info(f"订阅者 {subscriber_id} 已取消订阅")

    def get_frame(self, subscriber_id: str, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        获取最新帧（非阻塞）

        Args:
            subscriber_id: 订阅者标识
            timeout: 超时时间（秒）

        Returns:
            Optional[np.ndarray]: 帧图像，失败返回None
        """
        if subscriber_id not in self.subscribers:
            self.logger.warning(f"订阅者 {subscriber_id} 未注册")
            return None

        queue = self.subscribers[subscriber_id]

        try:
            # 清空队列，只保留最新帧
            frame = None
            while not queue.empty():
                frame = queue.get_nowait()

            # 如果队列为空，等待新帧
            if frame is None:
                frame = queue.get(timeout=timeout)

            return frame

        except Exception:
            return None

    def _capture_loop(self):
        """
        捕获循环（在独立线程中运行）
        """
        self.logger.info("摄像头捕获循环已启动")

        frame_interval = 1.0 / self.fps

        while self.is_running:
            loop_start = time.time()

            # 读取帧
            success, frame = self.cap.read()

            if not success:
                self.logger.warning("读取帧失败")
                time.sleep(0.1)
                continue

            self.frame_count += 1

            # 分发帧到所有订阅者
            with self.subscriber_lock:
                for subscriber_id, queue in self.subscribers.items():
                    try:
                        # 使用 put_nowait，如果队列满则丢弃旧帧
                        if queue.full():
                            # 丢弃最旧的帧
                            try:
                                queue.get_nowait()
                                self.dropped_frames += 1
                            except:
                                pass

                        # 每个订阅者获得独立的帧副本
                        queue.put_nowait(frame.copy())

                    except Full:
                        # 队列满，跳过这个订阅者
                        self.dropped_frames += 1

            # 控制帧率
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.logger.info("摄像头捕获循环已退出")

    def get_statistics(self) -> Dict:
        """
        获取统计信息

        Returns:
            Dict: 统计信息
        """
        runtime = time.time() - self.start_time if self.start_time else 0
        actual_fps = self.frame_count / runtime if runtime > 0 else 0

        return {
            'camera_index': self.camera_index,
            'is_opened': self.is_opened,
            'is_running': self.is_running,
            'frame_count': self.frame_count,
            'dropped_frames': self.dropped_frames,
            'runtime_seconds': runtime,
            'actual_fps': actual_fps,
            'target_fps': self.fps,
            'subscriber_count': len(self.subscribers),
            'subscribers': list(self.subscribers.keys())
        }

    def __enter__(self):
        """上下文管理器入口"""
        self.open()
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


# 使用示例
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建摄像头管理器
    cam_manager = CameraManager(camera_index=0, fps=30)

    if cam_manager.open():
        cam_manager.start()

        # 模拟两个订阅者
        cam_manager.subscribe("detector_1")
        cam_manager.subscribe("detector_2")

        print("按 Ctrl+C 停止...")

        try:
            frame_count = 0
            while frame_count < 100:  # 测试100帧
                # 订阅者1获取帧
                frame1 = cam_manager.get_frame("detector_1")
                if frame1 is not None:
                    cv2.putText(
                        frame1, "Detector 1", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )
                    cv2.imshow("Detector 1", frame1)

                # 订阅者2获取帧
                frame2 = cam_manager.get_frame("detector_2")
                if frame2 is not None:
                    cv2.putText(
                        frame2, "Detector 2", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                    )
                    cv2.imshow("Detector 2", frame2)

                frame_count += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\n收到中断信号")

        finally:
            cv2.destroyAllWindows()

            # 打印统计
            stats = cam_manager.get_statistics()
            print("\n摄像头统计:")
            print(f"  总帧数: {stats['frame_count']}")
            print(f"  丢帧数: {stats['dropped_frames']}")
            print(f"  实际FPS: {stats['actual_fps']:.1f}")
            print(f"  订阅者数: {stats['subscriber_count']}")

            cam_manager.close()
