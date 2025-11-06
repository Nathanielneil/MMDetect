#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态无人机集群人机交互系统 - 主程序
整合所有检测器并管理指令流
"""

import os
import sys
import argparse
import logging
import yaml
import threading
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from detectors.voice_detector import VoiceDetector
from detectors.gesture_detector import GestureDetector
from detectors.image_detector import ImageDetector
from detectors.mouse_detector import MouseDetector
from core.command_manager import CommandManager
from core.camera_manager import CameraManager
from visualization.unified_viewer import UnifiedViewer


class MultimodalDroneSystem:
    """
    多模态无人机集群系统主类
    """

    def __init__(self, config_path: str = "config/detector_config.yaml"):
        """
        初始化系统

        Args:
            config_path: 配置文件路径
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # 加载配置
        self.config = self._load_config(config_path)

        # 初始化检测器
        self.detectors = {}
        self.detector_threads = {}
        self.running = False

        # 初始化摄像头管理器
        self.camera_manager = None
        self.camera_enabled = False

        # 初始化统一查看器
        self.unified_viewer = None
        self.use_unified_viewer = False

        # 初始化指令管理器
        self.command_manager = CommandManager(
            self.config.get('command_manager', {})
        )

        # 注册指令回调
        self.command_manager.register_callback(self._on_command_received)

    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"配置文件加载成功: {config_path}")
            return config
        except Exception as e:
            self.logger.warning(f"配置文件加载失败: {str(e)},使用默认配置")
            return {}

    def initialize_detectors(self, enable_voice=False, enable_gesture=False,
                            enable_image=False, enable_mouse=False):
        """
        初始化检测器

        Args:
            enable_voice: 启用语音检测
            enable_gesture: 启用手势检测
            enable_image: 启用图像检测
            enable_mouse: 启用鼠标检测
        """
        # 如果启用了手势或图像检测，初始化摄像头管理器
        if enable_gesture or enable_image:
            self.logger.info("初始化摄像头管理器...")
            camera_index = self.config.get('gesture_detector', {}).get('camera_index', 0)
            self.camera_manager = CameraManager(camera_index=camera_index, fps=30)

            if self.camera_manager.open():
                self.camera_manager.start()
                self.camera_enabled = True
                self.logger.info("✓ 摄像头管理器初始化成功")
            else:
                self.logger.error("✗ 摄像头管理器初始化失败")
                self.camera_manager = None
                enable_gesture = False
                enable_image = False
        if enable_voice:
            self.logger.info("初始化语音检测器...")
            voice_config = self.config.get('voice_detector', {})
            self.detectors['voice'] = VoiceDetector(voice_config)
            if self.detectors['voice'].initialize():
                self.logger.info("✓ 语音检测器初始化成功")
            else:
                self.logger.error("✗ 语音检测器初始化失败")
                del self.detectors['voice']

        if enable_gesture:
            self.logger.info("初始化手势检测器...")
            gesture_config = self.config.get('gesture_detector', {})
            self.detectors['gesture'] = GestureDetector(gesture_config)
            if self.detectors['gesture'].initialize():
                self.logger.info("✓ 手势检测器初始化成功")
            else:
                self.logger.error("✗ 手势检测器初始化失败")
                del self.detectors['gesture']

        if enable_image:
            self.logger.info("初始化图像检测器...")
            image_config = self.config.get('image_detector', {})
            self.detectors['image'] = ImageDetector(image_config)
            if self.detectors['image'].initialize():
                self.logger.info("✓ 图像检测器初始化成功")
            else:
                self.logger.error("✗ 图像检测器初始化失败")
                del self.detectors['image']

        if enable_mouse:
            self.logger.info("初始化鼠标检测器...")
            mouse_config = self.config.get('mouse_detector', {})
            self.detectors['mouse'] = MouseDetector(mouse_config)
            if self.detectors['mouse'].initialize():
                self.logger.info("✓ 鼠标检测器初始化成功")
            else:
                self.logger.error("✗ 鼠标检测器初始化失败")
                del self.detectors['mouse']

        self.logger.info(f"共初始化 {len(self.detectors)} 个检测器")

    def start(self, use_unified_viewer: bool = False):
        """
        启动系统

        Args:
            use_unified_viewer: 是否使用统一查看器
        """
        self.running = True
        self.use_unified_viewer = use_unified_viewer

        self.logger.info("=" * 60)
        self.logger.info("多模态无人机集群系统启动")
        if use_unified_viewer:
            self.logger.info("模式: 统一可视化界面")
        self.logger.info("=" * 60)

        # 如果使用统一查看器，启动它
        if use_unified_viewer:
            self.unified_viewer = UnifiedViewer(
                camera_manager=self.camera_manager,
                detectors=self.detectors,
                command_manager=self.command_manager,
                window_size=(1280, 720)
            )
            self.unified_viewer.start()
            self.logger.info("✓ 统一查看器已启动")

        # 启动各检测器线程
        if 'voice' in self.detectors:
            self._start_voice_detector()

        if 'gesture' in self.detectors:
            self._start_gesture_detector()

        if 'image' in self.detectors:
            self._start_image_detector()

        if 'mouse' in self.detectors and not use_unified_viewer:
            # 如果使用统一查看器，不启动独立的鼠标窗口
            self._start_mouse_detector()

        self.logger.info("所有检测器已启动")

    def _start_voice_detector(self):
        """启动语音检测器线程"""
        def voice_loop():
            detector = self.detectors['voice']

            def on_voice_command(cmd):
                self.command_manager.add_command(cmd)

            detector.listen_continuous(on_voice_command, duration=None)

        thread = threading.Thread(target=voice_loop, daemon=True)
        thread.start()
        self.detector_threads['voice'] = thread
        self.logger.info("语音检测器线程已启动")

    def _start_gesture_detector(self):
        """启动手势检测器线程"""
        def gesture_loop():
            detector = self.detectors['gesture']

            if not self.camera_manager:
                self.logger.error("摄像头管理器未初始化")
                return

            # 订阅摄像头帧
            self.camera_manager.subscribe('gesture')
            self.logger.info("手势检测器已订阅摄像头")

            frame_count = 0
            while self.running:
                # 从摄像头管理器获取帧
                frame = self.camera_manager.get_frame('gesture', timeout=0.1)

                if frame is None:
                    continue

                frame_count += 1
                command = detector.run(frame)
                if command and command['confidence'] > 0.5:
                    self.command_manager.add_command(command)

                time.sleep(0.001)  # 减少延迟

            # 取消订阅
            self.camera_manager.unsubscribe('gesture')
            self.logger.info("手势检测器已停止")

        thread = threading.Thread(target=gesture_loop, daemon=True)
        thread.start()
        self.detector_threads['gesture'] = thread
        self.logger.info("手势检测器线程已启动")

    def _start_image_detector(self):
        """启动图像检测器线程"""
        def image_loop():
            detector = self.detectors['image']

            if not self.camera_manager:
                self.logger.error("摄像头管理器未初始化")
                return

            # 订阅摄像头帧
            self.camera_manager.subscribe('image')
            self.logger.info("图像检测器已订阅摄像头")

            frame_count = 0
            while self.running:
                # 从摄像头管理器获取帧
                frame = self.camera_manager.get_frame('image', timeout=0.1)

                if frame is None:
                    continue

                frame_count += 1
                command = detector.run(frame)
                if command and command['confidence'] > 0.5:
                    self.command_manager.add_command(command)

                time.sleep(0.001)  # 减少延迟

            # 取消订阅
            self.camera_manager.unsubscribe('image')
            self.logger.info("图像检测器已停止")

        thread = threading.Thread(target=image_loop, daemon=True)
        thread.start()
        self.detector_threads['image'] = thread
        self.logger.info("图像检测器线程已启动")

    def _start_mouse_detector(self):
        """启动鼠标检测器(使用OpenCV窗口)"""
        import cv2
        import numpy as np

        def mouse_loop():
            detector = self.detectors['mouse']
            canvas_cfg = self.config.get('mouse_detector', {})
            canvas = np.ones(
                (canvas_cfg.get('canvas_height', 600),
                 canvas_cfg.get('canvas_width', 800), 3),
                dtype=np.uint8
            ) * 255

            def mouse_callback(event, x, y, flags, param):
                mouse_data = {'x': x, 'y': y}

                if event == cv2.EVENT_LBUTTONDOWN:
                    mouse_data['event'] = 'down'
                    detector.run(mouse_data)  # 开始绘制，不发送命令
                elif event == cv2.EVENT_MOUSEMOVE:
                    mouse_data['event'] = 'move'
                    detector.run(mouse_data)  # 继续绘制，不发送命令
                elif event == cv2.EVENT_LBUTTONUP:
                    mouse_data['event'] = 'up'
                    command = detector.run(mouse_data)
                    # 只在松开鼠标时检查是否生成了有效的航点命令
                    if command and command['command'] == 'waypoint' and command['confidence'] > 0.5:
                        self.command_manager.add_command(command)
                        self.logger.info(
                            f"生成航点命令: {command['parameters']['num_waypoints']}个航点, "
                            f"总距离: {command['parameters']['total_distance']:.2f}m"
                        )
                        # 清除路径以便下一次绘制
                        detector.reset()
                else:
                    return

            cv2.namedWindow('Path Planner')
            cv2.setMouseCallback('Path Planner', mouse_callback)

            while self.running:
                vis_canvas = detector.visualize_path(canvas.copy())
                cv2.putText(
                    vis_canvas,
                    "Drag to draw path | 'r': reset | 'q': quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1
                )
                cv2.imshow('Path Planner', vis_canvas)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('r'):
                    detector.reset()
                    canvas[:] = 255
                elif key == ord('q'):
                    self.stop()
                    break

            cv2.destroyAllWindows()

        thread = threading.Thread(target=mouse_loop, daemon=True)
        thread.start()
        self.detector_threads['mouse'] = thread
        self.logger.info("鼠标检测器线程已启动")

    def _on_command_received(self, command: dict):
        """
        指令接收回调

        Args:
            command: 融合后的指令
        """
        self.logger.info(
            f"\n{'=' * 60}\n"
            f">>> 新指令: {command['command'].upper()}\n"
            f"    模态: {command['modality']}\n"
            f"    置信度: {command['confidence']:.2f}\n"
            f"    参数: {command.get('parameters', {})}\n"
            f"{'=' * 60}"
        )

        # 添加到统一查看器
        if self.unified_viewer:
            self.unified_viewer.add_command(command)

        # TODO: 在这里发送指令到ROS/Gazebo
        # self._send_to_ros(command)

    def stop(self):
        """停止系统"""
        self.logger.info("正在停止系统...")
        self.running = False

        # 停止统一查看器
        if self.unified_viewer:
            self.unified_viewer.stop()
            self.logger.info("统一查看器已关闭")

        # 等待所有线程结束
        time.sleep(0.5)

        # 关闭摄像头管理器
        if self.camera_manager:
            self.camera_manager.close()
            self.logger.info("摄像头管理器已关闭")

        # 关闭所有检测器
        for name, detector in self.detectors.items():
            detector.shutdown()

        # 打印统计
        self._print_statistics()

        self.logger.info("系统已停止")

    def _print_statistics(self):
        """打印统计信息"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("系统统计")
        self.logger.info("=" * 60)

        # 指令管理器统计
        cmd_stats = self.command_manager.get_statistics()
        self.logger.info(f"\n指令统计:")
        self.logger.info(f"  总接收: {cmd_stats['total_commands']}")
        self.logger.info(f"  已过滤: {cmd_stats['filtered_commands']}")
        self.logger.info(f"  已融合: {cmd_stats['fused_commands']}")
        self.logger.info(f"\n各模态指令数:")
        for modality, count in cmd_stats['modality_counts'].items():
            self.logger.info(f"  {modality}: {count}")

        # 各检测器统计
        self.logger.info(f"\n检测器性能:")
        for name, detector in self.detectors.items():
            stats = detector.get_statistics()
            self.logger.info(
                f"  {name}: {stats['detection_count']} 次检测, "
                f"平均延迟 {stats['average_latency_ms']:.1f}ms"
            )

        # 摄像头管理器统计
        if self.camera_manager and self.camera_enabled:
            cam_stats = self.camera_manager.get_statistics()
            self.logger.info(f"\n摄像头统计:")
            self.logger.info(f"  总帧数: {cam_stats['frame_count']}")
            self.logger.info(f"  丢帧数: {cam_stats['dropped_frames']}")
            self.logger.info(f"  实际FPS: {cam_stats['actual_fps']:.1f}")
            self.logger.info(f"  订阅者: {', '.join(cam_stats['subscribers'])}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="多模态无人机集群人机交互系统"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/detector_config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--voice',
        action='store_true',
        help='启用语音检测'
    )
    parser.add_argument(
        '--gesture',
        action='store_true',
        help='启用手势检测'
    )
    parser.add_argument(
        '--image',
        action='store_true',
        help='启用图像检测'
    )
    parser.add_argument(
        '--mouse',
        action='store_true',
        help='启用鼠标检测'
    )
    parser.add_argument(
        '--enable-all',
        action='store_true',
        help='启用所有检测器'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='日志级别'
    )
    parser.add_argument(
        '--unified-viewer',
        action='store_true',
        help='使用统一可视化界面（可切换多模态视图）'
    )

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 创建系统实例
    system = MultimodalDroneSystem(args.config)

    # 初始化检测器
    if args.enable_all:
        enable_voice = enable_gesture = enable_image = enable_mouse = True
    else:
        enable_voice = args.voice
        enable_gesture = args.gesture
        enable_image = args.image
        enable_mouse = args.mouse

    # 如果没有指定任何检测器,默认启用鼠标
    if not (enable_voice or enable_gesture or enable_image or enable_mouse):
        logging.info("未指定检测器,默认启用鼠标检测")
        enable_mouse = True

    system.initialize_detectors(
        enable_voice=enable_voice,
        enable_gesture=enable_gesture,
        enable_image=enable_image,
        enable_mouse=enable_mouse
    )

    # 启动系统
    system.start(use_unified_viewer=args.unified_viewer)

    # 如果使用统一查看器，打印使用说明
    if args.unified_viewer:
        print("\n" + "="*60)
        print("统一可视化界面 - 键盘控制")
        print("="*60)
        print("[1] 切换到手势识别视图")
        print("[2] 切换到图像检测视图")
        print("[3] 切换到鼠标路径视图")
        print("[4] 切换到四分屏视图")
        print("[0] 切换到概览视图")
        print("[R] 重置")
        print("[Q] 退出")
        print("="*60 + "\n")

    try:
        # 保持主线程运行
        while system.running:
            # 如果统一查看器停止了，也停止系统
            if args.unified_viewer and system.unified_viewer and not system.unified_viewer.is_running:
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        logging.info("\n收到中断信号")

    # 停止系统
    system.stop()


if __name__ == "__main__":
    main()
