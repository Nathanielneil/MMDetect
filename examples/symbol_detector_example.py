#!/usr/bin/env python3
"""
YOLOv8符号检测器集成示例

演示如何使用训练好的符号检测模型进行实时检测
包括基础用法、高级特性和错误处理
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from collections import deque
import time
import logging

from src.detectors.image_detector import ImageDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SymbolDetectorExample:
    """符号检测器使用示例"""

    def __init__(self, model_path, confidence=0.6):
        """初始化检测器"""
        config = ImageDetector.create_symbol_detector_config(
            model_path=model_path,
            confidence=confidence
        )
        self.detector = ImageDetector(config)

        if not self.detector.initialize():
            raise RuntimeError("检测器初始化失败")

        self.symbol_map = {
            'H': 'LAND (降落)',
            'T': 'TAKEOFF (起飞)',
            'F': 'FORMATION (编队)',
            'S': 'EXPLORE (搜索)',
            'L': 'HOVER (悬停)'
        }

    def example_1_basic_detection(self, frame):
        """示例1: 基础检测"""
        print("\n示例1: 基础检测")
        print("-" * 60)

        command = self.detector.run(frame)

        if command:
            print(f"检测到指令: {command['command']}")
            print(f"置信度: {command['confidence']:.3f}")
            print(f"符号: {command['parameters'].get('symbol')}")
            print(f"位置: {command['parameters'].get('target_center')}")
        else:
            print("未检测到任何符号")

        return command

    def example_2_specific_symbol(self, frame):
        """示例2: 检测特定符号"""
        print("\n示例2: 检测特定符号")
        print("-" * 60)

        symbols = ['H', 'T', 'F', 'S', 'L']

        for symbol in symbols:
            detection = self.detector.detect_symbol(frame, symbol)
            status = "检测到" if detection else "未检测到"
            conf = f"({detection['confidence']:.3f})" if detection else ""
            print(f"{symbol}: {status} {conf}")

    def example_3_multi_detection(self, frame):
        """示例3: 多目标检测"""
        print("\n示例3: 多目标检测")
        print("-" * 60)

        det_result = self.detector.detect(frame)
        detections = det_result['detections']

        if not detections:
            print("未检测到任何目标")
            return

        print(f"检测到 {len(detections)} 个目标:")
        for i, det in enumerate(detections, 1):
            print(f"  {i}. {det['class_name']}: 置信度 {det['confidence']:.3f}")

    def example_4_temporal_smoothing(self, video_source, window_size=5):
        """示例4: 时间平滑（多帧投票）"""
        print("\n示例4: 时间平滑检测")
        print("-" * 60)

        detection_history = deque(maxlen=window_size)
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            print("无法打开视频源")
            return

        frame_count = 0
        confirmed_detections = 0

        while cap.isOpened() and frame_count < 100:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 检测当前帧
            command = self.detector.run(frame)

            if command:
                symbol = command['parameters'].get('symbol')
                detection_history.append(symbol)

                # 投票决策
                if len(detection_history) == window_size:
                    symbol_counts = {}
                    for s in detection_history:
                        symbol_counts[s] = symbol_counts.get(s, 0) + 1

                    majority_symbol = max(symbol_counts, key=symbol_counts.get)
                    vote_confidence = symbol_counts[majority_symbol] / len(detection_history)

                    if vote_confidence > 0.6:
                        confirmed_detections += 1
                        print(f"第{frame_count}帧: 投票确认 {majority_symbol} "
                              f"(投票置信度: {vote_confidence:.2f})")

        cap.release()
        print(f"总确认检测: {confirmed_detections}/约{frame_count//window_size}")

    def example_5_confidence_adjustment(self, frame):
        """示例5: 动态调整置信度阈值"""
        print("\n示例5: 动态调整置信度阈值")
        print("-" * 60)

        thresholds = [0.5, 0.6, 0.7, 0.8]

        for threshold in thresholds:
            self.detector.confidence_threshold = threshold

            command = self.detector.run(frame)
            status = "检测到" if command else "未检测到"
            conf = f"({command['confidence']:.3f})" if command else ""

            print(f"阈值 {threshold}: {status} {conf}")

    def example_6_visualization(self, frame):
        """示例6: 可视化检测结果"""
        print("\n示例6: 可视化检测结果")
        print("-" * 60)

        det_result = self.detector.detect(frame)
        detections = det_result['detections']

        # 原始可视化
        vis_frame = self.detector.visualize(frame, detections)

        # 添加符号映射信息
        if detections:
            best_det = max(detections, key=lambda x: x['confidence'])
            symbol = best_det['class_name']
            command_text = self.symbol_map.get(symbol, "未知")

            cv2.putText(
                vis_frame,
                f"Command: {command_text}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        # 显示统计信息
        stats_text = f"检测数: {len(detections)} | 帧率: ~16 FPS"
        cv2.putText(
            vis_frame,
            stats_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            1
        )

        cv2.imshow('Symbol Detection Visualization', vis_frame)

        return vis_frame

    def example_7_performance_monitoring(self, video_source, num_frames=100):
        """示例7: 性能监测"""
        print("\n示例7: 性能监测")
        print("-" * 60)

        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            print("无法打开视频源")
            return

        frame_count = 0
        total_latency = 0
        detection_count = 0

        while cap.isOpened() and frame_count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 测量延迟
            start_time = time.time()
            command = self.detector.run(frame)
            latency = (time.time() - start_time) * 1000

            total_latency += latency

            if command:
                detection_count += 1

        cap.release()

        avg_latency = total_latency / frame_count if frame_count > 0 else 0
        fps = 1000 / avg_latency if avg_latency > 0 else 0
        detection_rate = (detection_count / frame_count) * 100 if frame_count > 0 else 0

        print(f"总帧数: {frame_count}")
        print(f"平均延迟: {avg_latency:.2f}ms")
        print(f"帧率: {fps:.1f} FPS")
        print(f"检测率: {detection_rate:.1f}%")
        print(f"检测数: {detection_count}")

    def example_8_real_time_loop(self, video_source=0):
        """示例8: 实时检测循环"""
        print("\n示例8: 实时检测循环")
        print("-" * 60)

        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            print("无法打开摄像头")
            return

        print("摄像头已打开，按'q'退出...")

        fps_history = deque(maxlen=30)
        prev_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 计算FPS
            current_time = time.time()
            fps_history.append(current_time - prev_time)
            prev_time = current_time

            avg_frame_time = sum(fps_history) / len(fps_history) if fps_history else 0
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

            # 检测
            command = self.detector.run(frame)

            # 可视化
            det_result = self.detector.detect(frame)
            vis_frame = self.detector.visualize(frame, det_result['detections'])

            # 显示信息
            if command:
                symbol = command['parameters'].get('symbol', '?')
                cmd_text = self.symbol_map.get(symbol, "未知")
                cv2.putText(
                    vis_frame,
                    f"Symbol: {symbol} ({cmd_text}) - Conf: {command['confidence']:.3f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            # 显示FPS
            cv2.putText(
                vis_frame,
                f"FPS: {fps:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                1
            )

            cv2.imshow('Real-time Symbol Detection', vis_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        print("实时检测循环已结束")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("  YOLOv8符号检测器集成示例")
    print("=" * 70)

    model_path = '/home/ubuntu/NGW/intern/MMDect/symbol_dataset/runs/detect/train2/weights/best.pt'

    # 初始化
    try:
        example = SymbolDetectorExample(model_path, confidence=0.6)
    except RuntimeError as e:
        print(f"错误: {e}")
        return

    print("\n可用示例:")
    print("1. 基础检测")
    print("2. 特定符号检测")
    print("3. 多目标检测")
    print("4. 时间平滑")
    print("5. 置信度调整")
    print("6. 可视化")
    print("7. 性能监测")
    print("8. 实时循环")
    print("9. 全部运行")

    choice = input("\n请选择示例 (1-9): ").strip()

    # 打开摄像头获取一帧作为测试
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头，某些示例将被跳过")
        test_frame = None
    else:
        ret, test_frame = cap.read()
        cap.release()

        if not ret:
            print("无法读取摄像头")
            test_frame = None

    if choice == '1' and test_frame is not None:
        example.example_1_basic_detection(test_frame)

    elif choice == '2' and test_frame is not None:
        example.example_2_specific_symbol(test_frame)

    elif choice == '3' and test_frame is not None:
        example.example_3_multi_detection(test_frame)

    elif choice == '4':
        example.example_4_temporal_smoothing(0, window_size=5)

    elif choice == '5' and test_frame is not None:
        example.example_5_confidence_adjustment(test_frame)

    elif choice == '6' and test_frame is not None:
        example.example_6_visualization(test_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif choice == '7':
        example.example_7_performance_monitoring(0, num_frames=100)

    elif choice == '8':
        example.example_8_real_time_loop(0)

    elif choice == '9':
        print("\n运行所有示例...")
        if test_frame is not None:
            example.example_1_basic_detection(test_frame)
            example.example_2_specific_symbol(test_frame)
            example.example_3_multi_detection(test_frame)
            example.example_5_confidence_adjustment(test_frame)
            example.example_6_visualization(test_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        example.example_4_temporal_smoothing(0, window_size=5)
        example.example_7_performance_monitoring(0, num_frames=100)

    else:
        print("无效选择")

    print("\n" + "=" * 70)
    print("示例执行完成")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()

