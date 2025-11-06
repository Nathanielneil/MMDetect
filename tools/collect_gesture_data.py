#!/usr/bin/env python3
"""
手势数据采集工具
用于采集手势训练数据
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime
import argparse


class GestureDataCollector:
    """手势数据采集器"""

    def __init__(self, save_dir='data/gestures'):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.dataset = []

    def collect(self, gesture_name, num_samples=100):
        """
        采集特定手势的数据

        Args:
            gesture_name: 手势名称
            num_samples: 采集样本数
        """
        cap = cv2.VideoCapture(0)
        count = 0
        skip_frames = 0  # 跳过的帧数,避免采集相似样本

        print(f"\n{'='*60}")
        print(f"开始采集手势: {gesture_name}")
        print(f"目标样本数: {num_samples}")
        print(f"说明:")
        print(f"  - 保持手势稳定")
        print(f"  - 尝试不同的角度和位置")
        print(f"  - 按'space'暂停/继续")
        print(f"  - 按'q'退出")
        print(f"{'='*60}\n")

        paused = False

        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取摄像头")
                break

            # 水平翻转,镜像显示
            frame = cv2.flip(frame, 1)

            # 转换为RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 检测手部
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks and not paused:
                # 绘制手部关键点
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )

                # 每隔5帧采集一次,避免样本过于相似
                if skip_frames == 0:
                    # 提取21个关键点的坐标
                    landmarks = []
                    for lm in results.multi_hand_landmarks[0].landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])

                    # 保存数据
                    self.dataset.append({
                        'landmarks': landmarks,  # 63维特征
                        'label': gesture_name,
                        'timestamp': datetime.now().isoformat()
                    })

                    count += 1
                    skip_frames = 5  # 重置跳帧计数
                else:
                    skip_frames -= 1

            # 显示信息
            info_text = f"Gesture: {gesture_name} | Samples: {count}/{num_samples}"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if paused:
                cv2.putText(frame, "PAUSED", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if results.multi_hand_landmarks:
                cv2.putText(frame, "Hand Detected", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No Hand Detected", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 进度条
            bar_width = 400
            bar_height = 20
            progress = int((count / num_samples) * bar_width)
            cv2.rectangle(frame, (10, frame.shape[0] - 40),
                         (10 + bar_width, frame.shape[0] - 40 + bar_height),
                         (100, 100, 100), 2)
            cv2.rectangle(frame, (10, frame.shape[0] - 40),
                         (10 + progress, frame.shape[0] - 40 + bar_height),
                         (0, 255, 0), -1)

            cv2.imshow('Gesture Data Collection', frame)

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused

        cap.release()
        cv2.destroyAllWindows()

        print(f"\n采集完成! 共采集 {count} 个样本")

    def save_dataset(self, filename=None):
        """保存数据集"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.save_dir}/gesture_dataset_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.dataset, f, indent=2)

        print(f"\n数据集已保存到: {filename}")
        print(f"总样本数: {len(self.dataset)}")

        # 统计各手势样本数
        from collections import Counter
        labels = [d['label'] for d in self.dataset]
        label_counts = Counter(labels)

        print("\n各手势样本数:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")

    def load_dataset(self, filename):
        """加载已有数据集"""
        with open(filename, 'r') as f:
            self.dataset = json.load(f)
        print(f"已加载数据集: {filename}")
        print(f"样本数: {len(self.dataset)}")


def main():
    parser = argparse.ArgumentParser(description='手势数据采集工具')
    parser.add_argument('--gesture', type=str, required=True,
                       help='手势名称')
    parser.add_argument('--samples', type=int, default=100,
                       help='采集样本数 (默认: 100)')
    parser.add_argument('--save-dir', type=str, default='data/gestures',
                       help='保存目录 (默认: data/gestures)')
    parser.add_argument('--load', type=str, default=None,
                       help='加载已有数据集继续采集')

    args = parser.parse_args()

    collector = GestureDataCollector(save_dir=args.save_dir)

    # 如果指定了已有数据集,先加载
    if args.load:
        collector.load_dataset(args.load)

    # 采集数据
    collector.collect(args.gesture, num_samples=args.samples)

    # 保存
    collector.save_dataset()


if __name__ == "__main__":
    # 交互式模式
    if len(os.sys.argv) == 1:
        print("手势数据采集工具 - 交互模式")
        print("="*60)

        collector = GestureDataCollector()

        # 预定义的手势列表
        gestures = [
            ('open_palm', '张开手掌', 150),
            ('fist', '握拳', 150),
            ('one_finger', '竖起一根手指', 150),
            ('v_sign', 'V字手势', 150),
            ('three_fingers', '三根手指', 150)
        ]

        print("\n将采集以下手势:")
        for i, (name, desc, samples) in enumerate(gestures, 1):
            print(f"  {i}. {desc} ({name}) - {samples}个样本")

        input("\n按Enter开始采集...")

        for name, desc, samples in gestures:
            print(f"\n准备采集: {desc}")
            input("摆好手势后按Enter继续...")
            collector.collect(name, num_samples=samples)

        collector.save_dataset()

        print("\n所有手势采集完成!")

    else:
        # 命令行模式
        main()
