#!/usr/bin/env python3
"""测试摄像头可用性"""
import cv2
import sys

def test_camera(index):
    """测试指定索引的摄像头"""
    print(f"\n测试摄像头 {index}...")
    cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        print(f"  ✗ 无法打开摄像头 {index}")
        return False

    print(f"  ✓ 摄像头 {index} 已打开")

    # 尝试读取一帧
    success, frame = cap.read()
    if not success:
        print(f"  ✗ 无法读取帧")
        cap.release()
        return False

    print(f"  ✓ 成功读取帧: {frame.shape}")
    cap.release()
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("摄像头可用性测试")
    print("=" * 60)

    available = []
    for i in range(4):  # 测试0-3
        if test_camera(i):
            available.append(i)

    print("\n" + "=" * 60)
    print(f"可用摄像头: {available}")
    print("=" * 60)
