#!/usr/bin/env python3
"""
摄像头测试工具
用于检测和测试可用的摄像头
"""

import cv2
import numpy as np
import sys


def find_cameras(max_id=10):
    """
    查找所有可用摄像头

    Args:
        max_id: 最大检测ID

    Returns:
        可用摄像头列表
    """
    print("="*60)
    print("  摄像头检测工具")
    print("="*60)
    print(f"\n正在检测摄像头 (ID: 0-{max_id})...\n")

    available_cameras = []

    for i in range(max_id):
        cap = cv2.VideoCapture(i)

        if cap.isOpened():
            # 读取一帧测试
            ret, frame = cap.read()

            if ret and frame is not None:
                h, w = frame.shape[:2]
                available_cameras.append({
                    'id': i,
                    'width': w,
                    'height': h,
                    'resolution': f"{w}x{h}"
                })
                print(f"✓ 摄像头 {i}: 可用 (分辨率: {w}x{h})")
            else:
                print(f"✗ 摄像头 {i}: 打开失败(无法读取帧)")

            cap.release()
        else:
            # 不打印不可用的,避免刷屏
            pass

    print("\n" + "="*60)

    if not available_cameras:
        print("❌ 未找到可用摄像头")
        print("\n可能的原因:")
        print("  1. 摄像头被其他程序占用(Zoom/Teams/浏览器等)")
        print("  2. 摄像头权限未授予")
        print("  3. 驱动程序问题")
        print("\n解决方案:")
        print("  - 关闭所有使用摄像头的程序")
        print("  - 检查系统设置中的摄像头权限")
        print("  - 重启电脑")
        return None

    print(f"✓ 找到 {len(available_cameras)} 个可用摄像头")
    print("="*60)

    return available_cameras


def test_camera(camera_id, duration=None):
    """
    测试指定摄像头

    Args:
        camera_id: 摄像头ID
        duration: 测试时长(秒),None=手动退出
    """
    print(f"\n正在打开摄像头 {camera_id}...")

    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 {camera_id}")
        return False

    # 获取摄像头信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"✓ 摄像头已打开")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps} fps")
    print(f"\n按 'q' 退出, 'c' 截图, 'i' 显示信息")

    frame_count = 0
    import time
    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("❌ 读取帧失败")
            break

        frame_count += 1

        # 计算实际FPS
        elapsed = time.time() - start_time
        if elapsed > 0:
            actual_fps = frame_count / elapsed
        else:
            actual_fps = 0

        # 添加信息显示
        info_frame = frame.copy()

        # 标题
        cv2.putText(info_frame, f"Camera {camera_id}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 分辨率
        cv2.putText(info_frame, f"Resolution: {width}x{height}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # FPS
        cv2.putText(info_frame, f"FPS: {actual_fps:.1f}", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 帧数
        cv2.putText(info_frame, f"Frames: {frame_count}", (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 提示
        cv2.putText(info_frame, "Press 'q' to quit, 'c' to capture, 'i' for info",
                   (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 显示
        cv2.imshow(f'Camera {camera_id} Test', info_frame)

        # 键盘控制
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n退出测试")
            break

        elif key == ord('c'):
            # 截图
            filename = f"camera_{camera_id}_capture_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✓ 截图已保存: {filename}")

        elif key == ord('i'):
            # 显示详细信息
            print(f"\n摄像头信息:")
            print(f"  ID: {camera_id}")
            print(f"  分辨率: {width}x{height}")
            print(f"  声明FPS: {fps}")
            print(f"  实际FPS: {actual_fps:.1f}")
            print(f"  已捕获帧数: {frame_count}")
            print(f"  运行时间: {elapsed:.1f}秒")

        # 检查时长限制
        if duration and elapsed >= duration:
            print(f"\n测试时长已到({duration}秒)")
            break

    cap.release()
    cv2.destroyAllWindows()

    # 最终统计
    elapsed = time.time() - start_time
    print(f"\n测试统计:")
    print(f"  总帧数: {frame_count}")
    print(f"  总时长: {elapsed:.1f}秒")
    print(f"  平均FPS: {frame_count/elapsed:.1f}")

    return True


def interactive_test():
    """交互式测试"""
    # 1. 查找摄像头
    cameras = find_cameras()

    if not cameras:
        return

    # 2. 显示推荐配置
    print("\n推荐配置 (复制到 config/detector_config.yaml):")
    print("-"*60)
    for cam in cameras:
        print(f"camera_index: {cam['id']}  # {cam['resolution']}")
    print("-"*60)

    # 3. 选择摄像头测试
    if len(cameras) == 1:
        print(f"\n将测试摄像头 {cameras[0]['id']}")
        choice = cameras[0]['id']
    else:
        print(f"\n找到 {len(cameras)} 个摄像头")
        print("请选择要测试的摄像头ID:")
        for cam in cameras:
            print(f"  {cam['id']}: {cam['resolution']}")

        try:
            choice = int(input("\n输入ID (直接回车测试第一个): ") or cameras[0]['id'])
        except ValueError:
            choice = cameras[0]['id']

    # 4. 测试摄像头
    test_camera(choice)


def quick_test():
    """快速测试(默认摄像头0)"""
    print("快速测试模式: 摄像头 0")
    test_camera(0)


def main():
    """主函数"""
    if len(sys.argv) > 1:
        if sys.argv[1] == '--find':
            # 只查找,不测试
            find_cameras()
        elif sys.argv[1] == '--test':
            # 测试指定摄像头
            camera_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            test_camera(camera_id)
        elif sys.argv[1] == '--quick':
            # 快速测试
            quick_test()
        else:
            print("用法:")
            print("  python tools/test_camera.py              # 交互模式")
            print("  python tools/test_camera.py --find       # 只查找摄像头")
            print("  python tools/test_camera.py --test [ID]  # 测试指定摄像头")
            print("  python tools/test_camera.py --quick      # 快速测试摄像头0")
    else:
        # 交互模式
        interactive_test()


if __name__ == "__main__":
    main()
