"""
手势识别高级可视化测试脚本
显示指尖轨迹、渐变骨架、发光效果等
"""

import cv2
import logging
import sys
from src.detectors.gesture_detector import GestureDetector

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    detector = GestureDetector()

    if detector.initialize():
        cap = cv2.VideoCapture(0)

        print("\n" + "="*60)
        print("  手势识别 - 高级可视化模式")
        print("="*60)
        print("\n支持的手势:")
        print("  🖐️  张开手掌  (5指) -> TAKEOFF   (起飞)")
        print("  ✊  握拳      (0指) -> LAND      (降落)")
        print("  ☝️  单指      (1指) -> HOVER     (悬停)")
        print("  ✌️  V字手势   (2指) -> EXPLORE   (探索)")
        print("  🤟  三指      (3指) -> FORMATION (编队)")
        print("\n可视化特效:")
        print("  ✨ 渐变色骨架 (根据手部深度)")
        print("  💫 发光关键点 (指尖脉冲效果)")
        print("  🌈 指尖轨迹   (紫色到粉色渐隐拖尾)")
        print("  💓 手掌脉冲   (青蓝色呼吸动画)")
        print("\n操作提示:")
        print("  - 距离摄像头 30-60cm")
        print("  - 手放在画面中心")
        print("  - 移动手指观察轨迹效果")
        print("  - 按 'q' 退出")
        print("="*60 + "\n")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("无法读取摄像头帧")
                break

            # 运行检测
            command = detector.run(frame)

            # 获取图像尺寸
            h, w = frame.shape[:2]

            # 使用高级可视化
            if hasattr(detector, '_last_detection_result') and detector._last_detection_result:
                landmarks = detector._last_detection_result.get('landmarks')
                if landmarks:
                    # 调用高级可视化方法 - 显示指尖轨迹等特效
                    frame = detector.visualize_advanced(frame, landmarks, h, w)

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
                        2
                    )

            # 显示指令信息
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

                color = (0, 255, 0) if stability > 0.8 else (0, 255, 255)

                cv2.putText(
                    frame,
                    display_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2
                )

                # 显示置信度
                cv2.putText(
                    frame,
                    f"Confidence: {command['confidence']:.2f}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

                # 终端打印
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
                    "No hand detected - Move hand into view",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # 添加标题
            cv2.putText(
                frame,
                "Advanced Gesture Visualization",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1
            )

            cv2.imshow('Gesture Detection - Advanced Mode', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        detector.shutdown()
        print("\n✓ 程序已退出")
    else:
        print("✗ 手势检测器初始化失败！")
        sys.exit(1)
