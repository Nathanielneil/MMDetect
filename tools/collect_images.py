#!/usr/bin/env python3
"""
图像数据采集工具
用于采集目标检测训练数据(如降落靶标)
"""

import cv2
import os
from datetime import datetime
import argparse


class ImageCollector:
    """图像采集器"""

    def __init__(self, save_dir='data/images', camera_id=0):
        """
        初始化

        Args:
            save_dir: 保存目录
            camera_id: 摄像头ID
        """
        self.save_dir = save_dir
        self.camera_id = camera_id
        os.makedirs(save_dir, exist_ok=True)

    def collect(self, class_name, num_images=100, auto_mode=False, interval=1.0):
        """
        采集图像

        Args:
            class_name: 类别名称(如landing_pad)
            num_images: 采集数量
            auto_mode: 自动模式(定时采集)
            interval: 自动模式间隔(秒)
        """
        # 创建类别目录
        class_dir = os.path.join(self.save_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 {self.camera_id}")
            return

        # 设置分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        count = 0
        last_save_time = datetime.now()

        print(f"\n{'='*60}")
        print(f"图像数据采集")
        print(f"{'='*60}")
        print(f"类别: {class_name}")
        print(f"目标数量: {num_images}")
        print(f"保存目录: {class_dir}")
        print(f"模式: {'自动' if auto_mode else '手动'}")
        if auto_mode:
            print(f"采集间隔: {interval}秒")
        print(f"\n操作说明:")
        print(f"  - 空格键: 保存当前帧")
        print(f"  - 'a': 切换自动/手动模式")
        print(f"  - 'q': 退出")
        print(f"{'='*60}\n")

        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取帧")
                break

            # 显示帧
            display_frame = frame.copy()

            # 添加信息
            info_text = f"Class: {class_name} | Images: {count}/{num_images}"
            mode_text = f"Mode: {'AUTO' if auto_mode else 'MANUAL'}"

            cv2.putText(display_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, mode_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                       (0, 255, 255) if auto_mode else (255, 255, 255), 2)

            # 绘制中心十字线(辅助对准)
            h, w = display_frame.shape[:2]
            cv2.line(display_frame, (w//2 - 50, h//2), (w//2 + 50, h//2),
                    (0, 255, 0), 2)
            cv2.line(display_frame, (w//2, h//2 - 50), (w//2, h//2 + 50),
                    (0, 255, 0), 2)

            # 进度条
            bar_width = 600
            bar_height = 30
            bar_x = (w - bar_width) // 2
            bar_y = h - 50

            progress = int((count / num_images) * bar_width)

            cv2.rectangle(display_frame, (bar_x, bar_y),
                         (bar_x + bar_width, bar_y + bar_height),
                         (100, 100, 100), 2)
            cv2.rectangle(display_frame, (bar_x, bar_y),
                         (bar_x + progress, bar_y + bar_height),
                         (0, 255, 0), -1)

            # 显示百分比
            percent_text = f"{int(count/num_images*100)}%"
            cv2.putText(display_frame, percent_text,
                       (bar_x + bar_width//2 - 30, bar_y + 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Image Collection', display_frame)

            # 自动模式
            if auto_mode:
                current_time = datetime.now()
                elapsed = (current_time - last_save_time).total_seconds()

                if elapsed >= interval:
                    # 保存图像
                    timestamp = current_time.strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"{class_name}_{timestamp}_{count:04d}.jpg"
                    filepath = os.path.join(class_dir, filename)

                    cv2.imwrite(filepath, frame)
                    print(f"[{count+1}/{num_images}] 已保存: {filename}")

                    count += 1
                    last_save_time = current_time

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' ') and not auto_mode:
                # 手动保存
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{class_name}_{timestamp}_{count:04d}.jpg"
                filepath = os.path.join(class_dir, filename)

                cv2.imwrite(filepath, frame)
                print(f"[{count+1}/{num_images}] 已保存: {filename}")

                count += 1
            elif key == ord('a'):
                # 切换模式
                auto_mode = not auto_mode
                last_save_time = datetime.now()
                print(f"切换到{'自动'if auto_mode else '手动'}模式")

        cap.release()
        cv2.destroyAllWindows()

        print(f"\n采集完成!")
        print(f"共保存 {count} 张图像到: {class_dir}")

    def preview_dataset(self, class_name):
        """预览数据集"""
        class_dir = os.path.join(self.save_dir, class_name)

        if not os.path.exists(class_dir):
            print(f"错误: 目录不存在 {class_dir}")
            return

        images = sorted([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png'))])

        if not images:
            print(f"目录为空: {class_dir}")
            return

        print(f"\n预览数据集: {class_name}")
        print(f"图像数量: {len(images)}")
        print(f"\n使用左右箭头键切换图像, 按'q'退出")

        idx = 0

        while True:
            img_path = os.path.join(class_dir, images[idx])
            img = cv2.imread(img_path)

            if img is None:
                print(f"无法读取: {images[idx]}")
                idx = (idx + 1) % len(images)
                continue

            # 调整大小以适应屏幕
            h, w = img.shape[:2]
            max_h, max_w = 800, 1200

            if h > max_h or w > max_w:
                scale = min(max_h/h, max_w/w)
                img = cv2.resize(img, (int(w*scale), int(h*scale)))

            # 添加信息
            info = f"[{idx+1}/{len(images)}] {images[idx]}"
            cv2.putText(img, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Dataset Preview', img)

            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                break
            elif key == 81 or key == 2:  # 左箭头
                idx = (idx - 1) % len(images)
            elif key == 83 or key == 3:  # 右箭头
                idx = (idx + 1) % len(images)

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='图像数据采集工具')
    subparsers = parser.add_subparsers(dest='command', help='命令')

    # 采集命令
    collect_parser = subparsers.add_parser('collect', help='采集图像')
    collect_parser.add_argument('--class', dest='class_name', type=str, required=True,
                               help='类别名称 (如: landing_pad)')
    collect_parser.add_argument('--num', type=int, default=100,
                               help='采集数量 (默认: 100)')
    collect_parser.add_argument('--auto', action='store_true',
                               help='自动模式')
    collect_parser.add_argument('--interval', type=float, default=1.0,
                               help='自动模式间隔(秒) (默认: 1.0)')
    collect_parser.add_argument('--save-dir', type=str, default='data/images',
                               help='保存目录 (默认: data/images)')
    collect_parser.add_argument('--camera', type=int, default=0,
                               help='摄像头ID (默认: 0)')

    # 预览命令
    preview_parser = subparsers.add_parser('preview', help='预览数据集')
    preview_parser.add_argument('--class', dest='class_name', type=str, required=True,
                               help='类别名称')
    preview_parser.add_argument('--save-dir', type=str, default='data/images',
                               help='数据集目录 (默认: data/images)')

    args = parser.parse_args()

    if args.command == 'collect':
        collector = ImageCollector(
            save_dir=args.save_dir,
            camera_id=args.camera
        )
        collector.collect(
            class_name=args.class_name,
            num_images=args.num,
            auto_mode=args.auto,
            interval=args.interval
        )

    elif args.command == 'preview':
        collector = ImageCollector(save_dir=args.save_dir)
        collector.preview_dataset(args.class_name)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
