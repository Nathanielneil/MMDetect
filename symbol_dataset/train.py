#!/usr/bin/env python3
"""
YOLOv8符号检测模型训练脚本
"""
from ultralytics import YOLO

def train_symbol_detector():
    # 加载预训练模型
    model = YOLO('yolov8n.pt')

    # 训练
    results = model.train(
        data='symbol_dataset/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        device=0,
        save=True,
        augment=True,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=15
    )

    print("\n训练完成!")
    print(f"最佳模型: {results.save_dir}/weights/best.pt")

    # 验证
    print("\n验证模型...")
    metrics = model.val()

    print("\n模型评估完成")
    return results

if __name__ == '__main__':
    results = train_symbol_detector()
