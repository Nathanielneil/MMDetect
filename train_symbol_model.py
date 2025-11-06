#!/usr/bin/env python3
"""
YOLOv8符号检测模型训练脚本 - 改进版本
训练H、T、F、S、L五个符号检测器
"""

import sys
import os
sys.path.insert(0, '/home/ubuntu/NGW/intern/MMDect')

from ultralytics import YOLO
import torch

def train_symbol_detector():
    """训练符号检测模型"""
    print("\n" + "="*70)
    print("  YOLOv8符号检测模型训练")
    print("="*70)

    # 检查GPU可用性
    cuda_available = torch.cuda.is_available()
    print(f"\nGPU可用: {cuda_available}")

    if cuda_available:
        try:
            print(f"GPU设备: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")
            device_id = 0
        except Exception as e:
            print(f"GPU初始化失败: {e}")
            print("强制使用CPU训练")
            device_id = 'cpu'
            cuda_available = False
    else:
        print("未检测到GPU，将使用CPU训练（速度较慢）")
        device_id = 'cpu'

    # 数据集路径
    data_yaml = '/home/ubuntu/NGW/intern/MMDect/symbol_dataset/data.yaml'

    # 检查数据集是否存在
    if not os.path.exists(data_yaml):
        print(f"错误: 数据集配置文件不存在: {data_yaml}")
        return False

    print(f"\n数据集配置: {data_yaml}")

    # 加载预训练模型
    print("\n加载预训练YOLOv8-nano模型...")
    model = YOLO('yolov8n.pt')

    print(f"模型加载完成")
    print(f"模型架构: {model.model}")

    # 训练配置
    print("\n" + "-"*70)
    print("训练配置:")
    print("-"*70)

    train_config = {
        'data': data_yaml,
        'epochs': 100,
        'imgsz': 640,
        'batch': 16 if cuda_available else 8,  # CPU的批大小更小
        'patience': 20,
        'device': device_id,  # 自动选择GPU或CPU
        'save': True,
        'augment': True,
        'flipud': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0 if cuda_available else 0.5,  # CPU上减少mosaic
        'close_mosaic': 15,
        'optimizer': 'SGD',
        'lr0': 0.01,  # 初始学习率
        'verbose': True
    }

    for key, value in train_config.items():
        print(f"  {key}: {value}")

    # 开始训练
    print("\n" + "="*70)
    print("开始训练...")
    print("="*70 + "\n")

    try:
        results = model.train(**train_config)

        print("\n" + "="*70)
        print("训练完成!")
        print("="*70)

        # 显示结果
        print(f"\n最佳模型路径:")
        print(f"  {results.save_dir}/weights/best.pt")

        print(f"\n训练日志:")
        print(f"  {results.save_dir}/results.csv")

        print(f"\n训练输出目录:")
        print(f"  {results.save_dir}")

        # 验证模型
        print("\n" + "-"*70)
        print("验证模型...")
        print("-"*70)

        metrics = model.val()

        print("\n验证完成!")
        print(f"\nmAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")

        return True

    except Exception as e:
        print(f"\n错误: 训练失败")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = train_symbol_detector()

    print("\n" + "="*70)
    if success:
        print("训练脚本执行成功!")
    else:
        print("训练脚本执行失败!")
    print("="*70 + "\n")

    sys.exit(0 if success else 1)

