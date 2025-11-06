#!/bin/bash
# 模型下载脚本

set -e

echo "================================================"
echo "  多模态无人机系统 - 模型下载脚本"
echo "================================================"

# 创建模型目录
mkdir -p models
cd models

# ===== 1. 下载YOLOv8-nano模型 =====
echo ""
echo "[1/2] 下载 YOLOv8-nano 模型..."

if [ -f "yolov8n.pt" ]; then
    echo "  ✓ yolov8n.pt 已存在，跳过下载"
else
    echo "  正在下载... (大小: ~6MB)"

    # 方法1: 从GitHub Release下载
    wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O yolov8n.pt

    # 如果上面失败，尝试从Ultralytics CDN下载
    # wget https://ultralytics.com/assets/yolov8n.pt -O yolov8n.pt

    if [ -f "yolov8n.pt" ]; then
        echo "  ✓ yolov8n.pt 下载完成"
        ls -lh yolov8n.pt
    else
        echo "  ✗ 下载失败"
        exit 1
    fi
fi

# ===== 2. 下载Vosk中文语音模型 =====
echo ""
echo "[2/2] 下载 Vosk 中文语音模型..."

if [ -d "vosk-model-small-cn" ]; then
    echo "  ✓ vosk-model-small-cn 已存在，跳过下载"
else
    echo "  正在下载... (大小: ~42MB)"

    # 下载
    wget https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip

    # 解压
    echo "  正在解压..."
    unzip -q vosk-model-small-cn-0.22.zip

    # 重命名
    mv vosk-model-small-cn-0.22 vosk-model-small-cn

    # 清理
    rm vosk-model-small-cn-0.22.zip

    if [ -d "vosk-model-small-cn" ]; then
        echo "  ✓ vosk-model-small-cn 下载完成"
        du -sh vosk-model-small-cn
    else
        echo "  ✗ 下载失败"
        exit 1
    fi
fi

# ===== 下载完成 =====
echo ""
echo "================================================"
echo "  ✓ 所有模型下载完成！"
echo "================================================"
echo ""
echo "模型文件："
ls -lh

echo ""
echo "下一步："
echo "  1. 运行系统: python src/main.py --mouse"
echo "  2. 启用所有检测器: python src/main.py --enable-all"
