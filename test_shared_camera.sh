#!/bin/bash
# 测试多检测器共享摄像头

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mmdetect

echo "=========================================="
echo "测试: 手势 + 图像检测器共享摄像头"
echo "=========================================="
echo ""
echo "现在可以同时启用手势和图像检测器了！"
echo "两个检测器将共享同一个摄像头数据流"
echo ""
echo "按 Ctrl+C 停止测试"
echo ""

# 同时启用手势和图像检测器（现在不会冲突了！）
python src/main.py --gesture --image
