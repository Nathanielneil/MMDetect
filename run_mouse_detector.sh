#!/bin/bash
# 运行鼠标检测器的便捷脚本

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mmdetect

# 运行鼠标检测器
echo "启动鼠标路径规划器..."
echo "使用说明:"
echo "  - 拖动鼠标绘制路径"
echo "  - 按 'r' 键重置"
echo "  - 按 'q' 键退出"
echo ""

python src/main.py --mouse
