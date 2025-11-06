#!/bin/bash
# 运行统一可视化界面

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mmdetect

echo "=========================================="
echo "MMDect - 统一可视化界面"
echo "=========================================="
echo ""
echo "功能特点："
echo "  ✓ 单窗口显示所有模态"
echo "  ✓ 键盘切换不同视图"
echo "  ✓ 实时显示检测结果"
echo "  ✓ 统计信息展示"
echo ""
echo "键盘控制："
echo "  [1] 手势识别视图"
echo "  [2] 图像检测视图"
echo "  [3] 鼠标路径视图"
echo "  [4] 四分屏视图"
echo "  [0] 概览视图"
echo "  [R] 重置"
echo "  [Q] 退出"
echo ""
echo "启动中..."
echo "=========================================="
echo ""

# 运行统一查看器模式
python src/main.py --gesture --image --mouse --unified-viewer
