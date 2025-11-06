#!/bin/bash
# 测试所有模态（包括共享摄像头）

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mmdetect

echo "=========================================="
echo "多模态无人机系统 - 完整测试"
echo "=========================================="
echo ""
echo "启用的检测器:"
echo "  ✓ 语音检测 (独立麦克风)"
echo "  ✓ 手势识别 (共享摄像头)"
echo "  ✓ 图像检测 (共享摄像头)"
echo "  ✓ 鼠标路径 (独立窗口)"
echo ""
echo "摄像头共享架构："
echo "  Camera 0 ──→ CameraManager ──┬──→ 手势检测器"
echo "                               └──→ 图像检测器"
echo ""
echo "按 Ctrl+C 停止"
echo ""

# 启用所有检测器（现在可以正常工作了！）
python src/main.py --voice --gesture --image --mouse
