#!/bin/bash
# 启用可视化的测试（会有Qt警告但不影响功能）

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mmdetect

echo "=========================================="
echo "多模态系统 - 启用可视化版本"
echo "=========================================="
echo ""
echo "注意："
echo "  - 会显示多个窗口（手势、图像、鼠标）"
echo "  - 可能有Qt线程警告（可忽略）"
echo "  - 所有检测器都可见"
echo ""
echo "按 Ctrl+C 停止"
echo ""

# 临时启用可视化（覆盖配置）
export MMDETECT_ENABLE_VISUALIZATION=1

python src/main.py --gesture --mouse
