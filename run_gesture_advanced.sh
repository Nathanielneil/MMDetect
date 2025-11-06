#!/bin/bash
# 运行高级手势识别（带完整可视化）

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mmdetect

echo "=========================================="
echo "高级手势识别系统"
echo "=========================================="
echo ""
echo "✨ 视觉特效："
echo "  ✓ 手部骨架实时显示"
echo "  ✓ 关键点高亮"
echo "  ✓ 指尖轨迹追踪"
echo "  ✓ 手势识别结果"
echo "  ✓ 置信度实时显示"
echo ""
echo "👋 支持的手势："
echo "  🖐️  张开手掌  (5指) → TAKEOFF   (起飞)"
echo "  ✊  握拳      (0指) → LAND      (降落)"
echo "  ☝️  单指      (1指) → HOVER     (悬停)"
echo "  ✌️  V字手势   (2指) → EXPLORE   (探索)"
echo "  🤟  三指      (3指) → FORMATION (编队)"
echo ""
echo "💡 使用技巧："
echo "  • 距离摄像头 30-60cm"
echo "  • 手放在画面中心"
echo "  • 每个手势保持 1-2秒"
echo "  • 按 'q' 键退出"
echo ""
echo "启动中..."
echo "=========================================="
echo ""

# 运行独立的手势检测器
python src/detectors/gesture_detector.py
