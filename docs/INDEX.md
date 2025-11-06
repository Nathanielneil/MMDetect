# MMDect 文档索引

本页面是MMDect项目的文档导航中心。

**最后更新**: 2025-11-06

---

## 📚 核心文档

### 1. [README.md](../README.md)
**项目概览和快速开始指南**

- 系统特性一览表
- 5分钟快速开始
- 快速命令参考
- 核心亮点介绍
- 常见问题快速解决

**适合人群**: 所有用户，首次接触项目必读

**阅读时间**: 5-10分钟

---

### 2. [DOCUMENTATION.md](../DOCUMENTATION.md) ⭐ 完整文档
**项目的统一完整文档 - 包含所有详细内容**

这是项目最重要的文档，整合了原来38个文档的所有内容。

**内容章节**:
1. **项目概述** - 系统简介、支持的指令、技术栈
2. **快速开始** - 最快测试、完整系统测试、命令行参数
3. **安装指南** - Python环境、依赖安装、模型下载
4. **使用手册** - 4种检测器详细使用方法
5. **系统架构** - 整体架构、核心模块、命令字典格式
6. **检测器详解** - 语音/手势/图像/鼠标检测器技术细节
7. **配置说明** - 完整配置文件解析
8. **故障排除** - 摄像头、模型、依赖、运行时错误解决
9. **开发指南** - 添加新检测器、融合策略、训练自定义模型
10. **性能指标** - 延迟测试、准确率、资源占用、模型大小
11. **项目状态** - 当前版本、已完成/待实现功能

**适合人群**:
- 想深入了解系统原理的开发者
- 需要自定义开发的研究者
- 需要完整技术规格的工程师

**阅读时间**: 1-2小时（完整阅读）

**快速查找**:
- 安装问题 → 第3章
- 使用方法 → 第4章
- 故障排除 → 第8章
- 开发扩展 → 第9章
- 性能数据 → 第10章

---

## 🎯 文档结构变化说明

### 之前（38个文档）
项目原有38个markdown文档分散在根目录和docs目录，包含大量重复内容和过时信息。

### 现在（2个核心文档）
我们将所有文档整合为：
1. **README.md** - 项目概览和快速入门
2. **DOCUMENTATION.md** - 完整详细文档

这样的好处：
- ✅ 消除了文档重复和冲突
- ✅ 更容易维护和更新
- ✅ 查找信息更快速
- ✅ 学习路径更清晰

---

## 🚀 推荐学习路径

### 路径A: 快速上手（30分钟）
适合想快速测试系统的用户

1. 阅读 [README.md](../README.md) （5分钟）
2. 运行快速开始命令 （20分钟）
3. 查看 [DOCUMENTATION.md](../DOCUMENTATION.md) 第4章：使用手册 （5分钟）

### 路径B: 深入理解（2小时）
适合想了解系统原理的开发者

1. 阅读 [DOCUMENTATION.md](../DOCUMENTATION.md) 完整文档 （60分钟）
2. 浏览源代码 `src/detectors/` （40分钟）
3. 测试各个检测器 （20分钟）

### 路径C: 自定义开发（半天）
适合想扩展系统功能的研究者

1. 完成路径B的所有内容 （2小时）
2. 阅读 [DOCUMENTATION.md](../DOCUMENTATION.md) 第9章：开发指南 （30分钟）
3. 训练自定义模型 （2-3小时）

---

## 📖 按场景查找

### 场景1: 我想快速测试手势识别
👉 [README.md](../README.md) → 快速命令 → 测试单个检测器

```bash
python tools/test_camera.py
python src/detectors/gesture_detector.py
python test_gesture_advanced.py  # 带指尖轨迹特效
```

### 场景2: 摄像头打不开怎么办？
👉 [DOCUMENTATION.md](../DOCUMENTATION.md) → 第8章：故障排除 → 8.1 摄像头问题

### 场景3: 如何下载预训练模型？
👉 [DOCUMENTATION.md](../DOCUMENTATION.md) → 第3章：安装指南 → 3.3 下载预训练模型

### 场景4: 手势识别不准确怎么调优？
👉 [DOCUMENTATION.md](../DOCUMENTATION.md) → 第8章：故障排除 → 8.1 Q3

### 场景5: 我想训练自己的目标检测模型
👉 [DOCUMENTATION.md](../DOCUMENTATION.md) → 第9章：开发指南 → 9.3 训练自定义模型

### 场景6: 如何理解多模态融合策略？
👉 [DOCUMENTATION.md](../DOCUMENTATION.md) → 第5章：系统架构 → 5.2 核心模块

### 场景7: 系统架构是怎样的？
👉 [DOCUMENTATION.md](../DOCUMENTATION.md) → 第5章：系统架构

### 场景8: 项目当前进度如何？
👉 [DOCUMENTATION.md](../DOCUMENTATION.md) → 第11章：项目状态

---

## 🔍 快速命令参考

### 测试命令
```bash
# 摄像头测试
python tools/test_camera.py

# 单个检测器测试
python src/detectors/gesture_detector.py  # 手势
python test_gesture_advanced.py           # 高级手势（带轨迹）
python src/detectors/voice_detector.py    # 语音
python src/detectors/image_detector.py    # 图像
python src/detectors/mouse_detector.py    # 鼠标
```

### 完整系统
```bash
# 启用所有检测器
python src/main.py --enable-all --unified-viewer

# 启用特定组合
python src/main.py --gesture --image --mouse

# 调试模式
python src/main.py --enable-all --log-level DEBUG
```

---

## 📞 获取帮助

### 问题查找顺序
1. [README.md](../README.md) - 快速命令和常见问题
2. [DOCUMENTATION.md](../DOCUMENTATION.md) 第8章 - 故障排除
3. [DOCUMENTATION.md](../DOCUMENTATION.md) 其他相关章节

### 联系方式
- 🐛 GitHub Issues: 问题反馈
- 💬 GitHub Discussions: 技术讨论

---

## 📝 文档维护

**文档版本**: v2.0 (整合版)
**整合日期**: 2025-11-06
**维护者**: MMDect开发团队

**更新日志**:
- 2025-11-06: 将38个文档整合为2个核心文档
- 2025-10-28: 创建原始文档体系

---

**快速跳转**:
- [返回首页](../README.md)
- [查看完整文档](../DOCUMENTATION.md)
- [查看源代码](../src/)

**最后更新**: 2025-11-06
