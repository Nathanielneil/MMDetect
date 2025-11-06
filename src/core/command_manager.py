"""
指令字典管理器
维护多模态指令的缓存队列和融合逻辑
"""

import time
import threading
from collections import deque
from typing import Dict, Any, Optional, List, Callable
import logging
from enum import Enum


class FusionStrategy(Enum):
    """指令融合策略"""
    LATEST = "latest"              # 使用最新指令
    HIGHEST_CONFIDENCE = "highest" # 使用置信度最高的指令
    VOTING = "voting"              # 投票机制
    PRIORITY = "priority"          # 基于模态优先级


class CommandManager:
    """
    指令字典管理器

    功能:
    1. 维护多模态指令缓存队列
    2. 指令过滤(置信度阈值)
    3. 多模态指令融合
    4. 指令历史记录
    5. 回调通知
    """

    # 模态优先级(数字越大优先级越高)
    MODALITY_PRIORITY = {
        'mouse': 4,     # 最精确
        'image': 3,
        'gesture': 2,
        'voice': 1
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化指令管理器

        Args:
            config: 配置字典,包含:
                - buffer_size: 缓存队列大小(默认100)
                - confidence_threshold: 置信度阈值(默认0.5)
                - fusion_strategy: 融合策略(默认HIGHEST_CONFIDENCE)
                - fusion_window: 融合时间窗口(秒,默认0.5)
                - enable_history: 是否启用历史记录(默认True)
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

        self.buffer_size = self.config.get('buffer_size', 100)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.fusion_strategy = FusionStrategy(
            self.config.get('fusion_strategy', 'highest')
        )
        self.fusion_window = self.config.get('fusion_window', 0.5)
        self.enable_history = self.config.get('enable_history', True)

        # 指令缓存队列(每个模态一个队列)
        self.command_buffers = {
            'voice': deque(maxlen=self.buffer_size),
            'gesture': deque(maxlen=self.buffer_size),
            'image': deque(maxlen=self.buffer_size),
            'mouse': deque(maxlen=self.buffer_size)
        }

        # 融合后的指令队列
        self.fused_commands = deque(maxlen=self.buffer_size)

        # 历史记录
        self.history = [] if self.enable_history else None

        # 当前活动指令
        self.current_command = None

        # 回调函数列表
        self.callbacks = []

        # 线程锁
        self.lock = threading.Lock()

        # 统计
        self.stats = {
            'total_commands': 0,
            'filtered_commands': 0,
            'fused_commands': 0,
            'modality_counts': {
                'voice': 0,
                'gesture': 0,
                'image': 0,
                'mouse': 0
            }
        }

    def add_command(self, command: Dict[str, Any]) -> bool:
        """
        添加新指令

        Args:
            command: 指令字典

        Returns:
            bool: 是否成功添加
        """
        with self.lock:
            modality = command.get('modality')

            if modality not in self.command_buffers:
                self.logger.warning(f"未知模态: {modality}")
                return False

            # 更新统计
            self.stats['total_commands'] += 1
            self.stats['modality_counts'][modality] += 1

            # 置信度过滤
            if command.get('confidence', 0) < self.confidence_threshold:
                self.stats['filtered_commands'] += 1
                self.logger.debug(
                    f"指令被过滤(置信度过低): {command['command']} "
                    f"({command.get('confidence', 0):.2f})"
                )
                return False

            # 添加到缓存
            self.command_buffers[modality].append(command)

            # 添加到历史
            if self.enable_history:
                self.history.append(command)

            self.logger.info(
                f"接收指令: [{modality}] {command['command']} "
                f"(置信度: {command.get('confidence', 0):.2f})"
            )

            # 触发融合
            self._fuse_commands()

            return True

    def _fuse_commands(self):
        """融合多模态指令"""
        current_time = time.time()

        # 收集时间窗口内的所有指令
        recent_commands = []

        for modality, buffer in self.command_buffers.items():
            for cmd in buffer:
                if current_time - cmd['timestamp'] <= self.fusion_window:
                    recent_commands.append(cmd)

        if not recent_commands:
            return

        # 根据策略融合
        if self.fusion_strategy == FusionStrategy.LATEST:
            fused = max(recent_commands, key=lambda x: x['timestamp'])

        elif self.fusion_strategy == FusionStrategy.HIGHEST_CONFIDENCE:
            fused = max(recent_commands, key=lambda x: x['confidence'])

        elif self.fusion_strategy == FusionStrategy.PRIORITY:
            fused = max(
                recent_commands,
                key=lambda x: self.MODALITY_PRIORITY.get(x['modality'], 0)
            )

        elif self.fusion_strategy == FusionStrategy.VOTING:
            fused = self._voting_fusion(recent_commands)

        else:
            fused = recent_commands[-1]

        # 检查是否与当前指令不同
        if (not self.current_command or
            fused['command'] != self.current_command.get('command') or
            fused['modality'] != self.current_command.get('modality')):

            self.current_command = fused
            self.fused_commands.append(fused)
            self.stats['fused_commands'] += 1

            self.logger.info(
                f"融合指令: {fused['command']} "
                f"[{fused['modality']}] "
                f"(置信度: {fused['confidence']:.2f})"
            )

            # 触发回调
            self._notify_callbacks(fused)

    def _voting_fusion(self, commands: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        投票融合策略

        Args:
            commands: 指令列表

        Returns:
            Dict: 融合后的指令
        """
        # 统计每种指令的投票(加权)
        votes = {}

        for cmd in commands:
            cmd_type = cmd['command']
            weight = cmd['confidence'] * self.MODALITY_PRIORITY.get(cmd['modality'], 1)

            if cmd_type not in votes:
                votes[cmd_type] = {'weight': 0, 'commands': []}

            votes[cmd_type]['weight'] += weight
            votes[cmd_type]['commands'].append(cmd)

        # 选择得票最高的指令
        winner = max(votes.items(), key=lambda x: x[1]['weight'])
        winner_commands = winner[1]['commands']

        # 返回该类型中置信度最高的指令
        return max(winner_commands, key=lambda x: x['confidence'])

    def get_current_command(self) -> Optional[Dict[str, Any]]:
        """
        获取当前活动指令

        Returns:
            Optional[Dict]: 当前指令,无则返回None
        """
        with self.lock:
            return self.current_command

    def get_command_history(
        self,
        modality: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        获取指令历史

        Args:
            modality: 过滤特定模态(可选)
            limit: 限制返回数量(可选)

        Returns:
            List[Dict]: 历史指令列表
        """
        with self.lock:
            if not self.enable_history:
                return []

            history = self.history

            if modality:
                history = [cmd for cmd in history if cmd['modality'] == modality]

            if limit:
                history = history[-limit:]

            return history

    def register_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        注册回调函数

        Args:
            callback: 回调函数,接收指令字典作为参数
        """
        self.callbacks.append(callback)
        self.logger.info(f"注册回调: {callback.__name__}")

    def _notify_callbacks(self, command: Dict[str, Any]):
        """
        通知所有回调函数

        Args:
            command: 指令字典
        """
        for callback in self.callbacks:
            try:
                callback(command)
            except Exception as e:
                self.logger.error(f"回调执行失败 {callback.__name__}: {str(e)}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            Dict: 统计数据
        """
        with self.lock:
            return {
                'total_commands': self.stats['total_commands'],
                'filtered_commands': self.stats['filtered_commands'],
                'fused_commands': self.stats['fused_commands'],
                'modality_counts': self.stats['modality_counts'].copy(),
                'buffer_status': {
                    modality: len(buffer)
                    for modality, buffer in self.command_buffers.items()
                },
                'fusion_strategy': self.fusion_strategy.value
            }

    def clear_buffers(self):
        """清空所有缓存"""
        with self.lock:
            for buffer in self.command_buffers.values():
                buffer.clear()
            self.fused_commands.clear()
            self.logger.info("已清空所有缓存")

    def export_history(self, filepath: str):
        """
        导出历史记录到文件

        Args:
            filepath: 导出文件路径
        """
        if not self.enable_history:
            self.logger.warning("历史记录未启用")
            return

        import json

        with self.lock:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)

        self.logger.info(f"历史记录已导出到: {filepath}")


# 使用示例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = {
        'confidence_threshold': 0.6,
        'fusion_strategy': 'highest',
        'fusion_window': 1.0
    }

    manager = CommandManager(config)

    # 注册回调
    def on_command(cmd):
        print(f"\n>>> 新指令: {cmd['command']} [{cmd['modality']}]")

    manager.register_callback(on_command)

    # 模拟多模态指令
    import time

    # 语音指令
    manager.add_command({
        'command': 'takeoff',
        'confidence': 0.8,
        'timestamp': time.time(),
        'modality': 'voice',
        'parameters': {}
    })

    time.sleep(0.2)

    # 手势指令(置信度更高)
    manager.add_command({
        'command': 'takeoff',
        'confidence': 0.95,
        'timestamp': time.time(),
        'modality': 'gesture',
        'parameters': {}
    })

    time.sleep(0.3)

    # 鼠标指令(不同指令)
    manager.add_command({
        'command': 'waypoint',
        'confidence': 0.9,
        'timestamp': time.time(),
        'modality': 'mouse',
        'parameters': {'waypoints': []}
    })

    # 打印统计
    print("\n统计信息:")
    import json
    print(json.dumps(manager.get_statistics(), indent=2))

    # 获取历史
    print("\n指令历史:")
    for cmd in manager.get_command_history(limit=5):
        print(f"  [{cmd['modality']}] {cmd['command']} - {cmd['confidence']:.2f}")
