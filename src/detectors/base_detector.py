"""
统一检测模板基类
所有检测器(语音、手势、图像、鼠标)都继承此基类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time
import logging
from enum import Enum


class CommandType(Enum):
    """支持的指令类型"""
    TAKEOFF = "takeoff"
    LAND = "land"
    HOVER = "hover"
    EXPLORE = "explore"
    FORMATION = "formation"
    WAYPOINT = "waypoint"
    EMERGENCY_STOP = "emergency_stop"


class ModalityType(Enum):
    """输入模态类型"""
    VOICE = "voice"
    GESTURE = "gesture"
    IMAGE = "image"
    MOUSE = "mouse"


class BaseDetector(ABC):
    """
    统一检测器基类

    所有检测器必须实现:
    1. initialize() - 初始化模型
    2. preprocess() - 预处理输入
    3. detect() - 执行检测
    4. postprocess() - 后处理结果
    """

    def __init__(self, modality: ModalityType, config: Optional[Dict[str, Any]] = None):
        """
        初始化检测器

        Args:
            modality: 输入模态类型
            config: 配置字典
        """
        self.modality = modality
        self.config = config or {}
        self.is_initialized = False
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # 性能统计
        self.detection_count = 0
        self.total_latency = 0.0
        self.last_detection_time = None

    @abstractmethod
    def initialize(self) -> bool:
        """
        初始化检测器(加载模型、配置等)

        Returns:
            bool: 初始化是否成功
        """
        pass

    @abstractmethod
    def preprocess(self, raw_input: Any) -> Any:
        """
        预处理输入数据

        Args:
            raw_input: 原始输入(音频/图像/坐标等)

        Returns:
            Any: 预处理后的数据
        """
        pass

    @abstractmethod
    def detect(self, preprocessed_input: Any) -> Dict[str, Any]:
        """
        执行检测/识别

        Args:
            preprocessed_input: 预处理后的输入

        Returns:
            Dict: 检测结果字典
        """
        pass

    @abstractmethod
    def postprocess(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        后处理检测结果

        Args:
            detection_result: 原始检测结果

        Returns:
            Dict: 标准化的指令字典
        """
        pass

    def run(self, raw_input: Any) -> Optional[Dict[str, Any]]:
        """
        完整检测流程

        Args:
            raw_input: 原始输入

        Returns:
            Optional[Dict]: 标准化的指令字典,失败返回None
        """
        if not self.is_initialized:
            self.logger.error("检测器未初始化,请先调用initialize()")
            return None

        start_time = time.time()

        try:
            # 1. 预处理
            preprocessed = self.preprocess(raw_input)

            # 2. 检测
            result = self.detect(preprocessed)

            # 3. 后处理
            command = self.postprocess(result)

            # 4. 添加元数据
            latency = (time.time() - start_time) * 1000  # ms
            command['metadata'] = {
                'modality': self.modality.value,
                'timestamp': time.time(),
                'latency_ms': latency,
                'detector': self.__class__.__name__
            }

            # 5. 更新统计
            self.detection_count += 1
            self.total_latency += latency
            self.last_detection_time = time.time()

            # 只在高置信度命令时输出日志，避免过多的hover日志
            if command.get('confidence', 0) > 0.5:
                self.logger.info(
                    f"检测成功: {command['command']} "
                    f"(置信度: {command.get('confidence', 0):.2f}, "
                    f"延迟: {latency:.1f}ms)"
                )

            return command

        except Exception as e:
            self.logger.error(f"检测失败: {str(e)}", exc_info=True)
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取性能统计

        Returns:
            Dict: 统计信息
        """
        avg_latency = self.total_latency / self.detection_count if self.detection_count > 0 else 0

        return {
            'modality': self.modality.value,
            'detection_count': self.detection_count,
            'average_latency_ms': avg_latency,
            'total_latency_ms': self.total_latency,
            'last_detection_time': self.last_detection_time
        }

    def create_command_dict(
        self,
        command: CommandType,
        confidence: float,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        创建标准化指令字典

        Args:
            command: 指令类型
            confidence: 置信度(0-1)
            parameters: 额外参数

        Returns:
            Dict: 标准化指令字典
        """
        return {
            'command': command.value,
            'confidence': max(0.0, min(1.0, confidence)),
            'parameters': parameters or {},
            'timestamp': time.time(),
            'modality': self.modality.value
        }

    def shutdown(self):
        """释放资源"""
        self.logger.info(f"{self.__class__.__name__} 正在关闭...")
        self.is_initialized = False


# 标准化指令字典示例
COMMAND_DICT_EXAMPLE = {
    'command': 'takeoff',           # 指令类型
    'confidence': 0.95,             # 置信度
    'parameters': {                 # 指令参数
        'altitude': 10.0,           # 高度(米)
        'formation': 'triangle'     # 编队类型
    },
    'timestamp': 1698765432.123,    # 时间戳
    'modality': 'voice',            # 输入模态
    'metadata': {                   # 元数据
        'latency_ms': 85.5,         # 检测延迟
        'detector': 'VoiceDetector' # 检测器名称
    }
}
