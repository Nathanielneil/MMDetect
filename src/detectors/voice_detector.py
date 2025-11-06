"""
语音指令检测器
使用Vosk进行轻量级语音识别
"""

import json
import numpy as np
from typing import Dict, Any, Optional
import logging

try:
    from vosk import Model, KaldiRecognizer
    import pyaudio
except ImportError:
    logging.warning("Vosk或pyaudio未安装,语音检测器将无法使用")

# 支持直接运行和模块导入两种方式
try:
    from .base_detector import BaseDetector, CommandType, ModalityType
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.detectors.base_detector import BaseDetector, CommandType, ModalityType


class VoiceDetector(BaseDetector):
    """
    语音指令检测器

    使用Vosk进行实时语音识别,将识别文本映射到指令
    """

    # 指令关键词映射
    COMMAND_KEYWORDS = {
        CommandType.TAKEOFF: ['起飞', '起飞', '升空', 'takeoff', 'take off'],
        CommandType.LAND: ['降落', '着陆', '下降', 'land', 'landing'],
        CommandType.HOVER: ['悬停', '停止', '保持', 'hover', 'hold'],
        CommandType.EXPLORE: ['探索', '搜索', '巡航', 'explore', 'search'],
        CommandType.FORMATION: ['编队', '队形', 'formation', 'form up'],
        CommandType.EMERGENCY_STOP: ['紧急停止', '急停', 'emergency', 'stop']
    }

    # 编队类型关键词
    FORMATION_KEYWORDS = {
        'triangle': ['三角', '三角形', 'triangle'],
        'line': ['一字', '直线', 'line', 'straight'],
        'circle': ['圆形', '圆圈', 'circle'],
        'square': ['方形', '正方形', 'square']
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化语音检测器

        Args:
            config: 配置字典,包含:
                - model_path: Vosk模型路径
                - sample_rate: 采样率(默认16000)
                - confidence_threshold: 置信度阈值(默认0.7)
        """
        super().__init__(ModalityType.VOICE, config)

        self.model_path = self.config.get('model_path', 'models/vosk-model-small-cn')
        self.sample_rate = self.config.get('sample_rate', 16000)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)

        self.model = None
        self.recognizer = None
        self.audio_stream = None

    def initialize(self) -> bool:
        """初始化Vosk模型和音频流"""
        try:
            self.logger.info(f"加载Vosk模型: {self.model_path}")
            self.model = Model(self.model_path)
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            self.recognizer.SetWords(True)

            self.is_initialized = True
            self.logger.info("语音检测器初始化成功")
            return True

        except Exception as e:
            self.logger.error(f"语音检测器初始化失败: {str(e)}")
            return False

    def preprocess(self, audio_data: bytes) -> bytes:
        """
        预处理音频数据

        Args:
            audio_data: 原始音频字节流(16-bit PCM)

        Returns:
            bytes: 预处理后的音频数据
        """
        # Vosk直接处理PCM数据,无需额外预处理
        return audio_data

    def detect(self, audio_data: bytes) -> Dict[str, Any]:
        """
        执行语音识别

        Args:
            audio_data: 预处理后的音频数据

        Returns:
            Dict: 识别结果
        """
        if self.recognizer.AcceptWaveform(audio_data):
            result = json.loads(self.recognizer.Result())
        else:
            result = json.loads(self.recognizer.PartialResult())

        return result

    def postprocess(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        将识别文本转换为指令

        Args:
            detection_result: Vosk识别结果

        Returns:
            Dict: 标准化指令字典
        """
        text = detection_result.get('text', '').lower()

        if not text:
            return self.create_command_dict(
                CommandType.HOVER,
                confidence=0.0,
                parameters={'reason': 'no_speech_detected'}
            )

        # 匹配指令
        matched_command = None
        max_confidence = 0.0

        for command_type, keywords in self.COMMAND_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    # 简单置信度计算:关键词在文本中的相对位置
                    confidence = 0.8 + 0.2 * (1 - text.index(keyword) / len(text))
                    if confidence > max_confidence:
                        max_confidence = confidence
                        matched_command = command_type

        if matched_command is None:
            self.logger.warning(f"无法识别指令: {text}")
            return self.create_command_dict(
                CommandType.HOVER,
                confidence=0.0,
                parameters={'raw_text': text, 'reason': 'unrecognized'}
            )

        # 提取参数
        parameters = {'raw_text': text}

        # 如果是编队指令,提取编队类型
        if matched_command == CommandType.FORMATION:
            for formation_type, keywords in self.FORMATION_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in text:
                        parameters['formation'] = formation_type
                        break

        return self.create_command_dict(
            matched_command,
            confidence=max_confidence,
            parameters=parameters
        )

    def listen_continuous(self, callback, duration: Optional[float] = None):
        """
        持续监听模式

        Args:
            callback: 检测到指令时的回调函数
            duration: 监听时长(秒),None表示无限监听
        """
        if not self.is_initialized:
            self.logger.error("检测器未初始化")
            return

        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=4000
        )

        self.logger.info("开始监听语音指令...")

        try:
            import time
            start_time = time.time()

            while True:
                if duration and (time.time() - start_time) > duration:
                    break

                data = stream.read(4000, exception_on_overflow=False)
                command = self.run(data)

                if command and command['confidence'] >= self.confidence_threshold:
                    callback(command)

        except KeyboardInterrupt:
            self.logger.info("停止监听")

        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

    def shutdown(self):
        """释放资源"""
        super().shutdown()
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()


# 使用示例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = {
        'model_path': 'models/vosk-model-small-cn',
        'confidence_threshold': 0.7
    }

    detector = VoiceDetector(config)

    if detector.initialize():
        def on_command(cmd):
            print(f"检测到指令: {cmd}")

        # 持续监听10秒
        detector.listen_continuous(on_command, duration=10)

        # 打印统计
        print(detector.get_statistics())
