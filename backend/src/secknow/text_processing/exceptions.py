from __future__ import annotations

"""4.1 文本处理模块异常定义。"""


class TextProcessingError(Exception):
    """文本处理模块基础异常。"""


class UnsupportedFileTypeError(TextProcessingError):
    """不支持的文件类型异常。"""


class DocumentLoadError(TextProcessingError):
    """文档加载阶段异常。"""


class ChunkingError(TextProcessingError):
    """分块阶段异常。"""


class EncodingError(TextProcessingError):
    """编码阶段异常。"""
