from __future__ import annotations


class InferenceError(Exception):
    """4.4 推理模块基础异常。"""


class QueryEncodingError(InferenceError):
    """查询无法完成编码时抛出。"""


class UnsupportedLanguageError(InferenceError):
    """代码扫描语言不受支持时抛出。"""
