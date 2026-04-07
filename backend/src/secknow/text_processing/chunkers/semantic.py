from __future__ import annotations

"""基于 LangChain Experimental SemanticChunker 的语义分块。

**本文件相对早期版本的改动要点**（供 Code Review / 接手同事速览）：
- 原先此处为「正则按句 + token 合并」的轻量实现；现改为 **真实嵌入驱动** 的语义断点。
- 依赖：`langchain-experimental` 的 `SemanticChunker` + `langchain-community` 的
  `HuggingFaceEmbeddings`，嵌入模型名与全局 `EMBEDDING_MODEL` 对齐（见 `config.get_embedding_model`）。
- **约束**：`EMBEDDING_MODE=fake` 时会直接报错——fake 向量无法表达语义距离。
- **后处理**：LangChain 产出的块可能仍长于 `max_tokens`，故统一再走 `chunk_fixed_window`
  截断，与 `hybrid`/`fixed_window` 等策略对 4.3 契约的长度预期一致。
- **可调参数**：见 `config.py` 中 `DEFAULT_SEMANTIC_*` 及对应 `get_semantic_*` / 环境变量说明。

与 LlamaIndex `SemanticSplitterNodeParser` 同属「嵌入相似度切分」思路；本仓库只落地 LangChain 一条链。

官方说明：https://python.langchain.com/docs/how_to/semantic-chunker/
"""

from functools import lru_cache
from typing import Literal, cast

from ..config import (
    DEFAULT_CHUNK_MAX_TOKENS,
    DEFAULT_CHUNK_OVERLAP,
    get_embedding_mode,
    get_embedding_model,
    get_semantic_breakpoint_amount,
    get_semantic_breakpoint_type,
    get_semantic_embedding_device,
    get_semantic_sentence_split_regex,
)
from ..exceptions import ChunkingError
from .fixed_window import _to_tokens, chunk_fixed_window

_BreakpointType = Literal[
    "percentile", "standard_deviation", "interquartile", "gradient"
]


@lru_cache(maxsize=8)
def _build_hf_embeddings(model_name: str, device: str):
    """懒加载 LangChain HuggingFaceEmbeddings，按 (model, device) 缓存。

    注意：这是 **独立于** `encoders/sbert.py` 里 SBERTEmbedder 的另一条加载路径，
    LangChain 需要自己的 Embeddings 包装类；两边应使用同一 `model_name` 以免语义空间不一致。
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def _enforce_max_tokens(
    chunks: list[str], max_tokens: int, overlap: int
) -> list[str]:
    """语义块可能过长，退回固定窗口以满足上游 token 上限。"""
    out: list[str] = []
    for piece in chunks:
        stripped = piece.strip()
        if not stripped:
            continue
        if _to_tokens(stripped).length <= max_tokens:
            out.append(stripped)
        else:
            out.extend(
                chunk_fixed_window(
                    stripped, max_tokens=max_tokens, overlap=overlap
                )
            )
    return out


def chunk_semantic(
    text: str,
    max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """LangChain SemanticChunker：按句向量语义断点切分，再约束 max_tokens。"""
    compact = text.strip()
    if not compact:
        return []

    # fake 模式用于联调/单测向量维度，不产生真实语义空间，SemanticChunker 无法工作。
    if get_embedding_mode() == "fake":
        raise ChunkingError(
            "semantic 分块依赖真实句向量，请将 EMBEDDING_MODE 设为 sbert "
            "或去掉 fake 后再使用 strategy=semantic"
        )

    try:
        # 延迟导入：避免仅使用 hybrid/fixed_window 的进程拉起 LangChain 依赖树。
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError as exc:  # pragma: no cover - 环境缺依赖时提示安装
        raise ChunkingError(
            "未安装 LangChain 语义分块依赖，请安装: "
            "langchain-core、langchain-community、langchain-experimental"
        ) from exc

    embeddings = _build_hf_embeddings(
        get_embedding_model(),
        get_semantic_embedding_device(),
    )
    # 单文档传入 list：SemanticChunker 在内部先按句 regex 切句，再算相邻句嵌入距离并合并/断开。
    chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=cast(
            _BreakpointType, get_semantic_breakpoint_type()
        ),
        breakpoint_threshold_amount=get_semantic_breakpoint_amount(),
        sentence_split_regex=get_semantic_sentence_split_regex(),
    )
    docs = chunker.create_documents([compact])
    # LangChain Document.page_content 即我们下游的「块文本」。
    raw_chunks = [
        d.page_content.strip()
        for d in docs
        if getattr(d, "page_content", None) and str(d.page_content).strip()
    ]
    if not raw_chunks:
        return []

    return _enforce_max_tokens(raw_chunks, max_tokens=max_tokens, overlap=overlap)
