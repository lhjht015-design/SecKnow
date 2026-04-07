from __future__ import annotations

"""固定窗口 / hybrid 分块实现。"""

from ..config import DEFAULT_CHUNK_MAX_TOKENS, DEFAULT_CHUNK_OVERLAP
from ..exceptions import ChunkingError


class _TokenSeq:
    """内部 token 序列结构，用于统一处理中英文分词差异。"""

    def __init__(self, tokens: list[str], joiner: str):
        self.tokens = tokens
        self.joiner = joiner

    @property
    def length(self) -> int:
        """返回 token 数量。"""
        return len(self.tokens)

    def to_text(self) -> str:
        """把 token 重新拼接为字符串。"""
        return self.joiner.join(self.tokens).strip()


def _to_tokens(text: str) -> _TokenSeq:
    """把文本转换为 token 序列。"""
    compact = text.strip()
    if not compact:
        return _TokenSeq([], "")

    # 含空格文本按词切分；无空格文本按字符切分，确保中文也能稳定分块。
    if " " in compact:
        return _TokenSeq(compact.split(), " ")
    return _TokenSeq(list(compact), "")


def _window_split(tokens: _TokenSeq, max_tokens: int, overlap: int) -> list[str]:
    """按固定窗口切分 token 序列。"""
    if max_tokens <= 0:
        raise ChunkingError("max_tokens 必须大于 0")
    if overlap >= max_tokens:
        raise ChunkingError("overlap 必须小于 max_tokens")

    if tokens.length == 0:
        return []

    chunks: list[str] = []
    step = max_tokens - overlap
    for start in range(0, tokens.length, step):
        end = min(start + max_tokens, tokens.length)
        piece = _TokenSeq(tokens.tokens[start:end], tokens.joiner).to_text()
        if piece:
            chunks.append(piece)
        if end >= tokens.length:
            break
    return chunks


def chunk_fixed_window(
    text: str,
    max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """固定窗口分块。"""
    return _window_split(_to_tokens(text), max_tokens=max_tokens, overlap=overlap)


def chunk_hybrid(
    text: str,
    max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """hybrid 分块：先段落聚合，超长段再退回固定窗口。"""
    paragraphs = [segment.strip() for segment in text.split("\n") if segment.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    buffer_tokens: list[str] = []
    buffer_joiner = ""

    for paragraph in paragraphs:
        token_seq = _to_tokens(paragraph)
        if token_seq.length == 0:
            continue

        # 超长段落直接按固定窗口切分，避免单块过大。
        if token_seq.length > max_tokens:
            if buffer_tokens:
                chunks.append(_TokenSeq(buffer_tokens, buffer_joiner).to_text())
                buffer_tokens = []
                buffer_joiner = ""
            chunks.extend(_window_split(token_seq, max_tokens=max_tokens, overlap=overlap))
            continue

        if not buffer_tokens:
            buffer_tokens = list(token_seq.tokens)
            buffer_joiner = token_seq.joiner
            continue

        if len(buffer_tokens) + token_seq.length <= max_tokens:
            # 优先拼接同类型文本，减少过短 chunk。
            if buffer_joiner == token_seq.joiner:
                buffer_tokens.extend(token_seq.tokens)
            else:
                chunks.append(_TokenSeq(buffer_tokens, buffer_joiner).to_text())
                buffer_tokens = list(token_seq.tokens)
                buffer_joiner = token_seq.joiner
        else:
            chunks.append(_TokenSeq(buffer_tokens, buffer_joiner).to_text())
            buffer_tokens = list(token_seq.tokens)
            buffer_joiner = token_seq.joiner

    if buffer_tokens:
        chunks.append(_TokenSeq(buffer_tokens, buffer_joiner).to_text())

    return [chunk for chunk in chunks if chunk]
