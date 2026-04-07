from __future__ import annotations

"""4.1 文本处理模块的基础配置。

除原有分块/嵌入默认值外，本文件还集中定义两类「后增能力」的配置（均可被环境变量覆盖）：

1. **MinHash 近似去重**（`dedupers/minhash.py` + `dedup_strategy="minhash"`）
   - 常量：`DEFAULT_MINHASH_*`
   - 环境变量：`DEDUP_MINHASH_THRESHOLD`、`DEDUP_MINHASH_PERM`、`DEDUP_MINHASH_SHINGLE_SIZE`

2. **LangChain 语义分块**（`chunkers/semantic.py` + `chunk_strategy="semantic"`）
   - 常量：`DEFAULT_SEMANTIC_*`
   - 环境变量：`SEMANTIC_BREAKPOINT_TYPE`、`SEMANTIC_BREAKPOINT_AMOUNT`、
     `SEMANTIC_EMBEDDING_DEVICE`、`SEMANTIC_SENTENCE_SPLIT_REGEX`

**PDF 抽取相关环境变量**见 `loaders/pdf_advanced.py` 模块文档（`PDF_OCR_*`、`PDF_TABLES_*`），
避免本文件过长，PDF 逻辑集中在该模块读取 `os.environ`。

流水线默认仍为 `DEFAULT_DEDUP_STRATEGY=exact`、`DEFAULT_CHUNK_STRATEGY=hybrid`，未改旧行为。
"""

import os

# 支持的文件扩展名到业务文件类型的映射。
SUPPORTED_EXTS: dict[str, str] = {
    ".txt": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".pdf": "pdf",
    ".docx": "docx",
    ".py": "code",
    ".js": "code",
    ".ts": "code",
    ".java": "code",
    ".go": "code",
    ".c": "code",
    ".cpp": "code",
    ".h": "code",
    ".hpp": "code",
    ".json": "text",
    ".yaml": "text",
    ".yml": "text",
    ".toml": "text",
    ".ini": "text",
    ".cfg": "text",
    ".log": "text",
}

# 默认 embedding 约定：与 README 的 Phase 1 契约保持一致。
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_EMBEDDING_MODE = "sbert"

DEFAULT_CHUNK_STRATEGY = "hybrid"
DEFAULT_DEDUP_STRATEGY = "exact"
DEFAULT_CHUNK_MAX_TOKENS = 300
DEFAULT_CHUNK_OVERLAP = 50

# --- MinHash 近似去重（设计方案中的 DataSketch/MinHash 路线；实现见 dedupers/minhash.py）---
# 新块与已保留块的估计 Jaccard ≥ 阈值则丢弃新块（保留先出现的版本）。
DEFAULT_MINHASH_THRESHOLD = 0.85
DEFAULT_MINHASH_PERM = 128
DEFAULT_MINHASH_SHINGLE_SIZE = 5

# --- LangChain SemanticChunker（实现见 chunkers/semantic.py；需 EMBEDDING_MODE!=fake）---
DEFAULT_SEMANTIC_BREAKPOINT_TYPE = "percentile"
DEFAULT_SEMANTIC_BREAKPOINT_AMOUNT = 90.0
DEFAULT_SEMANTIC_EMBEDDING_DEVICE = "cpu"
DEFAULT_SEMANTIC_SENTENCE_SPLIT_REGEX = r"(?<=[。！？．.?!])\s+"


def get_embedding_mode() -> str:
    """读取 embedding 模式：`sbert` 或 `fake`。"""
    return os.getenv("EMBEDDING_MODE", DEFAULT_EMBEDDING_MODE).strip().lower()


def get_embedding_model() -> str:
    """读取 embedding 模型名。"""
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()


def get_embedding_dim() -> int:
    """读取 embedding 维度，异常值自动回退到默认值。"""
    raw = os.getenv("EMBEDDING_DIM", str(DEFAULT_EMBEDDING_DIM)).strip()
    try:
        dim = int(raw)
    except ValueError:
        return DEFAULT_EMBEDDING_DIM
    return dim if dim > 0 else DEFAULT_EMBEDDING_DIM


def get_chunk_max_tokens() -> int:
    """读取分块最大 token 数。"""
    raw = os.getenv("CHUNK_MAX_TOKENS", str(DEFAULT_CHUNK_MAX_TOKENS)).strip()
    try:
        val = int(raw)
    except ValueError:
        return DEFAULT_CHUNK_MAX_TOKENS
    return val if val > 0 else DEFAULT_CHUNK_MAX_TOKENS


def get_chunk_overlap() -> int:
    """读取分块 overlap。"""
    raw = os.getenv("CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP)).strip()
    try:
        val = int(raw)
    except ValueError:
        return DEFAULT_CHUNK_OVERLAP
    return max(0, val)


def get_minhash_threshold() -> float:
    """读取 MinHash Jaccard 相似度阈值，高于此值的块视为重复。"""
    raw = os.getenv("DEDUP_MINHASH_THRESHOLD", str(DEFAULT_MINHASH_THRESHOLD)).strip()
    try:
        val = float(raw)
    except ValueError:
        return DEFAULT_MINHASH_THRESHOLD
    return min(1.0, max(0.0, val))


def get_minhash_perm() -> int:
    """读取 MinHash 排列数。"""
    raw = os.getenv("DEDUP_MINHASH_PERM", str(DEFAULT_MINHASH_PERM)).strip()
    try:
        val = int(raw)
    except ValueError:
        return DEFAULT_MINHASH_PERM
    return max(16, val)


def get_minhash_shingle_size() -> int:
    """读取字符级 shingle 长度。"""
    raw = os.getenv("DEDUP_MINHASH_SHINGLE_SIZE", str(DEFAULT_MINHASH_SHINGLE_SIZE)).strip()
    try:
        val = int(raw)
    except ValueError:
        return DEFAULT_MINHASH_SHINGLE_SIZE
    return max(2, val)


def get_semantic_breakpoint_type() -> str:
    """SemanticChunker 的 breakpoint_threshold_type。"""
    allowed = {"percentile", "standard_deviation", "interquartile", "gradient"}
    raw = os.getenv("SEMANTIC_BREAKPOINT_TYPE", DEFAULT_SEMANTIC_BREAKPOINT_TYPE).strip().lower()
    return raw if raw in allowed else DEFAULT_SEMANTIC_BREAKPOINT_TYPE


def get_semantic_breakpoint_amount() -> float:
    """断点阈值数量；percentile 模式下越小切得越碎。"""
    raw = os.getenv(
        "SEMANTIC_BREAKPOINT_AMOUNT", str(DEFAULT_SEMANTIC_BREAKPOINT_AMOUNT)
    ).strip()
    try:
        val = float(raw)
    except ValueError:
        return DEFAULT_SEMANTIC_BREAKPOINT_AMOUNT
    return val


def get_semantic_embedding_device() -> str:
    """HuggingFace 推理设备，例如 cpu、cuda。"""
    return os.getenv("SEMANTIC_EMBEDDING_DEVICE", DEFAULT_SEMANTIC_EMBEDDING_DEVICE).strip()


def get_semantic_sentence_split_regex() -> str:
    """句子切分正则，影响 SemanticChunker 的初始句子列表。"""
    return os.getenv(
        "SEMANTIC_SENTENCE_SPLIT_REGEX", DEFAULT_SEMANTIC_SENTENCE_SPLIT_REGEX
    ).strip() or DEFAULT_SEMANTIC_SENTENCE_SPLIT_REGEX
