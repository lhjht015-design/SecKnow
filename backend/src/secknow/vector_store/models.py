from __future__ import annotations

"""
4.3 向量数据库模块的数据契约定义（给 4.1 / 4.5 / 4.4 统一使用）。

可以把这个文件理解成“跨模块协议”：
1. 4.1 文本处理模块负责生产 ChunkRecord；
2. 4.3 负责存储、检索、导出；
3. 4.5 API 模块把请求/响应映射到这些模型；
4. 4.2 安全审查通过 BaselineBundle 获取分区基线向量。

如果你是 4.1 同事，最核心需要关注：
- ChunkMetadata：每个字段的含义和必填约束
- ChunkRecord：你最终交给 4.3 的入库记录结构
- generate_chunk_id()：chunk 主键的稳定生成规则
"""

from hashlib import sha1
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# 三个固定知识分区。后续如果要扩展分区，请先同步 4.3 和 4.2 的约束。
ZoneId = Literal["cyber", "ai", "crypto"]
# record_type 用于区分“普通知识块”和“安全基线块”。
RecordType = Literal["knowledge", "baseline"]


class ChunkMetadata(BaseModel):
    """
    每个文本块（chunk）的元数据。

    这是 4.1 -> 4.3 最关键的契约，字段应尽量稳定，避免频繁变更。
    """

    # 不允许出现未声明字段（防止上游字段漂移）；允许赋值时继续校验类型。
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # 文档级 ID：用于“按文档删除/重建/统计来源”。
    doc_id: str
    # 分区 ID：必须显式提供，不能依赖调用方“默认推断”。
    zone_id: ZoneId
    # 文件名（展示和过滤常用）。
    filename: str
    # 源文件路径（用于追溯来源和重建）。
    source_path: str
    # 扩展名，例如 ".md"、".py"。
    extension: str
    # 当前块在文档中的下标，从 0 开始。
    chunk_index: int = Field(ge=0)
    # 当前文档总块数，必须 > 0。
    chunk_count: int = Field(gt=0)
    # 当前文本块字符数（用于长度统计与展示）。
    char_len: int = Field(ge=0)
    # 内容哈希（用于幂等、去重、增量更新判断）。
    content_hash: str
    # 文件修改时间（Unix 时间戳，秒级）。
    mtime: int
    # 文件字节大小。
    size_bytes: int = Field(ge=0)
    # 文件类型：和 extension 不同，这里是归一化后的业务类型。
    file_type: Literal["text", "markdown", "code", "pdf", "docx"]
    # 可选语言标记，例如 "zh" / "en" / "python"。
    language: str | None = None
    # 记录类型：默认是知识块；baseline 用于安全审查基线。
    record_type: RecordType = "knowledge"
    # chunk 主键。允许上游不填，由 4.3 统一生成并回填。
    chunk_id: str | None = None


class ChunkRecord(BaseModel):
    """
    4.1 交付给 4.3 的“标准入库单元”。

    一个 ChunkRecord = 文本 + 向量 + 元数据
    """

    model_config = ConfigDict(extra="forbid")

    # 原文 chunk 内容。
    text: str
    # 与 text 对应的语义向量（维度需与分区索引一致）。
    vector: list[float]
    # 元数据（见 ChunkMetadata）。
    metadata: ChunkMetadata


class SearchHit(BaseModel):
    """
    稠密向量检索（ANN）或融合检索后统一返回的数据结构。
    """

    model_config = ConfigDict(extra="forbid")

    # 系统内部 chunk 主键（字符串）。
    chunk_id: str
    # 所属文档 ID。
    doc_id: str
    # 所属分区。
    zone_id: ZoneId
    # 相似度或融合分值（越大越相关，具体计算由检索实现决定）。
    score: float
    # 命中的文本内容。
    text: str
    # 扩展元数据（展示、过滤、调试都可能用到）。
    metadata: dict[str, Any] = Field(default_factory=dict)


class SparseHit(BaseModel):
    """
    稀疏检索（BM25/FTS5）返回的数据结构。

    字段刻意与 SearchHit 保持一致，方便 hybrid 融合层合并结果。
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    zone_id: ZoneId
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaselineBundle(BaseModel):
    """
    4.2 安全审查模块读取的“分区基线向量包”。

    常见用途：
    - 余弦相似度对比
    - 离群/偏离检测
    """

    model_config = ConfigDict(extra="forbid")

    zone_id: ZoneId
    vectors: list[list[float]] = Field(default_factory=list)
    metadatas: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def count(self) -> int:
        """基线向量条数（便于快速统计）。"""
        return len(self.vectors)


class ExportManifest(BaseModel):
    """
    分区导出包描述文件（manifest）对应的数据结构。

    导出时会把模型、维度、距离、分块策略等关键参数固化，
    以保证离线端和在线端配置可追溯、可校验。
    """

    model_config = ConfigDict(extra="forbid")

    zone_id: ZoneId
    engine: str
    embedding_model: str
    embedding_dim: int
    distance: str
    normalized: bool = True
    chunk_strategy: dict[str, Any] = Field(default_factory=dict)
    build_time: str
    record_count: int
    baseline_count: int
    schema_version: int = 1


class UpsertResult(BaseModel):
    """upsert 操作结果。"""

    model_config = ConfigDict(extra="forbid")

    zone_id: ZoneId
    attempted: int
    inserted: int
    chunk_ids: list[str] = Field(default_factory=list)


class DeleteResult(BaseModel):
    """delete 操作结果。"""

    model_config = ConfigDict(extra="forbid")

    zone_id: ZoneId
    requested: int
    deleted: int
    chunk_ids: list[str] = Field(default_factory=list)


def generate_chunk_id(metadata: ChunkMetadata) -> str:
    """
    生成稳定的 chunk_id。

    规则：sha1("{doc_id}:{chunk_index}:{content_hash}")
    目的：
    - 同一 chunk 重跑时 ID 保持稳定（幂等 upsert）
    - 避免直接用数据库自增 ID 造成跨环境不一致
    """
    stable = f"{metadata.doc_id}:{metadata.chunk_index}:{metadata.content_hash}"
    return sha1(stable.encode("utf-8")).hexdigest()
