from __future__ import annotations

"""近似去重（MinHash / datasketch），与设计方案中 DataSketch 路线一致。

**改动说明**：
- 此前 `dedup_minhash` 为占位 `NotImplementedError`；现提供基于 `datasketch.MinHash` 的实现。
- 与 `exact`（整段 SHA1 完全相同才删）不同：本策略对 **措辞略不同但字符 n-gram 高度重叠** 的块
  可能判为重复（由 `DEDUP_MINHASH_THRESHOLD` 控制敏感度）。
- 复杂度：当前实现为 O(n²) 与已保留签名两两比 Jaccard；块数量极大时需再考虑 LSH 桶或采样。

启用方式：`DocumentTextPipeline(dedup_strategy="minhash")` 或 `dedup_chunks(..., strategy="minhash")`。
"""

from datasketch import MinHash

from ..config import (
    get_minhash_perm,
    get_minhash_shingle_size,
    get_minhash_threshold,
)


def _shingles(text: str, k: int) -> list[bytes]:
    """字符级 shingle，空白先压成单空格以便近似匹配。"""
    compact = " ".join(text.split())
    if not compact:
        return []
    if len(compact) <= k:
        return [compact.encode("utf-8")]
    return [compact[i : i + k].encode("utf-8") for i in range(len(compact) - k + 1)]


def _minhash_for_text(text: str, *, num_perm: int, shingle_size: int) -> MinHash:
    # 每个 shingle 字节串 update 一次，得到可与其他块做 jaccard 估计的签名。
    mh = MinHash(num_perm=num_perm)
    for piece in _shingles(text, shingle_size):
        mh.update(piece)
    return mh


def dedup_minhash(
    chunks: list[str],
    *,
    threshold: float | None = None,
    num_perm: int | None = None,
    shingle_size: int | None = None,
) -> list[str]:
    """按 MinHash 估计 Jaccard 相似度去重，保留首次出现的块。"""
    th = get_minhash_threshold() if threshold is None else threshold
    perm = get_minhash_perm() if num_perm is None else num_perm
    k = get_minhash_shingle_size() if shingle_size is None else shingle_size

    # 顺序敏感：保留「先出现」的块，与 dedup_exact 的「保留首次」语义一致。
    normalized: list[str] = []
    signatures: list[MinHash] = []
    for chunk in chunks:
        stripped = chunk.strip()
        if not stripped:
            continue
        mh = _minhash_for_text(stripped, num_perm=perm, shingle_size=k)
        duplicate = False
        for kept in signatures:
            # datasketch 文档：jaccard 为基于 MinHash 的集合相似度估计。
            if mh.jaccard(kept) >= th:
                duplicate = True
                break
        if duplicate:
            continue
        signatures.append(mh)
        normalized.append(stripped)

    return normalized
