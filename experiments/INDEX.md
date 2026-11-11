# experiments/INDEX — 实验配置与结果

> 每次实验一个子目录：`experiments/EXP-NNNN-<slug>/`，内含 `config.json`、`results.json`、`report.md`、原始输出。

| id | 日期 | 主题 | 结论 | 报告 |
| --- | --- | --- | --- | --- |
| EXP-0000 | 2026-09-21 | 三家本地小模型 CTF 推理对比（预实验） | 3B 级全部不可用；phi4 结构好但 7 tok/s 速度不可接受 | `PROJECT_LOG.md` §3 结论 2 |
| EXP-0001 | 2026-09-21 | 离线推理解题可行性判决性实验 | 进行中（见下） | `EXP-0001/report.md` |
| EXP-0002 | 2026-09-21 | 带确定性验证的离线回答 | 验证闸门可正确拒绝无据方案 | `EXP-0002/report.md` |
| EXP-0003 | 2026-09-21 | 基于 meta WP 真值的检索质量评测（120 题） | tag_recall@8 = 0.593；Top-K 的 83% 来自 meta WP；"找相似题"受限于语料（仅 28% 的题有近似题） | `EXP-0003/report.md` |

### EXP-0003 的作用

AI 在比赛现场不能用，所以平台的核心指标不是「AI 能不能解题」，而是
**「拿到一道题，能不能快速准确地找到该看的 WP / 知识章节 / payload」**。
这件事有真值：meta WP 的 `tags` 就是标准答案，1156 篇可以大规模自动评分。

查询只用 `title + contest + difficulty + vuln_type` 拼成，**刻意排除**
`tags` / `one_liner` / `attack_chain` / `key_payload`——那些是答案本身。

**最终数据（完整语料 16361 块）**

| 指标 | 值 |
| --- | --- |
| `tag_recall@8` | 0.593 |
| `self_leak_rate` | 0.825 |
| `similar_wp_rate` | 0.067 |
| `similar_wp_rate_conditional`（仅在有近似题的子集上） | 0.20 |
| `top_k_composition` | meta WP 83% / 大部头 PDF 8.6% / 其它 8.4% |
| `mean_meta_wp_per_query` | 5.0 / 8，且 0 题命中 0 篇 WP |
| 单次检索耗时 | 约 3.8 秒（GPU） |

**必须和原始指标一起看的两个前提**（否则会得出错误的优化方向，见 `laws/LAW-0007`）：

1. `similar_wp_rate` 的上界不是 1.0：整库只有 **318/1156（28%）** 的题存在共享 ≥2 标签的近似题。
2. 把 OCR 出的 3 本书加入语料后指标**下降**了（tag_recall 0.638→0.593），
   因为书籍块挤占了固定的 Top-K 名额。这是**多样性权衡**，不是回归。

**漏检标签**几乎全是只出现一次的专有词（`PTRACE_POKETEXT`、`XXTEA`、`RSA-public-key-PEM`），
说明瓶颈在**语料覆盖**，不在排序算法，也不在模型。

---

## EXP-0001 — 离线推理解题可行性判决性实验

**文件**：`versions/v001_feasibility_probe.src`

**要回答的问题**：离线（纯本地）推理能不能产出「真的能解」的 CTF 解题路径？

**为什么不采用 LLM 自评**：浙密杯的教训是「AI 每次都说能解，现场解不了」。
让 LLM 评价自己等于重新引入同一个失效模式。所以判据全部客观。

| 区 | 内容 | 判据 |
| --- | --- | --- |
| A | 7 道自带已知 flag 的可执行题（Base64 链 / 单字节 XOR / 凯撒 / RSA 小明文 / 二进制转换 / 逆向逻辑 / CRC32 爆破） | 模型写的代码被**真实执行**，stdout 出现正确 flag 才算解出 |
| B | 4 道知识接地问答 | 引用有效性、关键词覆盖率、幻觉引用率 |
| C | 3 道硬负样本（库里根本没有答案） | 是否明确弃答；未弃答即记为 FALSE_CONFIDENCE |

**关键指标**
- `A_solve_rate`：真的被解出的比例
- `A_false_positive_rate`：给出了 flag 但是错的 —— **「假能解」的直接量化**
- `B_ungrounded_citation_rate`：引用了不存在的证据编号的比例
- `C_false_confidence_rate`：对无答案的问题仍然自信作答的比例

**自检（--oracle）**：用参考解验证 7 道题都可解且 flag 一致。
首次运行即抓到 A3 的正则 bug（`[a-z]{5}` 应为 `{4}`）。**oracle 不过，模型失败就不可信。**

**预实验数据**（qwen2.5:3b-instruct-q4，`A` 区 7 题）
```
A_solve_rate                0.0
A_solved_by_exec            0
A_false_positive_rate       0.714     ← 7 题里 5 题给出了一个错误的 flag
A_mean_latency_s            6.7
```
其中最典型的一题：模型直接把答案硬编码成 `print("flag{HELLO_WORLD}")`——
既没解码，也不是正确答案，而题目明确要求"不要写死答案"。

---

## EXP-0002 — 带确定性验证的离线回答

**文件**：`versions/v003_verified_answer.src`

**要验证的假设**：在生成之后插入一道**确定性**验证闸门，能否拦住「结构完整但解不了」的方案？

**做法**：模型按固定格式产出分步方案，每步带证据编号与关键载荷；然后纯字符串/集合运算检查
（a）引用的编号是否存在、（b）关键载荷是否出现在**它自己引用的**证据里、（c）是否整步无引用。

**逻辑自测**（合成答案，无模型参与）

| 输入 | 期望 | 实际 |
| --- | --- | --- |
| 全部有据 | SUPPORTED | SUPPORTED ✓ |
| 含假引用 `[E7]` + 编造载荷 `LD_PRELOAD=/tmp/evil.so` | UNSUPPORTED | UNSUPPORTED ✓，且精确列出无据载荷 |
| 整步无引用 | UNSUPPORTED | UNSUPPORTED ✓ |

**真实模型试跑**（qwen2.5:3b，问题「PHP 命令执行漏洞怎么利用？给出具体 payload」）
```
判定                     UNSUPPORTED
步骤                     7
假引用                   0/7
无据载荷                 7/15 (46.7%)
```
模型把 PHP 命令执行问题答成了 SSRF payload（`expect://`、`?url=`），
闸门识别出这些载荷在它自己引用的证据里根本不存在，**拒绝输出完整方案**，
改为输出「证据不足」并逐条列出被拒绝的步骤。

---

## 附件

- `EXP-0001/ocr_reindex.log` —— 4 本扫描书 OCR 重建索引的实时日志
- `EXP-0001/bulk_ingest_summary.json` —— 批量导入汇总（阶段 1–4）
- `EXP-0001/bulk_ingest_phase1.json` —— 仅解析阶段的汇总
