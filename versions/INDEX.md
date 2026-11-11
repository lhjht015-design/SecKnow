# versions/INDEX — 主体单文件代码版本

> 约定：**每个主体版本 = 一个可独立运行的单文件**。辅助代码只放在 `tools/`。
> 新版本不得修改旧版本文件，只能新增。旧版本永久保留以便复现。
>
> **后缀约定（重要，务必遵守）**：主体脚本一律命名为 `vNNN_主题.src`，用
> `python versions/vNNN_主题.src` 运行。原因见 [LAW-0003](../laws/LAW-0003.md)：
> 本机火绒 HIPS 会异步删除某些 `.py` 文件，换成 `.src` 后缀即可规避；
> Python 不关心扩展名，运行时行为完全一致。

| 版本 | 文件 | 主题 | 依赖 | 状态 |
| --- | --- | --- | --- | --- |
| v001 | [v001_feasibility_probe.src](v001_feasibility_probe.src) | 离线推理解题可行性判决性实验 | Ollama（可选 SecKnow 索引） | 可运行，oracle 7/7 通过 |
| v002 | [v002_bulk_ingest.src](v002_bulk_ingest.src) | 外部知识库批量导入（修复 O(n²)） | BGE-M3 权重（可选） | 可运行，dry-run 通过 |
| v003 | [v003_verified_answer.src](v003_verified_answer.src) | 带确定性验证的离线回答（实现 LAW-0001） | Ollama + SecKnow 索引 | 逻辑自测通过，待真实模型评测 |
| v004 | [v004_retrieval_bench.src](v004_retrieval_bench.src) | 用 meta WP 真值客观评测检索质量 | SecKnow 索引 | 已跑通，120 题数据已出 |

## 各版本说明

### v001_feasibility_probe.src

**回答的问题**：离线推理到底能不能产出「真的能解」的解题路径？

**为什么这样设计**：浙密杯的教训是「AI 每次都说能解，现场解不了」。所以本实验
**不采用 LLM 自评**，只用客观判据：

- Regime A（核心）：7 道自带已知 flag 的可执行题。模型必须写出代码，代码被真实执行，
  stdout 里出现正确 flag 才算解出。同时统计 `A_false_positive_rate`——
  模型给出 flag 但答案是错的，这就是「假能解」的量化。
- Regime B：知识接地问答，统计幻觉引用率。
- Regime C：硬负样本（库里根本没有答案），统计「仍然自信作答」的比例。

**自检**：`--oracle` 用参考解验证 7 道题都可解且 flag 一致。oracle 不过，模型失败就不可信。
（v001 首次自检就抓到 A3 的正则 bug：`[a-z]{5}` 应为 `{4}`——"flag" 只有 4 个字母。）

**运行**
```bash
./.venv/Scripts/python.exe versions/v001_feasibility_probe.src --oracle
./.venv/Scripts/python.exe versions/v001_feasibility_probe.src --regimes A --models qwen3:8b
./.venv/Scripts/python.exe versions/v001_feasibility_probe.src --regimes A,B,C --models qwen3:8b
```

**产出**：`experiments/EXP-0001/{results.json,report.md}`

### v002_bulk_ingest.src

**解决的问题**：`IngestionPipeline.ingest_path()` 每导入一个文档就重新编码全部已有块、
重建整张索引与图，导入 1200+ 文件时是 O(n²)，会从几分钟膨胀到几十小时。
本版本改成「全部解析切片 → 一次性编码 → 一次建索引/图」，降回 O(n)。

**有意偏离**：markdown 直接用纯 Python 的 `parse_markdown`，不优先走 unstructured
（unstructured 处理 1200 个文件太慢）。切片阶段不带 embed，因此语义边界检测被关闭，
退回规则切片——这是速度与质量的权衡，已记录待优化。

**额外产出**：`data/dense/<hash>.npz`——BGE-M3 dense 向量落到本地。
Milvus 未运行时，这是本地向量检索的唯一数据来源。

**运行**
```bash
./.venv/Scripts/python.exe versions/v002_bulk_ingest.src --dry-run --limit 30
./.venv/Scripts/python.exe versions/v002_bulk_ingest.src --skip-existing
```

**产出**：`data/{secknow.db,tfidf/<hash>/,graph/<hash>.json.gz,dense/<hash>.npz}`、
`experiments/EXP-0001/bulk_ingest_summary.json`

**参数**
- `--roots <路径…>`：语料根，默认 `book/Des-CTF-Knowledge`；处理扫描书时传 `--roots book`
- `--skip-existing`：跳过已激活版本的文档（增量导入的关键）
- `--ingest-only`：只做解析/切片/落库。用于先花时间做 OCR，等加速器就绪后再单独建索引
- `--no-encode`：跳过向量化，但仍然建 TF-IDF 与图
- `--dry-run` / `--limit N` / `--batch-size N`

### v003_verified_answer.src

**解决的问题**：把「生成即结论」改造成「验证后结论」——这是 `laws/LAW-0001` 的实现。
平台原 `answer/llm.py` 把 LLM 输出直接返回，没有任何环节能证伪它。于是"能解"变成一句
生成文本，而不是一个结论——这正是浙密杯的失败模式。

**做法**：在生成之后、返回之前插入一道**确定性**验证闸门。模型按固定格式产出分步方案，
每步必须带证据编号与关键载荷；然后逐项检查：

- 引用的 `[E编号]` 是否真的存在于本次检索结果里 → 抓幻觉引用
- 该步声称的关键命令/载荷，是否出现在**它自己引用的**那段证据原文里 → 抓无据断言
- 是否整步都没引用任何证据 → 抓凭空捏造

按未通过比例给出 `SUPPORTED / PARTIAL / UNSUPPORTED`。**UNSUPPORTED 时不输出确定性方案**，
改为输出「证据不足」＋已核实部分，并明确列出被拒绝的步骤。

**核心设计原则**：*验证器不能是被验证事物的同类*。用 LLM 去检查 LLM 会重新引入同一个
失效模式——它同样会说"验证通过"。所以这里的检查全部是确定性的字符串/集合运算，零模型参与。

**运行**
```bash
./.venv/Scripts/python.exe versions/v003_verified_answer.src --model qwen3:8b
./.venv/Scripts/python.exe versions/v003_verified_answer.src --model qwen3:8b --queries "ROP 链怎么构造"
```

**产出**：`experiments/EXP-0002/{results.json,report.md}`
