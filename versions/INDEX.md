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
| v005 | [v005_sag2_multihop.src](v005_sag2_multihop.src) | SAG2 式离线多跳检索（BM25 + 本地判官 + 词扩展） | SecKnow 索引 + reranker | 已实现，待评测 |
| v006 | [v006_code_units.src](v006_code_units.src) | 从知识库抽取代码单元并按题目召回（含拼装尝试） | 无（纯 Python） | 已跑通；拼装 0/5，暴露契约缺失 |
| v007 | [v007_solver_kit.src](v007_solver_kit.src) | 带显式契约的求解单元库 + 确定性拼装器 | 无（纯 Python） | **端到端 5/5** |
| v008 | [v008_shift_test.src](v008_shift_test.src) | 分布漂移测试（改名 / 换格式 / 抽象描述） | 无（纯 Python） | keyword 6/15，shape 15/15 |
| v009 | [v009_compose.src](v009_compose.src) | 多步组合：类型化搜索把单元串成流水线 | 无（纯 Python） | BFS 8/8→6/8；beam 8/8 |

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

### v004_retrieval_bench.src

**回答的问题**：比赛现场不能用 AI，那这个知识库到底找不找得到该看的东西？
用 1156 篇 meta WP 的 `tags` 当真值，可大规模自动评分。

**查询构造（防泄漏）**：只用 `title + contest + difficulty + vuln_type`，
**刻意排除** `tags` / `one_liner` / `attack_chain` / `key_payload`——那些就是答案。

**参数**：`--sample N`、`--top-k K`、`--no-dense`（A/B 关掉向量通道）

**产出**：`experiments/EXP-0003/{results.json,report.md}`

### v005_sag2_multihop.src

**回答的问题**：SAG2（arXiv:2610.22588，embedding-free 多跳检索）能不能离线用？
我们原来"转向量 + 向量重组推理"的路子还有没有戏？

**SAG2 的机制**：BM25(FTS5) 取候选 → JEV 判断哪些段落有用 → 被选中的段落提供新词 →
下一轮检索，默认 2 轮。建库零模型调用。

**关键判断**：SAG2 的**架构完全可离线**（就是 SQLite FTS5 + BM25），
但 **JEV 按发表形态是在线 API**（`api.typesafe.ai`，model `jev-latest`）。
所以本文件给判官做了两档，**都不使用生成式 LLM**：

| `--judge` | 判官 | 是否用神经模型 | 实测 excl_self | 单题耗时 |
| --- | --- | --- | --- | --- |
| `reranker`（默认） | `bge-reranker-v2-m3`（cross-encoder，判别式，不生成文本） | 是（非生成式） | 0.175 | 5.5 s |
| `none` | 无，直接按 BM25 分数排序 | **否，零模型** | 0.109 | **0.69 s** |

**链路**
```
BM25(FTS5) 取候选 → 判官（reranker 或 BM25）→ jieba 从选中段落抽新词 → 再检索 → RRF 合并
```

**参数**：`--rounds N`（0 = 单跳基线）、`--judge reranker|none`、
`--candidates`、`--judge-keep`、`--expand-n`

**产出**：`experiments/EXP-0004/{results.json,report.md}`

### v006_code_units.src

**回答的问题**：能不能把知识库代码按函数拆开，按题目召回函数，再拼成脚本？

**做法**：从 121 个 `.py` 与 markdown 内联代码块抽取 1165 个单元
（508 function / 572 snippet / 80 script / 5 class），建 FTS5 索引，
按题目召回；拼装时按"变量名像不像 n/e/c"猜槽位。

**结果：0/5。** 但诊断显示**正确单元都召回到了**，失败全在适配环节：
无关常量被当成必填槽位、函数无 driver、语义错配（把加密函数当解密用）。
→ 见 `laws/LAW-0010.md`。这个"失败"是有价值的，它定位了真正的瓶颈。

### v007_solver_kit.src

**回答的问题**：既然瓶颈在适配，显式契约能不能解决？

**做法**：每个单元声明 `params` / `returns` / `code` / `technique` / `demo`。
拼装器只做三件事：按技法召回 → 按参数名绑定 → 执行校验。
**不生成代码、不猜映射。**

**结果：**
- 参数由程序提供：**5/5**
- **端到端只给题目文本**（参数靠每个单元配的正则抽取器）：**5/5**，约 1 秒，零模型

**运行**
```bash
python versions/v007_solver_kit.src --list
python versions/v007_solver_kit.src --demo
python versions/v007_solver_kit.src --demo-text
```

**产出**：`experiments/EXP-0005/{solver_kit_results.json,solver_kit_text_results.json}`

**注意**：这 5 个单元是手工编写的最小验证集，**不是泛化能力的证明**。
真实成本在于把知识库里约 250 个可复用脚本改造成带契约的单元。

### v008_shift_test.src

**回答的问题**：当题目"不像我们出的题"时还能不能解？
（用户原话：主办方会换变量名、换函数名，实现逻辑很抽象。）

**三档陌生度**，同一批 5 个技术：
- **L1** 自编题：技术名明确、参数名标准
- **L2** 改名换格式：参数名改中文别名、数值改十六进制、数据放进 JSON
- **L3** 抽象描述：**完全不提技术名**，只描述行为

**三种召回策略对照**

| 策略 | L1 | L2 | L3 | 合计 |
| --- | --- | --- | --- | --- |
| keyword（只认技法词） | 5/5 | **1/5** | **0/5** | 6/15 |
| shape（只认数据形态） | 5/5 | 5/5 | 5/5 | 15/15 |
| hybrid | 5/5 | 5/5 | 5/5 | 15/15 |

**核心结论**：命名依赖可以消除（用数据形态路由），但**逻辑新颖性无法消除**——
题库里没有的变换，检索再准也找不出来。详见 `laws/LAW-0011.md`。

**诚实的限定**：L2/L3 仍是我写的题；而且我在跑测试的过程中修了 3 个自己抽取器的 bug，
所以 15/15 有一部分来自"看着失败样例调规则"，不能当作泛化证明。
真正有效的检验必须是**未见过的真实赛题**。

**运行**
```bash
python versions/v008_shift_test.src                # 三种策略全跑
python versions/v008_shift_test.src --router shape
```

**产出**：`experiments/EXP-0006/{results.json,report.md}`

### v009_compose.src

**回答的问题**：真实赛题大多是**多步**的（base64 → hex → XOR），
能不能在没有 LLM 的情况下自动规划出一条正确的单元序列并跑通？

**做法**：类型化的前向搜索。
- 每个单元声明 `accepts`（吃什么数据）与 `produces`（吐什么）；
- 规划器从载荷出发，每步选一个能接受当前类型的单元，**真实执行**，重新识别形态后继续；
- 结果为 flag 即成功；执行异常 / 状态重复 / 无变化即剪枝。
- `--search beam` 时按「解码 +2 / 编码 −5 / 可打印比例 / 深度」打分，每层只留前 k 个。

**结果**

| 搜索 | 5 单元 | 8 单元 | 12 单元 | 20 单元 |
| --- | --- | --- | --- | --- |
| BFS | 8/8 | 8/8 | **6/8** | **6/8** |
| beam | 8/8 | 8/8 | **8/8** | **8/8** |

- 深度 1–4 全部覆盖，包括 4 层嵌套（b64 → hex → b64 → XOR）。
- 两种题面（明说技术名 / 只描述行为）结果完全一致。
- **负样本（多字节 XOR，库里没有对应单元）在两种搜索下都正确失败**。

**运行**
```bash
python versions/v009_compose.src
python versions/v009_compose.src --sweep      # BFS vs beam 的可扩展性对照
```

**产出**：`experiments/EXP-0007/{results.json,report.md,scaling.json}`

**注意**：单元库只有 5–20 个，题目是合成链。**这验证的是"组合机制成立"，
不是"能解真实赛题"**——后者需要 50–200 个真实单元 + 未见过的留出题。
