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
| v010 | [v010_compose_mixed.src](v010_compose_mixed.src) | 混合场景：解码 → 解析参数 → RSA | 无（纯 Python） | 正样本 8/8，负样本 4/4 |
| v011 | [v011_file_payload.src](v011_file_payload.src) | 文件型载荷：魔数识别 + zip/png/pcap/elf | 无（纯 Python） | 12/12 正样本，4/4 负样本 |
| v012 | [v012_zip_members.src](v012_zip_members.src) | 多成员压缩包：全部成员入队 + 先验排序 | 无（纯 Python） | beam_k=32 时 12/12 |
| v013 | [v013_chain_coverage.src](v013_chain_coverage.src) | 用 726 条真实攻击链测单元库覆盖边界 | 无（纯 Python） | 61.7% 步骤属题目特有逻辑 |
| v014 | [v014_pwn_units.src](v014_pwn_units.src) | pwn 单元：ELF 分析 + gadget + 偏移探测 + 真实 exploit | pyelftools/capstone + WSL | 正样本 2/2，负样本 2/2 |
| v015 | [v015_leak_bypass.src](v015_leak_bypass.src) | canary / PIE 绕过：格式化字符串泄漏 + 同进程利用 | pyelftools/capstone + WSL | 3/3 绕过，负样本如实失败 |

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

### v010_compose_mixed.src

**回答的问题**：真实赛题是**异构**的——"先 base64 解码，解出来是 RSA 的 n/e/c"。
规划器的状态如果只会是字符串，就走不到结构化单元。

**改动**：
1. 状态值从 `str` 扩展为 `str | dict[str, int]`；
2. 新增类型 `int_fields`；
3. 新增单元 `parse_int_fields`（text → int_fields，认中英文标签、十进制/十六进制）；
4. 新增单元 `rsa_small_e`（int_fields → flag，`e>64` 时**主动拒绝**）。

**结果**

| 题 | 路径 | 结果 |
| --- | --- | --- |
| M1 base64("n/e/c") | `b64_decode → parse_int_fields → rsa_small_e` | ✓ |
| M2 hex("n/e/c") | `hex_decode → parse_int_fields → rsa_small_e` | ✓ |
| M3 三层编码 | `b64 → hex → b64 → parse_int_fields → rsa_small_e` | ✓ |
| M4 中文标签 + hex 数值 | `parse_int_fields → rsa_small_e` | ✓ |
| N1 e=65537（负样本） | 解析成功但 RSA 主动拒绝 | ✓ 正确失败 |
| N2 多字节 XOR（负样本） | 无对应单元 | ✓ 正确失败 |

正样本 **8/8**（4 题 × 2 种题面），负样本 **4/4 正确失败**。
加 9 个诱饵后：beam 仍 8/8，BFS 掉到 6/8——与 v009 的结论一致。

**运行**
```bash
python versions/v010_compose_mixed.src
python versions/v010_compose_mixed.src --search bfs --decoys 9
```

**产出**：`experiments/EXP-0008/{results.json,report.md}`

### v011_file_payload.src

**回答的问题**：真实比赛给的是**附件**（zip/png/pcap/elf），不是题面里的字符串。
规划器不认文件，前面所有能力在真题面前都等于零。

**改动**：
1. 状态值新增 `FileRef`；
2. 按**魔数**识别 zip / png / pdf / elf / pcap / jpeg / gif / bin；
3. 文件类单元：`unzip`（支持嵌套）、`read_text`、`png_text`（tEXt/iTXt/zTXt）、
   `pcap_strings`、`elf_strings`、`bin_strings`；
4. 载荷抽取支持"题面里给了附件路径"。

**结果**

| 题 | 实际路径 | 结果 |
| --- | --- | --- |
| F1 文本附件 | `read_text` | ✓ |
| F2 zip 内文本 | `unzip → read_text` | ✓ |
| F3 zip 内 PNG 的 tEXt 藏 base64 | `unzip → png_text → b64_decode` | ✓ |
| F4 zip 套 zip | `unzip → unzip → read_text` | ✓ |
| F5 pcap 明文 | `pcap_strings` | ✓ |
| F6 ELF 字符串表 | `elf_strings` | ✓ |
| N1 随机二进制 / N2 无关 PNG 文本块 | 无路径 | ✓ 正确失败 |

正样本 **12/12**（6 题 × 2 种题面），负样本 **4/4**。平均 2.9 个节点、约 1.1 ms/题。

**一个重要的自我修正**：首轮 `read_text` 接受所有文件类型，把二进制按 latin-1 解码，
flag 恰好是明文就被"蒙对"——F5/F6 一步解决，**等于没验证 pcap/elf 路由**。
收紧到只接受 `file:bin` 后重跑，路径才走对。见 `laws/LAW-0015.md`。

**运行**
```bash
python versions/v011_file_payload.src
python versions/v011_file_payload.src --search bfs
```

**产出**：`experiments/EXP-0009/{artifacts/,results.json,report.md}`

### v012_zip_members.src

**回答的问题**：真实 zip 常有十几甚至上百个成员，v011 只取第一个等于碰运气。

**改动**：
1. `unzip` 返回 `FileSet`（全部成员），不再只取第一个；
2. 规划器对每个成员**分叉**，成员先验（文件名像 flag/secret、体积小、类型有趣）
   只用于**排序**，不用于丢弃；
3. 防 zip-slip：跳过绝对路径与含 `..` 的成员；
4. **预算不足时显式告警** `members(N)>beam_k(K)`。

**结果**

| beam_k | 正样本 |
| --- | --- |
| 8 | 10/12（Z4 失败） |
| 16 | 10/12 |
| 32 | **12/12** |

- Z4 在 25 个成员中准确定位到名字毫无提示的 `zzz_9f3a.dat`；
- Z2 / N2 支持嵌套 zip；
- N1（10 个纯诱饵）在所有预算下都正确失败。

**关键认知**：成员先验只能排序，不能用来"猜一个"——靠名字猜会在 Z4 上直接漏掉。
而穷举要求**预算 ≥ 成员数**，否则会**静默漏解**（不报错，只是通过率下降）。
→ 见 `laws/LAW-0016.md`

**运行**
```bash
python versions/v012_zip_members.src
python versions/v012_zip_members.src --beam-k 8   # 观察告警
```

**产出**：`experiments/EXP-0010/{artifacts/,results.json,report.md}`

### v013_chain_coverage.src

**回答的问题**：用户指出 Des-CTF-Knowledge 里就是真实题。对——但它提供的是
**真实题目的解题链（attack_chain）**，不含可执行附件。所以它不能验证"能不能真的解出"，
但**非常适合验证"单元库够不够广"**。

**做法**：把 726 条真实攻击链（6795 个步骤）逐步归入技术族，
再对照我们已实现的单元族。纯文本统计，无模型。

**结果**

| 指标 | 值 |
| --- | --- |
| 总步骤 | 6795 |
| **无法归入任何标准技术族（题目特有逻辑）** | **4191（61.7%）** |
| 可归类步骤中落在已覆盖族 | 1035（约 32%） |
| 能由现有单元完整覆盖的整条链 | **81 / 726（11.2%）** |

**缺口排序（按真实步骤数）**：pwn 635、reverse 304、hash 190、block_cipher 174、
sql 162、rce 125、deserialize 94——目前只覆盖 encoding / binary / xor / image / rsa。

**抽查那 61.7%**：`orw shellcode (openat+mmap+writev)`、`sub_8001A8C 计算 a1=(a1>>6)&(a1>>2)`、
`torch.load('model.pth') 做 argmin 距离`、`输入 0o57355 按 ENTR 循环读 ROM`——
这些是**题目自定义的逻辑**，不是"没识别出的标准技术"。

→ 见 `laws/LAW-0017.md`

**运行**
```bash
python versions/v013_chain_coverage.src
```

**产出**：`experiments/EXP-0011/{results.json,report.md}`

### v014_pwn_units.src

**回答的问题**：LAW-0017 显示 pwn 是最大缺口（635 步）。pwn 的标准套路能不能单元化，
并且**真的打穿一个进程**？

**做法**：在 WSL(Ubuntu 22.04 + gcc 11.4) 里编译三个真实靶场，用四个单元串成确定性流程：

| 单元 | 作用 |
| --- | --- |
| `elf_info` | canary / PIE / NX / RELRO、符号地址、`.text` 布局 |
| `find_gadgets` | capstone 反汇编，按**指令边界**找 `pop reg; ret` |
| `probe_offset` | 用标记地址真实执行，从崩溃 RIP 反推 padding 长度 |
| `build_ret2win` | 拼 payload，自动处理 16 字节栈对齐 |

**结果**

| 题 | 类型 | 结果 |
| --- | --- | --- |
| T1 | ret2win | ✓ 真实执行拿到 flag |
| T2 | 带参数的 ret2win（需 `pop rdi`） | ✓ |
| N1 | 开 canary | ✓ 正确拒绝（`canary_present`） |
| N2 | 开 PIE | ✓ 正确拒绝（`pie_enabled`） |

平均 **475 ms/题**（偏移探测在 WSL 内批量执行，比逐次跨进程快约 15 倍）。

**五个必须踩过才知道的坑**（见 `laws/LAW-0018.md`）：
裸字节扫 gadget 会落在指令中间；内核把非规范地址归一化为 `nil`；
后门调用 libc 需要 16 字节栈对齐；canary 符号是未定义导入（`st_value=0`）；
跨进程探测慢一个数量级。

**运行**
```bash
python versions/v014_pwn_units.src
python versions/v014_pwn_units.src --only T2
```

**产出**：`experiments/EXP-0012/{pwn/,results.json,report.md}`

### v015_leak_bypass.src

**回答的问题**：真实比赛的 pwn 几乎都开 canary / PIE，v014 只能打两者都关的。怎么绕？

**核心约束**：**泄漏与利用必须在同一个进程里**——ASLR 每次运行都变，
"跑一次拿泄漏、再跑一次用泄漏"必然失败。所以必须交互式喂数据。

**新增四个单元**

| 单元 | 作用 |
| --- | --- |
| `leak_stack` | 发 `%p.`×60，解析泄漏值 |
| `find_canary` | 低字节 0、量级足够大的候选（栈地址排在后面） |
| `find_pie_base` | 用「低 12 位在 PIE 下不变」匹配 `main`/`stage1` 反推基址 |
| `frame_layout` | **从反汇编推导**缓冲区 / canary / 返回地址的偏移 |

**结果（连续 3 次稳定）**

| 题 | 保护 | 结果 |
| --- | --- | --- |
| C1 | canary | ✓（489 ms） |
| P1 | PIE | ✓（243 ms） |
| B1 | canary + PIE | ✓（260 ms） |
| N1 | 无泄漏原语 | ✓ 如实失败（`leak_stage_not_found`） |

**五个关键点**（见 `laws/LAW-0019.md`）：
泄漏与利用必须同进程；布局必须读反汇编（我按 64 算，实际是 72/88）；
capstone 把 `fs:0x28` 渲染成 `fs:[0x28]`；候选必须靠执行逐个验证；
没有泄漏原语时必须如实失败。

**运行**
```bash
python versions/v015_leak_bypass.src
python versions/v015_leak_bypass.src --only B1
```

**产出**：`experiments/EXP-0013/{pwn/,results.json,report.md}`
