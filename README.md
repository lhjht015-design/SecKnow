# SecKnow

SecKnow 是一个离线优先的安全知识检索与确定性图推理 Web 平台。它不是“只做检索”的原型：默认不调用生成式大模型，也能通过实体、关键词、章节、代码符号、语义相似边做邻居扩展和受限路径推理；可选的本地 LLM 只负责在证据之后生成带引用的答案。

当前工作区已经包含可运行后端、React 前端、9 本 `book/` 资料的索引、固定版本的 BGE-M3 与 bge-reranker 权重。运行时不会联网下载模型，也没有云端回退。

## 12 层实现

| 层 | 实现 |
| --- | --- |
| 1. 统一输入 | `DocumentDiscovery` 对 PDF、DOCX、TXT、PY、MD 做扩展名、magic、路径、SHA-256 和版本校验；LlamaIndex 作为统一文档/切片互操作层。 |
| 2. 结构解析 | Unstructured 解析 DOCX/Markdown；PyMuPDF + 可选 OCR 逐页审计 PDF；Python `ast` 和 `tree-sitter-language-pack` 只解析代码、不执行。 |
| 3. 语义切片 | LlamaIndex `SentenceSplitter` + BGE 相邻句余弦相似度；模型不可用时确定性降级为规则分句和重叠窗口。 |
| 4. 向量表示 | 固定版本 BAAI BGE-M3 同时输出 1024 维 dense 与 lexical sparse 权重，按模型指纹缓存。 |
| 5. 存储 | SQLite 保存版本、块、引用、FTS5 和作业；Milvus 保存 HNSW dense 与 sparse inverted index。 |
| 6. 图构建 | 安全领域规则 NER（CVE/CWE/CAPEC/ATT&CK/寄存器/函数等）+ 词典 + TF-IDF/TextRank + 章节/相邻/代码关系 + embedding 相似边；NetworkX 为本地权威快照，Neo4j 为可选镜像。 |
| 7. 基础查询 | SQLite FTS5、TF-IDF、BGE dense、BGE sparse 四类信号。 |
| 8. 混合查询 | Milvus `WeightedRanker` 先融合 dense/sparse；平台再以 `0.78 × 归一化加权 RRF + 0.22 × 加权通道覆盖度` 融合词法与向量召回。 |
| 9. 图扩展 | 从混合 Top-K 出发，按边权、实体 IDF、置信度和每跳 `0.68` 衰减做受限邻居/路径/跨文档实体扩展；默认 2 跳、160 节点。 |
| 10. 重排 | bge-reranker-v2-m3；最终分数为 `0.72 × sigmoid(reranker) + 0.28 × 原融合分`，失败时保持确定性融合排序。 |
| 11. 重组 | 内容哈希与字符 n-gram 去重、MMR 多样化、层次聚类，再按来源、图距离、页/行、原章节顺序重排。 |
| 12. 输出 | FastAPI 返回结构化证据、引用、通道分、图路径和告警；React 页面按主题簇渲染，支持原 PDF 页预览与查询子图。 |

没有人工相关性标注时，在线系统不会把“通道覆盖度”冒充真实召回率。准备 JSONL 标注集后，可用 `secknow evaluate` 计算真实 Precision@K、Recall@K、MRR 和 nDCG@K。

## 快速启动（Windows）

```powershell
Set-Location D:\SecKnow
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1 -Profile competition
powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1 -Mode competition
```

浏览器访问 <http://127.0.0.1:8765>。停止服务：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop.ps1
```

`bootstrap.ps1` 使用 uv 把 Python 3.11.15 安装到项目内的 `.uv-python/`，不会再依赖易失效的用户目录解释器。它会按 profile 安装锁定依赖并构建前端。

### Milvus 与 Neo4j

`competition`、`enhanced`、`llm` 模式会通过 `infra/compose.yml` 启动 Milvus 2.6、etcd、MinIO 和 Neo4j。没有 Docker 时，`competition` 会继续使用 SQLite/TF-IDF/NetworkX；`enhanced` 与 `llm` 会明确报错，不会伪装成完整模式。

检查真实能力：

```powershell
.\.venv\Scripts\python.exe -m secknow.cli doctor
```

Web 的“运行设置”页也显示解析器、BGE、reranker、OCR、Milvus/Neo4j 驱动和本地图是否真正就绪。

## 模型与完全离线运行

模型下载只允许显式执行，脚本固定到不可变 commit，并生成 `REVISION` 与 `SHA256SUMS`：

```powershell
$env:HF_HUB_DISABLE_XET = "1"
.\.venv\Scripts\python.exe .\scripts\download_models.py all --endpoint https://hf-mirror.com
```

当前固定版本：

- BGE-M3：`5617a9f61b028005a4858fdac845db406aefb181`
- bge-reranker-v2-m3：`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`

运行时只读取 `models/`，不访问 Hugging Face。`transformers` 被限制在 4.x，因为 FlagEmbedding 1.x 使用了 Transformers 5 已移除的 tokenizer API。

## 文档导入与索引

“知识源”页面支持一次上传最多 20 个 PDF、DOCX、TXT、PY 或 MD。后端执行：

- 文件名清理、目录穿越阻断、大小上限和 PDF/DOCX magic 校验；
- 流式 SHA-256、内容寻址保存和重复上传复用；
- 一个批量持久作业交给 worker，浏览器断开不影响索引；
- 原始文档、版本哈希、页码/行号、章节和边界框引用保持可追溯。

也可使用命令行：

```powershell
.\.venv\Scripts\python.exe -m secknow.cli ingest .
.\.venv\Scripts\python.exe -m secknow.cli search "ROP 链如何构造" --mode competition
```

## OCR 与当前 `book/` 状态

9 本 PDF 已进入文档审计；其中 5 本已有活动块，4 本因扫描页或字体映射损坏需要中文 Tesseract OCR。平台会隔离坏文本，绝不会把乱码静默写入索引。把可信的 Tesseract 运行时放到 `runtime/tesseract/`，并在 `.env` 中配置：

```dotenv
SECKNOW_TESSERACT_EXECUTABLE=D:/SecKnow/runtime/tesseract/tesseract.exe
SECKNOW_TESSERACT_TESSDATA_DIR=D:/SecKnow/runtime/tesseract/tessdata
SECKNOW_OCR_LANGUAGES=chi_sim+eng
```

随后在 Web 点击“开始增量索引”。`scripts/prepare-offline.ps1` 会要求 Tesseract 版本和完整 SHA-256 清单，避免无校验地打包第三方可执行文件。

## 评测与质量检查

标注文件每行格式：

```json
{"query":"ROP 链是什么？","relevant_chunk_ids":["chunk:..."]}
```

执行：

```powershell
.\.venv\Scripts\python.exe -m secknow.cli evaluate eval\gold.jsonl --top-k 10 --mode competition
powershell -ExecutionPolicy Bypass -File .\scripts\test.ps1
npm --prefix frontend run lint
npm --prefix frontend run build
```

## 运行模式

- `strict-lexical`：FTS5 + TF-IDF + NetworkX 图推理，不加载 Torch。
- `competition`：允许 BGE/Milvus，禁用生成式模型；组件不可用时带告警降级。
- `enhanced`：要求完整向量/图基础设施并启用 reranker，不生成答案。
- `llm`：在增强检索之后调用显式配置的本地/私有 OpenAI 兼容端点；默认关闭，拒绝公网端点。

所有模式均保留证据引用。生成式模型不是推理链路的必需组件。
