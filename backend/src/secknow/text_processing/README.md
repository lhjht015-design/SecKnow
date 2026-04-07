## Phase 2 边界（4.1）

已完成：

- 对外门面接口：`load_document / chunk_text / dedup / encode_chunks / run_pipeline`
- 主流水线：文件/内存文本 -> 清洗 -> 分块 -> 去重 -> 编码 -> `list[ChunkRecord]`
- 分块策略：`fixed_window / hybrid / paragraph / line / semantic`
- 去重策略：`exact / minhash`
- PDF 增强处理：PyMuPDF 文本层、可选 OCR、表格 Markdown、一阶页眉页脚弱化
- 4.1 -> 4.3 在线/离线联调脚本与分项检查脚本

部分满足（依赖环境或启发式限制）：

- OCR 质量依赖系统 `tesseract` 与 PDF 质量
- PDF 复杂版式/多栏/合并单元格场景为启发式处理
- Markdown/代码清洗为轻量规则，不是 AST 级重写

暂不做：

- 4.5 HTTP API 接入
- 前端联调
- LlamaIndex 双栈并行实现

# Phase 2 设计说明（4.1）

## 1. 目标

第二阶段目标是把 4.1 文本处理模块从“可跑通”升级为“可联调、可验收、可扩展”的标准化组件。  
核心仍是：稳定产出符合 4.3 契约的 `list[ChunkRecord]`。

## 2. 对 4.3 契约

4.1 仅负责产出，4.3 负责存储与检索。  
契约以 `secknow.vector_store.models` 为准，重点字段：

- `text`
- `vector`
- `metadata.doc_id / zone_id / chunk_index / chunk_count / content_hash`
- `metadata.record_type`（`knowledge` / `baseline`）

分区固定为：`cyber / ai / crypto`。

## 3. 对外接口（4.1）

模块入口：`secknow.text_processing`

- `load_document(file, normalize=False)`
- `chunk_text(text, strategy, max_tokens, overlap)`
- `dedup(chunks, strategy)`
- `encode_chunks(chunks, mode, model_name, dim)`
- `run_pipeline(file_path, zone_id, record_type, ...)`
- `DocumentTextPipeline.process_file(...)`
- `DocumentTextPipeline.process_text(...)`（纯内存，不写盘）

## 4. 模块分层

- 配置层：`config.py`
- 门面层：`facade.py`
- 加载层：`loaders/*`
- 清洗层：`cleaners/*`
- 分块层：`chunkers/*`
- 去重层：`dedupers/*`
- 编码层：`encoders/*`
- 元数据层：`metadata/*`
- 流水线层：`pipeline/run_pipeline.py`
- 单测：`tests/*`

## 5. 关键实现点

- 清洗改为按 `file_type` 分支串联（而不是单一 `basic_clean`）
- `process_file` 与 `process_text` 复用同一核心处理函数
- `semantic` 分块走 LangChain `SemanticChunker`（`EMBEDDING_MODE=fake` 下禁用）
- `minhash` 去重基于 `datasketch`
- 向量默认对齐 `all-MiniLM-L6-v2` 的 384 维

## 6. 依赖说明（与新增能力相关）

运行 Phase 2 建议至少包含：

- `sentence-transformers`
- `datasketch`
- `pymupdf`
- `pytesseract`
- `Pillow`
- `langchain-core / langchain-community / langchain-experimental`

其中 OCR 还需要系统安装 `tesseract` 可执行文件。

## 7. 联调与测试（2026-04-07）

基于你本地执行：

`PYTHONPATH=src python -m secknow.text_processing_other.scripts.run_all --embedding-mode fake --qdrant-host 127.0.0.1 --qdrant-port 6333`

结果摘要：

- `facade`: PASS
- `process_text_memory`: PASS
- `pdf_features`（helper）: PASS（未指定 `--pdf-file`，文件实测 SKIP）
- `strategies`: PASS
- `pipeline_contract`: PASS（`files=6`, `records=19`）
- `integration_offline`: PASS（`baseline_expected=4`, `baseline_count=4`）
- `integration_online`: PASS（`baseline_expected=4`, `baseline_count=6`，说明在线库存在历史残留 baseline）
- 总结：`ALL PASS run_all`

## 8. 与 4.3 的建议联调方式

推荐命令：

```bash
cd backend
PYTHONPATH=src python -m secknow.text_processing_other.scripts.run_all \
  --embedding-mode fake \
  --qdrant-host 127.0.0.1 \
  --qdrant-port 6333
```

若要覆盖语义分块真实路径：

```bash
cd backend
PYTHONPATH=src python -m secknow.text_processing_other.scripts.run_all \
  --embedding-mode sbert \
  --include-semantic \
  --qdrant-host 127.0.0.1 \
  --qdrant-port 6333
```

