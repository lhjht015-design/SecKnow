# 4.4 推理模块

`secknow.inference` 是 4.4 的编排层。

它位于以下模块之间：

- 4.5 API：接收面向用户的查询请求
- 4.3 `vector_store`：提供稠密检索、混合检索和基线读取
- 4.1 `text_processing`：提供可复用的查询编码路径

## 职责范围

4.4 负责：

- 语义检索流程编排
- 代码风险扫描流程编排
- 复用查询编码能力
- 检索结果重排
- 结果格式化输出

4.4 不负责：

- 文档入库和分块生产
- 向量存储底层实现
- HTTP 路由处理

## 当前目录结构

```text
inference/
├── __init__.py
├── README.md
├── config.py
├── exceptions.py
├── facade.py
├── models.py
├── retrieval/
├── scanners/
├── schemas/
├── services/
└── tests/
```

## 文件职责

- `facade.py`
  - 给 4.5 使用的 4.4 统一入口
  - 隔离 API 层与 4.4 内部服务装配细节
- `config.py`
  - 默认 `top_k`、重排权重、支持语言等配置
- `models.py`
  - 4.4 内部共享的数据模型
- `schemas/*`
  - 4.5 / 4.6 消费的请求响应契约
- `services/query_encoder.py`
  - 复用 4.1 的编码能力完成查询向量化
- `services/semantic_search.py`
  - 查询 -> 编码 -> 检索 -> 重排 -> 格式化
- `services/code_risk_scan.py`
  - 代码风险扫描主流程
- `services/reranker.py`
  - 本地结果重排策略
- `services/formatter.py`
  - 将原始检索命中统一转换为 UI / API 友好的输出
- `retrieval/semantic.py`
  - 对 4.3 `search()` / `hybrid_search()` 的轻量封装
- `retrieval/baseline.py`
  - 对 4.3 `get_baseline()` 的轻量封装
- `scanners/router.py`
  - 按语言选择代码扫描器
- `scanners/text_fallback.py`
  - 第一阶段不依赖 AST 的降级扫描器
- `scanners/treesitter_scan.py`
  - 预留给 AST 扫描的 `tree-sitter` 入口

## 对外接口

- `semantic_search(query, zone_id, top_k=10, filters=None)`
- `code_risk_scan(code_snippet, lang, zone_id="cyber", top_k=10)`

## 当前实现阶段

当前目录已完成 4.4 的初始骨架。

目前已具备：

- 包结构
- Schema 定义
- 服务装配
- 语义检索骨架
- 代码扫描骨架

下一步建议：

1. 在 4.5 中接入实际 API 层
2. 基于真实检索样本调整重排策略
3. 用 `tree-sitter` 切片替换当前降级扫描器
4. 补离线向量库集成测试
