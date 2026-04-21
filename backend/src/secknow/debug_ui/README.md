# Debug UI

这是一个仅用于当前联调阶段的本地调试界面。

目标：

- 帮助同学用界面完成单文件入库
- 帮助同学直接执行语义检索
- 清晰展示切块、向量维度、元数据和检索结果细节

约束：

- 它不是正式的 4.6 前端
- 它不替代 4.5 API
- 它只调用现有的 4.1 / 4.3 / 4.4 服务

## 启动方式

```bash
cd backend
PYTHONPATH=src /Users/zaochuan/Documents/code/python/.venvs/SecKnow/bin/python -m secknow.debug_ui.app
```

## 当前功能

- 单文件入库
  - 选择文件
  - 配置分区、记录类型、分块策略、编码模式
  - 查看 4.1 切块日志与 4.3 写库结果
- 语义检索
  - 输入 query
  - 选择分区与返回数量
  - 查看 query 向量维度、前几个向量值和检索结果
- 数据库结构
  - 读取当前 Qdrant 的 collection 结构
  - 查看向量维度、距离类型、点数量和 payload 示例

## 适用场景

- 调试阶段本地演示
- 给不熟悉 CLI 的同学做联调入口
- 验证 4.1 -> 4.3 -> 4.4 闭环是否正常
