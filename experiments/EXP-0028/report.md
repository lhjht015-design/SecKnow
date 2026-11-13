# EXP-0028 — 验证链面板（产品集成）

把 v027/v028 的入口发现与链推荐接进产品，形态是「入口清单 + 候选链 + 知识证据 + 验证状态」。

## 接口

| 接口 | 输入 | 输出 |
| --- | --- | --- |
| `POST /api/chain` | 源码文本 / 语料内相对路径 / 文件名 | 入口、候选链、证据、警告 |
| `POST /api/chain/binary` | multipart ELF 上传（≤64 MiB） | 同上，额外含二进制画像 |

分析在服务端**只读**完成：不执行目标程序，上传的 ELF 分析后即删除。

## 验证

- 源码含 `strcpy` + `printf(buf)` + `system(input)`：
  识别 3 个入口；候选链分别为「命令注入 / 格式化字符串 / 栈溢出」，状态
  `需执行验证 / 需泄漏链 / 需要二进制`，每条链附 4 条知识证据。
- 上传 EXP-0027 的 B1 二进制：
  识别 `unsafe_copy`，画像 PIE 关 / canary 关 / NX 开；
  后门 `grant_kkrx0y @ 0x4011f6`（flag 字符串交叉引用）；
  链状态 `needs-execution`，栈缓冲候选 280/288/72/80。
- `pytest tests/unit` 全绿（含 7 条 solving 测试）；前端 `tsc`、`eslint`、`vite build` 通过。

截图：`experiments/EXP-0028/chain-panel.jpg`

## 边界

- 产品内不做执行验证：所有利用步骤标为「需执行」，需在靶场复核；避免在 API 进程执行未知二进制。
- 入口识别是规则/静态特征，不做数据流；未命中已知 sink 时会返回「未识别到入口」。
- 二进制分析依赖 `pyelftools` / `capstone`（当前环境已装，未写入核心依赖），缺失时降级为源码分析并给出警告。
