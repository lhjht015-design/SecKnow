# SecKnow

## 整体仓库结构

结合当前项目规划，仓库整体结构如下：

```text
SecKnow/
├── frontend/                    # 4.6 Vue 3 + Vite 前端
│   ├── src/
│   │   ├── api/
│   │   ├── components/
│   │   ├── views/
│   │   └── store/
│   └── package.json
├── backend/                     # Python 后端
│   ├── src/secknow/
│   │   ├── text_processing/     # 4.1 文本处理模块
│   │   ├── safety/              # 4.2 安全审查模块
│   │   ├── vector_store/        # 4.3 向量存储模块
│   │   ├── inference/           # 4.4 推理模块
│   │   ├── api/                 # 4.5 FastAPI 接口层
│   │   ├── ui_report/           # 结果报告/前端联动辅助模块
│   │   ├── alert_board/         # 4.7 告警板模块
│   │   └── shared/              # 公共常量、日志、工具
│   ├── tests/
│   ├── scripts/
│   ├── requirements.txt
│   └── pyrightconfig.json
├── docker-compose.yml
└── README.md
```

### 当前模块分工

- `backend/src/secknow/text_processing/`
  - 4.1 的主要开发位置
  - 负责切分文本、抽取元数据、生成向量前的数据结构
- `backend/src/secknow/safety/`
  - 4.2 的主要开发位置
  - 负责读取 4.3 提供的基线向量，做安全审查比对
- `backend/src/secknow/vector_store/`
  - 4.3 的主要开发位置
  - 负责存储、检索、导出，以及对外统一接口
- `backend/src/secknow/inference/`
  - 4.4 的主要开发位置
  - 负责检索增强与推理调用
- `backend/src/secknow/api/`
  - 4.5 的主要开发位置
  - 负责把 HTTP 请求映射到 4.3 / 4.4 的服务接口

### 关于对接

**每个模块目录下，都写一份现阶段的README负责对接**

