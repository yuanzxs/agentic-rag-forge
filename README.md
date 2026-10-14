# agentic-rag-forge

Agentic RAG 实验与评测项目。

## 目录结构
```
agentic-rag-forge/
├── src/agentic_rag/
│   ├── config.py               # pydantic-settings，所有开关集中在这里（关键！）
│   ├── ingestion/              # loaders / chunkers（fixed, semantic, proposition, contextual）
│   ├── indexing/               # dense, sparse, hierarchical(RAPTOR), graph builders
│   ├── retrieval/              # hybrid, rrf, rerank, compression, segment_extraction, filters
│   ├── query/                  # rewrite, decompose, hyde, hype, router(classifier)
│   ├── agent/                  # graph.py(状态机), nodes/, tools/, state.py, policies.py(预算/重试)
│   ├── memory/                 # short_term, long_term(memorag-style), forgetting
│   ├── generation/             # prompts, llm clients, citation, structured output
│   ├── evaluation/             # golden set runner, metrics, ablations, failure taxonomy
│   └── serving/                # api.py, schemas.py, streaming, cache
├── tests/                      # pytest：单元 + 端到端冒烟
├── data/
│   ├── raw/                    # 你的领域文档（建议：选一个你真的懂的领域）
│   ├── golden_set.jsonl        # 冻结的评测集
│   └── results/                # 每周评测结果 CSV（w1_baseline.csv ...）
├── notebooks/                  # 你的实验笔记本（不是抄来的）
├── docs/
│   ├── week01.md ... week08.md # 每周复盘
│   ├── ablation.md             # 消融报告（核心资产）
│   ├── architecture.md         # 架构与设计取舍
│   └── failure_taxonomy.md     # 失败模式分析
├── reference/                  # ⚠️ gitignore：原仓库只读参考
├── docker-compose.yml
├── pyproject.toml
├── Makefile                    # make ingest / eval / serve / ablation
├── .env.example
└── README.md                   # 含架构图、消融表、一键复现
```

## 环境

- Python 3.11
- uv 管理依赖

## 快速开始

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
```

## 文档

- docs/week01.md ... week08.md — 每周复盘
- docs/ablation.md — 消融报告
- docs/architecture.md — 架构与设计取舍
- docs/failure_taxonomy.md — 失败模式分析