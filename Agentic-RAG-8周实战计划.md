# Agentic RAG 8 周实战计划

> 目标：以 [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) 为教材与素材来源，8 周 × 每天约 3 小时，产出一个**可写进简历、可被面试追问、可公开演示**的高质量 Agentic RAG 系统，并系统性掌握 Agentic RAG 全套技术。
> 投入：约 150–170 小时（每天 3h，每周 6 天学习 + 1 天复盘）。总产出：1 个开源仓库 + 1 份消融实验报告 + 1 篇技术文章 + 4 条简历 bullet。

---

## 0. 结论先行：8 周后你必须交出的四样东西

| # | 交付物 | 形态 | 验收标准 |
|---|---|---|---|
| 1 | **顶石项目** `agentic-rag-forge` | 可运行的开源仓库 | `docker compose up` + 一条命令，10 分钟内复现完整系统；含流式对话、引用溯源、检索轨迹 |
| 2 | **消融实验报告** | `docs/ablation.md` + 图表 | ≥ 9 组配置对照，含检索/重排/路由/反思/图谱各层增益，同时报告延迟与成本 |
| 3 | **分层评测集** | `data/golden_set.jsonl` | 200 题，覆盖单跳/多跳/全局摘要/否定/超纲五类，冻结版本并入库 |
| 4 | **简历与故事线** | 简历 bullet + 3 分钟 demo | 每条 bullet 都有仓库里的数字和图表支撑，任何一个数字都能被追问 3 层 |

**一句话项目定位（建议）**：
> 面向企业私有文档库的自适应、自校正 Agentic RAG 系统：混合检索 + 查询路由 + 多步反思 + 图谱增强 + 长期记忆，带 LangGraph 编排、FastAPI 服务、全链路可观测与自动化评测 CI。

---

## 1. 动手前必须知道的三件事

### 1.1 这个仓库是"教材"，不是"库"

仓库自述写得很明确：**41 个可运行 notebook，覆盖从基础检索、分块，到自适应、Agentic、图谱与自校正系统；它是教学仓库而非软件库，没有可安装的包，也没有公开 API。** 目录结构：

```
all_rag_techniques/                     # 41 个 notebook（8 大类）
├── 基础 RAG（6）      simple_rag / simple_csv_rag / local_rag_huggingface_faiss / choose_chunk_size ...
├── 分块与索引（8）    semantic_chunking / proposition_chunking / contextual_chunk_headers /
│                      hierarchical_indices / raptor / document_augmentation / context_enrichment_window
├── 查询变换（3）      query_transformations / HyDE / HyPE
├── 检索策略（9）      fusion_retrieval / reranking / multi_faceted_filtering / dartboard /
│                      relevant_segment_extraction / contextual_compression /
│                      adaptive_retrieval / explainable_retrieval
├── 迭代与自校正（6）  self_rag / crag / reliable_rag / retrieval_with_feedback_loop /
│                      Agentic_RAG / memorag
├── Graph RAG（5）     graph_rag / graph_rag_local_attribution / Microsoft_GraphRag /
│                      graphrag_with_milvus_vectordb / light_rag
├── 结构化与多模态（3）json_rag / multi_model_rag_with_captioning / multi_model_rag_with_colpali
└── evaluation/        define_evaluation_metrics / end-2-end_rag_evaluation / deep_eval / grouse / evalute_rag.py

all_rag_techniques_runnable_scripts/    # 约 19 个等价 .py 脚本，可直接当 CLI 跑
data/                                   # 气候报告 PDF、耐克年报 txt、客户 CSV、QA json
helper_functions.py                     # 共享的加载器 / 切分器 / 检索工具
llms.txt                                # 机器可读的完整索引（给 AI agent 用）
```

**实践含义**：不要 `pip install`，也不要指望 import。你的正确姿势是——**读 notebook 的思想 → 在它给的 `.py` 脚本上跑通 → 然后用自己的工程结构重写一遍**。文件名里的 `_with_llamaindex` 版本是同一技术的第二实现，用来对比框架差异，时间紧时可只读不写。

### 1.2 许可证是"自定义非商用"——这决定了你的二次开发策略

仓库使用 Nir Diamant 的自定义许可（LICENSE 原文）：授予**非商用**的复制、修改、分发权，但要求：

1. **署名**：注明作者、附上仓库链接；
2. **标注修改**：说明你做了哪些改动；
3. **禁止商用**：不得整体或部分用于任何商业目的；商业权利由作者独家保留，需书面许可。

**这对你的三条硬约束**：

- ✅ 个人学习、开源作品集（portfolio）属非商用，**可以做，也可以公开**；
- ✅ 写文章、做分享，注明来源即可；
- ❌ 不要把这个仓库的代码搬进公司产品或任何收费服务；面试时可以讲思路，但交付物必须是你自己的实现；
- ⚠️ **最干净的做法：只借鉴思想，代码全部自己写**。在原仓库 `reference/` 目录只作只读参考并写进 `.gitignore`，README 里写一句"技术选型参考 NirDiamant/RAG_Techniques"，既合规又显专业。

> 论文里的算法（Self-RAG、CRAG、RAPTOR 等）本身不受该许可限制，你实现的是论文，不需要授权。

### 1.3 仓库的五个盲区 = 你项目的五个价值点

教学 notebook 的共同特征，恰好是工业界最看重的部分，也就是你要补的：

| 仓库的现状 | 你的增量（简历亮点） |
|---|---|
| 单轮问答，无状态、无会话记忆 | 多轮会话 + 长期记忆（实体记忆/偏好记忆/遗忘策略） |
| 每个 notebook 各跑各的，技术之间不组合 | 统一状态机，把 8 类技术组合成可开关的 pipeline |
| 评测很浅（多为单例演示、无分层） | 200 题分层 golden set + RAGAS + 失败模式分类法 |
| 无成本/延迟工程（很多为演示而牺牲效率） | token 成本看板、语义缓存、P95 延迟优化、并发压测 |
| 无服务化、无可观测 | FastAPI + SSE 流式 + 引用溯源 + Langfuse 全链路追踪 |

**记住这句话**：面试官不关心你跑通了 41 个 notebook，只关心"你解决了什么别人解决不了的问题，以及你怎么证明它变好了"。**消融实验 + 分层评测 + 成本账**就是你和其他候选人的分水岭。

---

## 2. 前置自评（5 分钟，决定你要压缩哪几周）

| 技能项 | 自评问题 | 若"不会"的补救（占用你自己的弹性时间） |
|---|---|---|
| Python 工程 | 能写类型标注、虚拟环境、包结构、pytest 吗 | 第 1 周前两天补；必须会，否则后面全是阻力 |
| LLM API | 会调用 chat/completions、控制 JSON 输出、算 token 吗 | 半天即可上手，第 1 周 D2 顺带学 |
| 向量检索 | 知道 embedding、余弦相似度、HNSW、top-k 吗 | 第 1 周 D2–D3 边做边补，不必先学理论 |
| 异步/Web | 会用 asyncio、FastAPI 吗 | 可推迟到第 7 周现学，那里才真正需要 |
| LangChain/LlamaIndex | 用过任一个吗 | 用过更好；**没用过反而建议直接上 LangGraph**，别先啃 LangChain |
| 图谱 / Neo4j | 会 Cypher 吗 | 第 5 周现学，先用 networkx 起步再迁 Neo4j |

**压缩规则**（时间不够时按此顺序砍，不要砍评测和周 4）：

1. 所有 `*_with_llamaindex` notebook：只读不跑（省约 10h）；
2. `dartboard`、`multi_model_rag_with_colpali`、`Microsoft_GraphRag`：只读思想，实现跳过（省约 12h）；
3. `multi_model_rag_with_captioning`、`json_rag`：作为可选项，放第 8 周以后；
4. **绝不砍**：评测体系（周 1 D4、周 6 全周）、消融实验（周 8）、Agentic 核心（周 4）。

---

## 3. 技术栈与工程约定（第 1 天就要定死）

### 3.1 选型表

| 层次 | 首选 | 备选 / 说明 |
|---|---|---|
| 语言环境 | **Python 3.11 + uv** | 你本机是 3.9.13，太老：LangGraph、新版 pydantic、raptor 相关依赖都会出问题。**先装 3.11** |
| 生成模型 | DeepSeek-V3 / Qwen（OpenAI 兼容接口） | 成本约为 GPT-4o 的几十分之一；架构上用 `OpenAI(base_url=...)` 即可随时切换 |
| Embedding | **BGE-M3**（本地，免费，中英俱佳） | 备选 Qwen-Embedding / text-embedding-3-large；本地跑省掉评测期最大的开销 |
| 向量库 | **Qdrant**：开发期用 `QdrantClient(path="./qdrant_data")` 本地持久化模式（**无需 Docker**），第 7 周再切 server 模式 | 备选 Milvus（仓库有 graphrag+milvus 例子）；Milvus 更重但更接近工业部署 |
| 稀疏检索 | BM25（`bm25s` / `rank_bm25`） | 融合用 RRF，**自己实现 RRF**（20 行代码，面试常问） |
| 重排 | **bge-reranker-v2-m3**（本地 cross-encoder） | 备选 Cohere Rerank（有免费额度，省 GPU） |
| 编排 | **LangGraph**（状态机 + checkpointer + 流式） | 重点学它：条件边、循环、并行、中断、持久化 |
| 图谱 | networkx 起步 → **Neo4j**（Docker） | 落地时用 Neo4j，简历上更好看 |
| 缓存 | 内存 LRU → **Redis** + 语义缓存 | 语义缓存注意阈值误命中问题 |
| 评测 | **RAGAS** + DeepEval + 自研 LLM-judge | Grouse、Open-RAG-Eval 作为对比参考 |
| 服务 | **FastAPI + SSE** | 配 Pydantic 模型 + 依赖注入 |
| 可观测 | **Langfuse**（可自托管，Docker） | 备选 LangSmith（需 API key） |
| 部署 | Docker Compose 一键起 | api + qdrant + redis + neo4j + langfuse；**第 1–6 周不需要 Docker，第 7 周才装** |

> **环境前置（按当前机器实况校准）**：本机现有 Python 3.9（系统默认）+ Anaconda(3.13)，uv 与 Docker 均未安装，git 身份未配置（否则第一次 commit 会失败）。要点：**不要动 conda base**，用 uv 独立管理 3.11 环境；国内网络建议预设 `UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple` 与 `HF_ENDPOINT=https://hf-mirror.com`（第 2 天下载 BGE-M3 需要）。

### 3.2 仓库目录结构（第一天就建好，后面只填不重构）

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

### 3.3 三条铁律

1. **一切可开关**：`config.py` 里每个技术都是一个 `bool`/枚举开关（`use_rerank`, `use_hyde`, `use_agentic_loop`, `use_graph`...）。没有开关就没有消融实验，没有消融实验就没有简历。
2. **评测先行**：第 1 周就冻结评测集。**评测集一旦冻结，不许因为"跑分不好看"而修改**（只能新增分层，不能删难题）。
3. **每周出数字**：每周复盘必须更新 `data/results/*.csv`，第 8 周的消融表是"抄"出来的，不是"补"出来的。

### 3.4 预算与成本控制（务必第 1 天做）

- 用 **本地 embedding + 本地 reranker**，只把生成/评分交给 API，成本可降一个数量级；
- 评测重跑是最大的成本项：**评测用的小模型（如 Qwen-turbo 级）+ 缓存 + 只对 diff 重跑**；
- 在 `src/evaluation/` 里记录每次运行的 token 消耗与费用，`docs/` 里给出一张"每周花费表"——这本身就是简历素材；
- 概念验证阶段单次查询成本目标：**< ¥0.02**；8 周总 API 预算按你选定模型的实时价目核算，先设一个月度上限告警。

---

## 4. 8 周总览

| 周 | 主题 | 核心 notebook（仓库路径省略 `.ipynb`） | 本周交付 | 里程碑 |
|---|---|---|---|---|
| 1 | 地基 + 基线 + 评测尺子 | simple_rag、simple_rag_with_llamaindex、local_rag_huggingface_faiss、choose_chunk_size、evaluation/* | 工程骨架、naive RAG、50 题评测集、baseline 指标表 | `v0.1-baseline` |
| 2 | 分块、索引与检索质量 | semantic_chunking、proposition_chunking、contextual_chunk_headers、document_augmentation、fusion_retrieval、reranking、contextual_compression、hierarchical_indices、raptor | 混合检索 + 重排 + 层级索引，检索层消融 | `v0.2-retrieval` |
| 3 | 查询理解、路由与 Agent 基础 | query_transformations、HyDE、HyPE、adaptive_retrieval、explainable_retrieval、multi_faceted_filtering、relevant_segment_extraction | 查询变换 + 路由器 + LangGraph 骨架 + 工具集 | `v0.3-query-agent` |
| 4 | **Agentic RAG 核心** | **Agentic_RAG**、self_rag、crag、reliable_rag、retrieval_with_feedback_loop | 统一自校正状态机（plan→retrieve→grade→act→verify） | `v0.4-agentic` |
| 5 | 图谱、结构化与多模态 | graph_rag、graph_rag_local_attribution、light_rag、Microsoft_GraphRag、json_rag、multi_model_rag_with_captioning | 图谱通道 + 结构化通道，接入 Agent 成为工具 | `v0.5-graph` |
| 6 | 记忆 + 评测体系（差异化） | memorag、define_evaluation_metrics、end-2-end_rag_evaluation、evaluation_deep_eval、evaluation_grouse | 长期记忆 + 200 题分层评测 + 失败模式分类 + CI 门禁 | `v0.6-eval` |
| 7 | 工程化与产品化 | （无新 notebook，转向工程） | FastAPI 流式服务、引用溯源、语义缓存、Langfuse、Docker 一键起 | `v0.7-prod` |
| 8 | 消融、包装、发布 | — | 消融报告、架构文档、技术文章、简历 bullet、demo 视频 | `v1.0` |

**贯穿全程**：每周 Day6 跑一次全量评测并把结果写入 `data/results/weekNN.csv`。第 8 周只是把已有的数字画成图，而不是重新做实验。

---

## 5. 每周详细计划

> 每天的 3 小时固定拆成：**输入 40 min**（读 notebook/论文）→ **动手 100 min**（写进 `src/`）→ **输出 40 min**（更新 `docs/weekNN.md`，记指标与坑）。
> Day7 为复盘日：跑全量评测、写周报、打 tag、规划下周。若某天没完成，**优先用 Day7 的一半时间补，另一半保留复盘**。

---

### 第 1 周：地基、基线、评测尺子

**本周目标**：把"尺子"做出来。没有评测集的 RAG 项目，后面 7 周全是玄学。
**领域选择建议**：选一个你真的懂、且文档量在 50–500 页的领域（公司年报 / 某开源项目文档 / 你的专业教材 / 法规标准）。不要用仓库自带的《气候变化》PDF 当最终语料，它只适合练手。

| Day | 3h 拆解 | 学习材料 | 动手产出 | DoD（完成定义） |
|---|---|---|---|---|
| D1 | 输入 40：读原仓库 README 技术总表 + llms.txt<br>动手 100：装 uv + Python 3.11；建仓库骨架、pyproject、Makefile、`.env.example`、`config.py` 全开关；`docker-compose.yml` 只写占位**不启动**<br>输出 40：写 `docs/week01.md` 记录选型理由 | README、llms.txt、AGENTS.md | `agentic-rag-forge` 骨架 + `make help` 可用 | `python -m agentic_rag.cli --help` 能跑；`cli health` 能连上本地模式 Qdrant（`QdrantClient(path=...)`） |
| D2 | 输入 40：simple_rag + simple_rag_with_llamaindex<br>动手 100：自己写 `ingestion/fixed_chunker.py` + `retrieval/dense.py` + `generation/answer.py`，跑通最小闭环（先本地 BGE-M3 + Qdrant）<br>输出 40：记录首条端到端问答与延迟 | simple_rag、simple_rag_with_llamaindex、local_rag_huggingface_faiss | `make ask Q="..."` 可用 | 能对自有文档问 5 个问题，答案带引用 chunk id |
| D3 | 输入 40：choose_chunk_size<br>动手 100：写 chunk size 扫描脚本（256/512/1024 × overlap 0/15%），6 组配置各跑 20 题<br>输出 40：把结果画成表 | choose_chunk_size | `notebooks/w1_chunk_sweep.ipynb` + `results/w1_chunk.csv` | 能说出"我的语料最佳 chunk 是 X，因为 Y" |
| D4 | 输入 40：define_evaluation_metrics + evalute_rag.py<br>动手 100：定义 7 个指标（Faithfulness / Answer Relevancy / Context Precision / Context Recall / 引用准确率 / P95 延迟 / 单次成本）；写 50 条 golden QA（LLM 生成候选 + **人工逐条筛**）<br>输出 40：冻结 `golden_set.jsonl` v1 | evaluation/define_evaluation_metrics、evalute_rag.py | 50 题评测集（含标准答案 + 证据 span） | 评测集冻结并 git commit，后续只增不减 |
| D5 | 输入 40：end-2-end_rag_evaluation<br>动手 100：接 RAGAS，写 `evaluation/runner.py`（输入配置名 → 输出指标 CSV）<br>输出 40：跑出 baseline 全指标 | evaluation/end-2-end_rag_evaluation | `make eval CONFIG=baseline` | `results/w1_baseline.csv` 有 7 列指标 |
| D6 | 输入 40：无（纯工程）<br>动手 100：结构化输出 + 引用溯源（答案 → 句子级 → chunk → 文档/页码）；CLI 加 `--json`<br>输出 40：录一个 30 秒屏幕演示 | — | 引用可点击/可定位的问答 | 抽 5 题人工核对引用正确率 |
| D7 | 复盘：全量评测 → `docs/week01.md`（含选型取舍与首周数据）→ `git tag v0.1-baseline` | — | 周报 | 1 页周报，含 1 张 baseline 指标表 |

**本周论文**：RAG（Lewis et al., 2020）、DPR（Karpukhin et al., 2020）、Lost in the Middle（Liu et al., 2023，只看结论：为什么 top-k 不是越多越好）。
**验收**：一条命令跑出全部 baseline 指标；能口头解释每个指标在衡量什么、以及它的失败模式。

---

### 第 2 周：分块、索引与检索质量

**本周目标**：把召回率打上去。这一周的每一分提升都最"可解释"，是消融表的第一段。

| Day | 3h 拆解 | 学习材料 | 动手产出 | DoD |
|---|---|---|---|---|
| D1 | 输入 40：semantic_chunking + proposition_chunking<br>动手 100：抽象 `Chunker` 接口（fixed/semantic/proposition 三种实现，可插拔）<br>输出 40：对比三种分块在同一评测集上的 Context Recall | semantic_chunking、proposition_chunking | 可插拔 chunker | 分块方式切换只需改一行 config |
| D2 | 输入 40：contextual_chunk_headers + document_augmentation<br>动手 100：实现索引期增强（chunk 前拼接文档/章节上下文；为 chunk 生成"它能回答的问题"并一并索引）<br>输出 40：记录索引期额外成本（这是取舍点！） | contextual_chunk_headers、document_augmentation | 增强索引流水线 | 能报告"召回 +X，索引成本 +Y 元/千 chunk" |
| D3 | 输入 40：fusion_retrieval<br>动手 100：**自己实现 RRF**（不要抄库），接 BM25 + dense 双通道，加权重/无权重对比<br>输出 40：记录 k 值（RRF 常数）敏感性 | fusion_retrieval、fusion_retrieval_with_llamaindex | `retrieval/hybrid.py` + `rrf.py` | 能解释"为什么 RRF 比分数加权更鲁棒" |
| D4 | 输入 40：reranking + contextual_compression<br>动手 100：over-retrieve(k=30) → bge-reranker → top5；对比 cross-encoder 与 LLM 重排的延迟/收益；再加压缩<br>输出 40：画"重排深度 vs 指标 vs 延迟"曲线 | reranking、contextual_compression | 重排 + 压缩模块 | 曲线图能说明你为何选 k=30/5 |
| D5 | 输入 40：hierarchical_indices + raptor<br>动手 100：实现层级摘要索引（自下而上聚类 + 摘要树）或简化版 RAPTOR，支持"任一层次检索"<br>输出 40：在全局性问题上单独测一次 | hierarchical_indices、raptor | 层级索引 | 全局/摘要类问题指标有可见提升 |
| D6 | 输入 40：multi_faceted_filtering + relevant_segment_extraction<br>动手 100：加元数据过滤（来源/时间/章节）+ MMR 多样性 + 连续片段拼装<br>输出 40：跑全量评测并出本周消融表 | multi_faceted_filtering、relevant_segment_extraction、explainable_retrieval（预习） | 过滤与片段模块 | `results/w2_*.csv` + 消融表前 5 行 |
| D7 | 复盘：周报 + `git tag v0.2-retrieval` | — | 周报 | 能讲清"哪一层带来了多少增益" |

**本周论文**：Anthropic《Introducing Contextual Retrieval》(2024)、RAPTOR（Sarthi et al., 2024）、ColBERT/重排相关（选读）。
**验收**：Context Recall 相对 baseline 有明确提升（目标：+15pt 以上），且你能画出一条"检索质量 vs 延迟"的帕累托曲线。

---

### 第 3 周：查询理解、路由与 Agent 基础

**本周目标**：从"检索到就答"进化到"先想清楚要检索什么、去哪儿检索"。同时把散装 pipeline 重构成 LangGraph 状态机——这是周 4 的地基。

| Day | 3h 拆解 | 学习材料 | 动手产出 | DoD |
|---|---|---|---|---|
| D1 | 输入 40：query_transformations + HyDE<br>动手 100：实现查询改写、多查询扩展、查询分解；实现 HyDE（先让模型写假设答案，再拿假设答案去检索）<br>输出 40：对比 4 种策略的召回与延迟 | query_transformations、HyDe_Hypothetical_Document_Embedding | `query/` 模块 | 每种策略可单独开关并测出增益 |
| D2 | 输入 40：HyPE<br>动手 100：实现索引期假设问句生成（把"文档 → 该文档能回答的问题"入库）；与 HyDE 对比成本结构<br>输出 40：结论写进周报（查询期 vs 索引期，怎么选） | HyPE_Hypothetical_Prompt_Embeddings | HyPE 索引器 | 能说清何时用哪种 |
| D3 | 输入 40：adaptive_retrieval<br>动手 100：实现**路由器**：先用规则/小模型分类（不需检索 / 单跳 / 多跳 / 全局摘要 / 结构化），再加 LLM 分类器对比；记录分类成本<br>输出 40：路由混淆矩阵 | adaptive_retrieval | `query/router.py` | 路由准确率 + 成本对比表 |
| D4 | 输入 40：explainable_retrieval + dartboard（dartboard 只读）<br>动手 100：检索可解释性——为每个结果输出打分分解（稠密分/稀疏分/重排分/元数据命中/多样性惩罚）；实现检索轨迹记录<br>输出 40：做一个"为什么检索到它"的调试视图 | explainable_retrieval、dartboard | 可解释检索 | 任一结果都能回答"为什么是它" |
| D5 | 输入 40：LangGraph 官方文档（StateGraph / 条件边 / checkpointer / 流式）<br>动手 100：把现有 pipeline 重构为图：`state → retrieve → generate`，加 checkpointer 与会话状态，支持流式输出<br>输出 40：架构图 v1（draw.io / mermaid） | LangGraph docs、仓库的 AGENTS.md（工程约定参考） | `agent/graph.py` | 图能跑通且与旧 pipeline 指标一致（回归验证） |
| D6 | 输入 40：ReAct 论文<br>动手 100：实现工具层（`hybrid_search`、`graph_search`(占位)、`web_search`、`calculator`）+ 单 agent ReAct 循环，含最大步数与超时<br>输出 40：全量评测 + 成本/步数分布 | ReAct（Yao et al., 2022） | 工具 + Agent 循环 | 多跳问题准确率提升，且能报告平均步数 |
| D7 | 复盘：周报 + `git tag v0.3-query-agent` | — | 周报 | 能画出一张完整的 Agent 流程图 |

**本周论文**：HyDE（Gao et al., 2022）、Adaptive-RAG（Jeong et al., 2024）、Query2Doc、ReAct、RAG 综述（Gao et al., 2023）。
**验收**：路由器能减少无效检索（成本下降），Agent 循环能在多跳问题上带来增益。

---

### 第 4 周：Agentic RAG 核心（全项目重心，慢下来做）

**本周目标**：把 Self-RAG / CRAG / Reliable RAG / Agentic RAG 的思想**融合成一个统一状态机**，并且加上预算护栏。这是简历里最值钱的一周。

**统一状态机设计**：

```
                    ┌──────────────────────────────────────────┐
   question ──────► │  route（需要检索吗？哪条通路？）            │
                    └───────────────┬──────────────────────────┘
                                    ▼
                    ┌──────────────────────────────────────────┐
              ┌────►│  retrieve（hybrid / graph / structured）  │
              │     └───────────────┬──────────────────────────┘
              │                     ▼
              │     ┌──────────────────────────────────────────┐
              │     │  grade（每篇文档相关性分级：related/       │
              │     │         ambiguous/irrelevant）            │
              │     └───────┬──────────────┬───────────────────┘
              │             │ 足够          │ 不足
              │             ▼              ▼
              │     ┌──────────────┐  ┌─────────────────────────┐
              │     │  generate    │  │ rewrite / web / graph   │◄─┐
              │     └──────┬───────┘  │ fallback（受预算约束）   │  │
              │            ▼          └─────────────┬───────────┘  │
              │     ┌──────────────────────────────┐│              │
              │     │  verify（有无依据支撑？       ││              │
              │     │   答案是否回答了问题？）       ││              │
              │     └───┬──────────────────┬───────┘│              │
              │         │ 通过              │ 不通过 │              │
              │         ▼                  └────────┼──────────────┘
              │   ┌───────────┐                     │（重试，≤N 次 / ≤¥B / ≤T 秒）
              └───┤  answer   │◄────────────────────┘（预算耗尽则降级作答并标注不确定性）
                  │ + 引用    │
                  └───────────┘
```

| Day | 3h 拆解 | 学习材料 | 动手产出 | DoD |
|---|---|---|---|---|
| D1 | 输入 40：Agentic_RAG notebook（精读两遍）<br>动手 100：重写为 LangGraph 多步 agent，实现 plan → 多轮 retrieve → reflect → retry<br>输出 40：记录每步 token 与耗时，画"步数分布" | **Agentic_RAG** | 多步 Agent | 能展示一次完整的多步推理轨迹 |
| D2 | 输入 40：self_rag + Self-RAG 论文<br>动手 100：实现三段式 critic（检索必要性 / 证据支撑度 / 有用性），**用强制 JSON + 小模型 + 阈值**避免 critic 自身幻觉<br>输出 40：critic 与人工判断的一致性抽检（30 例） | self_rag | `agent/nodes/self_rag.py` | critic 三分类与人工标注一致率 ≥ 80%（达不到就调 prompt/阈值） |
| D3 | 输入 40：crag + CRAG 论文<br>动手 100：实现文档分级（correct/ambiguous/incorrect）+ 查询重写 + 外部（web）检索兜底，含 web 失败时的降级路径<br>输出 40：造 20 个"知识库无答案"的问题专门验证兜底 | crag | CRAG 节点 | 超纲问题不再编造，能明确说"库里没有" |
| D4 | 输入 40：reliable_rag + retrieval_with_feedback_loop<br>动手 100：加校验层（引用是否存在、数字是否与原文一致、是否答非所问）；实现用户反馈写回（👍/👎 + 修正 → 存入反馈库，影响后续排序/检索）<br>输出 40：反馈生效的对照小实验 | reliable_rag、retrieval_with_feedback_loop | 校验层 + 反馈闭环 | 抽 20 题，"无依据断言"数量下降 |
| D5 | 输入 40：无<br>动手 100：**融合成统一状态机**（上图），加预算护栏：最大 step 数、最大 token、最大时延；超预算时优雅降级并标注置信度<br>输出 40：mermaid 架构图 + README 架构章节草稿 | 你的设计 | `agent/graph.py`（统一版） | 一次请求的所有节点耗时/token 都能打印出来 |
| D6 | 输入 40：无<br>动手 100：**逐个开关跑消融**：baseline / +route / +grade / +rewrite / +web / +verify / 全开<br>输出 40：画"质量 vs 延迟 vs 成本"三维权衡图 | — | `results/w4_ablation.csv` | 能明确指出"哪个机制性价比最高、哪个最贵" |
| D7 | 复盘：周报（这周要写得最详细）+ `git tag v0.4-agentic` | — | 周报 | 能用 3 分钟说清你的 Agent 设计和每次取舍 |

**本周论文**：Self-RAG（Asai et al., 2023）、CRAG（Yan et al., 2024）、Reflexion（Shinn et al., 2023）、Agentic RAG 综述（Singh et al., 2025）、ReAct（重读）。
**验收**：Faithfulness 有可量化提升；超纲问题不再幻觉；单请求有硬性成本/时延上限。

---

### 第 5 周：图谱、结构化与多模态通道

**本周目标**：解决"多跳"和"全局摘要"这两类单跳检索天然做不好的问题，并把它们变成 Agent 的一个工具。

| Day | 3h 拆解 | 学习材料 | 动手产出 | DoD |
|---|---|---|---|---|
| D1 | 输入 40：graph_rag<br>动手 100：实现实体/关系抽取（结构化 JSON）→ 建图 → 多跳遍历问答；先用 networkx<br>输出 40：在 20 个多跳问题上与纯向量检索对比 | graph_rag | `indexing/graph_builder.py` | 多跳问题准确率提升可量化 |
| D2 | 输入 40：graph_rag_local_attribution<br>动手 100：**span 级溯源**（图上的边/节点映射回原文位置）；迁移到 Neo4j（Cypher 查询 + 索引）<br>输出 40：图谱+原文双视角的可视化（pyvis 或 Neo4j Bloom） | graph_rag_local_attribution | Neo4j 落地 + 溯源 | 图谱答案也能给出来源段落 |
| D3 | 输入 40：light_rag + Microsoft_GraphRag（只读思想）<br>动手 100：对比"社区摘要式全局检索"与"轻量 graph+vector 双通道"，做取舍分析（成本/延迟/效果）<br>输出 40：写一份 1 页对比结论 | light_rag、Microsoft_GraphRag | 方案对比文档 | 能回答"你为什么不做完整 GraphRAG" |
| D4 | 输入 40：json_rag + simple_csv_rag<br>动手 100：结构化数据通道（表格 → 语义化行 + 元数据）；可选加 Text2SQL 工具<br>输出 40：在结构化问题上验证 | json_rag、simple_csv_rag | 结构化通道 | 表格类问题不再靠段落检索硬凑 |
| D5 | 输入 40：multi_model_rag_with_captioning（ColPali 只读）<br>动手 100：图片通道——为文档插图生成 caption 并入索引，或直接抽图+描述；评估图文混排文档效果<br>输出 40：记录多模态带来的成本 | multi_model_rag_with_captioning、multi_model_rag_with_colpali | 多模态通道（可选） | 含图问题能正确引用图所在页 |
| D6 | 输入 40：无<br>动手 100：把 graph / structured 接进 Agent 作为工具，由路由器决定调用；跑全量评测<br>输出 40：本周消融表 + 多跳/全局分层的专项指标 | — | 多通道 Agent | 分层指标显示多跳类问题显著提升 |
| D7 | 复盘：周报 + `git tag v0.5-graph` | — | 周报 | 能画出一张"多通道检索"的总架构图 |

**本周论文**：GraphRAG（Edge et al., 2024）、LightRAG（Guo et al., 2024）、HippoRAG、ColPali（Faysse et al., 2024）。
**验收**：多跳类问题准确率明显提升；图谱构建的成本被如实报告。

---

### 第 6 周：长期记忆 + 评测体系（简历差异化周）

**本周目标**：把"评测"从跑分工具升级为**工程体系**：分层评测集、失败模式分类法、CI 门禁、以及 LLM-judge 的可信度校验。多数候选人没有这一周，这就是你的护城河。

| Day | 3h 拆解 | 学习材料 | 动手产出 | DoD |
|---|---|---|---|---|
| D1 | 输入 40：memorag<br>动手 100：实现全局记忆（对语料生成全局摘要/关键实体记忆），用记忆引导检索（记忆 → 生成线索 → 检索）<br>输出 40：对比"有记忆/无记忆"在全局性问题上的表现 | memorag | `memory/long_term.py` | 全局摘要类问题可量化提升 |
| D2 | 输入 40：记忆相关论文（选读）<br>动手 100：会话记忆 + 用户偏好记忆 + **遗忘策略**（TTL、重要性打分、冲突消解）；多轮指代消解（"它/这个"指什么）<br>输出 40：10 轮连续对话测试 | MemoRAG / LongMem | 记忆写回与遗忘 | 10 轮对话不丢上下文、不冲突 |
| D3 | 输入 40：define_evaluation_metrics（重读）<br>动手 100：**评测集扩到 200 题并分层**：单跳 80 / 多跳 50 / 全局摘要 30 / 否定与前提 20 / 超纲 20；标注证据 span 与难度<br>输出 40：golden set v2 冻结 + 数据卡（datasheet） | evaluation/* | 分层评测集 + 数据卡 | 每类都能单独报告指标 |
| D4 | 输入 40：evaluation_deep_eval + evaluation_grouse<br>动手 100：RAGAS / DeepEval / 自研 judge 三方对比；**做 LLM-judge 与人工标注的一致性检验（Cohen's kappa 或 Spearman）**<br>输出 40：写"我用哪个 judge、为什么、误差多大" | evaluation_deep_eval、evaluation_grouse、open-rag-eval-example | judge 可信度报告 | κ ≥ 0.6 才允许用该 judge 做门禁 |
| D5 | 输入 40：无<br>动手 100：**失败模式分类法（error taxonomy）**：检索缺失 / 排序错误 / 上下文冲突 / 分块割裂 / 路由错误 / 过度检索 / 幻觉 / 拒答过度；抽样 50 个失败案例逐条归类<br>输出 40：失败分布图 + 每类的修复方案 | — | `docs/failure_taxonomy.md` | 50 例全部归类，top3 类有修复计划 |
| D6 | 输入 40：无<br>动手 100：**CI 回归门禁**：GitHub Actions 跑分层评测，任一指标下降超阈值即 fail；成本/延迟预算断言；结果回写 README badge<br>输出 40：故意提交一次"退化"验证门禁真的会红 | — | `.github/workflows/eval.yml` | 门禁能在 10 分钟内跑完（用子集采样） |
| D7 | 复盘：评测报告成文 + `git tag v0.6-eval` | — | 周报 | 一份能给别人看的评测报告 |

**本周论文**：RAGAS（Es et al., 2023）、ARES（Saad-Falcon et al., 2023）、MemoRAG。
**验收**：任何一次代码改动都能用一条命令判断"变好了还是变坏了"。

---

### 第 7 周：工程化与产品化

**本周目标**：让它看起来像一个产品，而不是一个实验。这一周决定了面试官对"工程能力"的判断。

| Day | 3h 拆解 | 动手产出 | DoD |
|---|---|---|---|
| D1 | 输入 40：FastAPI + SSE 文档<br>动手 100：`/chat`（SSE 流式）、`/ingest`、`/feedback`、`/health`、`/trace/{id}`；Pydantic 请求/响应模型；依赖注入管理连接<br>输出 40：OpenAPI 文档可用 | FastAPI 服务 | `curl` 能看到逐 token 流式输出 |
| D2 | 输入 40：无<br>动手 100：引用与溯源 API（答案 → 引用卡片：文档/页码/原文片段/相关度）+ 检索轨迹 API；可选加 Streamlit/Gradio 极简前端<br>输出 40：录屏演示 | 引用 UI + 轨迹 API | 前端能显示可点击引用 |
| D3 | 输入 40：无<br>动手 100：缓存三层（embedding 缓存、检索结果缓存、语义缓存 + 相似度阈值）；token 统计与成本看板（Prometheus 指标或简单 dashboard）<br>输出 40：缓存命中率与省钱效果 | 缓存 + 成本看板 | 报告缓存命中率与降本比例 |
| D4 | 输入 40：Langfuse 文档<br>动手 100：接入 Langfuse：每次请求的完整 trace（节点耗时、token、成本、检索结果、失败原因）；错误与慢请求筛选<br>输出 40：截图进 README | Langfuse 可观测 | 任一线上问题能定位到具体节点 |
| D5 | 输入 40：Qdrant/HNSW 调优、异步并发最佳实践<br>动手 100：异步化 + 批量 embedding + 连接池 + Qdrant 索引参数（ef/M）调优；用 k6 或 locust 压测并出 QPS/P95 曲线<br>输出 40：优化前后对比表 | 性能优化 | P95 延迟有明确下降 |
| D6 | 输入 40：无<br>动手 100：Docker Compose 一键起（api + qdrant + redis + neo4j + langfuse）；README 写"10 分钟复现"步骤；补 pytest（单元 + 端到端冒烟）<br>输出 40：让一个朋友按 README 跑一遍，记录他卡在哪 | 一键部署 + 测试 + README | 陌生人 10 分钟内跑起来 |
| D7 | 复盘：周报 + `git tag v0.7-prod` | 周报 | 项目已达"可演示"状态 |

**验收**：有一个别人能打开就用的地址或一键脚本；有延迟、成本、可用性数字。

---

### 第 8 周：消融、包装、发布

**本周目标**：把 7 周的工程转化为"可被信任的故事"。技术再好，讲不出来就等于没做。

| Day | 3h 拆解 | 动手产出 | DoD |
|---|---|---|---|
| D1 | 输入 40：无<br>动手 100：**全量消融矩阵重跑**：9+ 组配置，固定随机种子，每组跑 3 次取均值与标准差（报告方差，这是专业度信号）<br>输出 40：结果落表 | `docs/ablation.md` 数据部分 | 矩阵完整、方差被报告 |
| D2 | 输入 40：无<br>动手 100：画图——帕累托前沿（质量 vs 延迟 vs 成本）、各技术增量瀑布图、分层评测雷达图、失败模式分布图<br>输出 40：图进 README | 4 张图 | 图能在 10 秒内被看懂 |
| D3 | 输入 40：优秀开源项目 README 参考<br>动手 100：写技术文章（中文，≥3000 字）：问题→架构→关键取舍→消融→失败分析→局限；写英文 README（含架构图、消融表、快速开始）<br>输出 40：发布到掘金/知乎/公众号 | 技术文章 + README | 文章有一张消融表和一段诚实的局限说明 |
| D4 | 输入 40：无<br>动手 100：**打磨简历**（见第 7 节）；准备面试问答 15 题，每题写下 3 句答案 + 一个具体数字/例子<br>输出 40：对着镜子讲 3 分钟 | 简历 bullet + 问答稿 | 每个数字都能追溯到仓库文件 |
| D5 | 输入 40：无<br>动手 100：代码质量收尾——类型标注、docstring、ruff/black 通过、测试覆盖率报告、清理死代码；**用你自己的实现重写一遍 README 的技术选型说明**（合规：注明参考来源）<br>输出 40：录 2–3 分钟 demo 视频 | 干净的仓库 + demo 视频 | CI 全绿；视频能独立看懂 |
| D6 | 输入 40：无<br>动手 100：发布 `v1.0`；可选部署 HuggingFace Space / 云服务器 demo；同步更新个人主页、GitHub Profile README<br>输出 40：在社区（Discord / 技术群）求一次反馈 | 公开发布 | 拿到至少 3 条外部反馈 |
| D7 | 复盘：8 周总复盘 + 下一步路线（多 Agent 协作 / SFT 微调检索器 / 生产化运维 / 论文复现） | 总复盘 | 写清"我下一步要学什么" |

**验收**：仓库有一个陌生人能在 10 分钟内跑起来并看懂价值。

---

## 6. 顶石项目规格书（可直接当 README 骨架）

### 6.1 架构总览

```
                 ┌──────────────────────── 接入层 ────────────────────────┐
  用户 ──► FastAPI /chat (SSE)  ──►  会话管理 + 身份 + 限流 + 语义缓存      │
                 └───────────────────────────┬─────────────────────────────┘
                                             ▼
                 ┌────────────────── Agent 编排层（LangGraph）──────────────┐
                 │  state: {question, history, memory, plan, docs, trace,   │
                 │          budget{steps,tokens,deadline}, verdict}         │
                 │  route → retrieve → grade → (rewrite|web|graph) →        │
                 │  generate → verify → (retry | answer)                    │
                 └───────┬──────────────────┬──────────────────┬────────────┘
                         ▼                  ▼                  ▼
              ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
              │ 检索通道      │   │ 记忆          │   │ 工具              │
              │ dense(BGE-M3)│   │ 会话短期      │   │ web_search       │
              │ sparse(BM25) │   │ 长期(MemoRAG) │   │ calculator       │
              │ RRF 融合      │   │ 用户偏好      │   │ sql / structured │
              │ rerank       │   │ 遗忘策略      │   │ graph_query      │
              │ 层级/RAPTOR   │   └──────────────┘   └──────────────────┘
              │ 图谱(Neo4j)   │
              └──────┬───────┘
                     ▼
        ┌───────────────────────── 存储层 ─────────────────────────┐
        │ Qdrant(稠密+元数据)  Redis(缓存/会话)  Neo4j(图谱)  PG(反馈) │
        └──────────────────────────────────────────────────────────┘
                     ▼
        ┌──────── 横切能力 ────────┐   ┌──────── 评测与观测 ────────┐
        │ 成本账本 / 预算护栏       │   │ Langfuse trace             │
        │ 引用溯源 / 结构化输出      │   │ RAGAS + 分层 golden set    │
        │ 降级与拒答策略            │   │ CI 回归门禁 / 消融 runner   │
        └──────────────────────────┘   └────────────────────────────┘
```

### 6.2 消融实验矩阵（`docs/ablation.md` 的核心表）

固定：同一 golden set（200 题）、同一模型、3 次运行取均值。**下表数字必须是你自己跑出来的真实值。**

| # | 配置 | 检索 | 重排 | 查询变换 | 路由 | 反思/校正 | 图谱 | Faithfulness | Ans.Relevancy | Ctx.Recall | 多跳准确率 | 幻觉率 | P95 延迟 | ¥/query |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Baseline（naive dense top-k） | dense k=5 | — | — | — | — | — | | | | | | | |
| 1 | + Hybrid + RRF | dense+BM25 | — | — | — | — | — | | | | | | | |
| 2 | + Cross-encoder 重排 | hybrid k=30→5 | ✓ | — | — | — | — | | | | | | | |
| 3 | + 上下文压缩/片段拼装 | hybrid+rerank | ✓ | — | — | — | — | | | | | | | |
| 4 | + 查询变换（HyDE/多查询） | 同上 | ✓ | ✓ | — | — | — | | | | | | | |
| 5 | + 自适应路由 | 同上 | ✓ | ✓ | ✓ | — | — | | | | | | | |
| 6 | + Self-RAG / CRAG 反思闭环 | 同上 | ✓ | ✓ | ✓ | ✓ | — | | | | | | | |
| 7 | + 图谱通道 | 同上 | ✓ | ✓ | ✓ | ✓ | ✓ | | | | | | | |
| 8 | **Full Agentic（+记忆+预算护栏）** | 全部 | ✓ | ✓ | ✓ | ✓ | ✓ | | | | | | | |

**分析要点（写进报告的文字比数字更值钱）**：
- 哪一层增益最大？哪一层最贵（延迟/成本）？性价比拐点在第几行？
- 有没有"负增益"？如果有，为什么（常见：查询改写把明确问题改模糊了；rerank 把长答案证据挤掉了）？**主动报告负增益会大幅提高可信度**。
- 延迟-质量帕累托前沿上，你为线上默认配置选了哪一点？为什么？

### 6.3 指标定义（写进 README，避免自说自话）

| 指标 | 定义 | 目标 |
|---|---|---|
| Faithfulness | 答案中的每个断言都能被检索上下文支撑的比例 | ≥ 0.90 |
| Answer Relevancy | 答案切题程度（反向生成问题与原始问题的相似度） | ≥ 0.88 |
| Context Precision | 检索结果中真正相关文档的排序质量 | ≥ 0.80 |
| Context Recall | 标准答案所需证据被召回的比例 | ≥ 0.85 |
| 引用准确率 | 引用标注指向的原文确实支撑该句的比例（人工抽检 50 条） | ≥ 0.95 |
| 拒答准确率 | 超纲题中正确拒答的比例；同时不误拒有答案的问题 | 拒答 ≥ 0.9 / 误拒 ≤ 0.05 |
| 幻觉率 | 抽检中含无依据断言的比例 | ≤ 5% |
| P95 延迟 | 端到端 95 分位 | ≤ 5s（含反思时 ≤ 8s） |
| 单次成本 | 平均每问 API 花费 | ≤ ¥0.02 |

---

## 7. 简历包装

### 7.1 项目描述模板（3 段式）

> **Agentic RAG 企业知识问答系统**（个人项目 · 开源）
> **背景**：企业私有文档问答存在三个痛点——检索召回不足导致答非所问、模型在证据不足时编造答案、以及质量无法量化因而无法迭代。
> **做了什么**：基于 LangGraph 设计自适应、自校正的多步 Agent 状态机，融合混合检索（BM25+BGE-M3+RRF）、Cross-Encoder 重排、查询路由、Self-RAG/CRAG 式反思闭环与图谱增强通道；建立 200 题分层评测集与 CI 回归门禁，并用消融实验驱动每一处技术选型。
> **结果**：Faithfulness 从 __ 提升至 __，多跳问题准确率从 __ 提升至 __，超纲问题幻觉率从 __% 降至 __%，同时单次查询成本控制在 ¥__ 以内。

### 7.2 简历 bullet（**所有数字必须替换成你仓库里跑出来的真实值**）

- 基于 **LangGraph** 构建自适应 Agentic RAG 状态机，实现「路由→检索→文档分级→查询重写/外部检索兜底→生成→答案校验」闭环；在自建 200 题分层评测集上，将 **Context Recall 从 __ 提升至 __、Faithfulness 从 __ 提升至 __**。
- 设计混合检索与重排链路（BM25 + BGE-M3 + 自研 **RRF** 融合 + Cross-Encoder 重排 + 上下文压缩），并给出 **质量-延迟-成本帕累托曲线**，为线上默认配置选定最优点（P95 __ ms）。
- 引入 **Self-RAG/CRAG** 式自校正机制与预算护栏（最大步数/Token/时延），**超纲问题幻觉率从 __% 降至 __%**，且单次查询成本 ≤ **¥__**；并通过失败模式分类法定位 top3 问题来源。
- 搭建 **分层评测与 CI 回归门禁**（RAGAS + 自研 LLM-judge，judge 与人工标注一致性 κ=__），使任意改动可在 __ 分钟内判定质量升降；产出 **9 组消融实验**证明各模块独立增益。
- （加分项）用 **Neo4j 知识图谱** + span 级溯源解决多跳与全局摘要问题，**多跳类问题准确率提升 __ pt**，并保留答案到原文段落的可追溯引用。

> 写法要点：动词用"设计/构建/量化/定位"，名词用真实技术栈，结果必须是**可核验的数字**。不要写"熟悉/了解/参与"，不要写"基于 GPT-4 实现了问答"，不要用"准确率 99%"这种没有定义的数字。

### 7.3 面试高频 15 问（每题准备好 3 句话 + 1 个数字/例子）

1. 为什么用 RRF 而不是加权求和？RRF 的常数 k 怎么定的，敏感吗？
2. chunk size 和 overlap 你是怎么定的？不同文档类型要变吗？
3. rerank 的收益和延迟代价怎么权衡？为什么 over-retrieve 到 30 而不是 50？
4. 你的路由器有多少类？用规则还是模型？路由错了会怎样，有兜底吗？
5. Self-RAG 的 critic 自己会不会幻觉？你怎么保证 grading 可靠？
6. 什么情况下你会**不**检索？不检索的收益是什么？
7. 如何评测多跳问题？单跳指标为什么不够？
8. 你的 LLM-judge 可信吗？怎么验证？κ 到多少才敢拿它当门禁？
9. 超纲问题怎么处理？怎么区分"库里没有"和"检索没召回"？
10. 上下文冲突（两份文档说法矛盾）怎么处理？
11. 成本怎么控制？语义缓存的误命中风险怎么防？
12. 流式输出和引用溯源怎么同时做？引用是生成前还是生成后确定？
13. 并发上去以后瓶颈在哪？Qdrant 索引参数怎么调？
14. 你的消融里有没有出现负增益的技术？为什么？你怎么决策的？
15. 如果文档量从 1 万涨到 1 亿，你的架构哪里先崩，怎么改？

### 7.4 简历避雷

- ❌ "跑通了 41 个 RAG 技术" —— 这是学习者，不是工程师；
- ❌ 只写技术名词堆砌，没有数字、没有取舍；
- ❌ 把 `.ipynb` 直接提交到仓库当项目 —— 面试官会认为你只会跟着教程走；
- ✅ 一定要有：可复现命令、消融表、失败分析、诚实的局限说明。**"我知道它在哪里不行"是加分项**。

---

## 8. 风险清单与对策

| 风险 | 症状 | 对策 |
|---|---|---|
| **教程地狱**（最大风险） | 每个 notebook 都跑通了，但什么都没留下 | 铁律：任何 notebook 必须"重写进 `src/` + 记录指标"才算完成；否则标记为"只读" |
| API 成本失控 | 评测重跑一次几十块 | 本地 embedding/rerank；评测用便宜模型；缓存；只重跑 diff |
| 评测自欺 | 指标很漂亮但答得还是烂 | 冻结评测集 + 人工抽检 50 条 + 报告方差 + 不看单一指标 |
| 过早优化 | 第 2 周就在纠结 HNSW 参数 | 严格按周递进：先质量、后评测、最后性能 |
| 范围蔓延 | 第 6 周还在加新通道 | 每周只做本周表格里的事；新想法记到 `docs/backlog.md`，第 8 周后再说 |
| 时间不够 | 某周只完成一半 | 用第 2 节的"压缩规则"砍范围，**但保住周 4（Agentic 核心）、周 6（评测）、周 8（消融）** |
| Docker 未安装 / 容器卡住 | 一整天在修 Qdrant 连接 | **第 1–6 周完全不依赖 Docker**：向量库用 `QdrantClient(path="./qdrant_data")` 本地模式，自研 RRF + 进程内 BM25 也不依赖服务端融合能力。Docker Desktop 留到第 7 周部署周再装（需 WSL2 + 管理员权限，可能重启）。设 45 分钟止损线 |
| 许可证风险 | 直接复制仓库代码 | 只借鉴思想、自己重写；README 注明参考来源与改动（见 1.2 节） |

---

## 9. 每日节奏与复盘模板

### 9.1 每天 3 小时（建议固定时段，减少启动成本）

| 时长 | 环节 | 具体动作 |
|---|---|---|
| 0:00–0:40 | **输入** | 读本周 notebook（先看标题/图/输出，再读代码）+ 相关论文摘要；**只记 3 个要点**，不做笔记搬运 |
| 0:40–2:20 | **动手** | 全部时间用于写进 `src/`；每 40 分钟 commit 一次；遇到报错先自己查 15 分钟 |
| 2:20–3:00 | **输出** | 更新 `docs/weekNN.md`：今天做了什么 / 我改了什么 / 指标变化 / 卡在哪 / 明天第一步 |

### 9.2 每次实验必须记录的四件事（缺一不可）

```
配置名:           例如 rerank_k30_top5
改动点:           一句话
指标:             faithfulness / ctx_recall / p95 / cost  (含运行时间与模型版本)
结论与坑:         为什么变好或变坏，下一个假设是什么
```

### 9.3 周日复盘清单（30 分钟）

- [ ] 全量评测跑完，`data/results/weekNN.csv` 已更新
- [ ] `docs/weekNN.md` 写完（含至少 1 张表或图）
- [ ] `git tag` 打完，README 进度部分更新
- [ ] 本周有一个"能对外讲 3 分钟"的成果
- [ ] 下周的 Day1 第一步已写在待办里

---

## 10. 资源索引

### 10.1 核心（本计划的主线）

- **教材主仓库**：[NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques)（41 个 notebook + `all_rag_techniques_runnable_scripts/` 下的等价 `.py` 脚本）
- **机器可读索引**：[llms.txt](https://github.com/NirDiamant/RAG_Techniques/blob/main/llms.txt)（丢给 AI 助手快速定位技术，很省时间）
- **工程约定参考**：[AGENTS.md](https://github.com/NirDiamant/RAG_Techniques/blob/main/AGENTS.md)
- **视频讲解**：仓库中有 4 个 notebook 配有作者本人的视频（RAG 为什么答错、表格语义检索、如何确认答案没编造、为什么返回整段而非一个事实）

### 10.2 同作者的其他仓库（第 5–8 周按需查阅）

- [GenAI_Agents](https://github.com/NirDiamant/GenAI_Agents)：单 Agent → 多 Agent 的教程
- [agents-towards-production](https://github.com/NirDiamant/agents-towards-production)：Agent 从原型到部署（第 7 周重点参考）
- [Agent_Memory_Techniques](https://github.com/NirDiamant/Agent_Memory_Techniques)：30 个记忆相关 notebook（第 6 周重点参考）

### 10.3 论文清单（按周绑定，只读摘要+方法+实验结论）

| 周 | 论文 |
|---|---|
| 1 | RAG (Lewis 2020)、DPR (Karpukhin 2020)、Lost in the Middle (Liu 2023) |
| 2 | Contextual Retrieval (Anthropic 2024)、RAPTOR (Sarthi 2024)、Sentence-BERT |
| 3 | HyDE (Gao 2022)、Adaptive-RAG (Jeong 2024)、ReAct (Yao 2022)、RAG Survey (Gao 2023) |
| 4 | **Self-RAG (Asai 2023)、CRAG (Yan 2024)**、Reflexion (Shinn 2023)、Agentic RAG Survey (Singh 2025) |
| 5 | GraphRAG (Edge 2024)、LightRAG (Guo 2024)、HippoRAG、ColPali (Faysse 2024) |
| 6 | **RAGAS (Es 2023)**、ARES (Saad-Falcon 2023)、MemoRAG |
| 7–8 | 工程类：LangGraph / Qdrant 调优 / 评测驱动开发（无必读论文） |

学习法：**先读摘要和结论 → 看方法图 → 再决定要不要读细节**。目标是能讲清"它解决什么问题、代价是什么"，不是复现。

---

## 附录 A：Makefile 目标（照着写，能省掉大量重复劳动）

```makefile
install:        # 建环境 + 装依赖
up:             # docker compose up -d（qdrant/redis/neo4j/langfuse）
ingest:         # 摄取 data/raw → 分块 → 索引（含可选增强）
ask:            # 单次问答（Q=...）
chat:           # 交互式多轮
eval:           # 全量评测（CONFIG=baseline 等）
ablation:       # 一键跑完 9 组消融，输出 docs/ablation.md 表格
serve:          # 起 FastAPI
test:           # pytest + 覆盖率
lint:           # ruff + black + mypy
```

## 附录 B：常见坑速查

| 现象 | 常见原因 | 排查顺序 |
|---|---|---|
| 检索不到明显相关的段落 | 分块割裂 / embedding 模型不适配中文 / 只用稠密检索 | 打印 chunk 原文 → 换 BGE-M3 → 加 BM25 |
| 答案里有原文没有的数字 | 模型补全倾向 / 上下文太长导致忽略证据 | 开校验层 → 缩短上下文 → 强制引用 |
| 加了 rerank 反而变差 | over-retrieve 太少 / 重排模型与语言不匹配 / 把长证据截断 | 提高候选数 → 换 bge-reranker-v2-m3 → 检查截断长度 |
| 多跳问题全错 | 单跳检索无法跨文档串证据 | 查询分解 + 图谱通道 |
| 评测分数高但人工看很差 | judge 太宽松 / 评测集与语料同源泄漏 | 换 judge + 一致性检验 + 人工抽检 |
| Agent 偶尔死循环 | 无预算护栏 / 重试条件过松 | 加最大步数、Token、时延上限 + 优雅降级 |
| 成本突然暴涨 | 反思循环每步都带全上下文 | 每步只传必要状态；对 critic 用小模型 |

---

## 附录 C：如果你只有 4 周（压缩版）

砍掉周 2 的 RAPTOR/层级、周 5 全部（图谱/多模态）、周 3 的 HyPE，保留：
**周 1（地基+评测）→ 周 2（混合检索+重排）→ 周 3–4（Agentic 核心 + 记忆 + 消融 + 包装）**。
项目定位改为"带自校正的混合检索 RAG"，仍然成立且有数字。

---

*本计划的仓库信息（41 个 notebook、8 大类目录结构、许可证条款）均来自对 [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) 的实际核查。计划中的技术路线为你自己的实现，原仓库仅作为学习参考。*
