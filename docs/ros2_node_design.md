# ROS2 节点拆分方案：RAG 文档智能问答系统

## 概述

将现有单体 RAG 系统 (`rag-system`) 拆分部署到 ROS2 分布式环境中。核心理念：将原本在同一进程内的方法调用，转换为 ROS2 节点间的 **Action** / **Service** / **Topic** 通信。

> **Cloud API 不要求本地替代**：DashScope 百炼（Router / Embedding）、DeepSeek（Answer Generation）保持云调用，仅将 API Client 封装在对应节点内。

---

## 拆分原则

| 原则 | 说明 |
|------|------|
| **I/O 边界拆分** | CPU 密集型（解析）、存储密集型（索引）、网络 IO 密集型（API 调用）分离到不同节点 |
| **生命周期解耦** | 解析节点按需启停，索引/检索/回答节点常驻 |
| **Cloud 保持云** | Router、Embedding、LLM、Qwen Realtime ASR/TTS 调用不变，节点内封装 API Client |
| **本地保留** | ChromaDB、SQLite 索引（BM25/Sparse/Manifest）、Whisper/FSMN-VAD 兜底模型 |

---

## 节点总览

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           ROS2 Node Graph                                  │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   ┌────────────────┐                                                     │
│   │  rag_api_node  │  FastAPI + Web UI（可选编排节点）                      │
│   │  (可选)         │                                                     │
│   └───┬────┬───┬───┘                                                     │
│       │    │   │                                                          │
│       │    │   │          Action: ParseDocument                           │
│       │    │   └──────────────────────────────────┐                       │
│       │    │                                      ▼                       │
│       │    │                          ┌──────────────────────┐            │
│       │    │                          │   rag_parser_node     │            │
│       │    │                          │   ─────────────────── │            │
│       │    │                          │   • 多格式解析 (auto)   │            │
│       │    │                          │   • 切片 (chunking)    │            │
│       │    │                          │   • 内嵌图片 OCR       │            │
│       │    │                          │   • SHA256 缓存        │            │
│       │    │                          │   • 表格/资源提取       │            │
│       │    │                          │   纯本地计算，无云依赖   │            │
│       │    │                          └──────────┬───────────┘            │
│       │    │                                      │                        │
│       │    │   Service: Search                    │ Topic:                 │
│       │    │   Service: Answer                    │ /parsed_document       │
│       │    │                                      ▼                        │
│       │    │                          ┌──────────────────────┐            │
│       │    ├─────────────────────────►│   rag_index_node      │            │
│       │    │                          │   ─────────────────── │            │
│       │    │                          │   • Embedding (百炼云) │            │
│       │    │                          │   • ChromaDB 向量库    │            │
│       │    │                          │   • BM25 关键词索引     │            │
│       │    │                          │   • Sparse 稀疏索引    │            │
│       │    │                          │   • Manifest 元数据    │            │
│       │    │                          │   • 图片向量索引        │            │
│       │    │                          │   数据唯一写入者        │            │
│       │    │                          └──────────┬───────────┘            │
│       │    │                                      │                        │
│       │    │   Service: Search                    │ 直接读取                │
│       │    │   Service: Answer                    │ 索引文件                │
│       │    │                                      ▼                        │
│       │    │                          ┌──────────────────────┐            │
│       │    ├─────────────────────────►│  rag_retrieval_node   │            │
│       │    │                          │   ─────────────────── │            │
│       │    │                          │   • Query Rewrite     │            │
│       │    │                          │     (DeepSeek 云)     │            │
│       │    │                          │   • 多路混合检索       │            │
│       │    │                          │     Dense+BM25+Sparse │            │
│       │    │                          │   • RRF 融合排序       │            │
│       │    │                          │   • Rerank (百炼云)   │            │
│       │    │                          │   • 质量评估 + 二次检索 │            │
│       │    │                          └──────────┬───────────┘            │
│       │    │                                      │                        │
│       │    │   Service: Answer                    │ Service: Search        │
│       │    │                                      │ (内部调用)              │
│       │    │                                      ▼                        │
│       │    │                          ┌──────────────────────┐            │
│       │    └─────────────────────────►│  rag_answer_node      │            │
│       │                               │   ─────────────────── │            │
│       │                               │   • Intent Router     │            │
│       │                               │     (DeepSeek 云)     │            │
│       │                               │   • Guard 预设回答     │            │
│       │                               │   • 上下文构建         │            │
│       │                               │   • Answer Generation │            │
│       │                               │     (DeepSeek 云)     │            │
│       │                               │   • 对话历史管理       │            │
│       │                               │   • 引用提取           │            │
│       │                               │   流式输出 → Topic     │            │
│       │                               └──────────────────────┘            │
│       │                                                                   │
│       │   Service: STT / TTS                                              │
│       │                                                                   │
│       ▼                                                                   │
│   ┌──────────────────────┐                                                │
│   │  rag_voice_node      │  ASR: Qwen Realtime / 本地 Whisper 兜底          │
│   │  (可选)               │  TTS: Qwen Realtime / edge-tts 兜底             │
│   └──────────────────────┘                                                │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 节点 1：`rag_parser_node` — 文档解析节点

### 职责

接收文件路径，返回解析后的结构化文档（`ParsedDocument`）。

### 生命周期

- **按需启动**：有解析任务时启动，完成后可休眠/销毁
- 建议使用 ROS2 **Managed Node**（LifecycleNode），配合 Action 触发

### ROS2 接口

```yaml
# Action: ParseDocument
# 长耗时任务，需要进度反馈

Request:
  file_path: string           # 输入文件路径
  backend: string = "auto"    # 解析后端: auto / light / docling / unstructured
  chunk_size: int32 = 2400
  chunk_overlap: int32 = 300
  enable_cache: bool = true
  ocr_languages: string = "eng+chi_sim"

Result:
  document: ParsedDocument    # 完整解析结果
  elapsed_seconds: float64
  diagnostics: string[]

Feedback:
  progress: float32           # 0.0 ~ 1.0
  current_stage: string       # "parsing" | "chunking" | "ocr" | "extracting"
  detail: string              # 当前处理的文件名或阶段描述
```

### 本地计算（无云依赖）

- 多格式解析：TXT / Markdown / HTML / PDF / Image / Docx / PPTX / XLSX / EPUB
- `auto` 模式智能选后端：小 PDF → Docling，扫描件 → Docling+OCR，大 PDF → pypdf
- 切片（`DocumentChunker`）
- SHA256 解析缓存
- 表格提取、资源（图片）提取
- 诊断信息构建

### 与现有代码映射

```
parsing/service.py  →  ParsingService  →  Action 回调中调用
parsing/models.py   →  ParsedDocument  →  自定义 ROS2 Message
```

---

## 节点 2：`rag_index_node` — 索引存储节点

### 职责

管理所有本地索引存储，是数据的**唯一写入者**。

### 生命周期

- **常驻**：索引数据需要随时可写入、可查询

### ROS2 接口

```yaml
# Topic 订阅
/parsed_document: rag_interfaces/msg/ParsedDocument   # 接收解析结果，自动触发索引

# Service: IndexDocument（手动触发单文档索引）
Request:
  document: ParsedDocument
---
Response:
  success: bool
  document_id: string
  chunk_count: int32
  vector_count: int32
  message: string

# Service: DeleteDocument
Request:  document_id: string
Response: success: bool, deleted_counts: DeleteCounts

# Service: SetDocumentEnabled
Request:  document_id: string, enabled: bool
Response: success: bool

# Service: GetStats
Response:
  document_count: int32
  chunk_count: int32
  vector_count: int32
  collection: string
  model: string
  dimensions: int32
  bm25_chunk_count: int32
  sparse_chunk_count: int32

# Service: ListDocuments
Request:  filter: string = ""
Response: documents: IndexedDocument[]

# Service: GetDocumentDetail
Request:  document_id: string
Response:
  document: IndexedDocument
  chunks: DocumentChunk[]
  images: ImageInfo[]
  counts: CountInfo

# 图片相关
# Service: IndexImages
# Service: SearchByImage
# Service: GetImageStats
# Action: IndexDocumentImages（含进度反馈）
```

### 本地存储

| 存储 | 技术 | 文件路径 |
|------|------|----------|
| 向量数据库 | ChromaDB（嵌入式） | `data/index/chroma/` |
| BM25 关键词索引 | SQLite | `data/index/bm25.sqlite3` |
| 稀疏向量索引 | SQLite | `data/index/sparse.sqlite3` |
| 文档元数据清单 | SQLite | `data/index/index.sqlite3` |
| 图片向量索引 | ChromaDB | `data/index/chroma_images/` |

### 云 API 调用

- **DashScope 百炼 Embedding API**：`text-embedding-v4`，文本 → 1024 维向量
- **DashScope 百炼 Multimodal Embedding**：`qwen3-vl-embedding`，图片 → 2048 维向量

### 与现有代码映射

```
indexing/service.py      →  IndexingService      →  Service 回调
indexing/image_service.py → ImageIndexingService  →  图片索引 Service
indexing/models.py       →  IndexedDocument 等   →  自定义 ROS2 Message
indexing/manifest.py     →  IndexManifest         →  文档管理 Service
```

---

## 节点 3：`rag_retrieval_node` — 检索节点

### 职责

接收查询字符串，执行混合检索，返回排序后的结果。

### 生命周期

- **常驻**：检索是高频操作

### ROS2 接口

```yaml
# Service: Search
Request:
  query: string
  top_k: int32 = 5
---
Response:
  results: SearchResult[]
  query_tokens: string[]
  rewrite:
    original_query: string
    rewritten_queries: string[]
    keywords: string[]
    intent: string
    method: string           # "deepseek" | "fallback"
    model: string
  diagnostics:
    retrieval_source: string # "hybrid" | "vector" | "bm25" | "sparse"
    rrf_score: float64
    rerank_score: float64
    matched_token_count: int32
    matched_token_ratio: float64
```

### 检索流水线

```
用户查询 query
    │
    ▼
┌─────────────────────────────┐
│  Query Rewrite (DeepSeek)   │  → 多条改写查询 + 关键词
│  deepseek-v4-flash          │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  多路并行检索                │
│  ┌──────────┬──────────┬───┐│
│  │ Dense    │ BM25     │Spa││
│  │ (Chroma) │ (SQLite) │rse││
│  └──────────┴──────────┴───┘│
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  RRF 融合排序 (k=60)         │  分数 = Σ(1/(k+rank))
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Rerank (百炼 qwen3-rerank)  │  或本地 BGE CrossEncoder
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  质量评估 + 二次检索(可选)    │  低质量时自动扩大召回
└─────────────┬───────────────┘
              │
              ▼
         Top-K 结果
```

### 本地计算

- 读取 ChromaDB + BM25 SQLite + Sparse SQLite（与索引节点共享 `data/index/` 目录）
- RRF 融合排序
- 检索质量评估（`_assess_retrieval_quality`）

### 云 API 调用

- **DeepSeek Query Rewrite**：`deepseek-v4-flash`，生成改写查询 + 关键词
- **DashScope Rerank**：`qwen3-rerank`，精排（可选本地 BGE CrossEncoder）

### 与现有代码映射

```
retrieval/query_rewrite.py → DeepSeekQueryRewriteClient  →  检索节点内调用
retrieval/rerank.py        → DashScopeReranker / Local   →  检索节点内调用
indexing/service.py        → IndexingService.search_*()  →  Service 回调
```

---

## 节点 4：`rag_answer_node` — 答案生成节点

### 职责

意图路由 + 上下文构建 + LLM 生成回答 + 引用提取。

### 生命周期

- **常驻**：维护对话历史状态

### ROS2 接口

```yaml
# Service: Answer（非流式）
Request:
  query: string
  top_k: int32 = 4
---
Response:
  query: string
  answer: string
  citations: Citation[]
  evidence: EvidenceChunk[]
  model: string
  confidence: string         # "high" | "medium" | "low"
  retrieval:                  # 检索诊断信息
    rewrite: QueryRewriteInfo
    results: SearchResult[]
    context_top_k: int32
    quality: RetrievalQuality
    second_pass: SecondPassInfo

# Topic: /answer_stream（流式输出）
# 消息类型: AnswerDelta
# type: "meta"     → 元信息（model, evidence, retrieval）
# type: "delta"    → 增量文本
# type: "final"    → 完整 AnswerResult
```

### 回答生成流程

```
用户问题 query
    │
    ▼
┌─────────────────────────────┐
│  Guard 预设回答匹配           │  问候/感谢等直接回复
│  命中 → 直接返回              │
└─────────────┬───────────────┘
              │ 未命中
              ▼
┌─────────────────────────────┐
│  Intent Router (DeepSeek)   │  四类路由:
│  deepseek-v4-flash          │  chat_direct → 不检索，直接对话
│                             │  rag_retrieve → 需要检索知识库
│                             │  app_help → 系统使用帮助
│                             │  clarify → 要求用户澄清
└─────────────┬───────────────┘
              │
    ┌─────────┴──────────┐
    │ chat_direct        │ rag_retrieve
    │ app_help           │
    │ clarify            │
    ▼                    ▼
┌────────────┐    ┌──────────────────────┐
│ 直接 LLM    │    │  调用检索节点          │
│ 对话        │    │  Search Service       │
│ (DeepSeek) │    └──────────┬───────────┘
└────────────┘               │
                             ▼
                    ┌──────────────────────┐
                    │  检索质量评估          │
                    │  低质量 → 二次检索     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  上下文构建            │
                    │  SearchResult         │
                    │  → EvidenceChunk      │
                    │  → Citation           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Answer Generation    │
                    │  (DeepSeek 云)         │
                    │  带检索上下文 + 历史    │
                    │  流式输出 → Topic      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  引用提取 + 置信度评估  │
                    │  → AnswerResult       │
                    └──────────────────────┘
```

### 本地计算

- 预设回答匹配（`guard.py`）
- 证据块构建（`build_evidence_chunks`）
- 对话历史管理（`ConversationHistory`，内存中维护）
- 检索质量评估 + 二次检索触发

### 云 API 调用

- **DeepSeek Intent Router**：`deepseek-v4-flash`，意图分类
- **DeepSeek Answer Generation**：`deepseek-v4-flash`，带上下文 RAG 生成
- （检索部分通过调用 `rag_retrieval_node` 的 Search Service 完成）

### 与现有代码映射

```
generation/service.py       → AnswerGenerationService  →  Service 回调 + Topic 发布
generation/intent_router.py → DeepSeekIntentRouter     →  答案节点内调用
generation/deepseek_chat.py → DeepSeekAnswerClient     →  答案节点内调用
generation/guard.py         → direct_answer_for_query  →  答案节点内调用
generation/history.py       → ConversationHistory      →  答案节点内维护
generation/context.py       → build_evidence_chunks    →  答案节点内调用
```

---

## 节点 5：`rag_voice_node` — 语音交互节点（可选）

### 职责

语音识别（STT）和语音合成（TTS）。

### 生命周期

- **按需**：有语音请求时启动

### ROS2 接口

```yaml
# Service: SpeechToText
Request:
  audio_data: uint8[]       # WAV 16-bit PCM 音频数据
  format: string = "wav"
---
Response:
  text: string

# Service: TextToSpeech
Request:
  text: string              # 待合成文本（最大 2000 字符）
---
Response:
  audio_data: uint8[]       # MP3 音频数据
  format: string = "mp3"
```

### 实现

| 功能 | 技术 | 部署位置 |
|------|------|----------|
| ASR | Qwen3-ASR-Flash-Realtime（主）/ Whisper + FSMN-VAD（兜底） | 云端流式 / 本地 CPU |
| TTS | Qwen3-TTS-Flash-Realtime（主）/ Microsoft edge-tts（兜底） | 云端流式 |

### 与现有代码映射

```
voice/service.py → VoiceService → Service 回调
```

---

## 节点 6：`rag_api_node` — Web API 编排节点（可选）

### 职责

保留原有的 FastAPI Web UI，作为各节点的 ROS2 Client 编排者。

### 设计要点

- **不包含业务逻辑**：仅做 ROS2 Action / Service / Topic 的调用编排
- 异步作业（解析、索引）通过 Action feedback 实现进度轮询
- 流式回答通过订阅 `/answer_stream` Topic 实现 SSE 输出
- 如果不需要 Web UI，可完全省略，改用 ROS2 CLI 或 RQT 调用

### 与现有代码映射

```
web/app.py → 重构：移除内部业务逻辑，改为 ROS2 Client 调用
```

---

## 自定义 ROS2 接口定义

### 消息类型 (msg)

```
rag_interfaces/msg/
├── ParsedSection.msg       # text + metadata[]
├── DocumentChunk.msg       # id + text + metadata[]
├── ParsedTable.msg         # table_id + rows[] + columns[] + text + html
├── ParsedAsset.msg         # asset_id + kind + caption + page + bbox + path
├── ParsedDocument.msg      # source + source_name + backend + sections[] + chunks[] + tables[] + assets[] + diagnostics[] + cache_key + elapsed_seconds + metadata[]
├── SearchResult.msg        # id + text + score + metadata[] + explanation[]
├── Citation.msg            # id + source_name + chunk_id + heading + heading_path + page + score + rrf_score + rerank_score + retrieval_source + quote
├── EvidenceChunk.msg       # id + search_result_id + text + source_name + chunk_id + heading + score
├── AnswerDelta.msg         # type + text + query + model  (流式片段)
├── AnswerResult.msg        # query + answer + citations[] + evidence[] + model + confidence + retrieval_data
├── IndexedDocument.msg     # document_id + source + source_name + chunk_count + vector_count + model + dimensions + file_type + status + enabled
├── IndexStats.msg          # document_count + chunk_count + vector_count + collection + model + dimensions + bm25_count + sparse_count
└── RetrievalDiagnostic.msg # retrieval_source + rrf_score + rerank_score + matched_token_count + matched_token_ratio + routes[]
```

### 服务 (srv)

```
rag_interfaces/srv/
├── Search.srv              # query + top_k → results[] + rewrite + diagnostics
├── Answer.srv              # query + top_k → answer + citations[] + evidence[] + confidence + retrieval
├── SpeechToText.srv        # audio_data + format → text
├── TextToSpeech.srv        # text → audio_data + format
├── ListDocuments.srv       # filter → documents[]
├── GetDocumentDetail.srv   # document_id → document + chunks[] + images[] + counts
├── SetDocumentEnabled.srv  # document_id + enabled → success
├── DeleteDocument.srv      # document_id → success + deleted_counts
├── GetIndexStats.srv       # → stats
└── GetImageStats.srv       # → image_count + vector_count + model + dimensions
```

### 动作 (action)

```
rag_interfaces/action/
├── ParseDocument.action    # file_path + config → document + elapsed + diagnostics
├── IndexDocument.action    # document → success + document_id + chunk_count + vector_count
└── BatchParse.action       # file_paths[] + config → documents[] + success_count + failed_count
```

---

## 部署拓扑

### 方案 A：单机部署（推荐）

```
┌────────────────────────────────────────────────────────────┐
│  设备：Jetson Orin / x86 工控机                              │
│                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ rag_parser   │  │ rag_index    │  │ rag_voice    │     │
│  │ _node        │  │ _node        │  │ _node        │     │
│  │ (按需)        │  │ (常驻)        │  │ (可选)        │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │ rag_retrieval│  │ rag_answer   │                        │
│  │ _node        │  │ _node        │                        │
│  │ (常驻)        │  │ (常驻)        │                        │
│  └──────────────┘  └──────────────┘                        │
│                                                            │
│  所有节点共享 data/ 目录                                     │
│  ┌────────────────────────────────────────────┐           │
│  │  data/                                      │           │
│  │  ├── uploads/         (上传文件)             │           │
│  │  ├── parser_cache/    (解析缓存 JSON)        │           │
│  │  └── index/                                 │           │
│  │      ├── chroma/      (ChromaDB 向量库)      │           │
│  │      ├── bm25.sqlite3 (BM25 索引)           │           │
│  │      ├── sparse.sqlite3 (稀疏索引)           │           │
│  │      └── index.sqlite3  (文档清单)           │           │
│  └────────────────────────────────────────────┘           │
│                                                            │
│  Internet ──→ DashScope 百炼 API  ──→  Embedding / Rerank │
│  Internet ──→ DeepSeek API         ──→ Rewrite / Intent / Answer │
│  Internet ──→ Microsoft Edge TTS   ──→ TTS                │
└────────────────────────────────────────────────────────────┘
```

### 方案 B：双机分离部署

```
┌── 边缘设备（机器人本体）────┐     ┌── 远程服务器 ────────────┐
│                            │     │                          │
│  rag_parser_node           │     │  rag_api_node            │
│  rag_voice_node            │     │  (FastAPI Web UI)        │
│                            │     │                          │
└────────────────────────────┘     └──────────────────────────┘
        │ (ROS2 DDS)
        ▼
┌── 计算节点 ────────────────┐
│                            │
│  rag_index_node            │
│  rag_retrieval_node        │
│  rag_answer_node           │
│                            │
│  共享 data/ 目录            │
│  Cloud API 访问             │
└────────────────────────────┘
```

> **注意**：跨机器时，检索节点不能直接读取索引文件。需要索引节点通过 Service 暴露 `QueryVectors` 接口，将向量数据传输过来，这会带来网络开销。推荐方案 A 同机部署核心节点。

---

## 关键设计决策

### Q: 为什么检索节点不嵌在答案节点内？

- **独立能力**：机器人其他模块可能只需要检索结果，不需要 LLM 生成回答
- **不同调用频率**：检索频繁但轻量，生成低频但 token 消耗大
- **可独立扩展**：如果检索量很大，可以部署多个检索节点

### Q: 为什么索引节点分离读写？

- 写入（索引构建）是低频操作，读取（检索）是高频操作
- ChromaDB 不支持真正的多进程并发写
- **方案**：`rag_index_node` 是唯一写入者；`rag_retrieval_node` 直接读取 ChromaDB 文件（同机部署）
- 跨机器时：索引节点通过 Service 暴露查询接口

### Q: 为什么不把 Cloud API 调用做成独立节点？

- 当前 Cloud API 调用是无状态的请求-响应模式
- 包装成独立节点会增加 ROS2 通信开销，收益很低
- 如果未来需要 API 调用限流 / 缓存 / 队列化，可以再单独拆分

### Q: 配置（API Key 等）如何管理？

- 使用 ROS2 **Parameter** 系统
- 敏感值（API Key）通过 launch 文件从环境变量注入
- 非敏感配置（chunk_size、top_k 等）可通过 `ros2 param set` 动态调整

---

## 实施步骤

| 步骤 | 内容 |
|------|------|
| 1 | 创建 ROS2 package：`rag_interfaces`，定义所有 msg / srv / action |
| 2 | 搭建 5 个 Python ROS2 Node 骨架，每个包装对应的现有 Service 类 |
| 3 | 适配 `settings.py` → ROS2 Parameters（API Key 等改为 Parameter，通过 launch 文件注入） |
| 4 | 实现 Action / Service 回调：将现有方法调用封装到 ROS2 接口中 |
| 5 | 实现跨节点数据流：`/parsed_document` Topic 连接 parser → index |
| 6 | 编写 `bringup.launch.py`：配置节点启动顺序、参数和 remapping |
| 7 | 编写 `config/params.yaml`：默认参数文件（模型名、维度、切片参数等） |
| 8 | 端到端测试：文档上传 → 解析 → 索引 → 检索 → 流式回答 |

---

## Launch 文件示例结构

```python
# launch/rag_bringup.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node, LifecycleNode

def generate_launch_description():
    return LaunchDescription([
        # 常驻节点
        Node(
            package='rag_index',
            executable='index_node',
            name='rag_index_node',
            parameters=['config/params.yaml'],
        ),
        Node(
            package='rag_retrieval',
            executable='retrieval_node',
            name='rag_retrieval_node',
            parameters=['config/params.yaml'],
        ),
        Node(
            package='rag_answer',
            executable='answer_node',
            name='rag_answer_node',
            parameters=['config/params.yaml'],
        ),
        # 可选节点
        Node(
            package='rag_voice',
            executable='voice_node',
            name='rag_voice_node',
            parameters=['config/params.yaml'],
        ),
        Node(
            package='rag_api',
            executable='api_node',
            name='rag_api_node',
            parameters=['config/params.yaml'],
        ),
    ])
```

---

## 验证方式

### 1. 启动所有节点

```bash
ros2 launch rag_system bringup.launch.py
```

### 2. 检查节点状态

```bash
ros2 node list
# 期望输出: /rag_index_node, /rag_retrieval_node, /rag_answer_node, ...

ros2 service list
ros2 action list
```

### 3. 文档解析测试

```bash
ros2 action send_goal /parse_document rag_interfaces/action/ParseDocument \
  "{file_path: '/home/user/docs/test.pdf', backend: 'auto', chunk_size: 2400, chunk_overlap: 300}" \
  --feedback
```

### 4. 检查解析结果自动索引

```bash
ros2 service call /get_index_stats rag_interfaces/srv/GetStats
# 验证 document_count 是否增加
```

### 5. 检索测试

```bash
ros2 service call /search rag_interfaces/srv/Search \
  "{query: '文档主要讲了什么', top_k: 5}"
```

### 6. 流式回答测试

```bash
# 终端 1：监听流式输出
ros2 topic echo /answer_stream

# 终端 2：发送回答请求
ros2 service call /answer rag_interfaces/srv/Answer \
  "{query: '总结一下文档的关键内容', top_k: 4}"
```

### 7. 语音测试（可选）

```bash
ros2 service call /stt rag_interfaces/srv/SpeechToText \
  "{audio_data: [...], format: 'wav'}"
```

---

## 与现有代码的详细映射

| 现有文件 | 迁移至节点 | 变更 |
|---------|-----------|------|
| `parsing/service.py` | `rag_parser_node` | Action 回调中调用 `ParsingService` |
| `parsing/models.py` | `rag_interfaces/msg/` | `ParsedDocument` → 自定义 Message |
| `parsing/chunking.py` | `rag_parser_node` | 无变更，Action 回调内调用 |
| `parsing/cache.py` | `rag_parser_node` | 无变更，缓存路径改为 Parameter |
| `indexing/service.py` | `rag_index_node` + `rag_retrieval_node` | 拆分为读写两个节点 |
| `indexing/store.py` | `rag_index_node` + `rag_retrieval_node` | index_node 写，retrieval_node 读 |
| `indexing/embeddings.py` | `rag_index_node` | 无变更（云 API 调用） |
| `indexing/manifest.py` | `rag_index_node` | 无变更 |
| `indexing/image_service.py` | `rag_index_node` | 无变更 |
| `retrieval/query_rewrite.py` | `rag_retrieval_node` | 无变更（云 API 调用） |
| `retrieval/rerank.py` | `rag_retrieval_node` | 无变更（云 API 调用） |
| `generation/service.py` | `rag_answer_node` | 流式输出改为 Topic 发布 |
| `generation/intent_router.py` | `rag_answer_node` | 无变更（云 API 调用） |
| `generation/deepseek_chat.py` | `rag_answer_node` | 无变更（云 API 调用） |
| `generation/history.py` | `rag_answer_node` | 无变更 |
| `voice/service.py` | `rag_voice_node` | 无变更 |
| `web/app.py` | `rag_api_node` | 重构为 ROS2 Client 编排者 |
| `settings.py` | 各节点 Parameters | 改为 ROS2 Parameter 管理 |
