# 4.0 合并版运行说明

所有合并、环境安装、模型和知识库迁移均在本 4.0 目录进行。3.0 的源码、环境、
原索引和 8000 端口服务保留；4.0 使用独立 `.venv-linux`、`data/` 和 8001 端口。

## 语音与表情

- 语音保留 4.0：本地唤醒 → 云端流式 ASR → 本地 EmbeddingGemma 2/Chroma/BM25 →
  Qwen Audio 3.1 同时返回文本和 24kHz PCM。默认回复不请求独立 TTS。
- PCM 的生成取消、缓冲清理、播放结束等待和独立对话历史继续使用 4.0 实现。
- 合入八种动态内容情绪：开心、关切、难过、惊讶、疑惑、害羞、自豪、鼓励。
  聆听、思考和错误优先显示；内容情绪后台分析，不阻塞文字和 PCM 播放。
- 模型超时或未配置密钥时使用本地规则。ROS DisplayNode 同样使用这组本地规则，
  不修改机器人语音网关或设备 drain 时序；机器人麦克风/扬声器仍需实机验收。
- 普通回复正文默认约 100 字（80—120 字），明确要求详细时可展开。

网页：<http://127.0.0.1:8001/>，动态表情：<http://127.0.0.1:8001/display/preview>，
专用屏：<http://127.0.0.1:8001/display>。副屏现使用独立的 4.0 浏览器窗口连接 8001；
3.0 服务继续保留。

## 安装与模型

```bash
bash scripts/install_all_linux.sh
.venv-linux/bin/python -m rag_system.indexing.embedding_server --download
.venv-linux/bin/python -m rag_system.model_setup verify --component all
```

新增 `embedding-local` extra，包含 SentenceTransformers >=6.1。
EmbeddingGemma 2 在当前 Transformers 5.17 稳定版中尚未注册，因此依赖固定到
Hugging Face 官方源码提交 `54c0fe02b61062fe2666add21744c27035549644`，避免每次安装跟随 main。
模型也固定到 `914f7f89142e33e77833254d9c9b90c3cef7303b`。

本地模型默认位于 `data/models/embeddinggemma-2`；原模型文件约 1.49GB。
只加载文本编码器，支持 CUDA 时使用 bfloat16，否则使用 float32；不使用 float16。
推理遵循官方 SearchQuery / Document 前缀、768 维和单位向量契约。
参考：[官方模型说明](https://huggingface.co/google/embeddinggemma-2)、
[官方运行建议](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2)。

## 启停

```bash
./scripts/start_linux.sh       # 先启动/复用向量服务，再启动 8001 网页
./scripts/stop_linux.sh        # 仅停止本目录记录的 Web PID
```

`RAG_SYSTEM_PORT` 可覆盖 8001；`EMBEDDINGGEMMA_MODEL_DIR` 可指定模型权重目录。
自定义 `embedding_url` 时启动器不启动本地模型，由调用方提供接口。
向量服务独立运行，Web 停止不会结束它；PID 和日志在 `data/runtime/embeddinggemma2.*`。
设置了 `RAG_SYSTEM_PORT` 后启动 ROS 网关时，`backend_url` 应指定相同端口。
不要同时启用网页与机器人两套麦克风采集。

## 知识库

本次先复制 3.0 原索引、上传文档与解析缓存到 4.0，再在副本内重新嵌入，
生成 `data/index/embeddinggemma2_768_v1`，保留分块 ID、文字和文档启用状态。
迁移范围为 1 个文档、3 个分块。回执在 `data/runtime/knowledge-migration.json`。
4.0 中保留旧向量快照，3.0 原文件不参与写入。

## 验证

```bash
.venv-linux/bin/python -m pip check
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-linux/bin/python -m pytest -q
node --test tests/browser/display_emotion_client.test.cjs
uv run --no-project --with playwright python tests/browser/verify_direct_voice_emotions.py
uv run --no-project --with playwright --with pillow python tests/browser/verify_tone_animations.py --url http://127.0.0.1:8001
```

浏览器验证使用受控回复/静音 PCM，不调用云端生成；另有一次正式 HTTP 云端回复
验收，统计保存在 `data/runtime/live-direct-voice-smoke.json`。3.0 源码校验清单保存在
`data/runtime/source-3.0-sha256.json`，未将真实密钥或运行数据纳入源码。

## 关键词表情增强

4.0 的 `emotion_rules.py` / `web/assets/display_emotion.js` 改编自 Project N.E.K.O.
提交 `30c6d7556d0a5f04ce30dd0b2aaec75e39f387de` 的关键词计分与否定过滤逻辑；
Apache-2.0 许可、原 NOTICE 保留在 `licenses/neko/`。没有引入其 Live2D 或其他运行依赖。

- 词库统一位于 `src/rag_system/web/assets/emotion_rules.json`，Python/ROS 与网页共用。
  开心、难过、惊讶词汇参考原项目，并补充关切、疑惑、害羞、自豪和鼓励规则。
- 每次回复文字增量都先在浏览器本地匹配，不等句号或模型请求；整句模型分析仍异步，
  每轮最多一次预判、一次终稿。强度不足的中性结果不覆盖明确关键词，迟到的旧句子
  分析不能覆盖新语句中的情绪。新问题、打断、错误会取消旧轮次结果。
- 同一关键词与其长短变体不重复加分；否定词只影响当前分句，转折切断否定作用域，
  `不开心`、`高兴不起来` 不触发开心，`不仅开心` 保留正向含义。
- 直接夸奖小绿时预置害羞、感谢时预置开心；不将用户的难过直接当作小绿的难过。
  纯事实仍可显示中性，不随机轮播情绪。
- 播放完成后保留约 3 秒。自动恢复麦克风聆听不立即抹掉表情，ASR 的 `speech_start`
  或非空实时转写会立即切回聆听。此保留时间只影响显示，不延迟录音、ASR 或 PCM。
- 端口仍为 8001；本次调整不修改 3.0 的源码、依赖或服务。

## Qwen 判断结束对话

结束意图现由 `generation/turn_control.py` 的 `QwenTurnController` 判断，使用百炼
`router_model`（本机为 `qwen3.7-flash`）与最近 3 轮上下文。只输出结构化布尔值和
置信度，不恢复旧身份识别、意图路由或改写流程；普通嗯啊语气词仍在本地过滤。

- 移除了本地精确匹配告别词表；“没事儿了”“好的先这样”等由模型理解。
- 若同句话还有新的问题/请求、否定结束或只在讨论告别词，则继续；单纯谢谢不强制结束。
- 只有布尔 `end_session=true` 且置信度至少 0.8 才结束；缺少密钥、网络失败、超时、
  非法输出或低置信度均保守继续，不回退到告别词匹配。
- 单次网络请求超时 1.5 秒、不重试。Qwen Audio 连接提前建立，与判断重叠；这仍会
  增加一段语义判断等待，本机 7 个真实样例约 0.8—1.3 秒。
- 命中后跳过知识检索，Qwen Audio 只生成简短道别。`meta.retrieval.reason=end_session`
  与最终结果都携带结束信号，网页/ROS 仍等待 PCM 播放完成才回到待唤醒。
- “LLM 实际输入”新增“Qwen 结束对话判断”；`retrieval.turn_control` 包含模型名、
  置信度、来源、耗时，便于排查。模型成功完成整轮道别后才清空对话历史；打断不清空。

验证：`tests/test_turn_control.py` 覆盖请求、解析、失败、取消；直接语音服务测试覆盖
上下文、检索绕过、历史清理、预热连接取消。真实模型样例保存在本机
`data/runtime/qwen-end-session-smoke.json`，不会写入源码或 3.0。
