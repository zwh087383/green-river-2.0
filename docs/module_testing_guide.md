# 模块独立测试指南

本文说明如何在不启动完整 Web 服务和机器人语音链路的情况下，单独验证一个功能模块。适用于本地开发、自测、问题定位和交付验收。

## 1. 测试原则

模块测试分为四层：

| 层级 | 验证目标 | 是否联网 | 是否使用真实模型 |
| --- | --- | --- | --- |
| 单元测试 | 接口、分支、异常和模块边界 | 否 | 否，使用 Fake 或内存实现 |
| 本地模型测试 | Wakeup、VAD、本地 ASR 等真实推理 | 否 | 是 |
| 在线模块测试 | DashScope、DeepSeek、Edge TTS 等 Provider | 是 | 是 |
| 链路测试 | Web、ROS2、语音状态机和上下游协议 | 视配置而定 | 视配置而定 |

单独测试模块时遵循以下约束：

1. 必须从仓库根目录运行命令，避免相对路径读写到错误位置。
2. RAG 模块默认使用 `data/acceptance/index`，不要直接操作正式的 `data/index`。
3. 单元测试不应依赖真实 API Key、网络、共享 `data/` 或已经运行的 Web 服务。
4. 真实在线测试只验证模块调用和协议，不替代准确率、音质和延迟验收。
5. 测试失败时先保留模块输出的 JSON、退出码和输入样本，再排查完整链路。

## 2. 环境准备

### Linux

项目已安装完整 Linux 环境时，先执行离线检查：

```bash
./scripts/run_linux.sh doctor
```

列出可独立运行的模块：

```bash
.venv-linux/bin/python -m rag_system.module_test list
```

运行完整测试套件：

```bash
./scripts/run_linux.sh test
```

`run_linux.sh test` 会禁用系统环境中自动发现的 pytest 插件，避免 ROS2 的 `launch_testing` 等外部插件干扰项目测试。

### Windows

```powershell
.\.venv\Scripts\python.exe -m rag_system.module_test doctor
.\.venv\Scripts\python.exe -m rag_system.module_test list
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

`doctor` 是只读离线检查。它检查 Python、依赖、本地模型、测试样本和 Key 是否已配置，但不会发起 API 请求，也不会输出 Key 明文。

## 3. 统一模块命令

所有模块都可以通过统一入口调用：

```bash
.venv-linux/bin/python -m rag_system.module_test <module> [arguments]
```

查看单个模块参数：

```bash
.venv-linux/bin/python -m rag_system.module_test wakeup --help
.venv-linux/bin/python -m rag_system.module_test rag-cloud --help
```

当前提供 7 个独立入口：

| 模块 | 输入 | 输出 | 网络要求 |
| --- | --- | --- | --- |
| `wakeup` | 16 kHz 单声道 PCM16 WAV | 唤醒检测 JSON | 离线 |
| `vad` | 16 kHz 单声道 PCM16 WAV | 语音片段 JSON/WAV | 离线 |
| `asr` | PCM WAV | 识别文本或 JSONL 事件 | 本地或在线 |
| `tts` | 文本 | MP3 和结果 JSON | 在线 |
| `rag-cloud` | 文档或文档 ID | 解析、入库、删除结果 | 视操作而定 |
| `rag-edge` | 查询或索引目录 | 检索、文档、统计结果 | 默认需要查询 Embedding |
| `agent` | 用户问题 | 完整回答或 JSONL 事件 | 在线 |

退出码约定：

| 退出码 | 含义 |
| --- | --- |
| `0` | 模块执行成功，且满足严格验收条件 |
| `1` | 参数、依赖、配置、模型或运行错误 |
| `2` | 模块运行完成，但未满足严格条件，例如没有检测到唤醒词、语音或指定 TTS Provider |

普通命令输出一个完整 JSON。流式 ASR 和 Agent 输出 JSON Lines，每一行都是一个可独立解析的 JSON 对象。

## 4. 逐模块运行测试

以下命令均从仓库根目录执行。

### 4.1 Wakeup

```bash
.venv-linux/bin/python -m rag_system.module_test wakeup \
  --input samples/audio/wakeup_xiaolv.wav \
  --wake-word 小绿小绿 \
  --require-detected
```

预期结果：

- 退出码为 `0`；
- JSON 中 `ok=true`；
- `output.detected=true`；
- 输出中包含检测到的关键词和时间信息。

只运行 Wakeup 单元测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-linux/bin/python -m pytest -q \
  tests/test_wakeup.py \
  tests/test_wakeup_models.py \
  tests/test_wakeup_package.py
```

### 4.2 VAD

```bash
mkdir -p data/acceptance/output

.venv-linux/bin/python -m rag_system.module_test vad \
  --input samples/audio/question_green_river.wav \
  --end-silence-ms 400 \
  --output-wav data/acceptance/output/utterance.wav \
  --require-speech
```

预期结果：`output.speech_detected=true`，并生成 `utterance.wav`。

静音反例：

```bash
.venv-linux/bin/python -m rag_system.module_test vad \
  --input samples/audio/silence_2s.wav
```

只运行 VAD 单元测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-linux/bin/python -m pytest -q \
  tests/test_vad.py \
  tests/test_vad_models.py \
  tests/test_vad_package.py
```

### 4.3 ASR

本地 ASR：

```bash
.venv-linux/bin/python -m rag_system.module_test asr \
  samples/audio/question_green_river.wav \
  --provider local
```

Qwen Realtime ASR：

```bash
.venv-linux/bin/python -m rag_system.module_test asr \
  samples/audio/question_green_river.wav \
  --provider qwen \
  --stream
```

Qwen 测试需要有效的 `DASHSCOPE_API_KEY`，或者已经在 `data/settings.json` 中保存 DashScope Key。`--stream` 只能和 `--provider qwen` 一起使用。

只运行 ASR 单元测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-linux/bin/python -m pytest -q \
  tests/test_asr_package.py \
  tests/test_dashscope_asr.py \
  tests/test_glossary.py
```

### 4.4 TTS

普通合成：

```bash
.venv-linux/bin/python -m rag_system.module_test tts \
  "绿色江河是一家环保组织" \
  --output data/acceptance/output/answer.mp3
```

严格验证 Qwen 流式 TTS：

```bash
.venv-linux/bin/python -m rag_system.module_test tts \
  "绿色江河是一家环保组织" \
  --stream \
  --require-provider qwen \
  --output data/acceptance/output/answer-qwen.mp3
```

预期结果：文件存在且 `bytes` 大于 `0`。指定 `--require-provider qwen` 后，如果实际回退到 Edge，命令返回退出码 `2`。

只运行 TTS 单元测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-linux/bin/python -m pytest -q \
  tests/test_tts_package.py \
  tests/test_dashscope_tts.py
```

### 4.5 RAG Cloud

只解析文档：

```bash
.venv-linux/bin/python -m rag_system.module_test rag-cloud \
  parse samples/acceptance/green_river.txt
```

解析并写入隔离索引：

```bash
.venv-linux/bin/python -m rag_system.module_test rag-cloud \
  index samples/acceptance/green_river.txt \
  --index-dir data/acceptance/index
```

删除测试文档：

```bash
.venv-linux/bin/python -m rag_system.module_test rag-cloud \
  delete <document_id> \
  --index-dir data/acceptance/index
```

`document_id` 从 `index` 命令返回的 JSON 中取得。需要保留 manifest 记录时添加 `--keep-manifest`。

只运行写侧单元测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-linux/bin/python -m pytest -q \
  tests/test_rag_cloud_package.py \
  tests/test_parsing.py \
  tests/test_model_assets.py
```

### 4.6 RAG Edge

执行检索：

```bash
.venv-linux/bin/python -m rag_system.module_test rag-edge \
  search "绿色江河是什么组织" \
  --top-k 3 \
  --index-dir data/acceptance/index
```

查看文档和索引统计：

```bash
.venv-linux/bin/python -m rag_system.module_test rag-edge \
  list --index-dir data/acceptance/index

.venv-linux/bin/python -m rag_system.module_test rag-edge \
  stats --index-dir data/acceptance/index
```

需要查看混合检索、改写、重排等内部信息时，在 `search` 命令中增加 `--verbose`。

只运行读侧单元测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-linux/bin/python -m pytest -q \
  tests/test_rag_edge_package.py \
  tests/test_indexing.py \
  tests/test_store.py \
  tests/test_rerank.py
```

### 4.7 Agent

测试完整回答：

```bash
.venv-linux/bin/python -m rag_system.module_test agent \
  answer "介绍一下绿色江河" \
  --index-dir data/acceptance/index
```

测试流式事件：

```bash
.venv-linux/bin/python -m rag_system.module_test agent \
  stream "介绍一下绿色江河" \
  --index-dir data/acceptance/index
```

当前 Agent 还负责判断输入属于普通问题、无意义内容还是明确结束语。流式测试应分别覆盖：

| 输入类别 | 示例 | 预期行为 |
| --- | --- | --- |
| 普通问题 | `绿色江河主要做什么` | 输出 `meta/delta/final` 回答事件 |
| 无意义内容 | `嗯`、`啊` | 输出忽略信号，不检索、不生成回答 |
| 明确结束语 | `没有别的问题了` | 跳过检索，流式生成结束语并结束会话 |

需要完整路由和检索诊断时添加 `--verbose`。

只运行 Agent 和生成单元测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-linux/bin/python -m pytest -q \
  tests/test_agent_package.py \
  tests/test_generation.py
```

## 5. 内部子系统独立测试

`parsing`、`indexing`、`retrieval`、`generation`、`voice`、`web` 不是统一测试目录中的 7 个顶层验收命令，但都可以单独运行 pytest。

| 子系统 | 推荐测试命令 |
| --- | --- |
| 文档解析 | `pytest tests/test_parsing.py tests/test_model_assets.py` |
| 文本索引与检索 | `pytest tests/test_indexing.py tests/test_store.py tests/test_rerank.py` |
| 图片索引 | `pytest tests/test_image_indexing.py` |
| 回答生成 | `pytest tests/test_generation.py` |
| 语音服务 | `pytest tests/test_voice.py tests/test_dashscope_asr.py tests/test_dashscope_tts.py` |
| 配置 | `pytest tests/test_settings.py` |
| Web API/WebSocket | `pytest tests/test_web_app.py` |
| 公共契约 | `pytest tests/test_contracts.py` |
| 模块入口与边界 | `pytest tests/test_module_test_cli.py tests/test_module_import_boundaries.py` |

Linux 示例：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-linux/bin/python -m pytest -q \
  tests/test_generation.py
```

## 6. ROS2 模块测试边界

当前 ROS2 默认测试包括：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-linux/bin/python -m pytest -q \
  tests/test_ros2_package_layout.py \
  tests/test_robot_audio_pipeline.py
```

这两组测试可以验证：

- ROS2 package、msg、srv、action 文件布局；
- Python 节点源码可以编译；
- ROS2 入口能够转发到完整 Python 环境；
- 本地音频状态机的唤醒、聆听、处理中、播放和复位行为。

它们目前不能替代真实 ROS2 节点测试。`answer_node`、`knowledge_node`、`bridge_node` 和 `audio_input_node` 的 topic、service、action、QoS、生命周期及多节点联调仍需在安装了 ROS2 Jazzy 的环境中单独验证。

## 7. 可选真实解析集成测试

默认测试会跳过需要真实重型解析后端和样本的集成测试。明确需要验证时执行：

```bash
RUN_PARSER_INTEGRATION=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.venv-linux/bin/python -m pytest -q tests/integration
```

这些测试可能加载 Docling、OCR 或其他重型后端，耗时和资源占用明显高于普通单元测试。

## 8. 判断测试是否通过

一次模块测试至少记录以下信息：

```text
模块名称：
输入样本：
命令：
退出码：
ok 字段：
Provider/模型：
输出文件或 JSON：
耗时：
结论：通过 / 不通过 / 环境阻塞
```

不要只根据“命令没有报错”判断通过。还应检查：

- 输出 JSON 的 `ok` 是否为 `true`；
- 实际 Provider 是否符合预期；
- 流式输出是否包含最终事件；
- 输出音频或索引文件是否非空；
- 测试是否意外读取或修改了正式 `data/index`；
- 无意义内容和结束语是否跳过了普通检索回答链路。

## 9. 常见问题

### pytest 启动时提示 `No module named lark`

这通常是系统 ROS2 pytest 插件被自动加载，而不是项目代码缺少依赖。使用项目脚本：

```bash
./scripts/run_linux.sh test
```

只运行某个测试文件时禁用插件自动加载：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-linux/bin/python -m pytest -q tests/test_generation.py
```

### 在线模块提示缺少 API Key

先执行：

```bash
./scripts/run_linux.sh doctor
```

确认输出中的 `dashscope_api_key`、`deepseek_api_key` 和对应模块 `ready` 状态。不要把真实 Key 写入测试代码、提交记录或测试报告。

### RAG Edge 搜不到刚入库的文档

确认 RAG Cloud 和 RAG Edge 使用了完全相同的 `--index-dir`。推荐测试目录统一使用：

```text
data/acceptance/index
```

### TTS 命令成功但 Provider 不符合预期

使用 `--require-provider qwen` 或 `--require-provider edge`。Provider 不匹配时命令返回退出码 `2`，可避免把降级调用误判为目标 Provider 测试通过。

### 本地语音样本通过但现场效果不好

仓库样本用于功能冒烟，不能代表真实距离、噪声、口音和设备条件。现场验收需要固定麦克风、距离、噪声等级和录音集合，并单独统计识别率、误唤醒率、漏唤醒率和端到端延迟。

## 10. 新增模块时的测试要求

新增可独立运行模块时，至少完成以下内容：

1. 在 `src/rag_system/<module>/` 中提供 `node.py`、`cli.py`、`__main__.py` 和公开 `__init__.py`。
2. Node 只依赖公共契约或注入的 Provider，不在导入模块时加载模型、连接网络或创建全局服务。
3. CLI 对成功和失败都输出结构化 JSON；流式协议使用 JSON Lines。
4. 在 `src/rag_system/module_test/catalog.py` 注册模块。
5. 在 `pyproject.toml` 增加命令入口和独立 optional dependency extra。
6. 在 `doctor.py` 增加只读依赖、模型和配置检查。
7. 增加 `tests/test_<module>_package.py`，覆盖正常输入、空输入、Provider 异常和输出协议。
8. 在 `tests/test_module_import_boundaries.py` 增加导入边界断言。
9. 使用 Fake Provider 完成单元测试，再使用固定样本执行本地模型或在线冒烟测试。
10. 文档中写明输入格式、输出格式、网络要求、退出码和不能覆盖的验收边界。

## 11. 使用时间探针

需要查看一次模块测试中每个函数和模块的耗时时，用独立探针包装原有命令：

```bash
.venv-linux/bin/python -m rag_system.time_probe \
  --output data/runtime/time_probe/module-test.json \
  -- rag_system.module_test rag-cloud parse samples/acceptance/green_river.txt
```

原有命令的 stdout 不会被探针混入，所以仍可以用 `jq` 或自动化脚本解析模块 JSON。探针报告包含 `summary`、`modules`、`functions` 和本次未执行到的 `inventory.unobserved_modules`；完整字段说明见 [本地时间探针](time_probe.md)。

相关设计与交付说明：

- [独立模块与甲方测试入口](module_packages.md)
- [代码架构总览](code_architecture_overview.md)
- [ROS2 节点设计](ros2_node_design.md)
- [ROS2 Linux 部署](ros2_linux_deployment.md)
