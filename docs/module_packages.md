# 独立模块与甲方测试入口

开发、自测和故障定位时，请同时参考[模块独立测试指南](module_testing_guide.md)。该指南包含 Linux/Windows 命令、逐模块 pytest、ROS2 测试边界和新增模块的测试要求。

## 交付边界

项目已经按职责拆为 7 个顶层模块，并提供统一测试入口。甲方测试某个模块时，无需启动 Web，也无需了解内部 Provider、模型类或索引类。

```text
rag_system/
  contracts/       公共音频与结果契约
  wakeup/          离线唤醒：WAV/PCM -> 唤醒事件
  vad/             离线切片：WAV/PCM -> 语音事件/切片 WAV
  asr/             本地或 Qwen Realtime：WAV -> 文本/流式事件
  tts/             Edge 或 Qwen：文本 -> MP3/流式 MP3 块
  rag_cloud/       写侧：解析、入库、删除
  rag_edge/        读侧：检索、文档列表、统计
  agent/           问答：完整回答/流式事件
  module_test/     面向甲方的统一命令分发器和环境自检
  time_probe/      按需开启的模块/函数耗时诊断和 JSON 报告
  voice/           完整 Web 语音编排与旧导入路径兼容层
  display          工控机显示壳、表情资源与 ROS2 显示模式控制
```

需要定位模块测试的端到端和函数耗时时，参考[本地时间探针](time_probe.md)。

当前完成的是源码包、导入边界、可选依赖和命令入口拆分，仍由一个 `rag-system` wheel 发布。以后需要端侧/云端分别版本发布时，可沿现有目录再拆独立 wheel，不需要重新设计模块 API。

## 甲方只需三步

在项目根目录打开 PowerShell：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[wakeup,vad,asr-local,asr-cloud,tts-edge,tts-cloud,rag-cloud,rag-edge,agent]"
.\module-test.ps1 doctor
.\module-test.ps1 list
```

`doctor` 是只读离线检查，不发送音频、文本或 API 请求，也不会输出密钥。语音样例位于 `samples/audio`，RAG 样例位于 `samples/acceptance`。

退出码约定：成功为 `0`，运行错误为 `1`；严格要求未命中（例如未唤醒、无语音、TTS Provider 不符）为 `2`。普通命令输出一个 JSON；流式命令输出 JSON Lines，每行均是完整 JSON。

## 可复制执行的模块测试

### 1. Wakeup（离线）

```powershell
.\module-test.ps1 wakeup --input .\samples\audio\wakeup_xiaolv.wav --wake-word 小绿小绿 --require-detected
```

预期：退出码 `0`，JSON 中 `ok=true`、`output.detected=true`。

### 2. VAD（离线，断句间隔 0.4 秒）

```powershell
.\module-test.ps1 vad --input .\samples\audio\question_green_river.wav --end-silence-ms 400 --output-wav .\data\acceptance\output\utterance.wav --require-speech
```

预期：`output.speech_detected=true`，并生成独立 WAV。可用 `silence_2s.wav` 验证静音不触发。

### 3. ASR

本地 CPU 模型：

```powershell
.\module-test.ps1 asr .\samples\audio\question_green_river.wav --provider local
```

Qwen ASR Realtime（真实流式输入和 JSONL 输出，需要 DashScope Key）：

```powershell
.\module-test.ps1 asr .\samples\audio\question_green_river.wav --provider qwen --stream
```

Qwen 固定样例的最终预期文本为“介绍一下绿色江河”。本地 base 模型只用于离线链路冒烟，不把该合成样例的逐字准确率作为验收结论。

### 4. TTS

Edge 在线女声：

```powershell
.\module-test.ps1 tts "绿色江河是一家环保组织" --output .\data\acceptance\output\answer.mp3
```

Qwen 流式 TTS（需要 DashScope Key，并禁止静默回退为 Edge）：

```powershell
.\module-test.ps1 tts "绿色江河是一家环保组织" --stream --require-provider qwen --output .\data\acceptance\output\answer-qwen.mp3
```

当前稳定文件契约是 MP3，命令会拒绝用 `.wav` 扩展名保存 MP3 数据。

### 5. RAG Cloud（写侧）

```powershell
.\module-test.ps1 rag-cloud parse .\samples\acceptance\green_river.txt
.\module-test.ps1 rag-cloud index .\samples\acceptance\green_river.txt
```

默认只写 `data/acceptance/index`，不会污染正式的 `data/index`。需要测试 DOCX 时，可将第一个参数替换为目标 DOCX；`rag-cloud` extra 已包含 Docling。

### 6. RAG Edge（读侧）

```powershell
.\module-test.ps1 rag-edge search "绿色江河是什么组织" --top-k 3
.\module-test.ps1 rag-edge stats
```

默认读取与 RAG Cloud 相同的 `data/acceptance/index`。只有明确添加 `--index-dir .\data\index` 时才会访问正式知识库。
默认结果只保留文本、分数和来源；开发排障时可添加 `--verbose` 查看完整排序明细。

### 7. Agent / LLM

```powershell
.\module-test.ps1 agent answer "介绍一下绿色江河"
.\module-test.ps1 agent stream "介绍一下绿色江河"
```

Agent 需要已配置的 LLM Key；`stream` 始终保持每个事件一行 JSON，即使传入 `--pretty` 也不会破坏 JSONL。默认读取 `data/acceptance/index`，输出只保留回答、模型、置信度和来源；添加 `--verbose` 才查看完整检索与路由诊断。

## 按模块安装

如需制作端侧或服务端精简环境，可以只安装对应 extra：

```powershell
pip install -e ".[wakeup]"
pip install -e ".[vad]"
pip install -e ".[asr-local]"
pip install -e ".[asr-cloud]"
pip install -e ".[tts-edge]"
pip install -e ".[tts-cloud]"
pip install -e ".[rag-cloud]"
pip install -e ".[rag-edge]"
pip install -e ".[agent]"
```

`voice` extra 继续保留，供完整 Web Demo 一次性安装。

## 验收分层

- `unit`：Fake Provider，不联网，验证接口和错误处理。
- `local_model`：真实 Wakeup、VAD、本地 ASR 模型和固定 WAV。
- `online`：真实 DashScope/LLM API，需要测试 Key。
- `acceptance`：双方确认的人声、距离、噪声样本及准确率/延迟报告。

仓库自带的语音是合成冒烟样例，只能证明模块可运行；真实识别率验收必须使用双方确认的录音集合。
