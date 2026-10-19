# ASR 语音识别模块

职责：将 WAV 音频识别为文字，支持本地批量识别和 Qwen Realtime 流式识别，并沿用简繁转换与易错字纠错配置。

- 最小输入：WAV 音频。
- 最小输出：本地模式输出单行 JSON；Qwen `--stream` 模式逐行输出 partial、completed 和 summary JSON 事件。
- 网络要求：本地模型可离线运行；下面的 Qwen 示例需要联网并配置 DashScope API Key。

在仓库根目录执行：

```powershell
./module-test.ps1 asr ./samples/audio/question_green_river.wav --provider qwen --stream
```
