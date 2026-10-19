# TTS 语音合成模块

职责：把文字合成为 MP3，支持流式 Qwen TTS，并能校验实际使用的 Provider，避免回退后误报通过。

- 最小输入：一段非空文字。
- 最小输出：MP3 文件和单行结果 JSON。
- 网络要求：Qwen 和 Edge-TTS 都需要联网；下面的 Qwen 示例还需要配置 DashScope API Key。

在仓库根目录执行：

```powershell
./module-test.ps1 tts "绿色江河是一家环保组织" --stream --require-provider qwen --output ./data/acceptance-output/tts_qwen.mp3
```
