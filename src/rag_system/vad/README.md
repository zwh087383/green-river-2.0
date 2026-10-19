# VAD 语音切片模块

职责：使用端侧 FSMN VAD 检测语音起止点，并按配置的静音间隔结束一句话。

- 最小输入：WAV 音频和断句间隔 `end-silence-ms`。
- 最小输出：单行 JSON，包含语音事件和时长；可额外保存切出的 WAV。
- 网络要求：模型下载完成后全程离线，不需要 API Key。

在仓库根目录执行：

```powershell
./module-test.ps1 vad --input ./samples/audio/question_green_river.wav --end-silence-ms 400 --output-wav ./data/acceptance-output/question_utterance.wav --require-speech
```
