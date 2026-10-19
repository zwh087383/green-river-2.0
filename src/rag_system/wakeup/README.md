# Wakeup 唤醒词模块

职责：使用端侧离线 KWS 模型检测自定义中文唤醒词；每次测试只处理一个 WAV 文件。

- 最小输入：WAV 音频（会统一转换为 16 kHz、单声道、PCM16）和唤醒词。
- 最小输出：单行 JSON，包含是否命中、命中的关键词和处理时长；`--require-detected` 未命中时返回退出码 `2`。
- 网络要求：模型下载完成后全程离线，不需要 API Key。

在仓库根目录执行：

```powershell
./module-test.ps1 wakeup --input ./samples/audio/wakeup_xiaolv.wav --wake-word 小绿小绿 --require-detected
```
