# 语音模块固定样例

这里的 WAV 均为单声道、16 kHz、PCM16，可直接用于甲方单模块冒烟测试：

- `wakeup_xiaolv.wav`：唤醒词“小绿小绿”。
- `question_green_river.wav`：问题“介绍一下绿色江河”。
- `wakeup_then_question.wav`：唤醒词、1 秒静音、问题。
- `silence_2s.wav`：2 秒静音，用于验证 VAD 不误触发。

这些音频由 TTS 合成，只用于验证接口、模型加载和数据链路是否正常，不能代表真实环境的识别准确率。准确率验收应另用双方确认的人声、距离和噪声样本。
