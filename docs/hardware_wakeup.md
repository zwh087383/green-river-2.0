# 机器人麦克风硬件唤醒

机器人默认由麦克风串口上报唤醒，复用现有 WebSocket、ASR、回答、播放器与打断处理。
本次不调整回声消除、音频参考接线或机械臂动作。控制页检测到机器人使用硬件唤醒后，
默认关闭浏览器自动监听，避免再启动第二路语音。用户仍可显式开启浏览器自己的软件唤醒，
该会话不会接收机器人的串口事件。

## 实机接口

- 音频设备：iflytek XFM-DP-V0.0.18，USB 输入 16 kHz、单声道 PCM16。
- 串口：`/dev/serial/by-id/usb-WCH.CN_USB_Single_Serial_0004-if00`，115200 8N1。
- 实测版本：`circle6:v1.0.2 engine:se_circle.1004; build date: Thu Jun 16 14:29:55 CST 2022`。
- 用户确认设备现有唤醒词为“小绿小绿”。尝试通过配置接口写入相同词时设备返回
  code 107 / Change keyword failed。本部署保留既有词，不刷固件，不在启动或重连时重新配置。
- v1.0.2 的版本信息和业务回执会出现在 0x04，不能将全部 0x04 当成唤醒。
  唤醒须为 `type=aiui_event`，旧固件还须 `content.eventType=4`，角度来自 `info.ivw`。
  同时兼容新版文档直接提供 beam/angle/confidence 的事件。

## 运行方式

`scripts/run_ipc_voice.sh` 默认导出 `RAG_ROBOT_WAKE_SOURCE=hardware`。
`voice_realtime.launch.py` 将 wake_source、wake_serial_port、wake_keyword 传入网关。
串口适配器是网关内独立线程，只用 Python 标准库，不需要启动厂商的在线识别/TTS 程序。

网关进程的 PULSE_SOURCE 固定到新麦克风的稳定 PulseAudio 源名称，保持现有扬声器输出。
可用 RAG_ROBOT_PULSE_SOURCE 覆盖，或通过 input_device 参数指定 PortAudio 设备。
该环境设置只影响网关，不更改系统默认声卡。

回退启动：

```bash
cd /home/nerv/green-river-4.0
./scripts/run_ipc_voice.sh wake_source:=software
```

其他串口可以传 `wake_serial_port:=/dev/serial/by-id/...`，或设置 RAG_ROBOT_WAKE_PORT。
硬件不可用时明确显示故障并重连串口，不静默启用第二套软件检测器。

## 协议与状态机

- 校验长度、帧头、校验和，支持分包、粘包与噪声重同步；串口通过 flock/TIOCEXCL 独占。
- 收到 0x01 时按原消息 ID 回复 0xFF 握手 ACK；运行期间只发送版本查询与握手 ACK。
- 设备消息去重并冷却；进入网关时只允许 armed/answer 阶段上报。
  提示音、收音、断线阶段的事件丢弃，不排队。
- 每条 WebSocket 有独立 connection_id 和单调事件 sequence，防止重复与跨连接旧事件。
- 硬件连接不创建软件唤醒模型会话。PCM 在待机/回答阶段丢弃，不送 ASR。
- 待机硬件事件产生原 wake_detected；回答阶段事件产生原 reason=interrupt，调用既有
  generation/cancel、HTTP close、PCM abort 和“我在”提示，提示结束后再开新 ASR。
- 提示音及 ASR 阶段忽略重复硬件控制消息；保持原有 continue/resume 连续对话规则。
- 硬件事件本身不直接发布手臂指令，沿用既有唤醒手势及打断只使旧动作请求失效的规则。

## 诊断与验证

- `GET /api/robot/voice/wakeup`：串口连接、固件版本、最后唤醒时间、唤醒计数、网关阶段。
- 主页面机械臂操作区显示“机器人唤醒：麦克风硬件 · 小绿小绿 · 已连接”；状态过期会显示未就绪。
- 原始状态：`data/runtime/hardware-wake.json`，由网关定期更新。
- 只读版本检查：在网关未占用串口时执行
  `.venv-linux/bin/python -m rag_system.hardware_wakeup version`。
  CLI configure 是显式持久配置操作，不属于启动流程。
- 自动测试覆盖 PTY 串口握手/版本/事件/去重/断线，坏帧与分包，回执不触发唤醒，
  WebSocket 硬件正常唤醒/打断/ASR隔离/重放，以及既有网关停播取消行为。
- 测试不自动播放唤醒词、不伪造真实语音到机器人动作链路；声音触发和未接回声参考时
  的误唤醒表现需现场验证。

## 结束对话后的状态显示

机器人网关在收到 armed 时将表情屏切回 idle，并显示待唤醒；收到 listening 时显示正在聆听。控制页硬件唤醒状态同步展示这两种状态，避免上一轮识别文字或回复在结束后造成仍在聆听的误解。结束语仍由主模型 end_conversation 工具判断，告别 PCM 播放完成后才 resume 回到待唤醒。网关日志记录 phase 和播完后的 end_session/下一状态，便于区分识别、判断和界面问题。
