# 工控机 4.0 部署

- 工控机：`172.16.40.215`，SSH 别名 `nerv`，用户 `nerv`。
- 项目目录：`/home/nerv/green-river-4.0`；独立 `.venv-linux`、模型、配置、知识库和 ROS2 overlay。
- Ubuntu 22.04 / Python 3.10.12，Intel Core Ultra 5 125H / 16GB 内存；向量模型运行在 CPU。
- 原 `/home/nerv/Voice` 与厂商 SDK 保留。2026-10-19 18:04 已按用户要求启动独立 4.0 语音、表情与机械臂联动，见文末记录。

## 当前服务

工控机本地网页：`http://127.0.0.1:8001/`；EmbeddingGemma 2：`http://127.0.0.1:8765/health`。
服务均绑定工控机回环地址。本机已启动 SSH 转发，可访问：

<http://127.0.0.1:8002/>

8002 是开发机到工控机 8001 的 SSH 转发，不是开发机自身的 4.0 服务。
开发机本来的 8001 页面及 3.0/8000 服务保留。
转发关闭后可重新运行：

```bash
ssh -N -L 127.0.0.1:8002:127.0.0.1:8001 nerv
```

网页使用麦克风时，应在工控机本地 localhost 打开，或通过上述 localhost 转发打开。
访问转发页面时采集的是当前浏览器所在电脑的麦克风；机器人麦克风由下面的 ROS2 入口负责。

## 在工控机启停

```bash
cd /home/nerv/green-river-4.0
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false ./scripts/start_linux.sh
./scripts/stop_linux.sh
```

Web PID/日志：`data/runtime/web.pid`、`data/runtime/web.log`；向量服务：
`data/runtime/embeddinggemma2.pid`、`data/runtime/embeddinggemma2.log`。
停止网页不会停止独立向量服务，也不会停止旧 Voice。

ROS2 接口已单独构建在 `ros2-install/`，构建文件位于 `data/ros2-build/`。
实际需要接管机器人语音时，使用新入口：

```bash
cd /home/nerv/green-river-4.0
./scripts/run_ipc_voice.sh
```

该入口显式使用本项目的 Python、ROS2 overlay、数据和 8001 后端，避免旧入口默认 8000。
此命令会启动机器人语音网关、显示和动作等节点；首次部署仅执行过 `--show-args`；现已正式启动并接入机器人麦克风和机械臂控制话题。

## 依赖和数据

以旧环境的文件副本为起点，仅在新目录安装/升级。Torch 2.12.1+cu130 检测不到 CUDA，
使用 CPU；未为此次部署新增整套 CUDA 运行库。
Transformers 为与开发机相同的固定源码版本 `5.21.0.dev0`，
源码提交 `54c0fe02b61062fe2666add21744c27035549644`。
工控机 GitHub 下载缓慢，因此传入本机已构建的纯 Python wheel：

`3ce4c5a04acadd3c1983cceebb0e1b55dabade1dea5d165ae2070ae493f9f459`

安装需求和 wheel 保存在工控机的 `data/runtime/requirements-deploy.txt`、`data/runtime/wheels/`。
SentenceTransformers 6.1.0，`pip check` 无冲突。

已同步唤醒、VAD、Whisper、Docling 和 EmbeddingGemma 2 模型。
向量权重 SHA256：`197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79`。
知识库 SQLite 使用在线备份，副本中的开发机绝对路径已改为工控机路径；配置仅保存在工控机
本机文件中，`data/settings.json` 权限 600。未复制旧 Web PID 或开发机浏览器会话。

## 验证

- 工控机 Python：479 passed，9 skipped；前端 Node：11 passed。
- Python 3.10 的 profiling hook 会在 await 暂停时报告 YIELD_FROM 前一条指令，
  本次对 TimeProbe 做了兼容修正。开发机 Python 3.12 的 12 个计时测试也通过。
- 模型文件校验与 doctor：7/7 模块就绪。
- ROS2：`rag_interfaces`、`voice` 构建成功，`rclpy`/接口/新 voice overlay 导入成功。
- 知识检索：3 条结果，实测约 148ms。
- 真实云端问答：Qwen 判断正常，41 段 PCM、783360 字节，首段音频约 2.29 秒。
- 本次仅验证音频生成，机器人麦克风、扬声器与机械臂需启动硬件链路后实机确认。

工控机验证日志均位于 `data/runtime/deploy-*`。前次代码复查的 4 个待修问题仍见
`docs/reviews/2026-10-19-code-review.md`，本次未顺带修改这些行为。


## 2026-10-19 语音与动作正式启用

18:04 启动新的后台进程组，launcher PID 18591；网关、显示、动作协调、知识服务四个节点均在运行，使用本项目 `.venv-linux`。进程脱离 SSH 会话运行。

- 日志：`data/runtime/ipc-voice-motion.log`。
- 启动 PID：`data/runtime/ipc-voice-motion.pid`。
- 话题连接证据：`data/runtime/ipc-voice-motion-verification.json`。
- 麦克风：系统默认 USB HK MIC；PulseAudio 可见 16 kHz 单声道输入流。输入 16 kHz、输出 24 kHz PCM16 格式检查通过。
- 话题链路已确认：`realtime_voice_gateway` → `/voice/arm_motion_request` → `voice_arm_motion_node` → `/arm_control/motion` → 现有 `ArmControlNode`；机械臂状态反馈也已双向连接。
- 保留既有配置：唤醒动作码 4，回答动作码 5，取消/完成后的预置复位码 1。此次未人工发布强制动作测试指令；用户正常唤醒和提问会按配置触发。

启动时修复了两处接线问题：网关命令改为协调节点所需的 `play_motion`；启动脚本在 SDK 环境加载后设置 `ROS_LOCALHOST_ONLY=1`，与现有机械臂服务保持一致（需要联网 DDS 时可用 `RAG_ROS_LOCALHOST_ONLY=0` 覆盖）。实时 launch 同时明确使用 `RAG_SYSTEM_PYTHON`，避免节点导入旧 Voice 的 Python 包。

网关和动作协调相关测试：21 项通过。更改后的 voice overlay 已重建，底层 SDK 服务未重启，3.0 未修改。此前主模型工具改造时保存的源文件哈希清单属于启用动作前的快照；本次接线修正另存 `data/runtime/ipc-motion-source-sha256-20261019.json`。


## 清理重复小绿并只启动最新版

按用户要求关闭旧小绿浏览器窗口，发送 `/speech_interaction/listening=false` 停用 SDK 旧语音监听；停止并重新启动 4.0 Web、向量服务、语音联动与全屏显示。保留机械臂、底盘等底层驱动。

最新启动快照：Web PID 20057，向量服务 PID 19973，ROS launcher PID 20140，语音网关 PID 20325，全屏显示 PID 20151。实际运行 PID 以 `data/runtime/` 中 PID 文件为准。

验证结果：4 个小绿 ROS 节点各只有 1 个；小绿相关音频采集只剩最新版网关，旧 Chrome 麦克风输入流已消失；机械臂控制话题已连接，Web 健康。20 个最新源文件与开发机 SHA256 一致。证据：`data/runtime/clean-restart-verification-20261019.json` 和 `data/runtime/latest-ipc-source-sha256-20261019.json`。
