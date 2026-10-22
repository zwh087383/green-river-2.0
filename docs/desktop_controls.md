# 机器人桌面控制

机器人 `/home/nerv/Desktop` 上有两个 Linux 桌面启动器：

- `启动全部项目进程.desktop`：启动网页、向量服务、语音网关、硬件唤醒、显示节点、知识节点、原生机械臂 SDK 和表情屏。项目已完整运行时直接提示，重复双击不会再启动一套。SDK 冷启动会给双臂上电并归位。
- `停止全部项目进程.desktop`：由统一进程管理器停止当前部署及旧 Voice 部署的项目进程，清理遗留子进程；根据命令、工作目录、用户和进程启动时间确认身份。

结果显示在终端，按回车关闭窗口。启动后的后台服务与此终端分离，关闭结果窗口不会停止服务。启动和停止操作互斥，避免连续点击互相干扰。

入口是 `scripts/desktop_runtime.py start` 和 `scripts/desktop_runtime.py stop`，内部复用 `scripts/run_ipc_voice.sh` 和 `scripts/robot_runtime.py`。启动日志为 `data/runtime/desktop-runtime.log`，控制页为 `http://127.0.0.1:8001/`。VAD 与断句配置沿用当前设置。
