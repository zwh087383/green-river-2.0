# 连续随机回答动作与播放结束收尾：交付记录

日期：2026-10-22。已部署到 `nerv:/home/nerv/green-river-4.0`，运行版本切换及一次有界实机收尾测试均完成。

## 生效行为

- 普通回答从首块 PCM 写入后启动伴随动作循环；每个动作完成、归位验证通过且声音仍在播放时，再随机选下一个。候选仍为现有回答动作池（演讲、摊手、举手、手指远方、亮肌肉，以及按原规则符合条件的自定义动作）；候选多于一个时避免连续重复。没有固定“一轮只做一个”的限制。
- 以音频设备 `output.stop()` 排空为正常结束依据。正常播完、失败、打断、被新轮替代和客户端销毁都会关闭对应作用域，不阻塞语音线程等待机械臂归位。
- 等待双臂空闲时，只要播放已结束就退出，不会等唤醒动作结束后补做过期回答动作。唤醒挥手也改走作用域管理，在进入回答的新 generation 时开始安全收尾。
- 已发送的动作完成当前已获准执行的双臂分步，再省略后续表演段。退出时回读并保存已走过的关节关键点，沿这些点反向回退、执行既有归位步骤并验证反馈；若已经在起始位，不会反向重走表演路径。
- 新轮可以先播放语音，等旧轮收尾、双臂重新 ready 后再启动自己的动作；旧轮结束不会关闭新轮，手动动作、归位及电源作业不属于播放作用域。
- SDK/驱动故障仍优先并锁定，不把取消当故障，也不会因取消自动断电、重启或重发动作。

告别动作只执行一次：结束语正常播完后允许动作完整执行并归位，不因音频排空触发收尾；打断、失败、新轮替代及故障仍按原机制处理。结束会话标记较晚到达时，当前动作也允许做完，但不再循环追加。连续随机循环用于普通回答。

2026-10-22 后续修订：本机与 nerv 的相关回归各 56 项通过；仅重载语音节点，SDK 宿主进程和实例保持不变，没有再次触发动作或上电归位。部署记录见 [结束语动作修订记录](/home/xue/.codex/visualizations/2026/10/22/01a16158-54d0-7281-953d-fef4d700eedf/closing-completion-deployment.json)。

## 协议与文件

`OPEN <instance>` 分配当前服务实例内单调递增且不复用的 scope；`KEEP <scope> <instance>` 续期；`END <scope> <instance>` 幂等关闭；`GESTURE <token> <preset> <scope> <instance>` 只接受仍有效的 scope。活动 scope 上限 64，租期 10 秒，客户端每秒续期；失联后在租期失效的下一个分步边界安全收尾。关闭的编号无法被 KEEP 或迟到 GESTURE 重新激活，服务重启通过 instance ID 隔离。动作 ACK 丢失只查询结果，不重放动作。

| 文件 | 内容 |
|---|---|
| [robot_motion.py](/home/xue/.codex/workspaces/green-river-4.0-nerv-motion-20261022/src/rag_system/robot_motion.py:179) | 播放作用域、循环随机调度、结束通知、结果观察 |
| [realtime_gateway_node.py](/home/xue/.codex/workspaces/green-river-4.0-nerv-motion-20261022/ros2/voice/voice/realtime_gateway_node.py:480) | 对接首块 PCM、实际排空、失败及打断 |
| [sdk_motion_worker.cpp](/home/xue/.codex/workspaces/green-river-4.0-nerv-motion-20261022/native/motion/sdk_motion_worker.cpp:588) | 分步准入、回读关键点、退出路径及 scope 协议 |
| [request_ledger.hpp](/home/xue/.codex/workspaces/green-river-4.0-nerv-motion-20261022/native/motion/request_ledger.hpp:29) | `cancelled_home` 终态与 `home_verified`，不将截断预设标成完整完成 |
| [test_playback_motion.py](/home/xue/.codex/workspaces/green-river-4.0-nerv-motion-20261022/tests/test_playback_motion.py) | 重复动作、结束竞态、旧 scope、租期、内部步骤取消及故障锁定测试 |

同时更新 gateway 回归与页面的“动作收尾归位中”状态显示。远程文件位于正式目录的对应相对位置，已安装的 gateway 副本也已更新。

## 验证结果与实际限制

- 本机、nerv 各完成相关回归：**219 passed**。后续针对生命周期和 gateway 的检查：**51 passed**。有一条原测试依赖 Starlette/httpx 的弃用提示。
- nerv 使用实际 SDK / ROS Humble 完整编译链接通过，包括未解析符号检查与 SDK 契约验证。
- 实机只做一次有界收尾试验：模拟一个播放作用域，强制选演讲；动作被接受后 3 秒发出播放结束。结果为 `cancelled_home`，`home_verified=true`，双臂 ready、归位，躯干归位，未出现驱动故障；没有重试动作。
- **该次从结束通知到归位验证约 12.767 秒**，动作总时长约 15.790 秒。当前 SDK 没有公开的通用 moveArm 轨迹取消接口，已经下发的较长分步必须完成，回退也需要时间。因此仍可能在语音结束后看到安全收尾运动，不能承诺立即静止。若还要明显压低这一时差，下一步需要制作短分步/短伴随动作并实测预算。
- 219 项回归中的重复循环由模拟 SDK 验证；本次实机验收测试的是一次演讲的中途收尾，没有播放真实 TTS 音频，也没有进行长时间连续随机动作可靠性试验。
- 正式服务恢复后：Web health 正常，各业务组件一份；startup 结果已验证；双臂与躯干 ready/home；空 scope 的 OPEN/END 验证通过，没有额外下发动作。

## 部署与回滚资料

正式插件 SHA-256：`207f199214bcdaf8562e701de6f22366a25f9fdbc475ed03f7d88e6e16f4195f`。已核对实际进程只加载正式目录的这份插件。

机器人备份：`/home/nerv/green-river-4.0/backups/before-voice-motion-sync-20261022-112935.tar.gz`。候选、构建产物、部署脚本与原始验收记录：`/home/nerv/green-river-4.0-voice-motion-sync-20261022/`。回滚需先停止业务与动作宿主，再恢复备份内源文件、插件、SDK 契约和已安装 gateway；重新启动会自动上电归位。

- [实机收尾数据](/home/xue/.codex/visualizations/2026/10/22/01a16158-54d0-7281-953d-fef4d700eedf/voice-motion-finish-trial.json)
- [实机 SDK 日志](/home/xue/.codex/visualizations/2026/10/22/01a16158-54d0-7281-953d-fef4d700eedf/voice-motion-finish-host.log)
- [部署哈希清单](/home/xue/.codex/visualizations/2026/10/22/01a16158-54d0-7281-953d-fef4d700eedf/voice-motion-deployment-manifest.json)
- [正式服务验收](/home/xue/.codex/visualizations/2026/10/22/01a16158-54d0-7281-953d-fef4d700eedf/voice-motion-active-validation.json)
- [相对本次部署前版本的增量补丁](/home/xue/.codex/visualizations/2026/10/22/01a16158-54d0-7281-953d-fef4d700eedf/voice-motion-sync.patch)
