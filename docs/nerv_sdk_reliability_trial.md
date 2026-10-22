# nerv 部署与动作试验验收

日期：2026-10-22（Asia/Shanghai）。已部署并完成本次受控动作试验，随后恢复整套项目服务。

## 结果

| 项目 | 结果 |
|---|---|
| 正式部署目录 | nerv:/home/nerv/green-river-4.0 |
| SDK / ROS | nerv 实际安装的 SDK，ROS Humble |
| 插件构建 | 完整编译、链接通过；启用未解析符号检查；SDK 头文件、库与活动配置契约通过 |
| 机器人离线回归 | 203 passed，1 个测试依赖弃用提示，73.36 秒 |
| 启动归位 | verified_completed，双臂关节和躯干归位反馈通过 |
| 挥手 | 一次，verified_completed，约 15.04 秒（含返回归位与结果查询） |
| 演讲 | 一次，verified_completed，约 24.82 秒（含返回归位与结果查询） |
| 正式服务恢复后 | ready，at_preset=true，左右臂电源均为 0（上电），trunk.at_home=true |
| Web | /api/health 正常；按 startup token + instance_id 查询返回 verified_completed |
| 进程 | 各业务组件均一份；另有 8 个正常 Chrome 子进程 |
| 已加载插件 | /home/nerv/green-river-4.0/ros2-install/greenriver_native_motion/lib/libgreenriver_native_motion.so |
| 插件 SHA-256 | f6ca8c9f2fc233d05e91e1ae99eb91616c846679a6790eefa8108601b84a3dab |

上述动作只各试验一次，不等同于长期重复运行的可靠性统计。验证范围为手臂关节位置/步骤、静止反馈及归位，不包括灵巧手力控或硬件采样时间戳。

## 实际采用的代码

nerv 当前运行版比原本地 2026-10-20 快照更新，已经包含自定义动作库、moveArm 分步执行、逐步目标反馈、躯干控制、首个驱动错误锁定及显式手臂恢复。此次基于 nerv 现版合并，没有用旧快照覆盖它。

此次保留既有动作目标、速度倍率、手指命令、躯干功能与抬头 0° 的现场避让设置；增加 SDK 能力/链接/哈希契约、速度静止判断、逐请求结果表、实例识别、结果查询 API 和后台客户端结果追踪。

nerv 的现版已逐步调用 SDK 并核对每步关节目标，因此本次使用该直接回读证据；旧本地方案中的 ROS 旁路观察器没有替换进该执行器。结果中的 observer_frames/reached_steps/expected_steps 为兼容字段，在该版本没有用作独立旁路观察计数；catalog 的执行进度仍由 step_index/step_count 提供。

## 文件位置

以下链接为与 nerv 本次合并代码对应的本机镜像：

- [SDK 原生执行和结果协议](/home/xue/.codex/workspaces/green-river-4.0-nerv-motion-20261022/native/motion/sdk_motion_worker.cpp)
- [逐请求记录表](/home/xue/.codex/workspaces/green-river-4.0-nerv-motion-20261022/native/motion/request_ledger.hpp)
- [SDK 适配](/home/xue/.codex/workspaces/green-river-4.0-nerv-motion-20261022/native/motion/sdk_compat.hpp)
- [SDK 能力与契约检查](/home/xue/.codex/workspaces/green-river-4.0-nerv-motion-20261022/scripts/check_motion_sdk.py)
- [原厂动作逐步读取](/home/xue/.codex/workspaces/green-river-4.0-nerv-motion-20261022/native/motion/vendor_motion_steps.hpp)
- [动作客户端与结果观察](/home/xue/.codex/workspaces/green-river-4.0-nerv-motion-20261022/src/rag_system/robot_motion.py)
- [Web 查询 API](/home/xue/.codex/workspaces/green-river-4.0-nerv-motion-20261022/src/rag_system/web/app.py)
- [新增查询回归](/home/xue/.codex/workspaces/green-river-4.0-nerv-motion-20261022/tests/test_sdk_request_results.py)

对应的远程文件均位于正式部署目录下的相同相对位置。

## 证据与备份

- [实机动作请求和反馈记录](/home/xue/.codex/visualizations/2026/10/22/01a16158-54d0-7281-953d-fef4d700eedf/nerv-hardware-trial.json)
- [实机 SDK 日志](/home/xue/.codex/visualizations/2026/10/22/01a16158-54d0-7281-953d-fef4d700eedf/nerv-hardware-trial.log)
- [正式服务最终检查](/home/xue/.codex/visualizations/2026/10/22/01a16158-54d0-7281-953d-fef4d700eedf/nerv-active-validation.json)
- [目标 SDK 构建契约](/home/xue/.codex/visualizations/2026/10/22/01a16158-54d0-7281-953d-fef4d700eedf/nerv-sdk-contract.json)
- [部署文件哈希清单](/home/xue/.codex/visualizations/2026/10/22/01a16158-54d0-7281-953d-fef4d700eedf/nerv-motion-deployment-manifest.json)
- [相对 nerv 原运行源码的补丁](/home/xue/.codex/visualizations/2026/10/22/01a16158-54d0-7281-953d-fef4d700eedf/nerv-motion-reliability.patch)

机器人上的原版备份：`/home/nerv/green-river-4.0/backups/before-sdk-reliability-20261022-105359.tar.gz`。

机器人上的候选源码、构建产物和原始验收记录：`/home/nerv/green-river-4.0-sdk-reliability-20261022/`。

若需回滚，应先通过既有项目停止入口停止业务和动作宿主，再恢复备份中的源文件、原插件和已安装语音节点；恢复后启动会再次上电归位，需要现场安排。此次没有自动触发故障恢复、重发动作或修改厂商 SDK 二进制。
