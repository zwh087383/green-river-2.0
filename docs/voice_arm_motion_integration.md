# 4.0 原生 SDK 动作控制

## 前端手臂电源与自动归位（2026-10-20）

主页面对话区增加“上电并归位”和“归位并断电”，同时轮询左右臂电源反馈。
操作复用当前原生动作宿主中的 ArmControl，不再创建第二个设备连接进程。
厂商网页通过 `/arm_control/connect` 的 Bool true/false 控制电源；已核对原厂
ArmControlNode::connectDevice 对应调用 connectDev()/disconnectDev()。
本实现直接在同一 SDK 执行线程调用这两个接口，不打开此前隔离的旧控制话题。

- 断电：仅在待机时接受，执行原厂 PresetPosition，等待双臂静止并验证全部 14 个关节，
  成功后才调用 disconnectDev()。归位失败、超时或故障时不继续提交断电调用。
- 上电：仅在本服务记录的断电状态接受，调用 connectDev()，验证电源与位置反馈，
  重新初始化 ArmMotion，再执行并核验原厂预置位，完成后才恢复手势。
- 电源切换期间拒绝手势、刷新归位和其他电源请求；断电状态下 F5 不上电、不归位。
  处于同一电源状态的重复操作不再执行归位；故障锁定不能通过电源按钮清除。
- 原厂 getPowerState 返回 0 上电、1 断电、2 急停、3 安全门打开、-1 未知。
  disconnectDev() 无返回值，因此调用返回与实际断电分别显示；未知反馈不显示为已断电。
  反馈附带年龄，页面超过 2 秒显示为过期。按钮是正常操作入口，不替代实体急停。
- POST `/api/robot/motion/power` 接受 `request_id` 和 `power: "on" | "off"`。
  202 仅表示接受；后续 GET `/api/robot/motion/state` 返回结果。
  请求去重、同源检查、忙碌拦截与故障锁定覆盖电源操作，不自动重试不明确的回执。
- SDK 异常时保留 facade，避免其析构隐式断开设备，将归位失败误变成断电操作。

验证使用假 SDK 覆盖归位/电源调用顺序、断电后刷新、忙碌、重复请求、归位失败、
超时晚返回、上电后归位失败和未知反馈；另有 API 与浏览器交互回归。
实际 SDK 插件只构建验证；部署检查不自动点击上电/断电，不以假 SDK 测试代替实机电源验收。

当前链路：语音播放事件 → `MotionClient` → Unix Socket → 厂商手臂驱动进程中的 C++ 插件 →
`wheeled_robot::ArmMotion::update()`。硬件执行使用厂商 SDK 原生接口，应用不再向
`/arm_control/motion` 发送动作码。SDK 自身仍依赖部分 ROS 动态库及生成的消息头，
这不等于应用通过 ROS 话题控制手臂。

## 行为

- 唤醒保持挥手；告别保持从挥手、比心、飞吻中随机选择。普通回答从当前目录的自定义 sequence 动作及 speech（演讲）里随机选一个，排除飞吻、比心、挥手。当前回答池为摊手、举手、手指远方、亮肌肉、演讲；已删除的挠头不参与。三个历史回答用途均使用同样的随机选择，候选为空时跳过动作、语音继续。
- 首答触发时如果本客户端接受的唤醒动作仍在运行，在独立动作线程只读轮询它的状态，待归位后只提交一次回答动作，不阻塞语音播放。等待最多 20 秒，新问题或语音打断令尚未发送的旧请求失效；故障、断电、初始化状态不提交动作，不自动上电。不重发未知回执，不积压多轮回答动作。
- SDK 内置动作按当前产品 arm_motion.json 的原始关节目标、手型、速度倍率与停顿逐段执行，避免一次 ArmMotion.update 内部继续派发后续段。每段核对位置和双臂停止反馈；已在目标位置的侧不重复提交零位移运动。手型透传沿用原厂 ArmMotion 的左 ID 40、右 ID 39、寄存器 0。
- SDK 宿主逐行读取本进程合并的原始 stdout/stderr，匹配左右臂 hardware_interface / arm_controller 日志，通过仅本机用户可访问的 Unix Socket DRIVER_LOG 通知原生控制器。保留原始时间戳并限制消息长度；普通 INFO 只转发一个用于验证链路，ERROR/FATAL 立即转发。活动运动的首个驱动错误锁定状态并清除本程序待发动作和待归位请求，后续帧、手型和自动归位均被 guard 拦截。保留原始驱动报错，迟到但时间戳早于通用故障的驱动错误可补全原因；后续超时或电源状态 1 不覆盖它。忽略无关节点、非错误级别及当前动作前的日志。已经进入 SDK/控制器的当前段不能靠软件锁定撤回，硬件保护继续生效。
- 故障后在原 SDK 线程可用时继续只读更新实际电源反馈，避免旧的“已上电”快照误导。恢复仍需要明确点击“重启手臂”，不自动重连或循环上电。
- 连接只建立一次，单一线程调用 SDK；忙时跳过新手势，不积压、不抢占。
- 语音打断只使尚未发送的旧请求失效，不追加预置位动作。
- SDK 错误、状态异常或默认 30 秒看门狗超时后进入锁定故障状态，不自动复位、重连或重试。
- 看门狗只能停止接受新请求，不能强行取消 SDK 内部正在执行的动作。实际停止须使用
  厂商支持的停止或急停功能。重启进程不代表消除了硬件故障。
- `accepted` 只表示请求已接收；`completed` 计数在 SDK 返回且双臂连续静止后增加，
  不构成每个关节到位的独立验收。底层 SDK 不提供本应用专用的完成回执。
- Socket 回执等待上限 300 ms，在独立线程执行；语音线程不等待动作回执。

当前预设及轨迹直接使用机器人 SDK 自带配置，未修改轨迹、速度或温度保护。

## 构建与启动

```bash
cd ~/green-river-4.0
bash scripts/build_sdk_motion.sh
bash scripts/run_ipc_voice.sh
```

需要 `g++`、厂商 SDK 与现有 ROS Humble 头文件，不增加 Python 第三方依赖。
构建产物位于 `ros2-install/greenriver_native_motion/lib/libgreenriver_native_motion.so`，
不包含于源码分发。`greenriver_native_motion` 通过标准 controller pluginlib 机制加载。
插件不声明关节命令接口，update 回调不执行动作；Socket 和 SDK 调用在独立线程运行。

4.0 launch 会启动 `scripts/run_sdk_motion.sh --takeover`。脚本仅优雅停止旧的
`arm_control_node`，再用原厂同一可执行程序和原配置启动专用手臂宿主，底盘、躯干不变。
启动时使用 controller-manager 服务加载原厂关节控制器及本插件；对话期间不通过
ROS 话题或 CLI 发送动作。

实机 SDK 的 `ArmControl::connectDev()` 依赖本进程的 `ArmRegistry` 和
`ControllerManagerRegistry`。安装包没有公开这两个注册表的初始化头文件，不能
把 ArmControl 直接放进空白可执行程序；已通过实机调试确认会停在 waitForArmAvailable。
因此采用原厂驱动进程加控制器插件，共享既有注册和驱动初始化机制，不猜测私有 ABI。

原厂 `/arm_control/motion` 和 `/arm_control/connect` 输入在专用宿主内重映射到
隔离入口，避免原手柄节点继续通过这两个旧入口抢占。不要同时用其他程序向关节轨迹、
跟随控制或隔离入口发送指令。

若旧控制器不能退出，拒绝接管，不强杀。当前原厂 launch 未给手臂节点配置自动重启；
如果更新 SDK 后启用了自动重启，应先调整该启动配置，不能让两个设备所有者并存。
SDK 宿主由启动监督程序管理，退出时先优雅停止，底层阻塞无法退出时终止此专用宿主，
不自动重启或同时建立第二个连接。

不要在专用动作宿主运行时重新启动整套厂商 SDK 服务，也不要启动其他直接连接手臂的程序。
`greenriver-arm-sdk.lock` 和启动锁防止本项目同时运行两个动作服务；它们不能约束
不使用这些锁的第三方程序。不支持运行中单独卸载/重载插件，恢复时重启专用宿主。

脚本工作目录为 `ROBOT_WS`（默认 `~/wheeled_robot_pkg`），保留 SDK 对
`install/product/config/arm_motion.json`、xCore 配置及现有设备许可的相对路径要求。

可配置环境变量：

- `GREENRIVER_MOTION_SOCKET`：默认 `/run/user/<uid>/greenriver-motion.sock`，所有客户端必须一致。
- `GREENRIVER_MOTION_TIMEOUT`：默认 30 秒；超时不自动解除故障。
- `GREENRIVER_PROJECT_ROOT`：launch 中的项目根目录，项目启动脚本自动设置。
- `ROBOT_WS`：厂商 SDK 目录。

## 状态与受控测试

```bash
.venv-linux/bin/python -m rag_system.robot_motion status
```

状态包括 `initializing`、`ready`、`running`、`fault`，同时返回最近请求、详情和完成计数。
生产版本返回 `backend: native_sdk`；测试编译版本返回 `backend: test`。

确认设备与运动空间允许后可显式请求一次动作：

```bash
.venv-linux/bin/python -m rag_system.robot_motion wake_wave
```

本次自动化回归使用编译期假 SDK，不连接设备；覆盖忙时跳过、去重、SDK 错误、
超时后不解锁、慢客户端、第二进程排他、语音打断和请求失败隔离。
真实动作仍须在现场验收，不能以 Socket 回执替代物理执行确认。

## 回退

停止 4.0 launch，确认专用动作宿主已退出，再恢复本次修改前的源码及 ROS overlay。
厂商 `arm_control_node` 源码和启动文件没有修改；通过原有厂商服务入口恢复其手臂节点。
恢复整套厂商服务会影响其管理的其他模块，应在合适时机执行。
当前工控机修改前的相关文件备份保存在项目 `backups/before-native-motion-*.tar.gz`。

## 本次部署验证（2026-10-20）

工控机 4.0 已构建并加载真实 C++ 插件，左右臂连接成功，状态为 `ready`、
`backend: native_sdk`。原动作协调节点已退出，旧动作与连接话题输入已隔离。

相关回归为 58 passed。100 次本机 Socket 状态查询中位耗时 0.035 ms，
p95 为 0.044 ms；此数值仅表示 IPC 状态查询，不是机械臂动作启动或完成时延。
语音后端健康检查返回 HTTP 200。本次未自动触发真实挥手或演讲，物理动作效果待现场确认。

## 程序重启与网页 F5 自动归位

- 原生服务连接成功后，先执行一次 `PresetPosition`（SDK 动作 1），状态为 `homing`。
  等 SDK 返回、双臂停止，并核对 14 个关节反馈与原厂 A/预置位配置后才进入 `ready`。
  到位检查容差为 0.05 rad；没有改动目标角度、速度或硬件保护。
- 主页面、机器人显示页、表情预览页的顶层 F5 刷新会发送一次 POST
  `/api/robot/motion/home`。初次打开页面、iframe 导航和浏览器后退缓存恢复不会重复归位。
- 正在执行手势时，仅保留一个待归位请求；多次刷新合并。归位期间跳过新手势。
- SDK 错误、急停或超时仍保持故障锁定，刷新不会清除故障、强制抢占或重复重试。
- `/api/robot/motion/state` 返回原生状态，`at_preset: true` 表示最近一次归位已通过
  关节位置反馈检查；开始新的手势后该标记清零。
- 归位请求只接受同源页面，读取页面和状态的 GET 请求本身不会触发动作。

本次补充验证：83 项相关回归通过。工控机真实启动后执行 `startup_home`，
网页实际刷新后执行 `refresh_home`；两次均通过关节反馈检查。最终状态为
`ready`、`at_preset: true`、`completed: 2`。网页验证使用临时页面，未改动原有显示页。

## 启动前清理旧实例

`scripts/run_ipc_voice.sh`、`ros2/voice/scripts/run_voice.sh` 和 `run_full_voice.sh`
统一进入 `scripts/robot_runtime.py`。启动器先保存专用表情屏的参数，再对本用户、
本项目及显式指定的旧 `~/Voice` 部署进程发送 TERM；10 秒未退出则 KILL。
清理覆盖旧启动器、Web、向量服务、语音、显示、知识、原生动作宿主及其子进程。
确认旧进程退出后才允许重新连接 SDK，不停止厂商底盘等无关服务。

启动锁串行处理并发请求，实例锁覆盖运行生命周期。每次信号操作核对 PID 的启动时间，
支持时使用 pidfd；PID 文件本身不能授权杀进程。各代进程继承独立实例 ID，旧启动器的
退出清理仅作用于自己这一代。正常停止后会清理该实例的后台节点；下次启动也会回收孤儿节点。
专用浏览器重开时保留原有屏幕、窗口和全屏参数，不影响其他浏览器会话。

本地进程集成测试覆盖连续启动、同时启动、孤儿节点、错误 PID 文件及无关进程保护，
使用临时假节点，不连接机械臂。

2026-10-20 验证：15 项相关回归通过。工控机先通过根目录入口启动，再通过已安装的
`run_full_voice.sh` 重启；两次成功启动之间旧实例完全退出。最终全局统计中 Web、向量、
语音网关、显示、知识、SDK 宿主、专用浏览器和各启动器均为 1 份，另有 Chrome 的
8 个正常子进程。后端健康检查正常，启动归位通过关节反馈验证，状态为 `ready`、
`at_preset: true`。结果保存在工控机 `data/runtime/single-runtime-validation.json`。
