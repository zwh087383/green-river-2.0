# 小绿语音联动手臂动作改造说明

本文记录这次本地代码修改、SDK 依据、工程策略、测试结果和 NERV-RMA05
上机实验步骤。

## 需求映射

| 语音场景 | 手臂动作 | SDK 代码 | 动作后 |
| --- | --- | ---: | --- |
| 正常唤醒“小绿小绿” | 挥手 | 4 | 动作完成后不重复发 1 |
| 普通回答开始播放 | 演讲 | 5 | 动作完成后不重复发 1 |
| 结束语回答开始播放 | 挥手 | 4 | 动作完成后不重复发 1 |

回答中再次说唤醒词会走现有 `interrupt` 流程，不再额外发一次挥手。如果当前
动作由本协调节点启动，打断时会请求回到预置位。

## 本次修改文件

- `ros2/voice/voice/arm_motion_node.py`：新增统一动作协调节点，独占
  `/arm_control/motion` 发布权，订阅 `/arm_control/arm_state`；本次另修正动作 4/5 完成后不重复发布预置位。
- `ros2/voice/voice/audio_input_node.py`：仅在正常 `wake` 事件中请求
  `wake_wave`；忙时 `interrupt` 不请求挥手。
- `ros2/voice/voice/bridge_node.py`：对有效最终回答，普通回答请求
  `answer_gesture`，`closing` 结束语请求 `closing_wave`。
- `ros2/voice/config/voice_rag.yaml`：新增动作请求话题、SDK 话题、动作码和
  超时参数。
- `ros2/voice/launch/voice_rag.launch.py`：默认启动 `voice_arm_motion_node`。
- `ros2/voice/voice/venv_entrypoints.py` 和 `ros2/voice/setup.py`：注册新的 ROS2
  可执行入口。
- `docs/ros2_linux_deployment.md`：补充动作协调节点和相关话题。
- `tests/test_voice_arm_motion_node.py`：覆盖 Byte 编码、动作映射、状态机、去重、
  过期消息、忙时跳过、打断、超时和发布失败。
- `tests/test_robot_audio_input_node.py`：覆盖正常唤醒发 4，忙时打断不重复发 4。
- `tests/test_ros2_voice_interrupt_nodes.py`：覆盖普通回答发 5、结束语发 4、
  无效回答不发动作，以及动作请求与 TTS 提交顺序。
- `tests/test_ros2_package_layout.py`：覆盖 setup、launch、YAML 和话题接线。

## PDF 明示依据

来源文件：

```text
/path/to/robot_sdk_interface_202603.pdf
```

已核对 PDF 物理第 5-8 页手臂接口和第 13 页 ROS2 参考代码：

- `/arm_control/motion` 类型为 `std_msgs/msg/Byte`，用于执行预定动作。
- 动作码 `0` 是“双臂垂下”，`1` 是“预置位”，`4` 是“挥手”，`5` 是“演讲”。
- `/arm_control/arm_state` 类型为 `wheeled_robot_msgs/msg/ArmState`，提供
  `left_arm_moving` 和 `right_arm_moving`。
- `/arm_control/connect` 类型为 `std_msgs/msg/Bool`，`true` 表示连接设备。
- 第 13 页参考顺序是：先 source ROS 环境，再向 `/arm_control/connect` 发
  `true`，最后向 `/arm_control/motion` 发一次 Byte 动作指令。
- PDF 第 23 页 C++ `isArmMoving` 说明 `false` 可表示“没有移动/移动完成”。
  因此单独一帧 `false` 不能证明这次动作刚刚完成，代码要求先观察到
  `true`，再观察到双臂 `false`。

PDF 参考命令原样语义如下：

```bash
source install/setup.bash
# 使能双臂
ros2 topic pub --once /arm_control/connect std_msgs/msg/Bool "{'data': true}"
# 双臂运动到预置位姿势
ros2 topic pub --once /arm_control/motion std_msgs/msg/Byte "{ 'data': {0x01} }"
```

## PDF 没有定义的部分

下列内容是工程策略，不是 SDK 承诺：

- PDF 没有 Python/rclpy 参考代码。
- PDF 没有预定动作完成的 service、action 或专用回执。
- PDF 没有说明动作 4/5 会自动回到 0 或 1。
- PDF 只把动作 1 称为“预置位”，没有把它称为“默认姿态”。
- SDK 语音接口没有“扬声器第一个字开始播放”回调。

因此当前将 `default_motion_code` 设为 `1`，表示需求中选定“预置位”作为
小绿默认姿态的候选。明天必须在实机上确认；如果现场认定的默认姿态是“双臂
垂下”，只需将 YAML 中该参数改为 `0`。

## 实现策略

### 单一动作所有者

`voice_arm_motion_node` 是唯一向 SDK `/arm_control/motion` 发布的节点。音频节点和
语音桥只向内部 `/voice/arm_motion_request` 发送带 `session_id`、`generation`和
`request_id` 的 `VoiceControl(command=play_motion)`，避免唤醒和回答同时直接抢占手臂。

内部 purpose 映射：

- `wake_wave -> 4`
- `answer_gesture -> 5`
- `closing_wave -> 4`

### Byte 编码

对外 SDK 话题严格使用 PDF 指定的 `std_msgs/msg/Byte`。目标 Humble 的 Python 字段
按单字节赋值：

```python
message.data = bytes([code])
```

不向 `Byte.data` 直接赋 Python 整数。专项测试桩也强制要求类型为 `bytes` 且长度为 1。

### 动作完成与归位

状态机：

```text
idle
  -> awaiting_motion_start
  -> motion_running       (任一手臂 moving=true)
  -> idle                 (双臂 moving=false；SDK 动作 4/5 已自行回到预置位)

interrupt / timeout
  -> awaiting_reset_start (发送预置位 1)
  -> reset_running        (预置位动作 moving=true，如果设备有上报)
  -> idle                 (双臂 moving=false)
```

- 发动作后收到的初始 `false` 不算完成。
- 已知任一手臂在移动，或本节点仍在处理前一动作时，新动作直接跳过，
  不排队、不抢占。
- 从未收到过 `arm_state` 时允许发第一次动作，依靠手动连接、现场安全检查和超时兜底。
- 动作码 `4/5` 正常完成不再追加预置位 `1`；其他动作码仍保留完成后归位。中断或超时最多发一次预置位，不循环发送。
- 本次修改不包含多段动作的完成归属或外部控制仲裁；后续仍需通过厂商完成回执或统一仲裁。
- 动作发布失败只记日志，不阻断唤醒、VAD、ASR、RAG 或 TTS。

### “回答首字”同步边界

在有效最终回答的同一个 bridge 回调中，代码先发内部动作请求，紧接着发
`TextToAudio`。这是当前 SDK 可实现的“逻辑开播近同步”，不保证物理扬声器首个音频
采样点与手臂完全同时。精确同步需要厂商播放节点在首个 PCM 写入时提供真实回调。

## 配置参数

`ros2/voice/config/voice_rag.yaml` 当前映射：

```yaml
voice_arm_motion_node:
  ros__parameters:
    enabled: true
    motion_request_topic: /voice/arm_motion_request
    interrupt_topic: /voice/interrupt
    control_topic: /voice/control
    arm_motion_topic: /arm_control/motion
    arm_state_topic: /arm_control/arm_state
    session_id: robot
    wake_motion_code: 4
    answer_motion_code: 5
    closing_motion_code: 4
    default_motion_code: 1
    motion_start_timeout_seconds: 3.0
    motion_finish_timeout_seconds: 15.0
    reset_start_timeout_seconds: 3.0
    reset_finish_timeout_seconds: 10.0
```

四个超时是本地初始值，PDF 没有给出动作时长，需根据明天实机
`/arm_control/arm_state` 时序调整。

## 本地验证结果

专项回归：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  tests/test_voice_arm_motion_node.py \
  tests/test_robot_audio_input_node.py \
  tests/test_ros2_voice_interrupt_nodes.py \
  tests/test_ros2_package_layout.py
```

结果：`44 passed`。

完整项目回归：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-linux/bin/python -m pytest -q
```

结果：`376 passed, 9 skipped`。

静态检查已通过：

```bash
python3 -m py_compile \
  ros2/voice/voice/arm_motion_node.py \
  ros2/voice/voice/audio_input_node.py \
  ros2/voice/voice/bridge_node.py \
  ros2/voice/voice/venv_entrypoints.py \
  ros2/voice/launch/voice_rag.launch.py
git diff --check
```

本机是 ROS2 Jazzy，且没有 `wheeled_robot_msgs`，所以没有用本机 Jazzy 构建结果代替目标
Humble 验证。需在 NERV 上用厂商 overlay 完成 colcon 构建。

## 明天 NERV-RMA05 上机步骤

### 1. 部署与构建

部署前先审查本地工作区变更。`deploy_voice_to_robot.sh` 使用 `rsync --delete`，会把机器人
`/home/nerv/Voice` 镜像成本地目录，不要在该目录保留未回传的机器人端独有文件。

```bash
ROBOT_HOST=nerv ./scripts/deploy_voice_to_robot.sh
ssh nerv 'bash /home/nerv/Voice/ros2/voice/scripts/install_linux.sh'
```

构建完后在机器人终端 source：

```bash
source /opt/ros/humble/setup.bash
source ~/wheeled_robot_pkg/install/setup.bash
source ~/wheeled_robot_pkg/sample/install/setup.bash
```

### 2. 安全与手动连接

确认手臂运动空间无人、无杂物，操作员可立即使用急停。语音程序不会自动连接或断开
手臂，按 PDF 参考代码手动使能：

```bash
ros2 topic pub --once /arm_control/connect std_msgs/msg/Bool "{'data': true}"
```

先检查 SDK 图：

```bash
ros2 node list | grep -E 'voice|arm_control'
ros2 topic info /arm_control/motion -v
ros2 topic info /arm_control/arm_state -v
ros2 topic echo /arm_control/arm_state
```

### 3. 先单独确认预置位 1

先不运行语音联动，使用 PDF 同样的 one-shot Byte 格式发动作 1：

```bash
ros2 topic pub --once /arm_control/motion std_msgs/msg/Byte "{ 'data': {0x01} }"
```

确认这是小绿需求中的“默认姿态”。如果不是，先停止试验，不要继续语音联动；根据现场
结论把 `default_motion_code` 改为 `0` 或其他经确认的动作码。

### 4. 启动语音联动

需要本地唤醒时：

```bash
cd ~/Voice
bash ros2/voice/scripts/run_full_voice.sh speak_answers:=true
```

另开终端观察：

```bash
source /opt/ros/humble/setup.bash
source ~/wheeled_robot_pkg/install/setup.bash
source ~/wheeled_robot_pkg/sample/install/setup.bash
ros2 topic echo /voice/arm_motion_request
ros2 topic echo /arm_control/motion
ros2 topic echo /arm_control/arm_state
```

### 5. 验收顺序

1. 待唤醒状态说“小绿小绿”。
   - 内部请求：`purpose: wake_wave`。
   - SDK 动作顺序：`4`，不再追加重复的预置位 `1`。
   - 挥手不得阻断后续 VAD 收音。
2. 询问一个普通问题。
   - 内部请求：`purpose: answer_gesture`。
   - SDK 动作顺序：`5`，不再追加重复的预置位 `1`。
   - 动作请求与最终回答 TTS 提交相邻。
3. 说已配置的结束语。
   - 最终 `VoiceAnswer.disposition` 为 `closing`。
   - 内部请求：`purpose: closing_wave`。
   - SDK 动作顺序：`4`，不再追加重复的预置位 `1`。
4. 回答播放中再说唤醒词。
   - 只走 `interrupt`，不新增第二次 `wake_wave`。
   - 如果协调节点正在执行自己的动作，请求回预置位。
5. 在手臂已由外部操作移动时触发语音。
   - 新动作应被跳过，不排队、不抢占。
   - 语音唤醒、识别和回答仍正常。

## 已知限制

- 真实“首字出声”精确同步需要厂商 TTS 播放开始回调，现有 SDK 无法单独实现。
- 预定动作无完成 action，只能根据 `moving` 状态和超时推断。
- 动作 1 是 PDF 的“预置位”，是否等于小绿的“默认姿态”必须实机确认。
- 仓库中没有厂商唤醒成功事件输入；唤醒挥手需在 `run_full_voice.sh` 本地 KWS 链路验证。
