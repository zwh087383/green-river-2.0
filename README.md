# GreenRiver 4.0 · 小绿

绿色江河语音讲解员：文档解析与知识检索、流式语音对话、动态像素表情，以及 ROS 2 机械臂手势联动。

## 当前流程

本地唤醒“小绿小绿” → Qwen 流式 ASR → EmbeddingGemma 2 / Chroma / BM25 检索 → Qwen Audio 3.1 同时输出文字与 24 kHz PCM。

- **结束对话**：由主模型调用 `end_conversation` 工具，确认结束后简短道别；正常提问直接继续，没有独立的前置结束判断请求。播放器等待告别音频播放完再回到唤醒监听。
- **动态表情**：开心、关切、难过、惊讶、疑惑、害羞、自豪、鼓励。共享关键词规则和语境触发，异步分析不阻塞语音；支持“展示害羞的表情”等请求。
- **机器人动作**：启动/刷新归位、唤醒招手和回答手势。通过本机 Socket 请求厂商驱动进程中的 C++ 服务，直接调用 ArmMotion 预设动作。
- **回复长度**：普通回答默认约 100 字，可按用户要求展开。
- **知识库**：上传文档后解析、索引；首次安装需要自行导入资料。

## Linux 安装与启动

需要 Python 3.10+。完整安装包含文档解析和语音依赖，部分组件需要系统库：

```bash
git clone https://github.com/xuetommy80-creator/greenriver4.0.git
cd greenriver4.0
bash scripts/install_all_linux.sh
mkdir -p data
cp data/settings.example.json data/settings.json
export DASHSCOPE_API_KEY='填写你自己的百炼 API Key'
```

下载本地模型并检查（权重不在仓库内）：

```bash
.venv-linux/bin/python -m rag_system.indexing.embedding_server --download
.venv-linux/bin/python -m rag_system.model_setup download
./scripts/run_linux.sh doctor
./scripts/start_linux.sh
```

- 管理页面：<http://127.0.0.1:8001/>
- 专用表情屏：<http://127.0.0.1:8001/display>
- 动态表情预览：<http://127.0.0.1:8001/display/preview>
- 停止 Web：`./scripts/stop_linux.sh`。向量服务是独立进程。

模型详情和可选下载见 [本地模型说明](docs/local_model_setup.md)。API Key 也可通过管理页面配置；真实设置保存在被 Git 忽略的 `data/settings.json`。

## ROS 2 工控机

现有机器人 SDK 和 ROS 2 Humble 需要提前安装。为本项目建立独立 overlay：

```bash
source /opt/ros/humble/setup.bash
source "${ROBOT_WS:-$HOME/wheeled_robot_pkg}/install/setup.bash"
export RAG_SYSTEM_PYTHON="$PWD/.venv-linux/bin/python"
colcon build --base-paths ros2 --build-base data/ros2-build --install-base ros2-install
./scripts/run_ipc_voice.sh
```

该入口使用项目独立 Python 环境、8001 后端、语音网关、显示节点和原生 SDK 动作服务。
每次启动会先结束本项目及旧 `~/Voice` 部署的旧进程，包括脱离父进程的显示、知识节点，
确认退出后才启动一套新实例。并发启动由文件锁串行处理；专用表情浏览器会恢复原来的全屏参数。
`ros2/voice/scripts/run_voice.sh` 和 `run_full_voice.sh` 也统一使用此入口。
浏览器自身会创建多个正常子进程；“单实例”指各功能节点只有一份。

查看或停止整套机器人程序：

```bash
python3 scripts/robot_runtime.py status --project-root "$PWD" --legacy-root "$HOME/Voice"
python3 scripts/robot_runtime.py stop --project-root "$PWD" --legacy-root "$HOME/Voice"
```

默认 `ROS_LOCALHOST_ONLY=1`；需要跨机器 DDS 通信时设置 `RAG_ROS_LOCALHOST_ONLY=0`，并与底层机器人配置一致。
不要同时开启网页和机器人两套麦克风监听。原生动作进程的构建、状态查询与回退见 `docs/voice_arm_motion_integration.md`。

## 目录

| 路径 | 内容 |
|---|---|
| `src/rag_system/` | 解析、索引、语音、主模型工具调用、Web 与显示状态 |
| `ros2/` | ROS 2 接口、语音网关、显示和动作节点 |
| `scripts/` | 安装、模型与服务启停脚本 |
| `tests/` | Python、Node 和浏览器验证 |
| `samples/` | 公开或合成的测试样例 |
| `data/settings.example.json` | 无密钥配置示例 |
| `licenses/neko/` | 表情关键词逻辑的第三方许可与归属 |

## 验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-linux/bin/python -m pytest -q
node --test tests/browser/display_emotion_client.test.cjs
```

浏览器回归脚本位于 `tests/browser/`，需额外安装 Playwright 并准备浏览器。

## 公开仓库范围

本仓库包含源码、测试、通用文档及样例，不包含真实 API Key、账号密码、上传的业务文档、知识库数据库、解析结果导出、运行日志、模型权重、Python 环境或 ROS 构建产物。

表情关键词规则改编自 Project N.E.K.O.，其 Apache-2.0 LICENSE 与 NOTICE 保留在 `licenses/neko/`；该许可说明针对对应的第三方部分。
