# 4.0 语音与表情代码复查（2026-10-19）

本次范围：4.0 的直接语音回复、Qwen 结束判断、上下文历史、手动/唤醒打断、
表情同步和动画、EmbeddingGemma 本地服务、Linux 启停脚本与依赖配置。
本次只新增复查报告和本机复现材料，不修改运行源码，不重启服务；3.0 不参与写入。

## 已确认的问题

### 1. [P1] 手动点击麦克风没有取消旧回复，新语音问题无法自动提交

位置：`src/rag_system/web/app.py:5384–5387`；相关位置 `5871–5899`、`6318–6320`。

触发条件：旧 `/api/answer/stream` 仍在生成或等待网络数据时，用户点击手动麦克风，
并在旧流结束前完成新的问题。`startVoiceRecording()` 只调用 `stopCurrentTTS()`，
既不 abort `currentAnswerController`，也不推进 `answerGeneration` 或清理 `isAnswering`。
新的录音可以开始，但识别完成时 `sendChatMessage()` 因 `isAnswering=true` 直接返回 false。
旧轮次的文字、表情、完成回调也仍被视为有效。

受控浏览器复现：旧音频确实入队，点击麦克风后停止音频，但旧请求 `aborted=false`、
旧回答仍运行、轮次号未变化；模拟识别完成后新问题提交返回 false，问题留在输入框。
复现只替换麦克风采集，不要求真实设备；保留了应用的录音入口、旧 NDJSON 流与提交逻辑。

建议：抽出统一的“取消当前轮次”函数，手动麦克风和唤醒打断共用。先推进 generation，
取消回答 HTTP、PCM 和表情请求，再进入录音；旧 finally 不得修改新录音/新轮次状态。
补一条“旧流不结束 → 手动录音 → 识别完成”的集成测试，而不只测音频 stop。

### 2. [P2] 道别音频尚未播放完，后端已清空历史

位置：`src/rag_system/generation/service.py:165–169`；网页等待播放结束发生在
`src/rag_system/web/app.py:6410–6429`，ROS 网关同样在接收最终事件之后才 drain 音频。

`response.done` 代表云端生成结束，服务随即执行 `conversation_history.clear()`，
然后发出 final。浏览器此时仍可能有几秒道别 PCM 没播完。若用户在缓冲播放期唤醒打断，
前端会取消退出动作并继续新问题，但原来的历史已经丢失；普通回复也会提前写入尚未听完的全文。

离线服务复现：先存一轮历史，收取道别 final，但不发送任何播放完成确认，历史长度已从 1 变 0。
现有取消测试只覆盖生成过程取消，未覆盖“生成已完成、物理播放未完成”的窗口。

建议：音频回复先记录 pending turn，以 response_id/generation 关联“播放完成”确认后再清空历史。
确认接口必须幂等、拒绝旧轮次；打断应撤销待完成的清空动作。纯文字回复可在 final 提交历史。

### 3. [P2] Qwen 判断的 1.5 秒 timeout 不是总耗时上限

位置：`src/rag_system/generation/turn_control.py:77–89`。

`httpx.Client(timeout=1.5)` 分别限制连接、读、写等网络阶段的等待，其中 read timeout
限制相邻读取间隔。只要服务端持续缓慢返回数据，总耗时就可以超过 1.5 秒，
而这里的调用是生成首个 meta/音频之前的同步必经路径。

本地慢速 HTTP 服务复现：将 timeout 缩放为 70ms，每 30ms 返回一小块 JSON，
请求仍成功耗时 295ms，没有触发超时。此复现没有调用百炼。

建议：给整个语义判断设置 monotonic 总截止时间，并让超时取消底层请求、继续对话；
不要只调低 read timeout，也不要用“不取消后台工作”的无界线程超时替代。
可复用项目已有异步 HTTP 总超时方案；补慢速分块响应和中途取消测试。

### 4. [P2] 启停脚本只检查 PID 存活，无法确认是否仍为本项目服务

位置：`scripts/start_linux.sh:17–21`；`scripts/stop_linux.sh:12–14`。

Web 崩溃或重启后留下 PID 文件，若该 PID 被其他进程复用，start 会误报服务已经运行，
stop 会直接向其他进程发 SIGTERM。指定新的 RAG_SYSTEM_PORT 时也会仅凭旧 PID
打印“新端口服务已运行”，即使该端口没有服务。

临时目录复现：复制 start 脚本到一次性 checkout，将 PID 文件指向测试自己启动的 sleep，
脚本返回 0 并声称 8001 服务正在运行。未对生产服务运行 stop，不影响 3.0/4.0 进程。

建议：同时校验 PID、进程启动时间/命令所属 checkout、实际监听端口与健康接口；
无法确认归属时拒绝 kill。进一步可用 systemd user service 管理进程生命周期。

## 改进方向（不属于已复现缺陷）

1. 优先统一轮次生命周期：把取消、生成、播放完成、下一轮聆听、历史提交纳入同一套明确状态。
   当前 `web/app.py` 约 6500 行，语音编排代码分散，导致手动打断和唤醒打断走不同路径。
2. 添加端到端耗时指标：ASR 完成、结束判断、向量检索、首段 PCM、物理播放结束分别记录。
   结束判断是新增串行等待点；可与安全的检索预取并行，并复用 HTTP 连接，降低额外握手成本。
   优化时保持 Qwen 负责语义判断，不恢复容易漏判的结束语词表。
3. 大知识库导入与交互检索分配调度优先级。当前本地 embedding 统一锁串行执行，文档批量
   编码可能挡住短问题检索；可拆小文档批次并优先处理 SearchQuery。未对当前 GPU 做大批次压力测试。
4. 建立可重复的口语样例集：结束/继续、含否定和转折、被夸/关切/鼓励、短回复、慢网络、
   播放期打断、手动麦克风打断。关键词表情仍可能误判反问、引语或复杂双重否定，需要实际样例校准。

## 验证与边界

- 现有 Python 全量：479 passed，9 skipped，20 warnings；前端 Node：11 passed。
- `pip check`：No broken requirements found。
- 新增复现覆盖上述 4 个问题，结果在 `data/runtime/code-review-20261019/`。
- 复现材料不写入 3.0，不改生产显示状态，不发送真实回答/情绪/结束判断云请求。
- 本轮不覆盖机器人真实麦克风、声卡、扬声器 drain 与机械臂实机，也未跑 GPU 极限压力测试。
- 通过现有测试不等于没有缺陷；以上问题正是现有测试未覆盖的状态边界。

复现命令（从 4.0 根目录执行）：

```bash
.venv-linux/bin/python data/runtime/code-review-20261019/reproduce_backend.py
uv run --no-project --with playwright python data/runtime/code-review-20261019/reproduce_manual_mic.py
```

注意：脚本的结果文件写入 `/tmp/green-river-review/`；若该目录不存在，请先创建。
