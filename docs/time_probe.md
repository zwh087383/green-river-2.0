# 本地时间探针

时间探针用于定位一次真实运行中慢的 Python 模块和函数。它只在显式通过探针命令启动时工作，普通 Web、ROS2、语音和 RAG 进程不会承担常驻性能开销。

## 快速开始

从仓库根目录运行一次无网络模块列表测试：

```bash
.venv-linux/bin/python -m rag_system.time_probe -- \
  rag_system.module_test list
```

目标命令的标准输出保持不变。探针摘要写入标准错误，完整 JSON 默认写入：

```text
data/runtime/time_probe/latest.json
```

指定报告文件和排序方式：

```bash
.venv-linux/bin/python -m rag_system.time_probe \
  --output data/runtime/time_probe/rag-cloud-parse.json \
  --sort self_ms \
  --limit 25 \
  -- \
  rag_system.module_test rag-cloud parse samples/acceptance/green_river.txt
```

Linux 包装脚本的等价写法：

```bash
./scripts/run_linux.sh probe \
  --output data/runtime/time_probe/rag-cloud-parse.json \
  -- rag_system.module_test rag-cloud parse samples/acceptance/green_river.txt
```

Windows PowerShell 可直接使用项目虚拟环境：

```powershell
.\.venv\Scripts\python.exe -m rag_system.time_probe `
  --output data\runtime\time_probe\rag-cloud-parse.json `
  -- rag_system.module_test rag-cloud parse samples\acceptance\green_river.txt
```

以可编辑模式重新安装项目后，Linux 和 Windows 也可直接使用 `rag-time-probe` console script，其参数与 `python -m rag_system.time_probe` 相同。

`--` 前是探针参数，`--` 后是要按 `python -m` 方式运行的目标模块和原始参数。目标命令的退出码会原样透传；即使目标失败，失败前的计时报告仍会落盘。

## 常用目标

统一模块测试入口可直接覆盖 Wakeup、VAD、ASR、TTS、RAG Cloud、RAG Edge 和 Agent。例如：

```bash
# Wakeup
.venv-linux/bin/python -m rag_system.time_probe -- \
  rag_system.module_test wakeup --input samples/audio/wakeup_xiaolv.wav

# VAD
.venv-linux/bin/python -m rag_system.time_probe -- \
  rag_system.module_test vad --input samples/audio/question_green_river.wav

# 检索
.venv-linux/bin/python -m rag_system.time_probe -- \
  rag_system.module_test rag-edge search "绿色江河是什么组织" \
  --index-dir data/acceptance/index

# 完整问答
.venv-linux/bin/python -m rag_system.time_probe -- \
  rag_system.module_test agent answer "绿色江河是什么组织" \
  --index-dir data/acceptance/index
```

测量 Web 进程时，用探针包装服务入口，完成本地操作后正常停止进程：

```bash
.venv-linux/bin/python -m rag_system.time_probe \
  --output data/runtime/time_probe/web-session.json \
  -- rag_system.web
```

也可包装 pytest，观察某个功能测试真正执行到的项目函数：

```bash
.venv-linux/bin/python -m rag_system.time_probe \
  --output data/runtime/time_probe/indexing-tests.json \
  -- pytest -q -p no:cacheprovider tests/test_indexing.py
```

## 报告字段

报告的 `modules` 和 `functions` 分别按 Python 模块与函数聚合，主要字段为：

| 字段 | 含义 |
| --- | --- |
| `cumulative_ms` | 从函数首次进入到最终返回的端到端时间，包括 `await` 等待和 generator 暂停 |
| `active_ms` | 函数 Python frame 实际处于执行状态的片段总时间 |
| `self_ms` | `active_ms` 扣除已探测项目子函数后的时间 |
| `calls` | 完整函数调用数；协程和 generator 多次恢复仍算一次 |
| `segments` | 协程或 generator 的实际执行片段数 |
| `incomplete_calls` | 探针停止时仍未最终返回的调用数 |

`cumulative_ms` 会在父子函数间重叠，因此不能把模块行的 `cumulative_ms` 当作进程总时间。进程总时间看 `summary.elapsed_ms`，CPU/Python 热点优先按 `self_ms` 查看，网络或流式等待优先按 `cumulative_ms` 查看。

`inventory.unobserved_modules` 列出本次未执行到的模块。`module_coverage_percent` 只表示运行路径覆盖了多少模块，不是代码行测试覆盖率。

## 边界与开销

- 只统计本次真正执行到的函数；不会为未运行的功能伪造耗时数据。
- 默认只记录 `rag_system` 模块。可重复传入 `--include MODULE` 扩展范围，用 `--exclude MODULE` 排除高频模块。
- C/C++ 扩展和三方 Python 函数默认不生成独立行；同步等待会归到调用它们的项目函数。
- 探针基于 Python profile hook，会增加运行开销。它适合本地诊断和相对排序，不应将探针运行时间当作无探针的精确基准。
- 报告不读取函数参数或局部变量，也不保存 query、回答、音频或 API Key。`--label` 由用户自行提供，不要在其中放入敏感信息。
