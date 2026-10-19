# Agent / LLM 问答模块

职责：组合检索结果、系统提示词、对话历史和可切换的 LLM Provider，输出完整回答或流式回答事件。

- 最小输入：一条用户问题和可选的 `top-k`。
- 最小输出：回答、模型、置信度和来源；`stream` 模式按 JSON Lines 输出回答事件。添加 `--verbose` 才显示检索与路由诊断。
- 网络要求：默认 LLM、查询改写和重排需要联网并配置相应 API Key；测试前应先完成验收知识库入库。
- 数据隔离：默认读取 `data/acceptance/index`，不会访问正式知识库。

在仓库根目录执行：

```powershell
./module-test.ps1 agent stream "介绍一下绿色江河" --top-k 3
```
