# RAG Cloud 写侧模块

职责：完成文档解析、切片、向量化、入库和删除，面向知识库的增删改发布流程。

- 最小输入：一个受支持的文档路径；删除时输入文档 ID。
- 最小输出：单行 JSON，包含操作状态、文档 ID、切片数和向量数等结果。
- 网络要求：纯解析可离线；向量化入库通常需要联网并配置 DashScope API Key。
- 数据隔离：甲方命令默认使用 `data/acceptance/index`，不会写入正式的 `data/index`。

在仓库根目录执行：

```powershell
./module-test.ps1 rag-cloud index ./samples/acceptance/green_river.txt --index-dir ./data/acceptance/index
```
