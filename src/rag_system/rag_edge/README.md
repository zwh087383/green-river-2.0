# RAG Edge 读侧模块

职责：只读查询端侧知识库，提供检索、文档列表和索引统计，不执行入库或删除。

- 最小输入：查询文字和可选的 `top-k`。
- 最小输出：单行 JSON，只包含命中文本、分数、来源和结果数量；添加 `--verbose` 才输出内部排序诊断。
- 网络要求：读取本地索引本身不联网；默认查询向量化、改写或重排 Provider 可能需要网络和 API Key。
- 数据隔离：默认读取 `data/acceptance/index`，应先运行 RAG Cloud 示例完成入库。

在仓库根目录执行：

```powershell
./module-test.ps1 rag-edge search "绿色江河是什么组织" --top-k 3 --index-dir ./data/acceptance/index
```
