# 文档解析阶段剩余问题

## 当前结论

当前解析器已经适合做“资料上传解析预览”和入库前检查：

- 支持本地网页上传、拖拽上传和解析结果查看
- 支持自动解析模式、缓存、PDF 预检、扫描 PDF OCR、HTML 清洗、表格抽取、中文 OCR
- 支持 sections / chunks / tables / assets / diagnostics / raw JSON 展示

但它还不是稳定批量生产入库系统。下面是后续仍需要处理的问题。

## 1. Web 界面还是本地工具级

当前 Web 页面主要用于本机调试和预览。

现状：

- job 状态保存在内存里
- 服务重启后页面历史任务会丢失
- 上传文件和解析缓存仍保留在 `data/uploads/` 和 `data/parser_cache/`

后续建议：

- 增加 job 持久化
- 增加历史任务列表
- 增加清理上传文件和缓存的入口

## 2. 复杂 PDF 首次解析仍然慢

缓存解决了重复解析问题，但第一次解析复杂 PDF 仍然可能很慢。

现状：

- 复杂 PDF 会优先走 Docling
- Docling 失败或超时会 fallback 到轻量 `PDFParser`
- 当前只有文件级进度，没有页级进度

后续建议：

- 增加页级进度
- 增加分页解析或分段解析
- 增加更明确的耗时预估
- 对超大 PDF 提供“只解析前 N 页”的选项

## 3. HTML 清洗是启发式规则

HTML 正文提取已经能移除常见导航、页脚和脚本噪声，但不是完美的网页正文抽取器。

现状：

- 会移除 `nav/header/footer/aside/script/style/form`
- 优先提取 `main/article`
- 找不到时按正文密度选择主体节点

风险：

- 结构特殊的网站可能误删正文
- 有些目录、侧边栏、广告仍可能残留

后续建议：

- 增加更多真实网页样本测试
- 增加用户可见的“清洗前/清洗后”对比
- 考虑接入专门正文抽取库

## 4. 表格结构还不够深

当前表格已经能抽出基础结构，但复杂表格还不够。

现状：

- 支持从 `text_as_html` 提取 rows / columns
- 支持 Markdown 表格
- XLSX 样本能提取为结构化表格

不足：

- 合并单元格没有完整表达
- Excel 公式没有单独保留
- 多 sheet 语义不完整
- 表格坐标和来源区域还不够细

后续建议：

- 增加 `sheet_name`
- 增加 `cell_range`
- 增加公式字段
- 增加 merged cells 表达

## 5. 图片资产提取还不完整

当前能提取一部分图片引用和 metadata，但还没有稳定导出所有嵌入图片。

现状：

- 支持 Markdown 图片引用 `![caption](path)`
- 支持部分 parser metadata 中的 image / figure 信息
- 支持 caption、page、bbox 字段

不足：

- PDF / DOCX / PPTX 内嵌图片还没有稳定导出为独立文件
- 图片和正文的上下文关系还比较弱
- 图表截图、表格截图还没有独立管理

后续建议：

- 从 Docling 原生 document 对象中导出 pictures
- 保存到 `data/parser_assets/`
- 增加 `asset_id -> section/chunk` 关联

## 6. OCR 质量不是工业级

当前 OCR 已经能跑英文和中文，但质量取决于图片清晰度。

现状：

- Tesseract 使用 `eng+chi_sim`
- 项目内已有 `data/tessdata/eng.traineddata`
- 项目内已有 `data/tessdata/chi_sim.traineddata`
- 中文样本已通过真实 OCR 验证

风险：

- 低分辨率图片效果差
- 倾斜、阴影、模糊扫描件效果差
- 混合排版和表格 OCR 质量不稳定

后续建议：

- 增加图像预处理：灰度、二值化、放大、去噪、纠偏
- 增加 OCR 置信度
- 对低置信度结果给出 warning

## 7. 缓存版本还偏手动

当前缓存 key 已包含文件 hash 和主要解析参数，但依赖库版本变化还没有自动纳入。

现状：

- cache key 包含文件 SHA256
- cache key 包含 backend、chunk 参数、OCR 语言、大小和页数限制
- cache version 当前需要手动 bump

风险：

- Docling / Unstructured / OCR 升级后，旧缓存可能仍被复用

后续建议：

- 自动把 parser 版本和关键依赖版本写入 cache key
- 提供清理缓存命令
- 提供缓存命中详情

## 8. 并发控制还可以加强

当前单个批次内部是顺序解析，但多个网页请求可能同时触发重解析。

风险：

- 多个复杂 PDF 同时解析会吃内存
- Docling 多进程并发可能拖慢机器

后续建议：

- 增加全局任务队列
- 对 Docling / Unstructured 设置全局并发上限
- 增加排队状态

## 9. 还没有接索引和问答

当前阶段只完成了解析和展示。

现状：

- 已经能输出 chunks
- 已经能展示解析结构
- 尚未写入 Chroma / BM25
- 尚未进入检索和 LLM 回答链路

后续建议：

- 增加 indexing 模块
- 把 chunks 写入向量库和 BM25
- 增加解析结果到索引的状态追踪
- 增加“解析后立即入库”的按钮

## 优先级建议

建议下一阶段优先做：

1. 缓存版本自动化
2. 全局解析队列和并发限制
3. 图片资产稳定导出
4. 表格结构增强
5. 索引模块接入

