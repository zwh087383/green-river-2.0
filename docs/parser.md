# 文档解析器说明

## 目标

解析器负责把外部文档统一转换成后续索引模块可以消费的结构化文本。

当前输入：

- `.txt` / `.log`
- `.md` / `.markdown`
- `.pdf`，需要可选依赖 `pypdf`
- `.png` / `.jpg` / `.jpeg` / `.webp` / `.bmp` / `.tiff`，需要 OCR 后端
- 复杂 PDF / DOCX / PPTX / HTML / 图片等，可选择 Docling 或 Unstructured 重解析后端

当前输出：

```python
DocumentChunk(
    id="manual.md:s1:c0:abcd1234ef56",
    text="用于检索和生成的正文片段",
    metadata={
        "source": "D:/docs/manual.md",
        "source_name": "manual.md",
        "parser": "markdown",
        "content_type": "markdown",
        "heading_path": "安装说明 > 环境要求",
        "section_index": 1,
        "chunk_index": 0,
    },
)
```

## 模块结构

```text
src/rag_system/parsing/
  base.py       # 解析器接口、OCR 接口、异常类型
  models.py     # ParsedSection / DocumentChunk 数据结构
  cache.py      # 文件 hash 解析缓存
  cleaning.py   # 保守文本清洗
  html_cleaning.py # HTML 正文提取和噪声清理
  pdf_utils.py  # PDF 页数 / 文字层预检
  structure.py  # tables / assets / diagnostics 抽取
  chunking.py   # ParsedSection -> DocumentChunk
  parsers.py    # Text / Markdown / PDF / Image 解析器
  heavy_parsers.py # Docling / Unstructured 重解析后端
  _docling_worker.py # Docling 子进程 worker，用于超时保护
  factory.py    # ParserRegistry 和默认注册表
  service.py    # ParsingService，对外统一入口
  __main__.py   # 可单独运行的解析 CLI
src/rag_system/web/
  app.py        # 本地上传解析网页
```

## 处理流程

```text
文件路径
  -> ParserRegistry 按扩展名选择解析器
  -> parser.parse() 生成 ParsedSection
  -> DocumentChunker 切分 section
  -> 输出 List[DocumentChunk]
```

解析器只负责提取和保留结构，不直接写 Chroma、BM25 或调用 LLM。这样未来索引、检索、生成模块可以独立替换。

## 解析器接口

新增解析器需要实现 `DocumentParser` 协议：

```python
class MyParser:
    supported_extensions = {".docx"}

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_extensions

    def parse(self, path: Path) -> list[ParsedSection]:
        return [
            ParsedSection(
                text="正文",
                metadata={
                    "parser": "docx",
                    "content_type": "docx",
                },
            )
        ]
```

注册方式：

```python
from rag_system.parsing import ParserRegistry, ParsingService

registry = ParserRegistry()
registry.register(MyParser())
service = ParsingService(registry=registry)
chunks = service.parse_to_chunks("example.docx")
```

## 后端选择

默认后端是 `light`：

```python
from rag_system.parsing import ParsingService

service = ParsingService.with_chunking(backend="light")
```

`light` 后端只使用项目内置解析器和轻依赖，适合普通 `.txt`、`.md` 和文字层 PDF。

`auto` 后端是当前推荐入口：

```python
service = ParsingService.with_chunking(backend="auto", enable_cache=True)
document = service.parse_document("manual.pdf")
```

它会按文件类型自动选择解析器：

```text
txt / md / html -> 轻量解析，其中 HTML 会先做正文清洗
普通小 PDF -> PDFParser
扫描版 PDF -> Docling OCR
复杂多页 PDF -> Docling，失败或超时 fallback 到 PDFParser
docx / pptx -> Docling
xlsx / csv / eml / epub -> Unstructured
图片 -> Tesseract OCR
```

如果需要复杂文档解析，可以显式选择重后端：

```python
docling_service = ParsingService.with_chunking(backend="docling")
unstructured_service = ParsingService.with_chunking(backend="unstructured")
```

重后端会优先接管它支持的文件类型。如果没有安装对应依赖，只有在真正解析文件时才会报 `OptionalDependencyMissing`。

## Markdown 策略

Markdown 解析器会按标题切分逻辑 section：

- 保留标题文本到 section 正文中
- 记录 `heading`、`heading_level`、`heading_path`
- 忽略代码块里的伪标题

这能让后续检索结果知道文本来自哪个章节。

## PDF 策略

PDF 解析器按页提取文本，每页对应一个 `ParsedSection`，metadata 中保留 1-based `page`。

依赖是懒加载的。没有安装 `pypdf` 时，只有真正解析 PDF 才会报错：

```powershell
pip install pypdf
```

`auto` 后端会先预检 PDF 前几页文字量。文字层太少时会判断为扫描版，自动改用 `DoclingParser(pdf_do_ocr=True)`。

默认保护：

```text
单文件大小上限：250 MB
PDF 页数硬上限：300 页
Docling PDF 页数上限：80 页，超过后自动走轻量 PDFParser
Docling 超时：180 秒
```

这些值可以通过 `ParsingService.with_chunking(...)` 参数调整：

```python
service = ParsingService.with_chunking(
    backend="auto",
    max_file_size_mb=250,
    max_pdf_pages=300,
    max_docling_pdf_pages=80,
)
```

## Docling 重解析后端

`DoclingParser` 面向复杂 PDF、Office 文档、图片和 HTML 等场景。它会调用 Docling 生成中间 Markdown，再复用 Markdown 分节逻辑输出 `ParsedSection`。

通过 `ParsingService.with_chunking(backend="docling")` 使用默认注册表时，Docling 会在子进程里运行，并设置 120 秒超时。这样复杂 PDF 卡住时不会拖死主进程。

默认注册表还给 PDF 配了轻量 fallback：如果 Docling 超时或解析失败，会回退到 `PDFParser`，并在 metadata 中标记：

```text
fallback_from = docling
fallback_reason = Docling timed out after ...
```

当前默认 `pdf_do_ocr=False`，也就是普通文字层 PDF 先不启动 OCR。这可以避免本地 OCR 模型初始化拖慢或失败。扫描版 PDF 后续可以用自定义 registry 显式打开 OCR：

```python
from rag_system.parsing import DoclingParser, ParserRegistry, ParsingService

registry = ParserRegistry([DoclingParser(pdf_do_ocr=True)])
service = ParsingService(registry=registry)
```

安装：

```powershell
pip install docling
```

CLI 使用：

```powershell
$env:PYTHONPATH="src"
python -m rag_system.parsing .\manual.pdf --backend docling
```

输出 metadata 中会带：

```text
parser = docling
content_type = pdf / docx / pptx / ...
intermediate_format = markdown
```

适合优先尝试的场景：

- 表格、标题层级、阅读顺序比较重要的 PDF
- DOCX / PPTX / HTML 需要统一转成 Markdown
- 后续需要把复杂文档统一送入 RAG 索引

## Unstructured 重解析后端

`UnstructuredParser` 调用 Unstructured 的 `partition` 接口，把文档拆成 Title、NarrativeText、Table 等元素，再逐个转成 `ParsedSection`。

安装：

```powershell
pip install "unstructured[all-docs]" python-magic-bin
pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
```

CLI 使用：

```powershell
$env:PYTHONPATH="src"
python -m rag_system.parsing .\manual.docx --backend unstructured
```

代码里可以透传 Unstructured 的 partition 参数：

```python
from rag_system.parsing import ParserRegistry, ParsingService, UnstructuredParser

registry = ParserRegistry([
    UnstructuredParser(strategy="fast"),
])
service = ParsingService(registry=registry)
```

输出 metadata 会尽量保留常用元素信息：

```text
parser = unstructured
content_type = pdf / docx / html / ...
element_type = Title / NarrativeText / Table / ...
page_number = 2
```

当前 Windows 环境已经实测：

```text
calibre_demo.docx -> 88 sections / 88 chunks
controlled_parser_sample.pptx -> 11 sections / 11 chunks
python_tutorial_introduction.html -> 149 sections / 149 chunks
controlled_table_sample.xlsx -> 1 section / 1 chunk，metadata 保留 text_as_html 表格
controlled_email_sample.eml -> 5 sections / 5 chunks
```

注意：HTML 解析会保留导航、目录等页面噪声，后续需要接正文清洗规则。
当前 `auto` 后端已经在解析 HTML 前移除常见 `nav/header/footer/aside/script/style/form` 噪声，并优先提取 `main/article` 正文区域。

## 图片 OCR 策略

`ImageParser` 支持注入 OCR 后端：

```python
class MyOCR:
    def extract_text(self, path: Path) -> str:
        return "OCR 结果"

parser = ImageParser(ocr_backend=MyOCR())
```

如果不注入，默认尝试使用 `pillow + pytesseract`：

```powershell
pip install pillow pytesseract
```

本机还需要安装 Tesseract 可执行程序。当前 Windows 环境可用：

```powershell
winget install --id tesseract-ocr.tesseract --exact
```

如果 Tesseract 不在 `PATH`，解析器会依次查找：

```text
TESSERACT_CMD 环境变量
PATH 中的 tesseract
项目本地 .tools/Tesseract-OCR/tesseract.exe
C:/Program Files/Tesseract-OCR/tesseract.exe
```

OCR 语言数据会优先查找：

```text
TESSDATA_PREFIX 环境变量
项目本地 data/tessdata/
C:/Program Files/Tesseract-OCR/tessdata/
```

当前项目已经在 `data/tessdata/` 放入 `eng.traineddata` 和 `chi_sim.traineddata`，图片 OCR 默认使用 `eng+chi_sim`。中文样本 `controlled_chinese_ocr_text_large.png` 已通过真实 OCR 验证。

## Docling OCR 策略

Docling 的图片和扫描 PDF OCR 使用 RapidOCR。当前 parser 显式设置 RapidOCR 后端为 `onnxruntime`，避免在 Windows 上退到不兼容的 torch OCR 配置。

需要的 Python 包：

```powershell
pip install docling onnxruntime
```

图片 OCR 默认开启：

```powershell
$env:PYTHONPATH="src"
python -m rag_system.parsing .\image.png --backend docling
```

普通文字层 PDF 默认关闭 OCR：

```python
DoclingParser(pdf_do_ocr=False)
```

扫描版 PDF 需要显式开启：

```python
from rag_system.parsing import DoclingParser, ParserRegistry, ParsingService

registry = ParserRegistry([DoclingParser(pdf_do_ocr=True)])
service = ParsingService(registry=registry)
sections = service.parse_file("scan.pdf")
```

## 单独运行

从项目根目录执行：

```powershell
$env:PYTHONPATH="src"
python -m rag_system.parsing .\docs\example.md
```

输出 chunks JSON。

如果只想看 parser 的原始 section：

```powershell
$env:PYTHONPATH="src"
python -m rag_system.parsing .\docs\example.md --sections
```

可调整 chunk 参数：

```powershell
$env:PYTHONPATH="src"
python -m rag_system.parsing .\docs --chunk-size 1000 --chunk-overlap 120
```

指定重解析后端：

```powershell
$env:PYTHONPATH="src"
python -m rag_system.parsing .\manual.pdf --backend auto --document --cache
```

```powershell
$env:PYTHONPATH="src"
python -m rag_system.parsing .\manual.docx --backend unstructured
```

## 本地网页窗口

启动上传解析窗口：

```powershell
$env:PYTHONPATH="src"
python -m rag_system.web
```

打开：

```text
http://127.0.0.1:8000
```

网页支持：

- 多文件选择
- 多文件拖拽上传
- 异步解析状态
- sections / chunks / tables / assets / diagnostics / raw JSON 查看

上传文件保存到：

```text
data/uploads/
```

解析缓存保存到：

```text
data/parser_cache/
```

## 测试

解析器测试可以独立运行：

```powershell
python -m pytest tests/test_parsing.py
```

当前测试覆盖：

- 文本清洗和 chunk 输出
- Markdown 标题层级解析
- 长文本 chunk overlap
- 未知扩展名拒绝
- 图片 OCR 后端注入
- Docling 后端注入测试
- Docling 失败 / 超时时的 PDF fallback
- Unstructured 后端注入测试
- light / docling / unstructured registry 优先级
- auto 后端缓存
- HTML 正文清洗
- 扫描 PDF 自动 OCR 分支选择
- PDF 页数限制和大文件保护
- 表格结构化抽取
- 批量解析失败隔离
- Web 上传 API

真实 OCR、真实 Docling、真实 Unstructured 已经做过手动集成验证。后续可以把这些样本固化成可选集成测试 fixture。

真实样本集成测试：

```powershell
$env:RUN_PARSER_INTEGRATION="1"
python -m pytest tests\integration -q
```
