from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pytest

from rag_system.parsing import (
    ChunkingConfig,
    DoclingParser,
    DocumentChunker,
    HTMLDocumentParser,
    ImageParser,
    MarkdownParser,
    PDFParser,
    ParseError,
    ParsedSection,
    ParserRegistry,
    ParsingService,
    TextParser,
    UnstructuredParser,
    UnsupportedFileType,
    default_parser_registry,
)
from rag_system.parsing.embedded_images import extract_embedded_image_ocr_sections
from rag_system.parsing.models import ParsingDiagnostic
from rag_system.parsing.structure import extract_assets, extract_tables


def test_text_parser_normalizes_text_and_service_chunks(tmp_path: Path) -> None:
    source = tmp_path / "note.txt"
    source.write_text("第一段。\r\n\r\n\r\n第二段。   \n", encoding="utf-8")

    service = ParsingService.with_chunking(chunk_size=20, chunk_overlap=5)
    chunks = service.parse_to_chunks(source)

    assert len(chunks) == 1
    assert chunks[0].text == "第一段。\n\n第二段。"
    assert chunks[0].metadata["parser"] == "text"
    assert chunks[0].metadata["source_name"] == "note.txt"
    assert chunks[0].id.startswith("note.txt:s0:c0:")


def test_markdown_parser_splits_by_headings_and_ignores_code_fence_heading(tmp_path: Path) -> None:
    source = tmp_path / "guide.md"
    source.write_text(
        "\n".join(
            [
                "# Guide",
                "Intro",
                "## Install",
                "Run install.",
                "```",
                "# not a heading",
                "```",
                "## Query",
                "Ask questions.",
            ]
        ),
        encoding="utf-8",
    )

    sections = MarkdownParser().parse(source)

    assert [section.metadata.get("heading_path") for section in sections] == [
        "Guide",
        "Guide > Install",
        "Guide > Query",
    ]
    assert "# not a heading" in sections[1].text


def test_chunker_splits_long_text_with_overlap(tmp_path: Path) -> None:
    source = tmp_path / "long.txt"
    source.write_text("alpha beta gamma delta epsilon", encoding="utf-8")
    chunker = DocumentChunker(ChunkingConfig(chunk_size=12, chunk_overlap=3))
    sections = TextParser().parse(source)

    chunks = chunker.chunk_sections(sections, Path("virtual.txt"))

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 12 for chunk in chunks)
    assert all(chunk.metadata["source_name"] == "virtual.txt" for chunk in chunks)


def test_registry_rejects_unknown_extension(tmp_path: Path) -> None:
    source = tmp_path / "data.bin"
    source.write_bytes(b"abc")

    registry = default_parser_registry()

    with pytest.raises(UnsupportedFileType):
        registry.parse(source)


def test_image_parser_accepts_injected_ocr_backend(tmp_path: Path) -> None:
    source = tmp_path / "screen.png"
    source.write_bytes(b"fake image bytes")

    class FakeOCR:
        def extract_text(self, path: Path) -> str:
            assert path == source
            return "识别出来的文字"

    sections = ImageParser(ocr_backend=FakeOCR()).parse(source)

    assert len(sections) == 1
    assert sections[0].text == "识别出来的文字"
    assert sections[0].metadata["content_type"] == "image"


def test_docling_parser_uses_injected_converter_and_preserves_markdown_structure(tmp_path: Path) -> None:
    source = tmp_path / "manual.pdf"
    source.write_bytes(b"fake pdf bytes")

    class FakeDocument:
        def export_to_markdown(self) -> str:
            return "# Manual\n\nIntro.\n\n## Install\n\nSteps."

    class FakeResult:
        document = FakeDocument()

    class FakeConverter:
        def convert(self, path: str) -> FakeResult:
            assert path == str(source)
            return FakeResult()

    sections = DoclingParser(converter=FakeConverter()).parse(source)

    assert [section.metadata.get("heading_path") for section in sections] == [
        "Manual",
        "Manual > Install",
    ]
    assert sections[0].metadata["parser"] == "docling"
    assert sections[0].metadata["content_type"] == "pdf"
    assert sections[0].metadata["intermediate_format"] == "markdown"


def test_docling_parser_defaults_pdf_ocr_off() -> None:
    parser = DoclingParser()
    assert parser.pdf_do_ocr is False
    assert parser.rapidocr_backend == "onnxruntime"
    assert DoclingParser(pdf_do_ocr=True).pdf_do_ocr is True


def test_docling_parser_falls_back_when_converter_fails(tmp_path: Path) -> None:
    source = tmp_path / "manual.pdf"
    source.write_bytes(b"fake pdf bytes")

    class FailingConverter:
        def convert(self, path: str) -> object:
            raise RuntimeError("boom")

    class FakeFallback:
        supported_extensions = {".pdf"}

        def can_parse(self, path: Path) -> bool:
            return path.suffix.lower() in self.supported_extensions

        def parse(self, path: Path) -> list[ParsedSection]:
            assert path == source
            return [
                ParsedSection(
                    text="fallback text",
                    metadata={"parser": "pdf", "content_type": "pdf"},
                )
            ]

    sections = DoclingParser(
        converter=FailingConverter(),
        fallback_parser=FakeFallback(),
    ).parse(source)

    assert sections[0].text == "fallback text"
    assert sections[0].metadata["parser"] == "pdf"
    assert sections[0].metadata["fallback_from"] == "docling"
    assert "Docling could not parse file" in sections[0].metadata["fallback_reason"]


def test_docling_parser_falls_back_when_worker_times_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "manual.pdf"
    source.write_bytes(b"fake pdf bytes")

    class FakeFallback:
        supported_extensions = {".pdf"}

        def can_parse(self, path: Path) -> bool:
            return path.suffix.lower() in self.supported_extensions

        def parse(self, path: Path) -> list[ParsedSection]:
            return [
                ParsedSection(
                    text="fallback after timeout",
                    metadata={"parser": "pdf", "content_type": "pdf"},
                )
            ]

    def fake_run(*args: object, **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr("rag_system.parsing.heavy_parsers.subprocess.run", fake_run)

    sections = DoclingParser(
        timeout_seconds=1,
        fallback_parser=FakeFallback(),
    ).parse(source)

    assert sections[0].text == "fallback after timeout"
    assert sections[0].metadata["fallback_from"] == "docling"
    assert "timed out after 1 seconds" in sections[0].metadata["fallback_reason"]


def test_unstructured_parser_uses_injected_partition_and_keeps_element_metadata(tmp_path: Path) -> None:
    source = tmp_path / "manual.docx"
    source.write_bytes(b"fake docx bytes")

    class FakeMetadata:
        def to_dict(self) -> dict[str, object]:
            return {
                "page_number": 2,
                "filename": "manual.docx",
                "ignored": "drop me",
            }

    class FakeTitle:
        text = "安装说明"
        metadata = FakeMetadata()

    class FakeNarrativeText:
        text = "第一步，准备环境。"
        metadata = {}

    def fake_partition(**kwargs: object) -> list[object]:
        assert kwargs["filename"] == str(source)
        assert kwargs["strategy"] == "fast"
        return [FakeTitle(), FakeNarrativeText()]

    sections = UnstructuredParser(partition_fn=fake_partition, strategy="fast").parse(source)

    assert [section.text for section in sections] == ["安装说明", "第一步，准备环境。"]
    assert sections[0].metadata["parser"] == "unstructured"
    assert sections[0].metadata["content_type"] == "docx"
    assert sections[0].metadata["element_type"] == "FakeTitle"
    assert sections[0].metadata["page_number"] == 2
    assert "ignored" not in sections[0].metadata


def test_heavy_backend_registry_prefers_requested_backend() -> None:
    docling_registry = default_parser_registry(backend="docling")
    unstructured_registry = default_parser_registry(backend="unstructured")

    assert docling_registry.get_parser(Path("manual.pdf")).__class__ is DoclingParser
    assert unstructured_registry.get_parser(Path("manual.pdf")).__class__ is UnstructuredParser
    assert default_parser_registry().get_parser(Path("manual.pdf")).__class__ is PDFParser


def test_html_parser_removes_navigation_and_keeps_main_content(tmp_path: Path) -> None:
    source = tmp_path / "page.html"
    source.write_text(
        """
        <html><body>
          <nav>index modules next previous</nav>
          <main><h1>Parser Guide</h1><p>Main content survives.</p></main>
          <footer>Footer noise</footer>
        </body></html>
        """,
        encoding="utf-8",
    )

    sections = HTMLDocumentParser().parse(source)

    assert len(sections) == 1
    assert "Parser Guide" in sections[0].text
    assert "Main content survives." in sections[0].text
    assert "Footer noise" not in sections[0].text
    assert "index modules" not in sections[0].text


def test_parse_document_uses_cache_on_second_parse(tmp_path: Path) -> None:
    source = tmp_path / "note.txt"
    source.write_text("cache me", encoding="utf-8")
    service = ParsingService.with_chunking(
        backend="auto",
        enable_cache=True,
        cache_dir=tmp_path / "cache",
    )

    first = service.parse_document(source)
    second = service.parse_document(source)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.sections[0].text == "cache me"


def test_parse_document_adds_embedded_docx_image_ocr_sections(tmp_path: Path) -> None:
    source = tmp_path / "with-image.docx"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("word/document.xml", "<w:document/>")
        archive.writestr("word/media/image1.png", b"fake image bytes")

    class FakeParser:
        supported_extensions = {".docx"}

        def can_parse(self, path: Path) -> bool:
            return path.suffix.lower() == ".docx"

        def parse(self, path: Path) -> list[ParsedSection]:
            return [ParsedSection("body text", {"parser": "fake", "content_type": "docx"})]

    class FakeOCR:
        effective_languages = "eng"

        def extract_text(self, path: Path) -> str:
            assert path.name.startswith("image1-")
            return "TEXT FROM EMBEDDED IMAGE"

    service = ParsingService(
        registry=ParserRegistry([FakeParser()]),
        embedded_image_ocr_backend=FakeOCR(),
        embedded_image_output_dir=tmp_path / "assets",
    )

    document = service.parse_document(source)

    assert any(section.text == "TEXT FROM EMBEDDED IMAGE" for section in document.sections)
    embedded = [section for section in document.sections if section.metadata.get("parser") == "embedded_image_ocr"]
    assert embedded[0].metadata["embedded_path"] == "word/media/image1.png"
    assert embedded[0].metadata["parent_content_type"] == "docx"
    assert embedded[0].metadata["image_path"]
    assert any(chunk.text == "TEXT FROM EMBEDDED IMAGE" for chunk in document.chunks)
    assert any(asset.path == embedded[0].metadata["image_path"] for asset in document.assets)


def test_embedded_image_ocr_supports_pptx_and_xlsx_media_paths(tmp_path: Path) -> None:
    class FakeOCR:
        effective_languages = "eng"

        def extract_text(self, path: Path) -> str:
            return f"ocr for {path.suffix}"

    cases = [
        ("deck.pptx", "ppt/media/image1.png", "pptx"),
        ("book.xlsx", "xl/media/image1.jpeg", "xlsx"),
    ]
    for filename, embedded_path, expected_parent_type in cases:
        source = tmp_path / filename
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr(embedded_path, b"fake image bytes")

        result = extract_embedded_image_ocr_sections(
            source,
            output_dir=tmp_path / "assets",
            ocr_backend=FakeOCR(),
        )

        assert len(result.sections) == 1
        assert result.sections[0].metadata["embedded_path"] == embedded_path
        assert result.sections[0].metadata["parent_content_type"] == expected_parent_type


def test_cache_hit_rewrites_source_to_current_file(tmp_path: Path) -> None:
    first_path = tmp_path / "one" / "same.txt"
    second_path = tmp_path / "two" / "same.txt"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    first_path.write_text("same content", encoding="utf-8")
    second_path.write_text("same content", encoding="utf-8")
    service = ParsingService.with_chunking(
        backend="auto",
        enable_cache=True,
        cache_dir=tmp_path / "cache",
    )

    service.parse_document(first_path)
    second = service.parse_document(second_path)

    assert second.cache_hit is True
    assert second.source == str(second_path)
    assert second.chunks[0].metadata["source"] == str(second_path)


def test_auto_pdf_uses_ocr_docling_for_scanned_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "scan.pdf"
    source.write_bytes(b"%PDF fake")

    class FakePreflight:
        page_count = 1
        extracted_chars = 0
        chars_per_page = 0.0
        likely_scanned = True

    class FakeDocling:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["pdf_do_ocr"] is True

        def parse(self, path: Path) -> list[ParsedSection]:
            return [ParsedSection("ocr text", {"parser": "docling", "content_type": "pdf"})]

    monkeypatch.setattr("rag_system.parsing.service.inspect_pdf", lambda path: FakePreflight())
    monkeypatch.setattr("rag_system.parsing.service.DoclingParser", FakeDocling)

    service = ParsingService.with_chunking(backend="auto")
    document = service.parse_document(source)

    assert document.sections[0].text == "ocr text"
    assert document.sections[0].metadata["auto_strategy"] == "FakeDocling"


def test_extract_tables_from_html_metadata(tmp_path: Path) -> None:
    sections = [
        ParsedSection(
            text="table",
            metadata={
                "parser": "unstructured",
                "text_as_html": "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>",
            },
        )
    ]

    tables = extract_tables(sections, tmp_path / "table.xlsx")

    assert len(tables) == 1
    assert tables[0].columns == ["A", "B"]
    assert tables[0].rows == [["1", "2"]]


def test_extract_assets_from_markdown_image_and_coordinates(tmp_path: Path) -> None:
    sections = [
        ParsedSection(
            text="![Figure 1](images/figure-1.png)",
            metadata={
                "parser": "docling",
                "page": 3,
                "bbox": {"l": 1, "t": 2, "r": 3, "b": 4},
            },
        )
    ]

    assets = extract_assets(sections, tmp_path / "manual.pdf")

    assert len(assets) == 1
    assert assets[0].caption == "Figure 1"
    assert assets[0].page == 3
    assert assets[0].bbox == {"l": 1, "t": 2, "r": 3, "b": 4}


def test_batch_parse_keeps_failures_per_file(tmp_path: Path) -> None:
    good = tmp_path / "good.txt"
    bad = tmp_path / "bad.txt"
    good.write_text("good", encoding="utf-8")
    bad.write_text("bad", encoding="utf-8")

    class SometimesFailingParser:
        supported_extensions = {".txt"}

        def can_parse(self, path: Path) -> bool:
            return path.suffix == ".txt"

        def parse(self, path: Path) -> list[ParsedSection]:
            if path.name == "bad.txt":
                raise RuntimeError("broken")
            return [ParsedSection("ok", {"parser": "fake"})]

    service = ParsingService(registry=ParserRegistry([SometimesFailingParser()]))
    result = service.parse_many_documents([tmp_path])

    assert result.success_count == 1
    assert result.failed_count == 1
    assert any(file_result.error and "broken" in file_result.error for file_result in result.results)


def test_parse_document_rejects_file_above_size_limit(tmp_path: Path) -> None:
    source = tmp_path / "large.txt"
    source.write_text("0123456789", encoding="utf-8")
    service = ParsingService.with_chunking(backend="auto", max_file_size_mb=0)

    with pytest.raises(ParseError, match="above the configured limit"):
        service.parse_document(source)


def test_auto_pdf_rejects_page_count_above_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "large.pdf"
    source.write_bytes(b"%PDF fake")

    class FakePreflight:
        page_count = 301
        extracted_chars = 1000
        chars_per_page = 100.0
        likely_scanned = False

    monkeypatch.setattr("rag_system.parsing.service.inspect_pdf", lambda path: FakePreflight())
    service = ParsingService.with_chunking(backend="auto", max_pdf_pages=300)

    with pytest.raises(ParseError, match="above the configured limit"):
        service.parse_file(source)
