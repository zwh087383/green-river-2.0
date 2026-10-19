from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

from rag_system.parsing.base import OCRBackend
from rag_system.parsing.base import DocumentParser, ParseError
from rag_system.parsing.cache import ParserCache, file_sha256
from rag_system.parsing.chunking import ChunkingConfig, DocumentChunker
from rag_system.parsing.embedded_images import extract_embedded_image_ocr_sections
from rag_system.parsing.factory import ParserRegistry, default_parser_registry
from rag_system.parsing.heavy_parsers import DoclingParser, UnstructuredParser
from rag_system.parsing.models import (
    BatchFileResult,
    BatchParseResult,
    DocumentChunk,
    ParsedDocument,
    ParsedSection,
    ParsingDiagnostic,
)
from rag_system.parsing.parsers import (
    HTMLDocumentParser,
    ImageParser,
    MarkdownParser,
    PDFParser,
    TextParser,
)
from rag_system.parsing.pdf_utils import inspect_pdf
from rag_system.parsing.structure import build_diagnostics, extract_assets, extract_tables


class ParsingService:
    """Facade used by CLI, tests, and future indexing code."""

    def __init__(
        self,
        registry: ParserRegistry | None = None,
        chunker: DocumentChunker | None = None,
        backend: str = "light",
        cache: ParserCache | None = None,
        enable_cache: bool = False,
        ocr_languages: str = "eng+chi_sim",
        embedded_image_ocr_backend: OCRBackend | None = None,
        enable_embedded_image_ocr: bool = True,
        embedded_image_output_dir: Path | str = Path("data/extracted_assets"),
        max_file_size_mb: int = 250,
        max_pdf_pages: int = 300,
        max_docling_pdf_pages: int = 80,
    ) -> None:
        self.registry = registry or default_parser_registry()
        self.chunker = chunker or DocumentChunker()
        self.backend = backend
        self.cache = cache or ParserCache()
        self.enable_cache = enable_cache
        self.ocr_languages = ocr_languages
        self.embedded_image_ocr_backend = embedded_image_ocr_backend
        self.enable_embedded_image_ocr = enable_embedded_image_ocr
        self.embedded_image_output_dir = Path(embedded_image_output_dir)
        self.max_file_size_mb = max_file_size_mb
        self.max_pdf_pages = max_pdf_pages
        self.max_docling_pdf_pages = max_docling_pdf_pages

    @classmethod
    def with_chunking(
        cls,
        chunk_size: int = 2400,
        chunk_overlap: int = 300,
        registry: ParserRegistry | None = None,
        backend: str = "light",
        enable_cache: bool = False,
        cache_dir: Path | str = Path("data/parser_cache"),
        ocr_languages: str = "eng+chi_sim",
        embedded_image_ocr_backend: OCRBackend | None = None,
        enable_embedded_image_ocr: bool = True,
        embedded_image_output_dir: Path | str = Path("data/extracted_assets"),
        max_file_size_mb: int = 250,
        max_pdf_pages: int = 300,
        max_docling_pdf_pages: int = 80,
    ) -> "ParsingService":
        registry_backend = "light" if backend == "auto" else backend
        return cls(
            registry=registry or default_parser_registry(backend=registry_backend),
            chunker=DocumentChunker(ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)),
            backend=backend,
            cache=ParserCache(cache_dir),
            enable_cache=enable_cache,
            ocr_languages=ocr_languages,
            embedded_image_ocr_backend=embedded_image_ocr_backend,
            enable_embedded_image_ocr=enable_embedded_image_ocr,
            embedded_image_output_dir=embedded_image_output_dir,
            max_file_size_mb=max_file_size_mb,
            max_pdf_pages=max_pdf_pages,
            max_docling_pdf_pages=max_docling_pdf_pages,
        )

    def parse_file(self, path: Path | str) -> list[ParsedSection]:
        resolved = Path(path)
        if self.backend == "auto":
            return self._parse_auto(resolved)
        return self.registry.parse(resolved)

    def parse_to_chunks(self, path: Path | str) -> list[DocumentChunk]:
        resolved = Path(path)
        sections = self.parse_file(resolved)
        return self.chunker.chunk_sections(sections, resolved)

    def parse_document(self, path: Path | str, use_cache: bool = True) -> ParsedDocument:
        resolved = Path(path)
        self._ensure_within_limits(resolved)
        cache_key = self.cache.build_key(
            resolved,
            backend=self.backend,
            chunk_size=self.chunker.config.chunk_size,
            chunk_overlap=self.chunker.config.chunk_overlap,
            ocr_languages=self.ocr_languages,
            max_file_size_mb=self.max_file_size_mb,
            max_pdf_pages=self.max_pdf_pages,
            max_docling_pdf_pages=self.max_docling_pdf_pages,
            enable_embedded_image_ocr=self.enable_embedded_image_ocr,
        )
        if self.enable_cache and use_cache:
            cached = self.cache.load(cache_key)
            if cached is not None:
                cached.source = str(resolved)
                cached.source_name = resolved.name
                cached.chunks = self.chunker.chunk_sections(cached.sections, resolved)
                cached.tables = extract_tables(cached.sections, resolved)
                cached.assets = extract_assets(cached.sections, resolved)
                cached.elapsed_seconds = 0.0
                cached.diagnostics = build_diagnostics(
                    cached.sections,
                    backend=cached.backend,
                    ocr_languages=self.ocr_languages,
                    cache_hit=True,
                )
                return cached

        started = time.perf_counter()
        diagnostics: list[ParsingDiagnostic] = []
        sections = self.parse_file(resolved)
        embedded_ocr = self._extract_embedded_image_ocr(resolved)
        if embedded_ocr.sections:
            sections.extend(embedded_ocr.sections)
        diagnostics.extend(embedded_ocr.diagnostics)
        chunks = self.chunker.chunk_sections(sections, resolved)
        tables = extract_tables(sections, resolved)
        assets = extract_assets(sections, resolved)
        diagnostics.extend(
            build_diagnostics(
                sections,
                backend=self.backend,
                ocr_languages=self.ocr_languages,
                cache_hit=False,
            )
        )

        metadata = {
            "file_sha256": file_sha256(resolved),
            "file_size": resolved.stat().st_size,
            "suffix": resolved.suffix.lower(),
        }
        if resolved.suffix.lower() == ".pdf":
            try:
                preflight = inspect_pdf(resolved)
                metadata["pdf_preflight"] = {
                    "page_count": preflight.page_count,
                    "extracted_chars": preflight.extracted_chars,
                    "chars_per_page": preflight.chars_per_page,
                    "likely_scanned": preflight.likely_scanned,
                }
            except Exception as exc:
                diagnostics.append(
                    ParsingDiagnostic(
                        level="warning",
                        code="pdf_preflight_failed",
                        message=f"Could not inspect PDF before parsing: {exc}",
                    )
                )

        document = ParsedDocument(
            source=str(resolved),
            source_name=resolved.name,
            backend=self.backend,
            sections=sections,
            chunks=chunks,
            tables=tables,
            assets=assets,
            diagnostics=diagnostics,
            cache_key=cache_key,
            cache_hit=False,
            elapsed_seconds=round(time.perf_counter() - started, 3),
            metadata=metadata,
        )
        if self.enable_cache:
            self.cache.save(cache_key, document)
        return document

    def parse_many_to_chunks(
        self,
        paths: Iterable[Path | str],
        recursive: bool = True,
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for file_path in self._iter_files(paths, recursive=recursive):
            chunks.extend(self.parse_to_chunks(file_path))
        return chunks

    def parse_many_documents(
        self,
        paths: Iterable[Path | str],
        recursive: bool = True,
    ) -> BatchParseResult:
        results: list[BatchFileResult] = []
        for file_path in self._iter_files(paths, recursive=recursive):
            started = time.perf_counter()
            try:
                document = self.parse_document(file_path)
                results.append(
                    BatchFileResult(
                        source=str(file_path),
                        status="success",
                        document=document,
                        elapsed_seconds=round(time.perf_counter() - started, 3),
                    )
                )
            except Exception as exc:
                results.append(
                    BatchFileResult(
                        source=str(file_path),
                        status="failed",
                        error=str(exc),
                        elapsed_seconds=round(time.perf_counter() - started, 3),
                    )
                )

        return BatchParseResult(
            results=results,
            success_count=sum(1 for result in results if result.status == "success"),
            failed_count=sum(1 for result in results if result.status == "failed"),
            cache_hit_count=sum(
                1
                for result in results
                if result.document is not None and result.document.cache_hit
            ),
        )

    def _iter_files(
        self,
        paths: Iterable[Path | str],
        recursive: bool,
    ) -> Iterable[Path]:
        supported_extensions = self._supported_extensions()
        for path in paths:
            resolved = Path(path)
            if resolved.is_file():
                if resolved.suffix.lower() in supported_extensions:
                    yield resolved
                continue

            if resolved.is_dir():
                iterator = resolved.rglob("*") if recursive else resolved.glob("*")
                for candidate in iterator:
                    if candidate.is_file() and candidate.suffix.lower() in supported_extensions:
                        yield candidate

    def _supported_extensions(self) -> set[str]:
        if self.backend != "auto":
            return self.registry.supported_extensions
        return {
            ".txt",
            ".text",
            ".log",
            ".md",
            ".markdown",
            ".html",
            ".htm",
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".bmp",
            ".tiff",
            ".tif",
            ".docx",
            ".pptx",
            ".xlsx",
            ".xls",
            ".csv",
            ".eml",
            ".epub",
        }

    def _parse_auto(self, path: Path) -> list[ParsedSection]:
        suffix = path.suffix.lower()
        self._ensure_within_limits(path)
        parser = self._select_auto_parser(path)
        sections = parser.parse(path)
        strategy = parser.__class__.__name__
        for section in sections:
            section.metadata = {
                **section.metadata,
                "auto_strategy": strategy,
            }
        return sections

    def _select_auto_parser(self, path: Path) -> DocumentParser:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".text", ".log"}:
            return TextParser()
        if suffix in {".md", ".markdown"}:
            return MarkdownParser()
        if suffix in {".html", ".htm"}:
            return HTMLDocumentParser()
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}:
            return ImageParser(ocr_languages=self.ocr_languages)
        if suffix == ".pdf":
            preflight = inspect_pdf(path)
            if preflight.page_count > self.max_pdf_pages:
                raise ParseError(
                    f"PDF has {preflight.page_count} pages, above the configured limit of {self.max_pdf_pages}: {path}"
                )
            if preflight.likely_scanned:
                return DoclingParser(
                    pdf_do_ocr=True,
                    timeout_seconds=180,
                    fallback_parser=PDFParser(),
                )
            if preflight.page_count > self.max_docling_pdf_pages:
                return PDFParser()
            if preflight.page_count > 3:
                return DoclingParser(timeout_seconds=180, fallback_parser=PDFParser())
            return PDFParser()
        if suffix in {".docx", ".pptx"}:
            return DoclingParser(timeout_seconds=180)
        if suffix in {".xlsx", ".xls", ".csv", ".eml", ".epub"}:
            return UnstructuredParser(strategy="fast")
        return self.registry.get_parser(path)

    def _ensure_within_limits(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            return
        max_bytes = self.max_file_size_mb * 1024 * 1024
        size = path.stat().st_size
        if size > max_bytes:
            raise ParseError(
                f"File is {size} bytes, above the configured limit of {max_bytes} bytes: {path}"
            )

    def _extract_embedded_image_ocr(self, path: Path):
        if not self.enable_embedded_image_ocr:
            from rag_system.parsing.embedded_images import EmbeddedImageOCRResult

            return EmbeddedImageOCRResult()
        return extract_embedded_image_ocr_sections(
            path,
            output_dir=self.embedded_image_output_dir,
            ocr_languages=self.ocr_languages,
            ocr_backend=self.embedded_image_ocr_backend,
        )
