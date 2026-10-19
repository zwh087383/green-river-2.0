from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rag_system.parsing.base import DocumentParser, OptionalDependencyMissing, ParseError
from rag_system.parsing.cleaning import normalize_text
from rag_system.parsing.model_assets import default_docling_artifacts_dir
from rag_system.parsing.models import ParsedSection
from rag_system.parsing.parsers import BasePathParser, MarkdownParser


class DoclingParser(BasePathParser):
    """Heavy parser backend powered by Docling.

    Docling is intentionally imported lazily so the light parser path still has
    zero heavyweight runtime dependencies.
    """

    supported_extensions = {
        ".pdf",
        ".docx",
        ".pptx",
        ".html",
        ".htm",
        ".md",
        ".markdown",
        ".png",
        ".jpg",
        ".jpeg",
        ".tiff",
        ".tif",
        ".bmp",
        ".webp",
    }
    parser_name = "docling"

    def __init__(
        self,
        converter: Any | None = None,
        pdf_do_ocr: bool = False,
        rapidocr_backend: str = "onnxruntime",
        timeout_seconds: float | None = None,
        fallback_parser: DocumentParser | None = None,
    ) -> None:
        self._converter = converter
        self.pdf_do_ocr = pdf_do_ocr
        self.rapidocr_backend = rapidocr_backend
        self.timeout_seconds = timeout_seconds
        self.fallback_parser = fallback_parser

    def parse(self, path: Path) -> list[ParsedSection]:
        self._ensure_file(path)
        try:
            if self.timeout_seconds is not None and self._converter is None:
                return self._parse_in_subprocess(path)

            return self._parse_in_process(path)
        except ParseError as exc:
            return self._parse_with_fallback(path, exc)

    def _parse_in_process(self, path: Path) -> list[ParsedSection]:
        converter = self._converter or self._build_converter()
        try:
            result = converter.convert(str(path))
            document = result.document
            markdown = document.export_to_markdown()
        except Exception as exc:  # pragma: no cover - dependency-specific errors vary
            raise ParseError(f"Docling could not parse file: {path}") from exc

        return _markdown_to_sections(
            markdown,
            parser_name=self.parser_name,
            content_type=_content_type_from_suffix(path),
        )

    def _parse_in_subprocess(self, path: Path) -> list[ParsedSection]:
        with tempfile.NamedTemporaryFile(prefix="rag-docling-", suffix=".json", delete=False) as output:
            output_path = Path(output.name)

        command = [
            sys.executable,
            "-m",
            "rag_system.parsing._docling_worker",
            str(path),
            "--output",
            str(output_path),
            "--rapidocr-backend",
            self.rapidocr_backend,
        ]
        if self.pdf_do_ocr:
            command.append("--pdf-do-ocr")

        env = os.environ.copy()
        src_root = str(Path(__file__).resolve().parents[2])
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"{src_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else src_root
        env.setdefault("PYTHONIOENCODING", "utf-8")

        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise ParseError(f"Docling timed out after {self.timeout_seconds:g} seconds: {path}") from exc

        try:
            if result.returncode != 0:
                stderr = normalize_text(result.stderr or "")
                detail = f" Details: {stderr[-1000:]}" if stderr else ""
                raise ParseError(f"Docling worker failed for file: {path}.{detail}")

            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ParseError(f"Docling worker did not produce output for file: {path}") from exc
        finally:
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                pass

        return [
            ParsedSection(
                text=str(item.get("text", "")),
                metadata=dict(item.get("metadata", {})),
            )
            for item in payload
            if normalize_text(str(item.get("text", "")))
        ]

    def _parse_with_fallback(self, path: Path, error: ParseError) -> list[ParsedSection]:
        fallback = self.fallback_parser
        if fallback is None or not fallback.can_parse(path):
            raise error

        try:
            sections = fallback.parse(path)
        except Exception as exc:
            raise error from exc

        reason = str(error)
        for section in sections:
            section.metadata = {
                **section.metadata,
                "fallback_from": self.parser_name,
                "fallback_reason": reason,
            }
        return sections

    def _build_converter(self) -> Any:
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
            from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
            from docling.document_converter import DocumentConverter
            from docling.document_converter import ImageFormatOption, PdfFormatOption
        except ImportError as exc:
            raise OptionalDependencyMissing(
                "Docling parsing requires the optional dependency 'docling'. "
                "Install it with: pip install docling"
            ) from exc

        artifacts_path = default_docling_artifacts_dir()
        accelerator_options = AcceleratorOptions(
            device=AcceleratorDevice.CPU,
            num_threads=max(1, min(8, os.cpu_count() or 4)),
        )

        pdf_options = PdfPipelineOptions()
        pdf_options.artifacts_path = artifacts_path
        pdf_options.accelerator_options = accelerator_options
        pdf_options.do_ocr = self.pdf_do_ocr
        if self.pdf_do_ocr:
            pdf_options.ocr_options = RapidOcrOptions(backend=self.rapidocr_backend)

        image_options = PdfPipelineOptions()
        image_options.artifacts_path = artifacts_path
        image_options.accelerator_options = accelerator_options
        image_options.do_ocr = True
        image_options.ocr_options = RapidOcrOptions(backend=self.rapidocr_backend)

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=image_options),
            }
        )


class UnstructuredParser(BasePathParser):
    """Heavy parser backend powered by Unstructured open source partitioning."""

    supported_extensions = {
        ".pdf",
        ".docx",
        ".pptx",
        ".html",
        ".htm",
        ".md",
        ".markdown",
        ".txt",
        ".text",
        ".log",
        ".png",
        ".jpg",
        ".jpeg",
        ".tiff",
        ".tif",
        ".bmp",
        ".webp",
        ".csv",
        ".xlsx",
        ".xls",
        ".eml",
        ".epub",
    }
    parser_name = "unstructured"

    def __init__(self, partition_fn: Callable[..., list[Any]] | None = None, **partition_kwargs: Any) -> None:
        self._partition_fn = partition_fn
        self._partition_kwargs = partition_kwargs

    def parse(self, path: Path) -> list[ParsedSection]:
        self._ensure_file(path)
        partition_fn = self._partition_fn or self._load_partition()
        try:
            elements = partition_fn(filename=str(path), **self._partition_kwargs)
        except Exception as exc:  # pragma: no cover - dependency-specific errors vary
            raise ParseError(f"Unstructured could not parse file: {path}") from exc

        sections: list[ParsedSection] = []
        for index, element in enumerate(elements):
            text = normalize_text(str(getattr(element, "text", "") or ""))
            if not text:
                continue
            metadata = {
                "parser": self.parser_name,
                "content_type": _content_type_from_suffix(path),
                "element_index": index,
                "element_type": element.__class__.__name__,
            }
            metadata.update(_extract_unstructured_metadata(element))
            sections.append(ParsedSection(text=text, metadata=metadata))

        return sections

    def _load_partition(self) -> Callable[..., list[Any]]:
        try:
            from unstructured.partition.auto import partition
        except ImportError as exc:
            raise OptionalDependencyMissing(
                "Unstructured parsing requires the optional dependency 'unstructured'. "
                "Install it with: pip install 'unstructured[all-docs]'"
            ) from exc

        return partition


def _markdown_to_sections(markdown: str, parser_name: str, content_type: str) -> list[ParsedSection]:
    markdown = normalize_text(markdown)
    if not markdown:
        return []

    sections = MarkdownParser()._parse_sections(markdown)
    converted: list[ParsedSection] = []
    for section in sections:
        converted.append(
            ParsedSection(
                text=section.text,
                metadata={
                    **section.metadata,
                    "parser": parser_name,
                    "content_type": content_type,
                    "intermediate_format": "markdown",
                },
            )
        )
    return converted


def _extract_unstructured_metadata(element: Any) -> dict[str, Any]:
    raw_metadata = getattr(element, "metadata", None)
    if raw_metadata is None:
        return {}

    if hasattr(raw_metadata, "to_dict"):
        metadata = raw_metadata.to_dict()
    elif isinstance(raw_metadata, dict):
        metadata = raw_metadata
    else:
        return {}

    kept_keys = {
        "coordinates",
        "page_number",
        "filename",
        "filetype",
        "languages",
        "category_depth",
        "parent_id",
        "text_as_html",
        "image_path",
        "image_base64",
        "image_mime_type",
        "detection_class_prob",
        "last_modified",
    }
    return {
        key: _metadata_value(value)
        for key, value in metadata.items()
        if key in kept_keys and value not in (None, "")
    }


def _content_type_from_suffix(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return suffix or "unknown"


def _metadata_value(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value
