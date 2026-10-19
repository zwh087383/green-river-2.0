from __future__ import annotations

import re
import os
import shutil
from pathlib import Path

from rag_system.parsing.base import OCRBackend, OptionalDependencyMissing, ParseError
from rag_system.parsing.cleaning import clean_pdf_text, normalize_text
from rag_system.parsing.html_cleaning import clean_html_file
from rag_system.parsing.models import ParsedSection


class BasePathParser:
    supported_extensions: set[str] = set()
    parser_name = "base"

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_extensions

    def _ensure_file(self, path: Path) -> None:
        if not path.exists():
            raise ParseError(f"File does not exist: {path}")
        if not path.is_file():
            raise ParseError(f"Expected a file path: {path}")


class TextParser(BasePathParser):
    supported_extensions = {".txt", ".text", ".log"}
    parser_name = "text"

    def parse(self, path: Path) -> list[ParsedSection]:
        self._ensure_file(path)
        text = normalize_text(_read_text_file(path))
        if not text:
            return []
        return [
            ParsedSection(
                text=text,
                metadata={
                    "parser": self.parser_name,
                    "content_type": "text",
                },
            )
        ]


class MarkdownParser(BasePathParser):
    supported_extensions = {".md", ".markdown"}
    parser_name = "markdown"
    _heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    def parse(self, path: Path) -> list[ParsedSection]:
        self._ensure_file(path)
        text = normalize_text(_read_text_file(path))
        if not text:
            return []
        return self._parse_sections(text)

    def _parse_sections(self, text: str) -> list[ParsedSection]:
        sections: list[ParsedSection] = []
        heading_stack: list[str] = []
        current_lines: list[str] = []
        current_metadata = {
            "parser": self.parser_name,
            "content_type": "markdown",
        }
        in_code_block = False

        def flush() -> None:
            section_text = normalize_text("\n".join(current_lines))
            if section_text:
                sections.append(
                    ParsedSection(
                        text=section_text,
                        metadata=dict(current_metadata),
                    )
                )

        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_code_block = not in_code_block

            match = self._heading_re.match(line) if not in_code_block else None
            if match:
                flush()
                level = len(match.group(1))
                heading = match.group(2).strip()
                heading_stack = heading_stack[: level - 1]
                heading_stack.append(heading)
                current_lines = [line]
                current_metadata = {
                    "parser": self.parser_name,
                    "content_type": "markdown",
                    "heading": heading,
                    "heading_level": level,
                    "heading_path": " > ".join(heading_stack),
                }
                continue

            current_lines.append(line)

        flush()
        return sections


class HTMLDocumentParser(BasePathParser):
    supported_extensions = {".html", ".htm"}
    parser_name = "html"

    def parse(self, path: Path) -> list[ParsedSection]:
        self._ensure_file(path)
        html = clean_html_file(path)
        markdown = self._html_to_markdown(html)
        sections = MarkdownParser()._parse_sections(markdown)
        for section in sections:
            section.metadata = {
                **section.metadata,
                "parser": self.parser_name,
                "content_type": "html",
                "cleaned_html": True,
            }
        return sections

    def _html_to_markdown(self, html: str) -> str:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise OptionalDependencyMissing(
                "HTML parsing requires the optional dependency 'beautifulsoup4'. "
                "Install it with: pip install beautifulsoup4"
            ) from exc

        soup = BeautifulSoup(html, "html.parser")
        lines: list[str] = []
        for node in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "table"]):
            if node.name and node.name.startswith("h") and len(node.name) == 2:
                level = int(node.name[1])
                text = node.get_text(" ", strip=True)
                if text:
                    lines.append(f"{'#' * level} {text}")
                continue
            if node.name == "table":
                table = _html_table_to_markdown(node)
                if table:
                    lines.append(table)
                continue
            text = node.get_text("\n" if node.name == "pre" else " ", strip=True)
            if text:
                prefix = "- " if node.name == "li" else ""
                lines.append(f"{prefix}{text}")
        return normalize_text("\n\n".join(lines))


class PDFParser(BasePathParser):
    supported_extensions = {".pdf"}
    parser_name = "pdf"

    def parse(self, path: Path) -> list[ParsedSection]:
        self._ensure_file(path)
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise OptionalDependencyMissing(
                "PDF parsing requires the optional dependency 'pypdf'. "
                "Install it with: pip install pypdf"
            ) from exc

        try:
            reader = PdfReader(str(path))
        except Exception as exc:  # pragma: no cover - pypdf exception types vary
            raise ParseError(f"Could not open PDF file: {path}") from exc

        sections: list[ParsedSection] = []
        for page_index, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:  # pragma: no cover - pypdf exception types vary
                raise ParseError(f"Could not extract text from PDF page {page_index + 1}: {path}") from exc
            page_text = clean_pdf_text(page_text)
            if not page_text:
                continue
            sections.append(
                ParsedSection(
                    text=page_text,
                    metadata={
                        "parser": self.parser_name,
                        "content_type": "pdf",
                        "page": page_index + 1,
                    },
                )
            )

        return sections


class ImageParser(BasePathParser):
    supported_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
    parser_name = "image"

    def __init__(self, ocr_backend: OCRBackend | None = None, ocr_languages: str = "eng+chi_sim") -> None:
        self.ocr_backend = ocr_backend
        self.ocr_languages = ocr_languages

    def parse(self, path: Path) -> list[ParsedSection]:
        self._ensure_file(path)
        backend = self.ocr_backend or _PytesseractOCRBackend(languages=self.ocr_languages)
        text = normalize_text(backend.extract_text(path))
        if not text:
            return []
        return [
            ParsedSection(
                text=text,
                metadata={
                    "parser": self.parser_name,
                    "content_type": "image",
                    "ocr_languages": getattr(backend, "effective_languages", self.ocr_languages),
                },
            )
        ]


class _PytesseractOCRBackend:
    def __init__(self, languages: str = "eng+chi_sim") -> None:
        self.languages = languages
        self.effective_languages = languages

    def extract_text(self, path: Path) -> str:
        try:
            from PIL import Image
            import pytesseract
        except ImportError as exc:
            raise OptionalDependencyMissing(
                "Image OCR requires optional dependencies 'pillow' and 'pytesseract', "
                "plus a local Tesseract executable. Install Python packages with: "
                "pip install pillow pytesseract"
            ) from exc

        tesseract_cmd = _find_tesseract_cmd()
        if tesseract_cmd is not None:
            pytesseract.pytesseract.tesseract_cmd = str(tesseract_cmd)

        tessdata_dir = _find_tessdata_dir()
        if tessdata_dir is not None:
            os.environ["TESSDATA_PREFIX"] = str(tessdata_dir)
        config = ""
        language = _available_tesseract_language_spec(self.languages, tessdata_dir)
        self.effective_languages = language
        with Image.open(path) as image:
            if language:
                return pytesseract.image_to_string(image, lang=language, config=config)
            return pytesseract.image_to_string(image, config=config)


def _html_table_to_markdown(node: object) -> str:
    rows: list[list[str]] = []
    for row in node.find_all("tr"):
        cells = [" ".join(cell.get_text(" ", strip=True).split()) for cell in row.find_all(["th", "td"])]
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    separator = ["---"] * width
    body = normalized[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _find_tesseract_cmd() -> Path | None:
    configured = os.environ.get("TESSERACT_CMD")
    if configured and Path(configured).exists():
        return Path(configured)

    path_cmd = shutil.which("tesseract")
    if path_cmd:
        return Path(path_cmd)

    candidates = [
        Path.cwd() / ".tools" / "Tesseract-OCR" / "tesseract.exe",
        Path(__file__).resolve().parents[3] / ".tools" / "Tesseract-OCR" / "tesseract.exe",
        Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def _find_tessdata_dir() -> Path | None:
    configured = os.environ.get("TESSDATA_PREFIX")
    if configured and Path(configured).exists():
        return Path(configured)

    candidates = [
        Path.cwd() / "data" / "tessdata",
        Path(__file__).resolve().parents[3] / "data" / "tessdata",
        Path("C:/Program Files/Tesseract-OCR/tessdata"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _available_tesseract_language_spec(requested: str, tessdata_dir: Path | None = None) -> str:
    try:
        import pytesseract

        if tessdata_dir is not None:
            os.environ["TESSDATA_PREFIX"] = str(tessdata_dir)
        config = ""
        available = set(pytesseract.get_languages(config=config))
    except Exception:
        return requested

    requested_parts = [part for part in requested.split("+") if part]
    effective = [part for part in requested_parts if part in available]
    if effective:
        return "+".join(effective)
    return "eng" if "eng" in available else ""


def _read_text_file(path: Path) -> str:
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise ParseError(f"Could not decode text file: {path}") from last_error
