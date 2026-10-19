from __future__ import annotations

import hashlib
import re
from html.parser import HTMLParser
import os
from pathlib import Path
from typing import Any

from rag_system.parsing.models import ParsedAsset, ParsedSection, ParsedTable, ParsingDiagnostic


def extract_tables(sections: list[ParsedSection], source_path: Path) -> list[ParsedTable]:
    tables: list[ParsedTable] = []
    for section_index, section in enumerate(sections):
        html = section.metadata.get("text_as_html")
        if isinstance(html, str) and "<table" in html.lower():
            rows = _html_table_rows(html)
            if rows:
                tables.append(_build_table(source_path, section_index, rows, section, html=html))
                continue

        for rows in _markdown_tables(section.text):
            tables.append(_build_table(source_path, section_index, rows, section, html=None))
    return tables


def extract_assets(sections: list[ParsedSection], source_path: Path) -> list[ParsedAsset]:
    assets: list[ParsedAsset] = []
    for section_index, section in enumerate(sections):
        for markdown_asset in _markdown_image_assets(section, source_path, section_index):
            assets.append(markdown_asset)

        element_type = str(section.metadata.get("element_type", "")).lower()
        kind = ""
        if "image" in element_type or "picture" in element_type or "figure" in element_type:
            kind = "image"
        elif "table" in element_type and section.metadata.get("image_path"):
            kind = "table_image"
        if not kind:
            continue

        page = section.metadata.get("page_number") or section.metadata.get("page")
        asset_id = _stable_id(source_path, "asset", section_index, section.text[:120])
        assets.append(
            ParsedAsset(
                asset_id=asset_id,
                kind=kind,
                caption=section.text or None,
                page=int(page) if isinstance(page, int) else None,
                bbox=_extract_bbox(section.metadata),
                path=_first_string(section.metadata, ["image_path", "filename"]),
                metadata={
                    "section_index": section_index,
                    "source_name": source_path.name,
                    **section.metadata,
                },
            )
        )
    return assets


def _markdown_image_assets(section: ParsedSection, source_path: Path, section_index: int) -> list[ParsedAsset]:
    assets: list[ParsedAsset] = []
    for match_index, match in enumerate(re.finditer(r"!\[([^\]]*)\]\(([^)]+)\)", section.text)):
        caption = match.group(1).strip() or None
        target = match.group(2).strip()
        page = section.metadata.get("page_number") or section.metadata.get("page")
        assets.append(
            ParsedAsset(
                asset_id=_stable_id(source_path, "asset", section_index, f"{match_index}:{target}"),
                kind="image",
                caption=caption,
                page=int(page) if isinstance(page, int) else None,
                bbox=_extract_bbox(section.metadata),
                path=target,
                metadata={
                    "section_index": section_index,
                    "source_name": source_path.name,
                    "origin": "markdown_image",
                    **section.metadata,
                },
            )
        )
    return assets


def build_diagnostics(
    sections: list[ParsedSection],
    *,
    backend: str,
    ocr_languages: str,
    cache_hit: bool,
) -> list[ParsingDiagnostic]:
    diagnostics = [
        ParsingDiagnostic(
            level="info",
            code="backend_used",
            message=f"Parsed with backend: {backend}",
            metadata={"cache_hit": cache_hit},
        )
    ]
    if any(section.metadata.get("fallback_from") for section in sections):
        diagnostics.append(
            ParsingDiagnostic(
                level="warning",
                code="parser_fallback",
                message="Heavy parser failed or timed out; returned fallback parser output.",
                metadata={"fallback_from": sections[0].metadata.get("fallback_from")},
            )
        )
    uses_tesseract_ocr = any(
        section.metadata.get("parser") == "image" or section.metadata.get("ocr_languages")
        for section in sections
    )
    if uses_tesseract_ocr and "chi_sim" in ocr_languages and not _tesseract_has_language("chi_sim"):
        diagnostics.append(
            ParsingDiagnostic(
                level="warning",
                code="missing_chinese_ocr_language",
                message="Tesseract Chinese language data chi_sim is not installed; Chinese OCR may be poor.",
            )
        )
    return diagnostics


def _build_table(
    source_path: Path,
    section_index: int,
    rows: list[list[str]],
    section: ParsedSection,
    html: str | None,
) -> ParsedTable:
    columns = rows[0] if rows else []
    body_rows = rows[1:] if len(rows) > 1 else rows
    text = "\n".join(" | ".join(row) for row in rows)
    return ParsedTable(
        table_id=_stable_id(source_path, "table", section_index, text[:160]),
        rows=body_rows,
        columns=columns,
        text=text,
        html=html,
        metadata={
            "section_index": section_index,
            "source_name": source_path.name,
            **section.metadata,
        },
    )


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"} and self._current_row is not None:
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._current_cell is not None and self._current_row is not None:
            cell = " ".join("".join(self._current_cell).split())
            self._current_row.append(cell)
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None:
            if any(cell for cell in self._current_row):
                self.rows.append(self._current_row)
            self._current_row = None


def _html_table_rows(html: str) -> list[list[str]]:
    parser = _TableHTMLParser()
    parser.feed(html)
    return parser.rows


def _markdown_tables(text: str) -> list[list[list[str]]]:
    lines = text.splitlines()
    tables: list[list[list[str]]] = []
    index = 0
    while index < len(lines) - 1:
        if "|" not in lines[index] or not _is_markdown_separator(lines[index + 1]):
            index += 1
            continue
        rows = [_split_markdown_row(lines[index])]
        index += 2
        while index < len(lines) and "|" in lines[index]:
            rows.append(_split_markdown_row(lines[index]))
            index += 1
        if rows:
            tables.append(rows)
    return tables


def _is_markdown_separator(line: str) -> bool:
    cells = _split_markdown_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _split_markdown_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [" ".join(cell.strip().split()) for cell in stripped.split("|")]


def _extract_bbox(metadata: dict[str, Any]) -> dict[str, Any] | None:
    bbox = metadata.get("bbox")
    if isinstance(bbox, dict):
        return bbox
    coordinates = metadata.get("coordinates")
    if isinstance(coordinates, dict):
        return coordinates
    if hasattr(coordinates, "to_dict"):
        return coordinates.to_dict()
    return None


def _first_string(metadata: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _stable_id(source_path: Path, kind: str, index: int, text: str) -> str:
    raw = f"{source_path.resolve()}:{kind}:{index}:{text}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{source_path.name}:{kind}:{index}:{digest}"


def _tesseract_has_language(language: str) -> bool:
    try:
        import pytesseract
        from rag_system.parsing.parsers import _find_tessdata_dir, _find_tesseract_cmd

        tesseract_cmd = _find_tesseract_cmd()
        if tesseract_cmd is not None:
            pytesseract.pytesseract.tesseract_cmd = str(tesseract_cmd)
        tessdata_dir = _find_tessdata_dir()
        if tessdata_dir is not None:
            os.environ["TESSDATA_PREFIX"] = str(tessdata_dir)
        return language in set(pytesseract.get_languages(config=""))
    except Exception:
        return False
