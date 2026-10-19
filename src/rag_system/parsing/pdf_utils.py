from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PDFPreflight:
    page_count: int
    extracted_chars: int
    chars_per_page: float
    likely_scanned: bool


def inspect_pdf(path: Path, *, scanned_chars_per_page_threshold: int = 5) -> PDFPreflight:
    try:
        from pypdf import PdfReader
    except ImportError:
        return PDFPreflight(page_count=0, extracted_chars=0, chars_per_page=0.0, likely_scanned=False)

    reader = PdfReader(str(path))
    page_count = len(reader.pages)
    extracted_chars = 0
    for page in reader.pages[: min(page_count, 5)]:
        try:
            extracted_chars += len((page.extract_text() or "").strip())
        except Exception:
            continue
    sampled_pages = min(page_count, 5) or 1
    chars_per_page = extracted_chars / sampled_pages
    return PDFPreflight(
        page_count=page_count,
        extracted_chars=extracted_chars,
        chars_per_page=chars_per_page,
        likely_scanned=page_count > 0 and chars_per_page < scanned_chars_per_page_threshold,
    )
