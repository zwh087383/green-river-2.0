from __future__ import annotations

import re

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_TOO_MANY_BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Normalize text without destroying technical content."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _TRAILING_SPACE_RE.sub("", text)
    text = _TOO_MANY_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def clean_pdf_text(text: str) -> str:
    """Apply conservative PDF-specific cleanup."""

    text = text.replace("-\n", "")
    text = normalize_text(text)
    return text
