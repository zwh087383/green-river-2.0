from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Protocol

from rag_system.parsing.models import ParsedSection


class ParseError(RuntimeError):
    """Raised when a file is supported but cannot be parsed."""


class UnsupportedFileType(ParseError):
    """Raised when no registered parser supports a file."""


class OptionalDependencyMissing(ParseError):
    """Raised when a parser needs an optional dependency that is not installed."""


class DocumentParser(Protocol):
    """Contract implemented by every file parser.

    Parsers return sections instead of chunks. Chunking is a separate step so
    future parsers can preserve richer structure before indexing decides how to
    split the content.
    """

    supported_extensions: ClassVar[set[str]]

    def can_parse(self, path: Path) -> bool:
        """Return whether this parser accepts the file path."""

    def parse(self, path: Path) -> list[ParsedSection]:
        """Parse a file into structured text sections."""


class OCRBackend(Protocol):
    """Small adapter boundary for OCR engines."""

    def extract_text(self, path: Path) -> str:
        """Extract text from an image path."""
