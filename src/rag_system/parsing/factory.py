from __future__ import annotations

from pathlib import Path

from rag_system.parsing.base import DocumentParser, UnsupportedFileType
from rag_system.parsing.heavy_parsers import DoclingParser, UnstructuredParser
from rag_system.parsing.models import ParsedSection
from rag_system.parsing.parsers import HTMLDocumentParser, ImageParser, MarkdownParser, PDFParser, TextParser


class ParserRegistry:
    """Registry that maps file paths to parser implementations."""

    def __init__(self, parsers: list[DocumentParser] | None = None) -> None:
        self._parsers: list[DocumentParser] = []
        for parser in parsers or []:
            self.register(parser)

    def register(self, parser: DocumentParser) -> None:
        self._parsers.append(parser)

    def get_parser(self, path: Path | str) -> DocumentParser:
        resolved = Path(path)
        for parser in self._parsers:
            if parser.can_parse(resolved):
                return parser
        raise UnsupportedFileType(f"No parser registered for file type: {resolved.suffix or '<none>'}")

    def parse(self, path: Path | str) -> list[ParsedSection]:
        resolved = Path(path)
        return self.get_parser(resolved).parse(resolved)

    @property
    def supported_extensions(self) -> set[str]:
        extensions: set[str] = set()
        for parser in self._parsers:
            extensions.update(parser.supported_extensions)
        return extensions


def default_parser_registry(backend: str = "light") -> ParserRegistry:
    """Create the default registry.

    Heavy dependencies are imported lazily inside individual parsers.
    """

    backend = backend.lower()
    if backend not in {"light", "docling", "unstructured"}:
        raise ValueError("backend must be one of: light, docling, unstructured")

    parsers: list[DocumentParser] = []
    if backend == "docling":
        parsers.append(DoclingParser(timeout_seconds=120, fallback_parser=PDFParser()))
    elif backend == "unstructured":
        parsers.append(UnstructuredParser())

    parsers.extend(
        [
            MarkdownParser(),
            HTMLDocumentParser(),
            TextParser(),
            PDFParser(),
            ImageParser(),
        ]
    )

    return ParserRegistry(
        parsers=parsers,
    )
