"""Document parsing interfaces and default parser implementations."""

from rag_system.parsing.base import (
    DocumentParser,
    OCRBackend,
    OptionalDependencyMissing,
    ParseError,
    UnsupportedFileType,
)
from rag_system.parsing.chunking import ChunkingConfig, DocumentChunker
from rag_system.parsing.factory import ParserRegistry, default_parser_registry
from rag_system.parsing.heavy_parsers import DoclingParser, UnstructuredParser
from rag_system.parsing.models import (
    BatchFileResult,
    BatchParseResult,
    DocumentChunk,
    ParsedAsset,
    ParsedDocument,
    ParsedSection,
    ParsedTable,
    ParsingDiagnostic,
)
from rag_system.parsing.parsers import HTMLDocumentParser, ImageParser, MarkdownParser, PDFParser, TextParser
from rag_system.parsing.service import ParsingService

__all__ = [
    "BatchFileResult",
    "BatchParseResult",
    "ChunkingConfig",
    "DocumentChunk",
    "DocumentChunker",
    "DocumentParser",
    "DoclingParser",
    "HTMLDocumentParser",
    "ImageParser",
    "MarkdownParser",
    "OCRBackend",
    "OptionalDependencyMissing",
    "PDFParser",
    "ParseError",
    "ParsedAsset",
    "ParsedDocument",
    "ParsedSection",
    "ParsedTable",
    "ParserRegistry",
    "ParsingDiagnostic",
    "ParsingService",
    "TextParser",
    "UnstructuredParser",
    "UnsupportedFileType",
    "default_parser_registry",
]
