from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ParsedSection:
    """A logical section produced by a parser before chunking."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentChunk:
    """Index-ready chunk with stable source metadata."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedTable:
    """Structured table extracted from parser metadata or table-like text."""

    table_id: str
    rows: list[list[str]]
    columns: list[str] = field(default_factory=list)
    text: str = ""
    html: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedAsset:
    """Image, figure, or other visual asset discovered during parsing."""

    asset_id: str
    kind: str
    caption: str | None = None
    page: int | None = None
    bbox: dict[str, Any] | None = None
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsingDiagnostic:
    """Non-fatal parser signal shown to users and logs."""

    level: str
    code: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedDocument:
    """Full parse result used by indexing, batch parsing, and the web UI."""

    source: str
    source_name: str
    backend: str
    sections: list[ParsedSection] = field(default_factory=list)
    chunks: list[DocumentChunk] = field(default_factory=list)
    tables: list[ParsedTable] = field(default_factory=list)
    assets: list[ParsedAsset] = field(default_factory=list)
    diagnostics: list[ParsingDiagnostic] = field(default_factory=list)
    cache_key: str | None = None
    cache_hit: bool = False
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BatchFileResult:
    """Per-file result for batch parsing."""

    source: str
    status: str
    document: ParsedDocument | None = None
    error: str | None = None
    elapsed_seconds: float = 0.0


@dataclass(slots=True)
class BatchParseResult:
    """Batch parse summary with independent per-file failures."""

    results: list[BatchFileResult] = field(default_factory=list)
    success_count: int = 0
    failed_count: int = 0
    cache_hit_count: int = 0
