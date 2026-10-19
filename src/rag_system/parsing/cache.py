from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rag_system.parsing.models import (
    DocumentChunk,
    ParsedAsset,
    ParsedDocument,
    ParsedSection,
    ParsedTable,
    ParsingDiagnostic,
)


CACHE_VERSION = "parser-cache-v3"


class ParserCache:
    """Simple JSON cache keyed by file hash and parser options."""

    def __init__(self, root: Path | str = Path("data/parser_cache")) -> None:
        self.root = Path(root)

    def build_key(
        self,
        path: Path,
        *,
        backend: str,
        chunk_size: int,
        chunk_overlap: int,
        ocr_languages: str,
        max_file_size_mb: int = 250,
        max_pdf_pages: int = 300,
        max_docling_pdf_pages: int = 80,
        enable_embedded_image_ocr: bool = True,
    ) -> str:
        payload = {
            "version": CACHE_VERSION,
            "file_sha256": file_sha256(path),
            "source_name": path.name,
            "suffix": path.suffix.lower(),
            "backend": backend,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "ocr_languages": ocr_languages,
            "max_file_size_mb": max_file_size_mb,
            "max_pdf_pages": max_pdf_pages,
            "max_docling_pdf_pages": max_docling_pdf_pages,
            "enable_embedded_image_ocr": enable_embedded_image_ocr,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def load(self, cache_key: str) -> ParsedDocument | None:
        cache_path = self._cache_path(cache_key)
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        document = parsed_document_from_dict(payload)
        document.cache_key = cache_key
        document.cache_hit = True
        return document

    def save(self, cache_key: str, document: ParsedDocument) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = asdict(document)
        payload["cache_key"] = cache_key
        payload["cache_hit"] = False
        self._cache_path(cache_key).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _cache_path(self, cache_key: str) -> Path:
        return self.root / f"{cache_key}.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parsed_document_from_dict(payload: dict[str, Any]) -> ParsedDocument:
    return ParsedDocument(
        source=str(payload.get("source", "")),
        source_name=str(payload.get("source_name", "")),
        backend=str(payload.get("backend", "")),
        sections=[ParsedSection(**section) for section in payload.get("sections", [])],
        chunks=[DocumentChunk(**chunk) for chunk in payload.get("chunks", [])],
        tables=[ParsedTable(**table) for table in payload.get("tables", [])],
        assets=[ParsedAsset(**asset) for asset in payload.get("assets", [])],
        diagnostics=[
            ParsingDiagnostic(**diagnostic)
            for diagnostic in payload.get("diagnostics", [])
        ],
        cache_key=payload.get("cache_key"),
        cache_hit=bool(payload.get("cache_hit", False)),
        elapsed_seconds=float(payload.get("elapsed_seconds", 0.0) or 0.0),
        metadata=dict(payload.get("metadata", {})),
    )
