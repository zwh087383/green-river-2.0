from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class IndexedDocument:
    document_id: str
    source: str
    source_name: str
    file_sha256: str
    chunk_count: int
    vector_count: int
    model: str
    dimensions: int
    indexed_at: str
    file_type: str = ""
    file_size: int = 0
    status: str = "indexed"
    parse_status: str = "success"
    index_status: str = "success"
    image_index_status: str = "unknown"
    enabled: bool = True
    bm25_count: int = 0
    sparse_count: int = 0
    image_count: int = 0
    image_vector_count: int = 0
    cache_key: str | None = None
    created_at: str = ""
    updated_at: str = ""
    last_error: str = ""


@dataclass(slots=True)
class IndexingResult:
    status: str
    document_id: str
    source: str
    source_name: str
    chunk_count: int
    vector_count: int
    model: str
    dimensions: int
    collection: str
    indexed_at: str
    message: str = ""
    bm25_count: int = 0
    sparse_count: int = 0


@dataclass(slots=True)
class IndexStats:
    document_count: int
    chunk_count: int
    vector_count: int
    collection: str
    model: str
    dimensions: int
    bm25_chunk_count: int = 0
    sparse_chunk_count: int = 0


@dataclass(slots=True)
class SearchResult:
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    explanation: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ImageIndexingResult:
    status: str
    image_count: int
    vector_count: int
    model: str
    dimensions: int
    collection: str
    indexed_at: str
    images: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""


@dataclass(slots=True)
class ImageIndexStats:
    image_count: int
    vector_count: int
    collection: str
    model: str
    dimensions: int


@dataclass(slots=True)
class ImageSearchResult:
    image_id: str
    image_path: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
