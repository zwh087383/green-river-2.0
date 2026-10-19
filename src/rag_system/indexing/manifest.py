from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from rag_system.indexing.document_ids import document_id_for_document, file_sha256_for_document
from rag_system.indexing.models import IndexedDocument
from rag_system.parsing.models import ParsedDocument


class IndexManifest:
    def __init__(self, path: Path | str = Path("data/index/index.sqlite3")) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def upsert_document(self, document: IndexedDocument) -> None:
        now = _now()
        created_at = document.created_at or now
        updated_at = document.updated_at or now
        with self._connect() as conn:
            conn.execute(
                """
                insert into indexed_documents (
                    document_id, source, source_name, file_sha256, file_type, file_size,
                    status, parse_status, index_status, image_index_status, enabled,
                    chunk_count, vector_count, bm25_count, sparse_count, image_count,
                    image_vector_count, model, dimensions, cache_key, indexed_at,
                    created_at, updated_at, last_error
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(document_id) do update set
                    source = excluded.source,
                    source_name = excluded.source_name,
                    file_sha256 = excluded.file_sha256,
                    file_type = excluded.file_type,
                    file_size = excluded.file_size,
                    status = 'indexed',
                    parse_status = 'success',
                    index_status = 'success',
                    enabled = indexed_documents.enabled,
                    chunk_count = excluded.chunk_count,
                    vector_count = excluded.vector_count,
                    bm25_count = excluded.bm25_count,
                    sparse_count = excluded.sparse_count,
                    model = excluded.model,
                    dimensions = excluded.dimensions,
                    cache_key = coalesce(excluded.cache_key, indexed_documents.cache_key),
                    indexed_at = excluded.indexed_at,
                    updated_at = excluded.updated_at,
                    last_error = ''
                """,
                (
                    document.document_id,
                    document.source,
                    document.source_name,
                    document.file_sha256,
                    document.file_type,
                    document.file_size,
                    document.status,
                    document.parse_status,
                    document.index_status,
                    document.image_index_status,
                    int(document.enabled),
                    document.chunk_count,
                    document.vector_count,
                    document.bm25_count,
                    document.sparse_count,
                    document.image_count,
                    document.image_vector_count,
                    document.model,
                    document.dimensions,
                    document.cache_key,
                    document.indexed_at,
                    created_at,
                    updated_at,
                    document.last_error,
                ),
            )

    def record_parsed_document(self, document: ParsedDocument) -> IndexedDocument:
        document_id = document_id_for_document(document)
        now = _now()
        file_size = int(document.metadata.get("file_size") or _file_size(document.source))
        file_type = str(document.metadata.get("suffix") or Path(document.source_name).suffix.lower())
        file_sha256 = file_sha256_for_document(document)
        with self._connect() as conn:
            conn.execute(
                """
                insert into indexed_documents (
                    document_id, source, source_name, file_sha256, file_type, file_size,
                    status, parse_status, index_status, image_index_status, enabled,
                    chunk_count, vector_count, bm25_count, sparse_count, image_count,
                    image_vector_count, model, dimensions, cache_key, indexed_at,
                    created_at, updated_at, last_error
                ) values (?, ?, ?, ?, ?, ?, 'parsed', 'success', 'pending', 'unknown', 1,
                    ?, 0, 0, 0, 0, 0, '', 0, ?, '', ?, ?, '')
                on conflict(document_id) do update set
                    source = excluded.source,
                    source_name = excluded.source_name,
                    file_sha256 = excluded.file_sha256,
                    file_type = excluded.file_type,
                    file_size = excluded.file_size,
                    status = 'parsed',
                    parse_status = 'success',
                    index_status = 'pending',
                    chunk_count = excluded.chunk_count,
                    cache_key = excluded.cache_key,
                    updated_at = excluded.updated_at,
                    last_error = ''
                """,
                (
                    document_id,
                    document.source,
                    document.source_name,
                    file_sha256,
                    file_type,
                    file_size,
                    len(document.chunks),
                    document.cache_key,
                    now,
                    now,
                ),
            )
        record = self.get_document(document_id)
        if record is None:
            raise RuntimeError(f"Failed to register parsed document {document_id}")
        return record

    def record_parse_failure(self, *, document_id: str, source: str, source_name: str, file_sha256: str, error: str) -> None:
        now = _now()
        path = Path(source)
        with self._connect() as conn:
            conn.execute(
                """
                insert into indexed_documents (
                    document_id, source, source_name, file_sha256, file_type, file_size,
                    status, parse_status, index_status, image_index_status, enabled,
                    chunk_count, vector_count, bm25_count, sparse_count, image_count,
                    image_vector_count, model, dimensions, cache_key, indexed_at,
                    created_at, updated_at, last_error
                ) values (?, ?, ?, ?, ?, ?, 'failed', 'failed', 'pending', 'unknown', 1,
                    0, 0, 0, 0, 0, 0, '', 0, null, '', ?, ?, ?)
                on conflict(document_id) do update set
                    source = excluded.source,
                    source_name = excluded.source_name,
                    file_sha256 = excluded.file_sha256,
                    file_type = excluded.file_type,
                    file_size = excluded.file_size,
                    status = 'failed',
                    parse_status = 'failed',
                    index_status = 'pending',
                    updated_at = excluded.updated_at,
                    last_error = excluded.last_error
                """,
                (
                    document_id,
                    source,
                    source_name,
                    file_sha256,
                    path.suffix.lower(),
                    _file_size(source),
                    now,
                    now,
                    error,
                ),
            )

    def update_image_index(
        self,
        document_id: str,
        *,
        image_count: int,
        image_vector_count: int,
        status: str = "success",
        error: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                update indexed_documents
                set image_index_status = ?,
                    image_count = ?,
                    image_vector_count = ?,
                    updated_at = ?,
                    last_error = case when ? = '' then last_error else ? end
                where document_id = ?
                """,
                (status, image_count, image_vector_count, _now(), error, error, document_id),
            )

    def update_index_error(self, document_id: str, error: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                update indexed_documents
                set status = 'failed',
                    index_status = 'failed',
                    updated_at = ?,
                    last_error = ?
                where document_id = ?
                """,
                (_now(), error, document_id),
            )

    def set_document_enabled(self, document_id: str, enabled: bool) -> IndexedDocument | None:
        with self._connect() as conn:
            conn.execute(
                """
                update indexed_documents
                set enabled = ?, updated_at = ?
                where document_id = ?
                """,
                (int(enabled), _now(), document_id),
            )
        return self.get_document(document_id)

    def get_document(self, document_id: str) -> IndexedDocument | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                select *
                from indexed_documents
                where document_id = ?
                """,
                (document_id,),
            ).fetchone()
        return self._row_to_document(row) if row is not None else None

    def list_documents(self) -> list[IndexedDocument]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                select *
                from indexed_documents
                order by updated_at desc, indexed_at desc, created_at desc
                """
            ).fetchall()
        return [self._row_to_document(row) for row in rows]

    def delete_document(self, document_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("delete from indexed_documents where document_id = ?", (document_id,))
        return cursor.rowcount > 0

    def disabled_document_ids(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute("select document_id from indexed_documents where enabled = 0").fetchall()
        return {str(row[0]) for row in rows}

    def totals(self) -> tuple[int, int]:
        with self._connect() as conn:
            row = conn.execute(
                "select count(*), coalesce(sum(chunk_count), 0) from indexed_documents"
            ).fetchone()
        return int(row[0]), int(row[1])

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                create table if not exists indexed_documents (
                    document_id text primary key,
                    source text not null,
                    source_name text not null,
                    file_sha256 text not null,
                    chunk_count integer not null,
                    vector_count integer not null,
                    model text not null,
                    dimensions integer not null,
                    indexed_at text not null
                )
                """
            )
            self._add_missing_columns(conn)
            conn.execute(
                "create index if not exists idx_indexed_documents_source on indexed_documents(source)"
            )
            conn.execute(
                "create index if not exists idx_indexed_documents_sha on indexed_documents(file_sha256)"
            )
            conn.execute(
                "create index if not exists idx_indexed_documents_enabled on indexed_documents(enabled)"
            )

    def _add_missing_columns(self, conn: sqlite3.Connection) -> None:
        existing = {row[1] for row in conn.execute("pragma table_info(indexed_documents)").fetchall()}
        columns: dict[str, str] = {
            "file_type": "text not null default ''",
            "file_size": "integer not null default 0",
            "status": "text not null default 'indexed'",
            "parse_status": "text not null default 'success'",
            "index_status": "text not null default 'success'",
            "image_index_status": "text not null default 'unknown'",
            "enabled": "integer not null default 1",
            "bm25_count": "integer not null default 0",
            "sparse_count": "integer not null default 0",
            "image_count": "integer not null default 0",
            "image_vector_count": "integer not null default 0",
            "cache_key": "text",
            "created_at": "text not null default ''",
            "updated_at": "text not null default ''",
            "last_error": "text not null default ''",
        }
        for name, definition in columns.items():
            if name not in existing:
                conn.execute(f"alter table indexed_documents add column {name} {definition}")
        now = _now()
        conn.execute("update indexed_documents set created_at = indexed_at where created_at = ''")
        conn.execute("update indexed_documents set updated_at = indexed_at where updated_at = ''")
        conn.execute("update indexed_documents set created_at = ? where created_at = ''", (now,))
        conn.execute("update indexed_documents set updated_at = ? where updated_at = ''", (now,))

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    @staticmethod
    def _row_to_document(row: Iterable[object] | sqlite3.Row) -> IndexedDocument:
        data: dict[str, Any]
        if isinstance(row, sqlite3.Row):
            data = {key: row[key] for key in row.keys()}
        else:
            values = list(row)
            keys = [
                "document_id",
                "source",
                "source_name",
                "file_sha256",
                "chunk_count",
                "vector_count",
                "model",
                "dimensions",
                "indexed_at",
            ]
            data = dict(zip(keys, values, strict=False))
        return IndexedDocument(
            document_id=str(data.get("document_id", "")),
            source=str(data.get("source", "")),
            source_name=str(data.get("source_name", "")),
            file_sha256=str(data.get("file_sha256", "")),
            chunk_count=int(data.get("chunk_count") or 0),
            vector_count=int(data.get("vector_count") or 0),
            model=str(data.get("model") or ""),
            dimensions=int(data.get("dimensions") or 0),
            indexed_at=str(data.get("indexed_at") or ""),
            file_type=str(data.get("file_type") or ""),
            file_size=int(data.get("file_size") or 0),
            status=str(data.get("status") or "indexed"),
            parse_status=str(data.get("parse_status") or "success"),
            index_status=str(data.get("index_status") or "success"),
            image_index_status=str(data.get("image_index_status") or "unknown"),
            enabled=bool(data.get("enabled", 1)),
            bm25_count=int(data.get("bm25_count") or 0),
            sparse_count=int(data.get("sparse_count") or 0),
            image_count=int(data.get("image_count") or 0),
            image_vector_count=int(data.get("image_vector_count") or 0),
            cache_key=data.get("cache_key"),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
            last_error=str(data.get("last_error") or ""),
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_size(path: Path | str) -> int:
    try:
        resolved = Path(path)
        return resolved.stat().st_size if resolved.exists() else 0
    except OSError:
        return 0
