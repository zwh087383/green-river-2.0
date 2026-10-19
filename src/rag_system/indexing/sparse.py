from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_system.indexing.embeddings import SparseEmbedding, SparseEmbeddingEntry


@dataclass(slots=True)
class SparseHit:
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    matched_sparse: list[dict[str, Any]] = field(default_factory=list)


class SparseVectorIndex:
    def __init__(self, path: Path | str = Path("data/index/sparse.sqlite3")) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def upsert_chunks(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]],
        sparse_vectors: list[SparseEmbedding],
    ) -> int:
        if not ids:
            return 0
        if len(ids) != len(sparse_vectors):
            raise RuntimeError(f"Sparse vector count mismatch: expected {len(ids)}, got {len(sparse_vectors)}")
        indexed_at = datetime.now(timezone.utc).isoformat()
        rows = []
        for item_id, text, metadata, sparse_vector in zip(ids, texts, metadatas, sparse_vectors):
            rows.append(
                (
                    item_id,
                    str(metadata.get("document_id", "")),
                    str(metadata.get("source_name", "")),
                    text,
                    json.dumps(metadata, ensure_ascii=False, default=str),
                    json.dumps(_sparse_to_json(sparse_vector), ensure_ascii=False),
                    str(metadata.get("text_hash", "")),
                    indexed_at,
                )
            )
        with self._connect() as conn:
            conn.executemany(
                """
                insert into sparse_chunks (
                    id, document_id, source_name, text, metadata_json,
                    sparse_json, text_hash, indexed_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(id) do update set
                    document_id = excluded.document_id,
                    source_name = excluded.source_name,
                    text = excluded.text,
                    metadata_json = excluded.metadata_json,
                    sparse_json = excluded.sparse_json,
                    text_hash = excluded.text_hash,
                    indexed_at = excluded.indexed_at
                """,
                rows,
            )
        return len(rows)

    def search(
        self,
        query_vector: SparseEmbedding,
        top_k: int = 5,
        *,
        excluded_document_ids: set[str] | None = None,
    ) -> list[SparseHit]:
        if not query_vector:
            return []
        query_map = _sparse_to_map(query_vector)
        query_tokens = {entry.index: entry.token for entry in query_vector}
        excluded = excluded_document_ids or set()
        hits: list[SparseHit] = []
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                select id, text, metadata_json, sparse_json
                from sparse_chunks
                """
            ).fetchall()
        for row in rows:
            metadata = _loads_json_dict(row["metadata_json"])
            if str(metadata.get("document_id") or "") in excluded:
                continue
            sparse_vector = _sparse_from_json(row["sparse_json"])
            score, matches = _dot_score(query_map, sparse_vector, query_tokens)
            if score <= 0:
                continue
            hits.append(
                SparseHit(
                    id=str(row["id"]),
                    text=str(row["text"] or ""),
                    score=round(score, 6),
                    metadata=metadata,
                    matched_sparse=matches[:20],
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[: max(1, top_k)]

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("select count(*) from sparse_chunks").fetchone()
        return int(row[0])

    def count_document(self, document_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute("select count(*) from sparse_chunks where document_id = ?", (document_id,)).fetchone()
        return int(row[0])

    def delete_document(self, document_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute("delete from sparse_chunks where document_id = ?", (document_id,))
        return int(cursor.rowcount or 0)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                create table if not exists sparse_chunks (
                    id text primary key,
                    document_id text not null,
                    source_name text not null,
                    text text not null,
                    metadata_json text not null,
                    sparse_json text not null,
                    text_hash text not null,
                    indexed_at text not null
                )
                """
            )
            conn.execute("create index if not exists idx_sparse_chunks_document on sparse_chunks(document_id)")
            conn.execute("create index if not exists idx_sparse_chunks_text_hash on sparse_chunks(text_hash)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)


def _dot_score(
    query_map: dict[int, float],
    sparse_vector: SparseEmbedding,
    query_tokens: dict[int, str],
) -> tuple[float, list[dict[str, Any]]]:
    score = 0.0
    matches = []
    for entry in sparse_vector:
        query_value = query_map.get(entry.index)
        if query_value is None:
            continue
        contribution = query_value * entry.value
        score += contribution
        matches.append(
            {
                "index": entry.index,
                "token": entry.token or query_tokens.get(entry.index, ""),
                "query_value": round(query_value, 6),
                "chunk_value": round(entry.value, 6),
                "contribution": round(contribution, 6),
            }
        )
    matches.sort(key=lambda item: float(item["contribution"]), reverse=True)
    return score, matches


def _sparse_to_map(sparse_vector: SparseEmbedding) -> dict[int, float]:
    return {entry.index: entry.value for entry in sparse_vector}


def _sparse_to_json(sparse_vector: SparseEmbedding) -> list[dict[str, Any]]:
    return [
        {
            "index": entry.index,
            "value": entry.value,
            "token": entry.token,
        }
        for entry in sparse_vector
    ]


def _sparse_from_json(value: str) -> SparseEmbedding:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    sparse_vector: SparseEmbedding = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
            score = float(item.get("value"))
        except (TypeError, ValueError):
            continue
        sparse_vector.append(SparseEmbeddingEntry(index=index, value=score, token=str(item.get("token") or "")))
    return sparse_vector


def _loads_json_dict(value: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}
