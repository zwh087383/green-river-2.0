from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+[A-Za-z0-9]*(?:[-_./][A-Za-z0-9]+)*|\d+(?:\.\d+)?|[\u4e00-\u9fff]+",
    re.UNICODE,
)
_CJK_PATTERN = re.compile(r"^[\u4e00-\u9fff]+$")


@dataclass(slots=True)
class BM25Hit:
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    tokens: list[str] = field(default_factory=list)
    matched_tokens: list[dict[str, int | str]] = field(default_factory=list)


class BM25Index:
    def __init__(self, path: Path | str = Path("data/index/bm25.sqlite3")) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._documents: list[dict[str, Any]] = []
        self._bm25: Any | None = None
        self._loaded = False
        self._ensure_schema()

    def upsert_chunks(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> int:
        if not ids:
            return 0
        indexed_at = datetime.now(timezone.utc).isoformat()
        rows = []
        for item_id, text, metadata in zip(ids, texts, metadatas):
            tokens = tokenize(text)
            rows.append(
                (
                    item_id,
                    str(metadata.get("document_id", "")),
                    str(metadata.get("source_name", "")),
                    text,
                    json.dumps(metadata, ensure_ascii=False, default=str),
                    json.dumps(tokens, ensure_ascii=False),
                    str(metadata.get("text_hash", "")),
                    indexed_at,
                )
            )
        with self._connect() as conn:
            conn.executemany(
                """
                insert into bm25_chunks (
                    id, document_id, source_name, text, metadata_json,
                    tokens_json, text_hash, indexed_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(id) do update set
                    document_id = excluded.document_id,
                    source_name = excluded.source_name,
                    text = excluded.text,
                    metadata_json = excluded.metadata_json,
                    tokens_json = excluded.tokens_json,
                    text_hash = excluded.text_hash,
                    indexed_at = excluded.indexed_at
                """,
                rows,
            )
        self._loaded = False
        return len(rows)

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        excluded_document_ids: set[str] | None = None,
    ) -> tuple[list[str], list[BM25Hit]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return [], []
        self._load()
        if not self._documents or self._bm25 is None:
            return query_tokens, []

        excluded = excluded_document_ids or set()
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            (
                (index, float(score))
                for index, score in enumerate(scores)
                if float(score) > 0.0 and str(self._documents[index]["metadata"].get("document_id") or "") not in excluded
            ),
            key=lambda item: item[1],
            reverse=True,
        )[: max(1, top_k)]

        hits = []
        for index, score in ranked:
            document = self._documents[index]
            tokens = list(document["tokens"])
            hits.append(
                BM25Hit(
                    id=str(document["id"]),
                    text=str(document["text"]),
                    score=round(score, 6),
                    metadata=dict(document["metadata"]),
                    tokens=tokens,
                    matched_tokens=matched_tokens(query_tokens, tokens),
                )
            )
        return query_tokens, hits

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("select count(*) from bm25_chunks").fetchone()
        return int(row[0])

    def count_document(self, document_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute("select count(*) from bm25_chunks where document_id = ?", (document_id,)).fetchone()
        return int(row[0])

    def delete_document(self, document_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute("delete from bm25_chunks where document_id = ?", (document_id,))
        self._loaded = False
        return int(cursor.rowcount or 0)

    def list_document_chunks(self, document_id: str) -> list[BM25Hit]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                select id, text, metadata_json, tokens_json
                from bm25_chunks
                where document_id = ?
                order by indexed_at asc, id asc
                """,
                (document_id,),
            ).fetchall()
        chunks: list[BM25Hit] = []
        for row in rows:
            chunks.append(
                BM25Hit(
                    id=str(row["id"]),
                    text=str(row["text"] or ""),
                    score=0.0,
                    metadata=_loads_json_dict(row["metadata_json"]),
                    tokens=_loads_json_list(row["tokens_json"]),
                )
            )
        return chunks

    def _load(self) -> None:
        if self._loaded:
            return
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                select id, text, metadata_json, tokens_json
                from bm25_chunks
                order by indexed_at asc, id asc
                """
            ).fetchall()
        documents: list[dict[str, Any]] = []
        corpus: list[list[str]] = []
        for row in rows:
            tokens = _loads_json_list(row["tokens_json"])
            metadata = _loads_json_dict(row["metadata_json"])
            documents.append(
                {
                    "id": str(row["id"]),
                    "text": str(row["text"] or ""),
                    "metadata": metadata,
                    "tokens": tokens,
                }
            )
            corpus.append(tokens)
        self._documents = documents
        self._bm25 = _build_bm25(corpus) if corpus else None
        self._loaded = True

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                create table if not exists bm25_chunks (
                    id text primary key,
                    document_id text not null,
                    source_name text not null,
                    text text not null,
                    metadata_json text not null,
                    tokens_json text not null,
                    text_hash text not null,
                    indexed_at text not null
                )
                """
            )
            conn.execute("create index if not exists idx_bm25_chunks_document on bm25_chunks(document_id)")
            conn.execute("create index if not exists idx_bm25_chunks_text_hash on bm25_chunks(text_hash)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(text):
        value = match.group(0).strip()
        if not value:
            continue
        if _CJK_PATTERN.match(value):
            tokens.extend(_tokenize_cjk(value))
        else:
            tokens.extend(_expand_token(value.lower()))
    return [token for token in tokens if _keep_token(token)]


def matched_tokens(query_tokens: list[str], chunk_tokens: list[str]) -> list[dict[str, int | str]]:
    query_counts = Counter(query_tokens)
    chunk_counts = Counter(chunk_tokens)
    matches = []
    for token, query_count in query_counts.items():
        chunk_count = chunk_counts.get(token, 0)
        if chunk_count:
            matches.append(
                {
                    "token": token,
                    "query_count": int(query_count),
                    "chunk_count": int(chunk_count),
                }
            )
    return matches


def _tokenize_cjk(value: str) -> list[str]:
    try:
        import jieba
    except ImportError:
        return [value]
    return [item.strip().lower() for item in jieba.lcut(value) if item.strip()]


def _expand_token(value: str) -> list[str]:
    expanded = [value]
    if any(separator in value for separator in ("-", "_", "/", ".")):
        expanded.extend(part for part in re.split(r"[-_/.]+", value) if part)
    return expanded


def _keep_token(token: str) -> bool:
    if not token:
        return False
    if len(token) == 1 and token.isascii() and token.isalpha():
        return False
    return True


def _build_bm25(corpus: list[list[str]]) -> Any:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise RuntimeError("rank-bm25 package is required for BM25 keyword indexing") from exc
    return BM25Okapi(corpus)


def _loads_json_list(value: str) -> list[str]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload]


def _loads_json_dict(value: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}
