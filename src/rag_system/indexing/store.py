from __future__ import annotations

import math
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from rag_system.indexing.models import SearchResult


# Chroma 1.5.x creates a process-wide System for each persistent path.  Its
# check-then-create startup sequence is not atomic, so concurrent first-use
# requests can observe a RustBindingsAPI before it has finished starting.
# Keep client and collection creation in one critical section; normal store
# operations remain concurrent after initialization.
_CHROMA_INITIALIZATION_LOCK = Lock()


class VectorStore(Protocol):
    collection_name: str

    def upsert(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        ...

    def query(self, embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        ...

    def list_all(self) -> list[SearchResult]:
        ...

    def delete_where(self, where: dict[str, Any]) -> int:
        ...

    def count(self) -> int:
        ...


class ChromaVectorStore:
    def __init__(
        self,
        path: Path | str = Path("data/index/chroma"),
        collection_name: str = "rag_chunks",
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("chromadb package is required for persistent vector indexing") from exc

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        with _CHROMA_INITIALIZATION_LOCK:
            self._client = chromadb.PersistentClient(path=str(self.path))
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def upsert(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=[_clean_metadata(metadata) for metadata in metadatas],
        )

    def query(self, embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=max(1, top_k),
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        texts = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        search_results: list[SearchResult] = []
        for item_id, text, metadata, distance in zip(ids, texts, metadatas, distances):
            score = 1.0 - float(distance)
            search_results.append(
                SearchResult(
                    id=str(item_id),
                    text=str(text or ""),
                    score=round(score, 6),
                    metadata=dict(metadata or {}),
                )
            )
        return search_results

    def count(self) -> int:
        return int(self._collection.count())

    def list_all(self) -> list[SearchResult]:
        result = self._collection.get(include=["documents", "metadatas"])
        ids = result.get("ids", [])
        texts = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        items: list[SearchResult] = []
        for item_id, text, metadata in zip(ids, texts, metadatas):
            items.append(
                SearchResult(
                    id=str(item_id),
                    text=str(text or ""),
                    score=0.0,
                    metadata=dict(metadata or {}),
                )
            )
        return items

    def delete_where(self, where: dict[str, Any]) -> int:
        if not where:
            return 0
        result = self._collection.get(where=_clean_metadata(where), include=[])
        ids = [str(item_id) for item_id in result.get("ids", [])]
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        return len(ids)


class InMemoryVectorStore:
    def __init__(self, collection_name: str = "rag_chunks") -> None:
        self.collection_name = collection_name
        self._items: dict[str, tuple[str, list[float], dict[str, Any]]] = {}

    def upsert(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for item_id, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
            self._items[item_id] = (text, embedding, _clean_metadata(metadata))

    def query(self, embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        scored: list[SearchResult] = []
        for item_id, (text, stored_embedding, metadata) in self._items.items():
            scored.append(
                SearchResult(
                    id=item_id,
                    text=text,
                    score=round(_cosine_similarity(embedding, stored_embedding), 6),
                    metadata=dict(metadata),
                )
            )
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._items)

    def list_all(self) -> list[SearchResult]:
        return [
            SearchResult(id=item_id, text=text, score=0.0, metadata=dict(metadata))
            for item_id, (text, _embedding, metadata) in self._items.items()
        ]

    def delete_where(self, where: dict[str, Any]) -> int:
        if not where:
            return 0
        deleted = 0
        for item_id, (_text, _embedding, metadata) in list(self._items.items()):
            if all(metadata.get(key) == value for key, value in where.items()):
                del self._items[item_id]
                deleted += 1
        return deleted


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    cleaned: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, bool | int | float | str):
            cleaned[str(key)] = value
        else:
            cleaned[str(key)] = str(value)
    return cleaned


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
