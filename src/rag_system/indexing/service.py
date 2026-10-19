from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_system.indexing.bm25 import BM25Hit, BM25Index, matched_tokens, tokenize
from rag_system.indexing.document_ids import document_id_for_document, file_sha256_for_document, text_hash
from rag_system.indexing.embeddings import (
    AliyunEmbeddingClient,
    AliyunSparseEmbeddingClient,
    EmbeddingClient,
    MissingApiKeyError,
    SparseEmbeddingClient,
)
from rag_system.indexing.manifest import IndexManifest
from rag_system.indexing.models import IndexedDocument, IndexStats, IndexingResult, SearchResult
from rag_system.indexing.sparse import SparseHit, SparseVectorIndex
from rag_system.indexing.store import ChromaVectorStore, VectorStore
from rag_system.parsing.models import ParsedDocument
from rag_system.settings import SettingsStore


@dataclass(slots=True)
class QueryEmbeddingBundle:
    """Best-effort query vectors fetched once per request."""

    dense: list[float] | None = None
    sparse: list[Any] | None = None
    dense_error: str | None = None
    sparse_error: str | None = None

    @property
    def errors(self) -> dict[str, str]:
        return {
            key: value[:300]
            for key, value in (("dense", self.dense_error), ("sparse", self.sparse_error))
            if value
        }


class IndexingService:
    def __init__(
        self,
        settings_store: SettingsStore | None = None,
        vector_store: VectorStore | None = None,
        manifest: IndexManifest | None = None,
        embedding_client: EmbeddingClient | None = None,
        bm25_index: BM25Index | None = None,
        sparse_embedding_client: SparseEmbeddingClient | None = None,
        sparse_index: SparseVectorIndex | None = None,
        query_rewrite_client: Any | None = None,
        reranker: Any | None = None,
        index_dir: Path | str = Path("data/index"),
        collection_name: str = "rag_chunks",
    ) -> None:
        self.settings_store = settings_store or SettingsStore()
        self.index_dir = Path(index_dir)
        self.collection_name = collection_name
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.sparse_embedding_client = sparse_embedding_client
        self.sparse_index = sparse_index
        # Legacy injections are accepted but intentionally unused.
        self.query_rewrite_client = query_rewrite_client
        self.reranker = reranker
        self.manifest = manifest or IndexManifest(self.index_dir / "index.sqlite3")
        self.embedding_client = embedding_client

    def index_document(self, document: ParsedDocument) -> IndexingResult:
        embedding_client = self._embedding_client()
        self.manifest.record_parsed_document(document)
        prepared = self._prepare_chunks(document, embedding_client)
        if not prepared:
            raise ValueError(f"No non-empty chunks to index for {document.source_name}")

        ids = [item["id"] for item in prepared]
        texts = [item["text"] for item in prepared]
        metadatas = [item["metadata"] for item in prepared]
        vectors = embedding_client.embed_texts(texts)
        if len(vectors) != len(texts):
            raise RuntimeError(f"Embedding count mismatch: expected {len(texts)}, got {len(vectors)}")
        for vector in vectors:
            if len(vector) != embedding_client.dimensions:
                raise RuntimeError(
                    f"Embedding dimension mismatch: expected {embedding_client.dimensions}, got {len(vector)}"
                )

        vector_store = self._vector_store()
        vector_store.upsert(ids=ids, texts=texts, embeddings=vectors, metadatas=metadatas)
        bm25_count = self._bm25_index().upsert_chunks(ids=ids, texts=texts, metadatas=metadatas)
        sparse_count = self._index_sparse_chunks(ids=ids, texts=texts, metadatas=metadatas)

        indexed_at = datetime.now(timezone.utc).isoformat()
        document_id = str(metadatas[0]["document_id"])
        source = document.source
        source_name = document.source_name
        file_sha256 = self._file_sha256(document)
        file_size = int(document.metadata.get("file_size") or _safe_file_size(document.source))
        file_type = str(document.metadata.get("suffix") or Path(document.source_name).suffix.lower())
        manifest_document = IndexedDocument(
            document_id=document_id,
            source=source,
            source_name=source_name,
            file_sha256=file_sha256,
            chunk_count=len(texts),
            vector_count=len(vectors),
            model=embedding_client.model,
            dimensions=embedding_client.dimensions,
            indexed_at=indexed_at,
            file_type=file_type,
            file_size=file_size,
            status="indexed",
            parse_status="success",
            index_status="success",
            bm25_count=bm25_count,
            sparse_count=sparse_count,
            cache_key=document.cache_key,
            updated_at=indexed_at,
        )
        self.manifest.upsert_document(manifest_document)
        return IndexingResult(
            status="success",
            document_id=document_id,
            source=source,
            source_name=source_name,
            chunk_count=len(texts),
            vector_count=len(vectors),
            model=embedding_client.model,
            dimensions=embedding_client.dimensions,
            collection=vector_store.collection_name,
            indexed_at=indexed_at,
            message=f"Indexed {len(vectors)} vectors, {bm25_count} BM25 chunks, and {sparse_count} sparse chunks",
            bm25_count=bm25_count,
            sparse_count=sparse_count,
        )

    def prefetch_query_embeddings(self, query: str) -> QueryEmbeddingBundle:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Search query cannot be empty")

        bundle = QueryEmbeddingBundle()
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-query") as executor:
            dense_future = executor.submit(self.prefetch_dense_query_embedding, cleaned)
            sparse_future = executor.submit(self.prefetch_sparse_query_embedding, cleaned)
            try:
                bundle.dense = dense_future.result()
            except Exception as exc:
                bundle.dense_error = str(exc)[:300]
            try:
                bundle.sparse = sparse_future.result()
            except Exception as exc:
                bundle.sparse_error = str(exc)[:300]
        return bundle

    def prefetch_dense_query_embedding(self, query: str) -> list[float]:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Search query cannot be empty")
        vectors = self._embedding_client().embed_texts([cleaned])
        if len(vectors) != 1:
            raise RuntimeError(f"Dense embedding count mismatch: expected 1, got {len(vectors)}")
        return list(vectors[0])

    def prefetch_sparse_query_embedding(self, query: str) -> list[Any]:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Search query cannot be empty")
        if not self.settings_store.load().sparse_embedding_enabled:
            return []
        vectors = self._sparse_embedding_client().embed_sparse_texts([cleaned], text_type="query")
        if len(vectors) != 1:
            raise RuntimeError(f"Sparse embedding count mismatch: expected 1, got {len(vectors)}")
        return list(vectors)

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        query_embeddings: QueryEmbeddingBundle | None = None,
        dense_vector: list[float] | None = None,
        sparse_vector: Any | None = None,
    ) -> list[SearchResult]:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Search query cannot be empty")
        bundle = query_embeddings if query_embeddings is not None else None
        if bundle is None and (dense_vector is not None or sparse_vector is not None):
            bundle = QueryEmbeddingBundle(
                dense=dense_vector,
                sparse=[sparse_vector] if sparse_vector is not None else [],
            )
        if bundle is None:
            bundle = self.prefetch_query_embeddings(cleaned)
        expanded_top_k = min(max(top_k * 4, top_k), 80)
        self._sync_bm25_from_vector_store_if_needed()
        self._sync_sparse_from_vector_store_if_needed()
        excluded_document_ids = self.manifest.disabled_document_ids()
        route_results: list[dict[str, Any]] = []
        if bundle.dense is not None:
            vector_candidates = self._filter_searchable(
                self._vector_store().query(bundle.dense, top_k=expanded_top_k),
                excluded_document_ids,
            )
            route_results.append(
                {
                    "route": "vector_original",
                    "kind": "vector",
                    "query": cleaned,
                    "results": vector_candidates,
                }
            )
        _query_tokens, bm25_candidates = self._bm25_index().search(
            cleaned,
            top_k=expanded_top_k,
            excluded_document_ids=excluded_document_ids,
        )
        route_results.append({"route": "bm25_original", "kind": "bm25", "query": cleaned, "results": bm25_candidates})
        if bundle.sparse:
            sparse_candidates = self._sparse_index().search(
                    bundle.sparse[0],
                    top_k=expanded_top_k,
                    excluded_document_ids=excluded_document_ids,
                )
            route_results.append({"route": "sparse_original", "kind": "sparse", "query": cleaned, "results": sparse_candidates})
        query_tokens = tokenize(cleaned)
        fused = self._fuse_route_results(
            query_tokens=query_tokens or tokenize(cleaned),
            route_results=route_results,
        )
        deduped = self._deduplicate_search_results(fused, top_k=min(max(top_k * 4, top_k), 40))
        return self._deduplicate_search_results(deduped, top_k=top_k)

    def tokenize_query(self, query: str) -> list[str]:
        return tokenize(query.strip())

    def stats(self) -> IndexStats:
        settings = self.settings_store.load()
        document_count, chunk_count = self.manifest.totals()
        vector_count = self._safe_vector_count()
        bm25_chunk_count = self._sync_bm25_from_vector_store_if_needed(vector_count=vector_count)
        sparse_chunk_count = self._safe_sparse_count()
        return IndexStats(
            document_count=document_count,
            chunk_count=chunk_count,
            vector_count=vector_count,
            collection=self.collection_name,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            bm25_chunk_count=bm25_chunk_count,
            sparse_chunk_count=sparse_chunk_count,
        )

    def list_documents(self) -> list[IndexedDocument]:
        return self.manifest.list_documents()

    def get_document_detail(self, document_id: str) -> dict[str, Any] | None:
        record = self.manifest.get_document(document_id)
        if record is None:
            return None
        chunks = self._document_vector_chunks(document_id)
        if not chunks:
            chunks = [
                {
                    "id": hit.id,
                    "text": hit.text,
                    "metadata": hit.metadata,
                    "tokens": hit.tokens[:120],
                }
                for hit in self._bm25_index().list_document_chunks(document_id)
            ]
        return {
            "document": asdict(record),
            "chunks": chunks,
            "counts": {
                "vector_count": len(self._document_vector_chunks(document_id)),
                "bm25_count": self._bm25_index().count_document(document_id),
                "sparse_count": self._sparse_index().count_document(document_id),
            },
        }

    def set_document_enabled(self, document_id: str, enabled: bool) -> IndexedDocument | None:
        return self.manifest.set_document_enabled(document_id, enabled)

    def delete_document(self, document_id: str, *, remove_manifest: bool = True) -> dict[str, Any]:
        deleted: dict[str, Any] = {
            "document_id": document_id,
            "text_vectors": 0,
            "bm25_chunks": 0,
            "sparse_chunks": 0,
            "manifest": False,
        }
        try:
            deleted["text_vectors"] = self._vector_store().delete_where({"document_id": document_id})
        except RuntimeError:
            deleted["text_vectors"] = 0
        deleted["bm25_chunks"] = self._bm25_index().delete_document(document_id)
        deleted["sparse_chunks"] = self._sparse_index().delete_document(document_id)
        if remove_manifest:
            deleted["manifest"] = self.manifest.delete_document(document_id)
        return deleted

    def _prepare_chunks(
        self,
        document: ParsedDocument,
        embedding_client: EmbeddingClient,
    ) -> list[dict[str, Any]]:
        document_id = self._document_id(document)
        prepared: list[dict[str, Any]] = []
        seen_text_hashes: set[str] = set()
        for fallback_index, chunk in enumerate(document.chunks):
            text = chunk.text.strip()
            if not text:
                continue
            text_hash_value = self._text_hash(text)
            if text_hash_value in seen_text_hashes:
                continue
            seen_text_hashes.add(text_hash_value)
            vector_id = self._vector_id(document_id, chunk.id, fallback_index)
            metadata = {
                **chunk.metadata,
                "document_id": document_id,
                "chunk_id": chunk.id,
                "vector_id": vector_id,
                "source": document.source,
                "source_name": document.source_name,
                "file_sha256": self._file_sha256(document),
                "text_hash": text_hash_value,
                "model": embedding_client.model,
                "dimensions": embedding_client.dimensions,
            }
            prepared.append({"id": vector_id, "text": text, "metadata": metadata})
        return prepared

    def _deduplicate_search_results(
        self,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        deduped: list[SearchResult] = []
        seen_hashes: dict[str, SearchResult] = {}
        duplicate_counts: dict[str, int] = {}
        for result in results:
            text_hash_value = self._text_hash(result.text)
            existing_duplicate_count = int(result.metadata.get("duplicate_count") or 1)
            if text_hash_value in seen_hashes:
                duplicate_counts[text_hash_value] = duplicate_counts.get(text_hash_value, 1) + existing_duplicate_count
                continue
            result.metadata = {
                **result.metadata,
                "text_hash": text_hash_value,
                "duplicate_count": existing_duplicate_count,
            }
            seen_hashes[text_hash_value] = result
            duplicate_counts[text_hash_value] = existing_duplicate_count
            deduped.append(result)

        for result in deduped:
            text_hash = str(result.metadata.get("text_hash", ""))
            result.metadata["duplicate_count"] = duplicate_counts.get(text_hash, 1)
        return deduped[:top_k]

    def _fuse_search_results(
        self,
        query_tokens: list[str],
        vector_results: list[SearchResult],
        bm25_results: list[BM25Hit],
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        return self._fuse_route_results(
            query_tokens=query_tokens,
            route_results=[
                {"route": "vector_original", "kind": "vector", "query": "", "results": vector_results},
                {"route": "bm25_original", "kind": "bm25", "query": "", "results": bm25_results},
            ],
            rrf_k=rrf_k,
        )

    def _fuse_route_results(
        self,
        query_tokens: list[str],
        route_results: list[dict[str, Any]],
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        candidates: dict[str, dict[str, Any]] = {}

        def candidate_for(item_id: str, text: str, metadata: dict[str, Any]) -> dict[str, Any]:
            candidate = candidates.setdefault(
                item_id,
                {
                    "id": item_id,
                    "text": text,
                    "metadata": dict(metadata),
                    "vector_rank": None,
                    "vector_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "sparse_rank": None,
                    "sparse_score": None,
                    "matched_sparse": [],
                    "tokens": [],
                    "matched_tokens": [],
                    "routes": [],
                },
            )
            if text and not candidate["text"]:
                candidate["text"] = text
            candidate["metadata"] = {**candidate["metadata"], **metadata}
            return candidate

        for route_payload in route_results:
            route = str(route_payload.get("route") or "")
            kind = str(route_payload.get("kind") or "")
            route_query = str(route_payload.get("query") or "")
            for rank, result in enumerate(route_payload.get("results") or [], start=1):
                candidate = candidate_for(result.id, result.text, result.metadata)
                score = float(result.score)
                if kind == "vector":
                    if candidate["vector_rank"] is None or rank < int(candidate["vector_rank"]):
                        candidate["vector_rank"] = rank
                    if candidate["vector_score"] is None or score > float(candidate["vector_score"]):
                        candidate["vector_score"] = result.score
                elif kind == "bm25":
                    if candidate["bm25_rank"] is None or rank < int(candidate["bm25_rank"]):
                        candidate["bm25_rank"] = rank
                    if candidate["bm25_score"] is None or score > float(candidate["bm25_score"]):
                        candidate["bm25_score"] = result.score
                    if isinstance(result, BM25Hit):
                        candidate["tokens"] = result.tokens
                        candidate["matched_tokens"] = result.matched_tokens
                elif kind == "sparse":
                    if candidate["sparse_rank"] is None or rank < int(candidate["sparse_rank"]):
                        candidate["sparse_rank"] = rank
                    if candidate["sparse_score"] is None or score > float(candidate["sparse_score"]):
                        candidate["sparse_score"] = result.score
                    if isinstance(result, SparseHit):
                        candidate["matched_sparse"] = result.matched_sparse
                candidate["routes"].append(
                    {
                        "route": route,
                        "kind": kind,
                        "query": route_query,
                        "rank": rank,
                        "score": result.score,
                        "rrf_contribution": round(1.0 / (rrf_k + rank), 6),
                    }
                )

        fused: list[SearchResult] = []
        unique_query_tokens = set(query_tokens)
        for candidate in candidates.values():
            vector_rank = candidate["vector_rank"]
            bm25_rank = candidate["bm25_rank"]
            sparse_rank = candidate["sparse_rank"]
            vector_contribution = sum(
                float(route["rrf_contribution"]) for route in candidate["routes"] if route["kind"] == "vector"
            )
            bm25_contribution = sum(
                float(route["rrf_contribution"]) for route in candidate["routes"] if route["kind"] == "bm25"
            )
            sparse_contribution = sum(
                float(route["rrf_contribution"]) for route in candidate["routes"] if route["kind"] == "sparse"
            )
            rrf_score = vector_contribution + bm25_contribution + sparse_contribution
            tokens = candidate["tokens"] or tokenize(str(candidate["text"]))
            matches = candidate["matched_tokens"] or matched_tokens(query_tokens, tokens)
            retrieval_source = self._retrieval_source(vector_rank, bm25_rank, sparse_rank)
            matched_ratio = len(matches) / len(unique_query_tokens) if unique_query_tokens else 0.0
            metadata = {
                **candidate["metadata"],
                "retrieval_source": retrieval_source,
                "rrf_score": round(rrf_score, 6),
                "vector_rank": vector_rank,
                "bm25_rank": bm25_rank,
                "sparse_rank": sparse_rank,
                "vector_score": candidate["vector_score"],
                "bm25_score": candidate["bm25_score"],
                "sparse_score": candidate["sparse_score"],
                "vector_contribution": round(vector_contribution, 6),
                "bm25_contribution": round(bm25_contribution, 6),
                "sparse_contribution": round(sparse_contribution, 6),
                "matched_token_count": len(matches),
                "matched_token_ratio": round(matched_ratio, 6),
                "matched_sparse_count": len(candidate["matched_sparse"]),
                "routes": [route["route"] for route in candidate["routes"]],
                "route_details": candidate["routes"],
                "route_count": len(candidate["routes"]),
            }
            fused.append(
                SearchResult(
                    id=str(candidate["id"]),
                    text=str(candidate["text"]),
                    score=round(rrf_score, 6),
                    metadata=metadata,
                    explanation={
                        "query_tokens": query_tokens,
                        "chunk_tokens_preview": tokens[:80],
                        "matched_tokens": matches,
                        "matched_sparse": candidate["matched_sparse"],
                        "routes": candidate["routes"],
                    },
                )
            )

        fused.sort(
            key=lambda result: (
                result.score,
                int(result.metadata.get("matched_token_count", 0)),
                float(result.metadata.get("vector_score") or 0.0),
                float(result.metadata.get("bm25_score") or 0.0),
                float(result.metadata.get("sparse_score") or 0.0),
            ),
            reverse=True,
        )
        return fused

    def _combined_query_tokens(self, queries: list[str]) -> list[str]:
        tokens: list[str] = []
        seen: set[str] = set()
        for query in queries:
            for token in tokenize(query):
                if token not in seen:
                    seen.add(token)
                    tokens.append(token)
        return tokens

    @staticmethod
    def _retrieval_source(vector_rank: int | None, bm25_rank: int | None, sparse_rank: int | None = None) -> str:
        active = [name for name, rank in (("vector", vector_rank), ("bm25", bm25_rank), ("sparse", sparse_rank)) if rank]
        if len(active) > 1:
            return "hybrid"
        return active[0] if active else "unknown"

    def _embedding_client(self) -> EmbeddingClient:
        if self.embedding_client is not None:
            return self.embedding_client
        self.embedding_client = AliyunEmbeddingClient.from_settings(self.settings_store.load())
        return self.embedding_client

    def _vector_store(self) -> VectorStore:
        if self.vector_store is None:
            self.vector_store = ChromaVectorStore(
                path=self.index_dir / "chroma",
                collection_name=self.collection_name,
            )
        return self.vector_store

    def _bm25_index(self) -> BM25Index:
        if self.bm25_index is None:
            self.bm25_index = BM25Index(self.index_dir / "bm25.sqlite3")
        return self.bm25_index

    def _sparse_embedding_client(self) -> SparseEmbeddingClient:
        if self.sparse_embedding_client is not None:
            return self.sparse_embedding_client
        self.sparse_embedding_client = AliyunSparseEmbeddingClient.from_settings(self.settings_store.load())
        return self.sparse_embedding_client

    def _sparse_index(self) -> SparseVectorIndex:
        if self.sparse_index is None:
            self.sparse_index = SparseVectorIndex(self.index_dir / "sparse.sqlite3")
        return self.sparse_index

    def _safe_vector_count(self) -> int:
        try:
            return self._vector_store().count()
        except RuntimeError:
            return 0

    def _safe_bm25_count(self) -> int:
        try:
            return self._bm25_index().count()
        except RuntimeError:
            return 0

    def _safe_sparse_count(self) -> int:
        try:
            return self._sparse_index().count()
        except RuntimeError:
            return 0

    def _filter_searchable(
        self,
        results: list[SearchResult],
        excluded_document_ids: set[str],
    ) -> list[SearchResult]:
        if not excluded_document_ids:
            return results
        filtered = []
        for result in results:
            document_id = str(result.metadata.get("document_id") or "")
            if document_id and document_id in excluded_document_ids:
                continue
            filtered.append(result)
        return filtered

    def _document_vector_chunks(self, document_id: str) -> list[dict[str, Any]]:
        try:
            items = self._vector_store().list_all()
        except RuntimeError:
            return []
        chunks = [
            {
                "id": item.id,
                "text": item.text,
                "score": item.score,
                "metadata": item.metadata,
            }
            for item in items
            if str(item.metadata.get("document_id") or "") == document_id
        ]
        chunks.sort(key=lambda item: str(item["metadata"].get("chunk_id") or item["id"]))
        return chunks

    def _sync_bm25_from_vector_store_if_needed(self, vector_count: int | None = None) -> int:
        bm25_count = self._safe_bm25_count()
        vector_total = self._safe_vector_count() if vector_count is None else vector_count
        if vector_total <= 0 or bm25_count >= vector_total:
            return bm25_count
        try:
            items = self._vector_store().list_all()
        except RuntimeError:
            return bm25_count

        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for item in items:
            text = item.text.strip()
            if not text:
                continue
            metadata = dict(item.metadata)
            metadata.setdefault("vector_id", item.id)
            metadata.setdefault("chunk_id", metadata.get("chunk_id", item.id))
            metadata.setdefault("text_hash", self._text_hash(text))
            ids.append(item.id)
            texts.append(text)
            metadatas.append(metadata)
        if not ids:
            return bm25_count
        self._bm25_index().upsert_chunks(ids=ids, texts=texts, metadatas=metadatas)
        return self._safe_bm25_count()

    def _sync_sparse_from_vector_store_if_needed(self, vector_count: int | None = None) -> int:
        if not self.settings_store.load().sparse_embedding_enabled:
            return self._safe_sparse_count()
        sparse_count = self._safe_sparse_count()
        vector_total = self._safe_vector_count() if vector_count is None else vector_count
        if vector_total <= 0 or sparse_count >= vector_total:
            return sparse_count
        try:
            items = self._vector_store().list_all()
        except RuntimeError:
            return sparse_count
        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for item in items:
            text = item.text.strip()
            if not text:
                continue
            metadata = dict(item.metadata)
            metadata.setdefault("vector_id", item.id)
            metadata.setdefault("chunk_id", metadata.get("chunk_id", item.id))
            metadata.setdefault("text_hash", self._text_hash(text))
            ids.append(item.id)
            texts.append(text)
            metadatas.append(metadata)
        if not ids:
            return sparse_count
        try:
            self._index_sparse_chunks(ids=ids, texts=texts, metadatas=metadatas)
        except Exception:
            return sparse_count
        return self._safe_sparse_count()

    def _index_sparse_chunks(self, ids: list[str], texts: list[str], metadatas: list[dict[str, Any]]) -> int:
        if not self.settings_store.load().sparse_embedding_enabled:
            return 0
        try:
            sparse_vectors = self._sparse_embedding_client().embed_sparse_texts(texts, text_type="document")
        except Exception:
            return 0
        return self._sparse_index().upsert_chunks(
            ids=ids,
            texts=texts,
            metadatas=metadatas,
            sparse_vectors=sparse_vectors,
        )

    def _embed_sparse_queries(self, queries: list[str]):
        if not self.settings_store.load().sparse_embedding_enabled:
            return []
        try:
            return self._sparse_embedding_client().embed_sparse_texts(queries, text_type="query")
        except Exception:
            return []

    def _document_id(self, document: ParsedDocument) -> str:
        return document_id_for_document(document)

    def _vector_id(self, document_id: str, chunk_id: str, fallback_index: int) -> str:
        digest = hashlib.sha1(f"{document_id}:{chunk_id}:{fallback_index}".encode("utf-8")).hexdigest()[:20]
        return f"{document_id}:{digest}"

    def _file_sha256(self, document: ParsedDocument) -> str:
        return file_sha256_for_document(document)

    def _text_hash(self, text: str) -> str:
        return text_hash(text)


def indexing_result_to_dict(result: IndexingResult) -> dict[str, Any]:
    return asdict(result)


def _safe_file_size(path: Path | str) -> int:
    try:
        resolved = Path(path)
        return resolved.stat().st_size if resolved.exists() else 0
    except OSError:
        return 0
