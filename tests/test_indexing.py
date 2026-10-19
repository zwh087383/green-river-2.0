from __future__ import annotations

import pytest

from rag_system.indexing import (
    BM25Index,
    IndexingService,
    InMemoryVectorStore,
    MissingApiKeyError,
    SparseEmbeddingEntry,
    SparseVectorIndex,
    tokenize,
)
from rag_system.indexing.manifest import IndexManifest
from rag_system.indexing.models import SearchResult
from rag_system.parsing.models import DocumentChunk, ParsedDocument
from rag_system.retrieval.query_rewrite import QueryRewriteResult
from rag_system.settings import SettingsStore


class FakeEmbeddingClient:
    model = "text-embedding-v4"
    dimensions = 4

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        lower = text.lower()
        return [
            1.0 if "alpha" in lower else 0.0,
            1.0 if "beta" in lower else 0.0,
            1.0 if "gamma" in lower else 0.0,
            1.0,
        ]


class OrderedVectorStore(InMemoryVectorStore):
    def query(self, embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        results = super().query(embedding, top_k=top_k)
        results.sort(key=lambda result: 0 if "semantic-only" in result.text else 1)
        return results[:top_k]


class FakeQueryRewriteClient:
    model = "deepseek-v4-flash"

    def rewrite(self, query: str) -> QueryRewriteResult:
        return QueryRewriteResult(
            original_query=query,
            rewritten_queries=["qwen3-vl-embedding exact keyword"],
            keywords=["qwen3-vl-embedding", "keyword"],
            intent="find exact embedding keyword",
            method="deepseek",
            model=self.model,
        )


class FakeSparseEmbeddingClient:
    model = "text-embedding-v4"

    def embed_sparse_texts(self, texts: list[str], *, text_type: str = "document"):
        vectors = []
        for text in texts:
            lower = text.lower()
            entries = []
            if "alpha" in lower:
                entries.append(SparseEmbeddingEntry(index=1, value=1.0, token="alpha"))
            if "qwen3" in lower or "qwen3-vl-embedding" in lower:
                entries.append(SparseEmbeddingEntry(index=2, value=2.0, token="qwen3-vl-embedding"))
            if "exact" in lower:
                entries.append(SparseEmbeddingEntry(index=3, value=1.5, token="exact"))
            vectors.append(entries)
        return vectors


class FakeReranker:
    model = "fake-reranker"

    def rerank(self, query: str, results: list[SearchResult], *, top_k: int) -> list[SearchResult]:
        ordered = sorted(results, key=lambda result: 0 if "rerank winner" in result.text else 1)
        for index, result in enumerate(ordered):
            result.metadata = {
                **result.metadata,
                "reranked": True,
                "reranker_model": self.model,
                "rerank_score": float(100 - index),
            }
        return ordered[:top_k]


def test_index_document_writes_vectors_manifest_and_searches(tmp_path) -> None:
    service = IndexingService(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        vector_store=InMemoryVectorStore(),
        manifest=IndexManifest(tmp_path / "index.sqlite3"),
        embedding_client=FakeEmbeddingClient(),
        bm25_index=BM25Index(tmp_path / "bm25.sqlite3"),
    )
    document = ParsedDocument(
        source=str(tmp_path / "demo.docx"),
        source_name="demo.docx",
        backend="auto",
        chunks=[
            DocumentChunk(id="chunk-alpha", text="alpha project introduction", metadata={"chunk_index": 0}),
            DocumentChunk(id="chunk-beta", text="beta deployment notes", metadata={"chunk_index": 1}),
        ],
        metadata={"file_sha256": "abc123"},
    )

    result = service.index_document(document)
    stats = service.stats()
    hits = service.search("alpha", top_k=1)

    assert result.status == "success"
    assert result.chunk_count == 2
    assert result.vector_count == 2
    assert result.dimensions == 4
    assert stats.document_count == 1
    assert stats.chunk_count == 2
    assert stats.vector_count == 2
    assert stats.bm25_chunk_count == 2
    assert hits[0].metadata["source_name"] == "demo.docx"
    assert "alpha" in hits[0].text
    assert hits[0].explanation["query_tokens"] == ["alpha"]


def test_index_document_skips_duplicate_chunk_text_and_search_deduplicates(tmp_path) -> None:
    store = InMemoryVectorStore()
    service = IndexingService(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        vector_store=store,
        manifest=IndexManifest(tmp_path / "index.sqlite3"),
        embedding_client=FakeEmbeddingClient(),
        bm25_index=BM25Index(tmp_path / "bm25.sqlite3"),
    )
    document = ParsedDocument(
        source=str(tmp_path / "repeated.docx"),
        source_name="repeated.docx",
        backend="auto",
        chunks=[
            DocumentChunk(id="chunk-alpha-1", text="alpha repeated text", metadata={"chunk_index": 0}),
            DocumentChunk(id="chunk-alpha-2", text="alpha   repeated\ntext", metadata={"chunk_index": 1}),
            DocumentChunk(id="chunk-beta", text="beta different text", metadata={"chunk_index": 2}),
        ],
        metadata={"file_sha256": "duplicate123"},
    )

    result = service.index_document(document)
    hits = service.search("alpha", top_k=5)

    assert result.chunk_count == 2
    assert result.vector_count == 2
    assert result.bm25_count == 2
    assert store.count() == 2
    assert [hit.text for hit in hits].count("alpha repeated text") == 1


def test_search_deduplicates_existing_duplicate_vectors(tmp_path) -> None:
    store = InMemoryVectorStore()
    store.upsert(
        ids=["a", "b", "c"],
        texts=["alpha same text", "alpha   same\ntext", "beta other text"],
        embeddings=[[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
        metadatas=[{"source_name": "one.docx"}, {"source_name": "one.docx"}, {"source_name": "two.docx"}],
    )
    service = IndexingService(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        vector_store=store,
        manifest=IndexManifest(tmp_path / "index.sqlite3"),
        embedding_client=FakeEmbeddingClient(),
        bm25_index=BM25Index(tmp_path / "bm25.sqlite3"),
    )

    hits = service.search("alpha", top_k=5)

    assert len(hits) == 2
    assert hits[0].metadata["duplicate_count"] == 2
    assert "alpha" in hits[0].text


def test_stats_backfills_bm25_from_existing_vector_store_items(tmp_path) -> None:
    store = InMemoryVectorStore()
    store.upsert(
        ids=["legacy-a", "legacy-b", "legacy-c"],
        texts=[
            "legacy OCR content from word/media images",
            "legacy Chroma vector content",
            "legacy qwen3-vl-embedding image content",
        ],
        embeddings=[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
        metadatas=[
            {"source_name": "legacy.docx", "text_hash": "legacy-a"},
            {"source_name": "legacy.md", "text_hash": "legacy-b"},
            {"source_name": "legacy-image.md", "text_hash": "legacy-c"},
        ],
    )
    bm25_index = BM25Index(tmp_path / "bm25.sqlite3")
    service = IndexingService(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        vector_store=store,
        manifest=IndexManifest(tmp_path / "index.sqlite3"),
        embedding_client=FakeEmbeddingClient(),
        bm25_index=bm25_index,
    )

    stats = service.stats()
    _, hits = bm25_index.search("word media OCR", top_k=1)

    assert stats.vector_count == 3
    assert stats.bm25_chunk_count == 3
    assert hits[0].id == "legacy-a"


def test_bm25_tokenize_preserves_mixed_model_and_system_tokens() -> None:
    tokens = tokenize("ROS2 text-embedding-v4 qwen3-vl-embedding 维度2048 图片OCR识别")

    assert "ros2" in tokens
    assert "text-embedding-v4" in tokens
    assert "qwen3-vl-embedding" in tokens
    assert "2048" in tokens
    assert "图片" in tokens
    assert "ocr" in tokens
    assert "识别" in tokens


def test_bm25_index_persists_rebuilds_and_explains_matches(tmp_path) -> None:
    path = tmp_path / "bm25.sqlite3"
    index = BM25Index(path)
    index.upsert_chunks(
        ids=["ocr", "vector", "image"],
        texts=[
            "DOCX embedded image OCR extracts text from word/media images.",
            "Chroma vector database stores text-embedding-v4 vectors.",
            "qwen3-vl-embedding indexes image vectors for visual search.",
        ],
        metadatas=[
            {"source_name": "parser.docx", "text_hash": "a"},
            {"source_name": "vector.md", "text_hash": "b"},
            {"source_name": "image.md", "text_hash": "c"},
        ],
    )

    rebuilt = BM25Index(path)
    query_tokens, hits = rebuilt.search("word media OCR", top_k=2)

    assert query_tokens == ["word", "media", "ocr"]
    assert hits[0].id == "ocr"
    assert hits[0].metadata["source_name"] == "parser.docx"
    assert {item["token"] for item in hits[0].matched_tokens} == {"word", "media", "ocr"}


def test_sparse_index_scores_dot_product_and_explains_matches(tmp_path) -> None:
    sparse_index = SparseVectorIndex(tmp_path / "sparse.sqlite3")
    sparse_index.upsert_chunks(
        ids=["exact", "other"],
        texts=["qwen3-vl-embedding exact keyword", "alpha content"],
        metadatas=[{"source_name": "one.md"}, {"source_name": "two.md"}],
        sparse_vectors=[
            [
                SparseEmbeddingEntry(index=2, value=2.0, token="qwen3-vl-embedding"),
                SparseEmbeddingEntry(index=3, value=1.0, token="exact"),
            ],
            [SparseEmbeddingEntry(index=1, value=1.0, token="alpha")],
        ],
    )

    hits = sparse_index.search(
        [
            SparseEmbeddingEntry(index=2, value=3.0, token="qwen3-vl-embedding"),
            SparseEmbeddingEntry(index=3, value=1.0, token="exact"),
        ],
        top_k=1,
    )

    assert sparse_index.count() == 2
    assert hits[0].id == "exact"
    assert hits[0].score == 7.0
    assert hits[0].matched_sparse[0]["token"] == "qwen3-vl-embedding"


def test_hybrid_search_rrf_promotes_chunk_found_by_vector_and_bm25(tmp_path) -> None:
    service = IndexingService(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        vector_store=OrderedVectorStore(),
        manifest=IndexManifest(tmp_path / "index.sqlite3"),
        embedding_client=FakeEmbeddingClient(),
        bm25_index=BM25Index(tmp_path / "bm25.sqlite3"),
    )
    document = ParsedDocument(
        source=str(tmp_path / "hybrid.md"),
        source_name="hybrid.md",
        backend="auto",
        chunks=[
            DocumentChunk(id="semantic", text="semantic-only alpha fallback", metadata={"chunk_index": 0}),
            DocumentChunk(id="exact", text="qwen3-vl-embedding exact keyword", metadata={"chunk_index": 1}),
            DocumentChunk(id="neutral", text="neutral gamma content", metadata={"chunk_index": 2}),
        ],
        metadata={"file_sha256": "hybrid123"},
    )

    service.index_document(document)
    hits = service.search("qwen3-vl-embedding", top_k=2)

    assert hits[0].text == "qwen3-vl-embedding exact keyword"
    assert hits[0].metadata["retrieval_source"] == "hybrid"
    assert hits[0].metadata["bm25_rank"] == 1
    assert hits[0].metadata["vector_rank"] == 2
    assert hits[0].metadata["matched_token_count"] == 4


def test_search_uses_sparse_route_when_sparse_embeddings_are_available(tmp_path) -> None:
    service = IndexingService(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        vector_store=InMemoryVectorStore(),
        manifest=IndexManifest(tmp_path / "index.sqlite3"),
        embedding_client=FakeEmbeddingClient(),
        bm25_index=BM25Index(tmp_path / "bm25.sqlite3"),
        sparse_embedding_client=FakeSparseEmbeddingClient(),
        sparse_index=SparseVectorIndex(tmp_path / "sparse.sqlite3"),
    )
    document = ParsedDocument(
        source=str(tmp_path / "sparse.md"),
        source_name="sparse.md",
        backend="auto",
        chunks=[
            DocumentChunk(id="exact", text="qwen3-vl-embedding exact keyword", metadata={"chunk_index": 0}),
            DocumentChunk(id="noise", text="beta unrelated content", metadata={"chunk_index": 1}),
        ],
        metadata={"file_sha256": "sparse123"},
    )

    result = service.index_document(document)
    _rewrite, hits = service.search_with_rewrite("qwen3 exact", top_k=1)

    assert result.sparse_count == 2
    assert hits[0].text == "qwen3-vl-embedding exact keyword"
    assert "sparse_original" in hits[0].metadata["routes"]
    assert hits[0].metadata["sparse_rank"] == 1
    assert hits[0].explanation["matched_sparse"]


def test_reranker_can_reorder_rrf_candidates(tmp_path) -> None:
    service = IndexingService(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        vector_store=InMemoryVectorStore(),
        manifest=IndexManifest(tmp_path / "index.sqlite3"),
        embedding_client=FakeEmbeddingClient(),
        bm25_index=BM25Index(tmp_path / "bm25.sqlite3"),
        reranker=FakeReranker(),
    )
    document = ParsedDocument(
        source=str(tmp_path / "rerank.md"),
        source_name="rerank.md",
        backend="auto",
        chunks=[
            DocumentChunk(id="alpha", text="alpha ordinary content", metadata={"chunk_index": 0}),
            DocumentChunk(id="winner", text="beta rerank winner", metadata={"chunk_index": 1}),
        ],
        metadata={"file_sha256": "rerank123"},
    )

    service.index_document(document)
    _rewrite, hits = service.search_with_rewrite("alpha", top_k=1)

    assert hits[0].text == "beta rerank winner"
    assert hits[0].metadata["reranked"] is True
    assert hits[0].metadata["reranker_model"] == "fake-reranker"


def test_search_with_deepseek_rewrite_uses_rewritten_query_routes(tmp_path) -> None:
    service = IndexingService(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        vector_store=InMemoryVectorStore(),
        manifest=IndexManifest(tmp_path / "index.sqlite3"),
        embedding_client=FakeEmbeddingClient(),
        bm25_index=BM25Index(tmp_path / "bm25.sqlite3"),
        query_rewrite_client=FakeQueryRewriteClient(),
    )
    document = ParsedDocument(
        source=str(tmp_path / "rewrite.md"),
        source_name="rewrite.md",
        backend="auto",
        chunks=[
            DocumentChunk(id="exact", text="qwen3-vl-embedding exact keyword", metadata={"chunk_index": 0}),
            DocumentChunk(id="noise", text="unrelated beta content", metadata={"chunk_index": 1}),
            DocumentChunk(id="more-noise", text="another gamma paragraph", metadata={"chunk_index": 2}),
        ],
        metadata={"file_sha256": "rewrite123"},
    )

    service.index_document(document)
    rewrite, hits = service.search_with_rewrite("图片模型怎么搜", top_k=1)

    assert rewrite.method == "deepseek"
    assert rewrite.rewritten_queries == ["qwen3-vl-embedding exact keyword"]
    assert hits[0].text == "qwen3-vl-embedding exact keyword"
    assert "bm25_rewrite_1" in hits[0].metadata["routes"]
    assert hits[0].metadata["route_count"] >= 1


def test_search_with_rewrite_falls_back_without_deepseek_key(tmp_path) -> None:
    service = IndexingService(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        vector_store=InMemoryVectorStore(),
        manifest=IndexManifest(tmp_path / "index.sqlite3"),
        embedding_client=FakeEmbeddingClient(),
        bm25_index=BM25Index(tmp_path / "bm25.sqlite3"),
    )
    document = ParsedDocument(
        source=str(tmp_path / "fallback.md"),
        source_name="fallback.md",
        backend="auto",
        chunks=[DocumentChunk(id="alpha", text="alpha content", metadata={"chunk_index": 0})],
        metadata={"file_sha256": "fallback123"},
    )

    service.index_document(document)
    rewrite, hits = service.search_with_rewrite("alpha", top_k=1)

    assert rewrite.method == "fallback"
    assert rewrite.rewritten_queries == []
    assert rewrite.error
    assert hits[0].text == "alpha content"


def test_document_disable_and_delete_control_search_indexes(tmp_path) -> None:
    store = InMemoryVectorStore()
    manifest = IndexManifest(tmp_path / "index.sqlite3")
    bm25_index = BM25Index(tmp_path / "bm25.sqlite3")
    sparse_index = SparseVectorIndex(tmp_path / "sparse.sqlite3")
    service = IndexingService(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        vector_store=store,
        manifest=manifest,
        embedding_client=FakeEmbeddingClient(),
        bm25_index=bm25_index,
        sparse_embedding_client=FakeSparseEmbeddingClient(),
        sparse_index=sparse_index,
    )
    document = ParsedDocument(
        source=str(tmp_path / "managed.md"),
        source_name="managed.md",
        backend="auto",
        chunks=[DocumentChunk(id="alpha", text="alpha managed content", metadata={"chunk_index": 0})],
        metadata={"file_sha256": "managed123"},
    )

    indexed = service.index_document(document)
    assert service.search("alpha", top_k=1)

    disabled = service.set_document_enabled(indexed.document_id, False)
    assert disabled is not None
    assert disabled.enabled is False
    assert service.search("alpha", top_k=1) == []

    enabled = service.set_document_enabled(indexed.document_id, True)
    assert enabled is not None
    assert enabled.enabled is True
    assert service.search("alpha", top_k=1)

    deleted = service.delete_document(indexed.document_id)

    assert deleted["text_vectors"] == 1
    assert deleted["bm25_chunks"] == 1
    assert deleted["sparse_chunks"] == 1
    assert store.count() == 0
    assert bm25_index.count() == 0
    assert sparse_index.count() == 0
    assert manifest.get_document(indexed.document_id) is None


def test_index_document_requires_api_key_without_injected_embedding_client(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    service = IndexingService(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        vector_store=InMemoryVectorStore(),
        manifest=IndexManifest(tmp_path / "index.sqlite3"),
        bm25_index=BM25Index(tmp_path / "bm25.sqlite3"),
    )
    document = ParsedDocument(
        source=str(tmp_path / "demo.txt"),
        source_name="demo.txt",
        backend="auto",
        chunks=[DocumentChunk(id="chunk", text="hello")],
    )

    with pytest.raises(MissingApiKeyError):
        service.index_document(document)
