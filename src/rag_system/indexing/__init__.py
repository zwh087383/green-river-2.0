"""Vector indexing and retrieval support for parsed documents."""

from rag_system.indexing.bm25 import BM25Hit, BM25Index, tokenize
from rag_system.indexing.embeddings import (
    AliyunEmbeddingClient,
    AliyunSparseEmbeddingClient,
    EmbeddingClient,
    MissingApiKeyError,
    SparseEmbeddingClient,
    SparseEmbeddingEntry,
)
from rag_system.indexing.image_embeddings import ImageEmbeddingClient, QwenImageEmbeddingClient
from rag_system.indexing.image_service import ImageIndexingService
from rag_system.indexing.models import IndexedDocument, IndexStats, IndexingResult, SearchResult
from rag_system.indexing.sparse import SparseHit, SparseVectorIndex
from rag_system.indexing.service import IndexingService, QueryEmbeddingBundle
from rag_system.indexing.store import ChromaVectorStore, InMemoryVectorStore, VectorStore

__all__ = [
    "AliyunEmbeddingClient",
    "AliyunSparseEmbeddingClient",
    "BM25Hit",
    "BM25Index",
    "ChromaVectorStore",
    "EmbeddingClient",
    "ImageEmbeddingClient",
    "ImageIndexingService",
    "IndexedDocument",
    "IndexStats",
    "IndexingResult",
    "IndexingService",
    "QueryEmbeddingBundle",
    "InMemoryVectorStore",
    "MissingApiKeyError",
    "QwenImageEmbeddingClient",
    "SearchResult",
    "SparseEmbeddingClient",
    "SparseEmbeddingEntry",
    "SparseHit",
    "SparseVectorIndex",
    "tokenize",
    "VectorStore",
]
