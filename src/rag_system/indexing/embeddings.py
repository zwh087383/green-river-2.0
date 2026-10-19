from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Protocol

from rag_system.settings import AppSettings


class MissingApiKeyError(RuntimeError):
    """Raised when an embedding request needs an API key but none is configured."""


class EmbeddingClient(Protocol):
    model: str
    dimensions: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass(slots=True)
class EmbeddingGemmaClient:
    """Local 768d search/document embeddings; never falls back to another model."""

    endpoint_url: str = "http://127.0.0.1:8765/v1/embeddings"
    model: str = "google/embeddinggemma-2"
    dimensions: int = 768
    batch_size: int = 16
    timeout: float = 20.0

    @classmethod
    def from_settings(cls, settings: AppSettings) -> "EmbeddingGemmaClient":
        return cls(endpoint_url=settings.embedding_url)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "Document")

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "SearchQuery")

    def _embed(self, texts: list[str], prompt_name: str) -> list[list[float]]:
        import httpx

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding inputs must be non-empty text")
        vectors: list[list[float]] = []
        with httpx.Client(timeout=self.timeout, trust_env=False) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start:start + self.batch_size]
                response = client.post(self.endpoint_url, json={
                    "model": self.model, "input": batch, "dimensions": self.dimensions,
                    "encoding_format": "float", "prompt_name": prompt_name,
                })
                response.raise_for_status()
                payload = response.json()
                if payload.get("model") != self.model:
                    raise RuntimeError("Embedding server returned an unexpected model")
                items = sorted(payload.get("data", []), key=lambda item: item["index"])
                if [item["index"] for item in items] != list(range(len(batch))):
                    raise RuntimeError("Embedding response indexes do not match inputs")
                for item in items:
                    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in item["embedding"]):
                        raise RuntimeError("Embedding response contains non-numeric values")
                    vector = [float(value) for value in item["embedding"]]
                    if len(vector) != self.dimensions or not all(math.isfinite(v) for v in vector):
                        raise RuntimeError("Embedding response has invalid dimensions or values")
                    norm = math.sqrt(sum(value * value for value in vector))
                    if not math.isfinite(norm) or norm <= 0:
                        raise RuntimeError("Embedding response contains a zero vector")
                    vectors.append([value / norm for value in vector])
        return vectors


@dataclass(slots=True)
class SparseEmbeddingEntry:
    index: int
    value: float
    token: str = ""


SparseEmbedding = list[SparseEmbeddingEntry]
TextType = Literal["document", "query"]


class SparseEmbeddingClient(Protocol):
    model: str

    def embed_sparse_texts(self, texts: list[str], *, text_type: TextType = "document") -> list[SparseEmbedding]:
        ...


@dataclass(slots=True)
class AliyunEmbeddingClient:
    api_key: str
    base_url: str
    model: str
    dimensions: int = 1024
    batch_size: int = 10

    @classmethod
    def from_settings(cls, settings: AppSettings) -> "AliyunEmbeddingClient":
        if not settings.dashscope_api_key:
            raise MissingApiKeyError("DASHSCOPE_API_KEY is not configured")
        return cls(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        cleaned = [text.strip() for text in texts if text and text.strip()]
        if not cleaned:
            return []

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for Aliyun embeddings") from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=60.0)
        vectors: list[list[float]] = []
        for start in range(0, len(cleaned), self.batch_size):
            batch = cleaned[start : start + self.batch_size]
            response = client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimensions,
                encoding_format="float",
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend([list(item.embedding) for item in ordered])

        if len(vectors) != len(cleaned):
            raise RuntimeError(f"Embedding count mismatch: expected {len(cleaned)}, got {len(vectors)}")
        return vectors


@dataclass(slots=True)
class AliyunSparseEmbeddingClient:
    api_key: str
    endpoint_url: str
    model: str
    dimensions: int = 1024
    batch_size: int = 10

    @classmethod
    def from_settings(cls, settings: AppSettings) -> "AliyunSparseEmbeddingClient":
        if not settings.dashscope_api_key:
            raise MissingApiKeyError("DASHSCOPE_API_KEY is not configured")
        return cls(
            api_key=settings.dashscope_api_key,
            endpoint_url=settings.dashscope_text_embedding_url,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )

    def embed_sparse_texts(self, texts: list[str], *, text_type: TextType = "document") -> list[SparseEmbedding]:
        cleaned = [text.strip() for text in texts if text and text.strip()]
        if not cleaned:
            return []

        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("requests package is required for Aliyun sparse embeddings") from exc

        sparse_vectors: list[SparseEmbedding] = []
        for start in range(0, len(cleaned), self.batch_size):
            batch = cleaned[start : start + self.batch_size]
            response = requests.post(
                self.endpoint_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": {"texts": batch},
                    "parameters": {
                        "text_type": text_type,
                        "dimension": self.dimensions,
                        "output_type": "sparse",
                    },
                },
                timeout=60,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"Aliyun sparse embedding request failed: {response.status_code} {response.text[:300]}")
            payload = response.json()
            sparse_vectors.extend(_extract_sparse_embeddings(payload, expected_count=len(batch)))

        if len(sparse_vectors) != len(cleaned):
            raise RuntimeError(f"Sparse embedding count mismatch: expected {len(cleaned)}, got {len(sparse_vectors)}")
        return sparse_vectors


def _extract_sparse_embeddings(payload: dict[str, Any], *, expected_count: int) -> list[SparseEmbedding]:
    embeddings = payload.get("output", {}).get("embeddings", [])
    if not isinstance(embeddings, list):
        raise RuntimeError("Aliyun sparse embedding response missing output.embeddings")
    ordered = sorted(embeddings, key=lambda item: int(item.get("text_index", item.get("index", 0))))
    sparse_vectors = [_parse_sparse_embedding(item.get("sparse_embedding", [])) for item in ordered]
    if len(sparse_vectors) != expected_count:
        raise RuntimeError(f"Sparse embedding response count mismatch: expected {expected_count}, got {len(sparse_vectors)}")
    return sparse_vectors


def _parse_sparse_embedding(value: Any) -> SparseEmbedding:
    if not isinstance(value, list):
        return []
    entries: SparseEmbedding = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
            score = float(item.get("value"))
        except (TypeError, ValueError):
            continue
        token = str(item.get("token") or item.get("word") or "")
        entries.append(SparseEmbeddingEntry(index=index, value=score, token=token))
    return entries
