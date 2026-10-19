from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import requests

from rag_system.indexing.embeddings import MissingApiKeyError
from rag_system.settings import AppSettings


class ImageEmbeddingClient(Protocol):
    model: str
    dimensions: int

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass(slots=True)
class QwenImageEmbeddingClient:
    api_key: str
    endpoint: str
    model: str = "qwen3-vl-embedding"
    dimensions: int = 2048
    timeout_seconds: int = 60

    @classmethod
    def from_settings(cls, settings: AppSettings) -> "QwenImageEmbeddingClient":
        if not settings.dashscope_api_key:
            raise MissingApiKeyError("DASHSCOPE_API_KEY is not configured")
        return cls(
            api_key=settings.dashscope_api_key,
            endpoint=settings.dashscope_multimodal_url,
            model=settings.image_embedding_model,
            dimensions=settings.image_embedding_dimensions,
        )

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for path in image_paths:
            vectors.append(self._embed_content({"image": image_to_data_uri(path)}))
        return vectors

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            cleaned = text.strip()
            if cleaned:
                vectors.append(self._embed_content({"text": cleaned}))
        return vectors

    def _embed_content(self, content: dict[str, str]) -> list[float]:
        payload = {
            "model": self.model,
            "input": {"contents": [content]},
            "parameters": {
                "enable_fusion": True,
                "dimension": self.dimensions,
            },
        }
        response = requests.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Image embedding request failed: HTTP {response.status_code} {response.text[:300]}")
        vector = _extract_embedding_vector(response.json())
        if len(vector) != self.dimensions:
            raise RuntimeError(f"Image embedding dimension mismatch: expected {self.dimensions}, got {len(vector)}")
        return vector


def image_to_data_uri(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


def _extract_embedding_vector(payload: dict[str, Any]) -> list[float]:
    output = payload.get("output")
    if isinstance(output, dict):
        embedding = output.get("embedding")
        if isinstance(embedding, list):
            return [float(value) for value in embedding]
        embeddings = output.get("embeddings")
        if isinstance(embeddings, list) and embeddings:
            first = embeddings[0]
            if isinstance(first, dict) and isinstance(first.get("embedding"), list):
                return [float(value) for value in first["embedding"]]
            if isinstance(first, list):
                return [float(value) for value in first]

    data = payload.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and isinstance(first.get("embedding"), list):
            return [float(value) for value in first["embedding"]]

    raise RuntimeError(f"Could not find embedding vector in response keys: {sorted(payload.keys())}")
