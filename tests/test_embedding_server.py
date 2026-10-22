from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag_system.indexing.embedding_server import create_app


class Encoder:
    device = "test"

    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return [[1.0] + [0.0] * 767 for _ in texts]


def test_local_embedding_server_preserves_prompt_and_vector_contract():
    encoder = Encoder()
    with TestClient(create_app(model_factory=lambda: encoder)) as client:
        assert client.get("/health").json()["dimensions"] == 768
        for prompt in ("Document", "SearchQuery"):
            response = client.post("/v1/embeddings", json={"input": ["绿色江河", "环保"], "prompt_name": prompt})
            assert response.status_code == 200
            assert response.json()["model"] == "google/embeddinggemma-2"
            assert [item["index"] for item in response.json()["data"]] == [0, 1]
            assert encoder.calls[-1][1]["prompt_name"] == prompt
            assert encoder.calls[-1][1]["normalize_embeddings"] is True


@pytest.mark.parametrize("payload", [
    {"input": ""}, {"input": []}, {"input": ["x"] * 17},
    {"input": "x", "model": "another-model"},
    {"input": "x", "dimensions": 1024}, {"input": "x", "prompt_name": "wrong"},
])
def test_local_embedding_server_rejects_invalid_requests(payload):
    with TestClient(create_app(model_factory=Encoder)) as client:
        assert client.post("/v1/embeddings", json=payload).status_code in (400, 422)


def test_local_embedding_server_fails_on_nonfinite_or_zero_vectors():
    encoder = Encoder()
    with TestClient(create_app(model_factory=lambda: encoder)) as client:
        encoder.encode = lambda *args, **kwargs: [[float("nan")] * 768]
        assert client.post("/v1/embeddings", json={"input": "query"}).status_code == 500
