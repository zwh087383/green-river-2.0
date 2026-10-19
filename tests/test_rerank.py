from __future__ import annotations

from rag_system.indexing.models import SearchResult
from rag_system.retrieval.rerank import DashScopeReranker, _extract_rerank_items


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def test_dashscope_reranker_orders_candidates_and_marks_metadata(monkeypatch) -> None:
    captured: dict = {}

    def fake_post(url, *, headers, json, timeout):
        captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.92},
                    {"index": 0, "relevance_score": 0.31},
                ],
                "model": "qwen3-rerank",
            }
        )

    monkeypatch.setattr("requests.post", fake_post)
    reranker = DashScopeReranker(
        api_key="sk-test-rerank",
        endpoint_url="https://dashscope.aliyuncs.com/compatible-api/v1/reranks",
        model="qwen3-rerank",
    )
    results = [
        SearchResult(id="a", text="less relevant", score=0.1, metadata={}),
        SearchResult(id="b", text="more relevant", score=0.2, metadata={}),
    ]

    reranked = reranker.rerank("which passage is relevant", results, top_k=2)

    assert [result.id for result in reranked] == ["b", "a"]
    assert reranked[0].metadata["reranked"] is True
    assert reranked[0].metadata["reranker_model"] == "qwen3-rerank"
    assert reranked[0].metadata["rerank_score"] == 0.92
    assert captured["json"]["query"] == "which passage is relevant"
    assert captured["json"]["documents"] == ["less relevant", "more relevant"]
    assert captured["json"]["top_n"] == 2
    assert captured["headers"]["Authorization"] == "Bearer sk-test-rerank"


def test_extract_rerank_items_accepts_legacy_output_shape() -> None:
    payload = {
        "output": {
            "results": [
                {"index": 2, "relevance_score": 0.7},
                {"index": 0, "relevance_score": 0.4},
            ]
        }
    }

    assert _extract_rerank_items(payload) == [
        {"index": 2, "relevance_score": 0.7},
        {"index": 0, "relevance_score": 0.4},
    ]
