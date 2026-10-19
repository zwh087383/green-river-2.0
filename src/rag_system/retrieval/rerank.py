from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from rag_system.indexing.models import SearchResult
    from rag_system.settings import AppSettings


class Reranker(Protocol):
    model: str

    def rerank(self, query: str, results: list[SearchResult], *, top_k: int) -> list[SearchResult]:
        ...


@dataclass(slots=True)
class DashScopeReranker:
    api_key: str
    endpoint_url: str
    model: str = "qwen3-rerank"
    instruct: str = "Given a web search query, retrieve relevant passages that answer the query."
    timeout_seconds: int = 60
    max_document_chars: int = 12000

    @classmethod
    def from_settings(cls, settings: AppSettings) -> "DashScopeReranker":
        if not settings.dashscope_api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not configured")
        return cls(
            api_key=settings.dashscope_api_key,
            endpoint_url=settings.dashscope_rerank_url,
            model=settings.reranker_model,
        )

    def rerank(self, query: str, results: list[SearchResult], *, top_k: int) -> list[SearchResult]:
        if not results:
            return []
        documents = [result.text[: self.max_document_chars] for result in results]
        payload = self._request_payload(query=query, documents=documents, top_n=min(top_k, len(results)))
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("requests package is required for DashScope reranking") from exc

        response = requests.post(
            self.endpoint_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"DashScope rerank request failed: {response.status_code} {response.text[:300]}")
        ranked_items = _extract_rerank_items(response.json())
        if not ranked_items:
            raise RuntimeError("DashScope rerank response returned no results")

        reranked: list[SearchResult] = []
        for item in ranked_items:
            original_index = int(item["index"])
            if original_index < 0 or original_index >= len(results):
                continue
            result = results[original_index]
            score = float(item["relevance_score"])
            result.metadata = {
                **result.metadata,
                "reranked": True,
                "reranker_model": self.model,
                "rerank_score": round(score, 6),
                "rerank_index": original_index,
            }
            reranked.append(result)

        if not reranked:
            raise RuntimeError("DashScope rerank response did not match any candidate index")
        reranked.sort(key=lambda result: float(result.metadata.get("rerank_score") or 0.0), reverse=True)
        return reranked[:top_k]

    def _request_payload(self, *, query: str, documents: list[str], top_n: int) -> dict[str, Any]:
        if self.model == "qwen3-rerank":
            return {
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "instruct": self.instruct,
            }
        return {
            "model": self.model,
            "input": {
                "query": query,
                "documents": documents,
            },
            "parameters": {
                "top_n": top_n,
                "return_documents": False,
            },
        }


@dataclass(slots=True)
class LocalCrossEncoderReranker:
    model: str = "BAAI/bge-reranker-base"
    local_files_only: bool = True
    max_length: int = 512
    batch_size: int = 8
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)

    def rerank(self, query: str, results: list[SearchResult], *, top_k: int) -> list[SearchResult]:
        if not results:
            return []
        tokenizer, model = self._load()
        scores: list[float] = []
        for start in range(0, len(results), self.batch_size):
            batch = results[start : start + self.batch_size]
            encoded = tokenizer(
                [query] * len(batch),
                [result.text for result in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            outputs = model(**encoded)
            logits = outputs.logits.detach().cpu()
            for item in logits:
                if item.numel() == 1:
                    scores.append(float(item.reshape(-1)[0]))
                else:
                    scores.append(float(item.reshape(-1)[-1]))

        reranked = []
        for result, score in zip(results, scores):
            result.metadata = {
                **result.metadata,
                "reranked": True,
                "reranker_model": self.model,
                "rerank_score": round(score, 6),
            }
            reranked.append(result)
        reranked.sort(key=lambda result: float(result.metadata.get("rerank_score") or 0.0), reverse=True)
        return reranked[:top_k]

    def _load(self):
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers package is required for local reranking") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.model, local_files_only=self.local_files_only)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model,
            local_files_only=self.local_files_only,
        )
        self._model.eval()
        return self._tokenizer, self._model


def _extract_rerank_items(payload: dict[str, Any]) -> list[dict[str, float | int]]:
    raw_results = payload.get("results")
    if raw_results is None:
        raw_results = payload.get("output", {}).get("results")
    if not isinstance(raw_results, list):
        raise RuntimeError("DashScope rerank response missing results")

    items: list[dict[str, float | int]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item["index"])
            relevance_score = float(item["relevance_score"])
        except (KeyError, TypeError, ValueError):
            continue
        items.append({"index": index, "relevance_score": relevance_score})
    return items


def mark_rerank_fallback(
    results: list[SearchResult],
    *,
    model: str,
    error: str,
) -> list[SearchResult]:
    for result in results:
        result.metadata = {
            **result.metadata,
            "reranked": False,
            "reranker_model": model,
            "rerank_error": error,
        }
    return results
