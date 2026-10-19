from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Confidence = Literal["high", "medium", "low"]
Route = Literal["chat_direct", "rag_retrieve", "app_help", "clarify"]


@dataclass(slots=True)
class ConversationTurn:
    user: str
    assistant: str


@dataclass(slots=True)
class EvidenceChunk:
    id: int
    search_result_id: str
    text: str
    source_name: str
    chunk_id: str
    heading: str = ""
    heading_path: str = ""
    page: str | int | None = None
    section_index: int | None = None
    chunk_index: int | None = None
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Citation:
    id: int
    source_name: str
    chunk_id: str
    heading: str = ""
    heading_path: str = ""
    page: str | int | None = None
    section_index: int | None = None
    chunk_index: int | None = None
    score: float = 0.0
    rrf_score: float | None = None
    rerank_score: float | None = None
    retrieval_source: str = ""
    matched_token_ratio: float | None = None
    matched_token_count: int | None = None
    route_count: int | None = None
    quote: str = ""


@dataclass(slots=True)
class GeneratedAnswer:
    answer: str
    citation_ids: list[int] = field(default_factory=list)
    confidence: Confidence = "low"
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IntentRouteResult:
    route: Route
    needs_retrieval: bool
    confidence: float
    reason: str = ""
    standalone_query: str = ""
    rewrite_hints: list[str] = field(default_factory=list)
    model: str = ""
    method: str = "fallback"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnswerResult:
    query: str
    answer: str
    citations: list[Citation]
    evidence: list[EvidenceChunk]
    model: str
    confidence: Confidence = "low"
    retrieval: dict[str, Any] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for citation in payload.get("citations", []):
            if isinstance(citation, dict):
                citation.pop("rerank_score", None)
        return payload
