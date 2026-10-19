from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol, runtime_checkable


@runtime_checkable
class AnswerService(Protocol):
    """Minimal generation contract hidden behind :class:`AgentNode`."""

    def answer(self, query: str, *, top_k: int | None = None) -> Any: ...

    def stream_answer_events(
        self,
        query: str,
        *,
        top_k: int | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> Iterator[dict[str, Any]]: ...


class AgentNode:
    """Customer-facing answer node that hides retrieval and LLM orchestration."""

    module_name = "agent"

    def __init__(
        self,
        *,
        service: AnswerService | None = None,
        service_factory: Callable[[], AnswerService] | None = None,
    ) -> None:
        self._service_instance = service
        self._service_factory = service_factory or _default_service

    def answer(
        self,
        query: str,
        *,
        top_k: int | None = None,
        include_details: bool = False,
    ) -> dict[str, Any]:
        """Return one complete answer as a JSON-ready result."""

        cleaned = _validate_query(query)
        validated_top_k = _validate_top_k(top_k)
        result = _to_jsonable(self._service().answer(cleaned, top_k=validated_top_k))
        return _success(
            self.module_name,
            "answer",
            result if include_details else _compact_answer(result),
        )

    def stream(
        self,
        query: str,
        *,
        top_k: int | None = None,
        include_details: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Yield stable event envelopes while the answer is generated."""

        cleaned = _validate_query(query)
        validated_top_k = _validate_top_k(top_k)
        for event in self._service().stream_answer_events(cleaned, top_k=validated_top_k):
            payload = _to_jsonable(event)
            yield _success(
                self.module_name,
                "stream",
                payload if include_details else _compact_stream_event(payload),
            )

    def _service(self) -> AnswerService:
        if self._service_instance is None:
            self._service_instance = self._service_factory()
        return self._service_instance


def build_default_node(
    *,
    index_dir: Path | str = Path("data/index"),
) -> AgentNode:
    resolved_index_dir = Path(index_dir)
    return AgentNode(service_factory=lambda: _default_service(index_dir=resolved_index_dir))


def _default_service(*, index_dir: Path | str = Path("data/index")) -> AnswerService:
    import logging

    import jieba

    from rag_system.generation import AnswerGenerationService

    jieba.setLogLevel(logging.ERROR)
    return AnswerGenerationService(index_dir=index_dir)


def _validate_query(query: str) -> str:
    cleaned = str(query or "").strip()
    if not cleaned:
        raise ValueError("query cannot be empty")
    return cleaned


def _validate_top_k(top_k: int | None) -> int | None:
    if top_k is None:
        return None
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("top_k must be an integer")
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")
    return top_k


def _success(module: str, operation: str, data: Any) -> dict[str, Any]:
    return {"ok": True, "module": module, "operation": operation, "data": data}


def _compact_answer(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    compact = {
        key: value[key]
        for key in ("query", "answer", "model", "confidence")
        if key in value
    }
    citations = value.get("citations")
    if isinstance(citations, list):
        compact["citation_count"] = len(citations)
        compact["sources"] = list(
            dict.fromkeys(
                str(item.get("source_name"))
                for item in citations
                if isinstance(item, dict) and item.get("source_name")
            )
        )
    return compact


def _compact_stream_event(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    event_type = value.get("type")
    if event_type == "meta":
        evidence = value.get("evidence")
        return {
            key: item
            for key, item in {
                "type": event_type,
                "query": value.get("query"),
                "model": value.get("model"),
                "evidence_count": len(evidence) if isinstance(evidence, list) else 0,
            }.items()
            if item is not None
        }
    if event_type == "final" and "result" in value:
        return {"type": "final", "result": _compact_answer(value["result"])}
    return value


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(asdict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _to_jsonable(to_dict())
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    raise TypeError(f"{type(value).__name__} is not JSON serializable")
