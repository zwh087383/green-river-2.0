from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class KnowledgeReader(Protocol):
    """Read-only methods needed from the existing indexing service."""

    def search(self, query: str, top_k: int = 5) -> list[Any]: ...

    def list_documents(self) -> list[Any]: ...

    def stats(self) -> Any: ...


class RAGEdgeNode:
    """Read-only facade intended for an edge device or ROS adapter."""

    module_name = "rag-edge"

    def __init__(
        self,
        *,
        reader: KnowledgeReader | None = None,
        reader_factory: Callable[[], KnowledgeReader] | None = None,
    ) -> None:
        self._reader_instance = reader
        self._reader_factory = reader_factory or _default_reader

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        include_details: bool = False,
    ) -> dict[str, Any]:
        """Search the local knowledge base without exposing retrieval internals."""

        cleaned = str(query or "").strip()
        if not cleaned:
            raise ValueError("query cannot be empty")
        limited_top_k = _validate_top_k(top_k)
        results = self._reader().search(cleaned, top_k=limited_top_k)
        output_results = (
            _to_jsonable(results)
            if include_details
            else [_compact_search_result(result) for result in results]
        )
        return _success(
            self.module_name,
            "search",
            {
                "query": cleaned,
                "top_k": limited_top_k,
                "count": len(results),
                "results": output_results,
            },
        )

    def list(self) -> dict[str, Any]:
        """List locally available documents."""

        documents = self._reader().list_documents()
        return _success(
            self.module_name,
            "list",
            {"count": len(documents), "documents": _to_jsonable(documents)},
        )

    def list_documents(self) -> dict[str, Any]:
        """Explicit alias for callers that avoid the short ``list`` name."""

        return self.list()

    def stats(self) -> dict[str, Any]:
        """Return local index statistics."""

        return _success(self.module_name, "stats", _to_jsonable(self._reader().stats()))

    def _reader(self) -> KnowledgeReader:
        if self._reader_instance is None:
            self._reader_instance = self._reader_factory()
        return self._reader_instance


def build_default_node(
    *,
    index_dir: Path | str = Path("data/index"),
) -> RAGEdgeNode:
    resolved_index_dir = Path(index_dir)
    return RAGEdgeNode(reader_factory=lambda: _default_reader(index_dir=resolved_index_dir))


def _default_reader(*, index_dir: Path | str = Path("data/index")) -> KnowledgeReader:
    import logging

    import jieba

    from rag_system.indexing import IndexingService

    jieba.setLogLevel(logging.ERROR)
    return IndexingService(index_dir=index_dir)


def _validate_top_k(top_k: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("top_k must be an integer")
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")
    return top_k


def _success(module: str, operation: str, data: Any) -> dict[str, Any]:
    return {"ok": True, "module": module, "operation": operation, "data": data}


def _compact_search_result(value: Any) -> Any:
    payload = _to_jsonable(value)
    if not isinstance(payload, dict):
        return payload
    compact = {
        key: payload[key]
        for key in ("id", "text", "score")
        if key in payload
    }
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key in ("document_id", "source_name"):
            if metadata.get(key) not in (None, ""):
                compact[key] = metadata[key]
    return compact


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
