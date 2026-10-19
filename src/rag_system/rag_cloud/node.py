from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class DocumentParser(Protocol):
    """Small write-side contract required by :class:`RAGCloudNode`."""

    def parse_document(self, source: Path | str) -> Any: ...


@runtime_checkable
class DocumentIndexer(Protocol):
    """Small index contract required by :class:`RAGCloudNode`."""

    def index_document(self, document: Any) -> Any: ...

    def delete_document(self, document_id: str, *, remove_manifest: bool = True) -> Any: ...


class RAGCloudNode:
    """Simple cloud/write-side facade for parsing and knowledge-base updates.

    Existing parsing and indexing services are injected instead of being exposed to
    callers.  Lazy factories keep importing the package side-effect free and let a
    customer test this node with small in-memory fakes.
    """

    module_name = "rag-cloud"

    def __init__(
        self,
        *,
        parser: DocumentParser | None = None,
        indexer: DocumentIndexer | None = None,
        parser_factory: Callable[[], DocumentParser] | None = None,
        indexer_factory: Callable[[], DocumentIndexer] | None = None,
    ) -> None:
        self._parser_instance = parser
        self._indexer_instance = indexer
        self._parser_factory = parser_factory or _default_parser
        self._indexer_factory = indexer_factory or _default_indexer

    def parse(self, source: Path | str) -> dict[str, Any]:
        """Parse one document and return a JSON-ready result."""

        source_value = _require_source(source)
        document = self._parser().parse_document(source_value)
        return _success(self.module_name, "parse", _to_jsonable(document))

    def index(self, source_or_document: Path | str | Any) -> dict[str, Any]:
        """Parse and index a path, or directly index an already parsed document."""

        if isinstance(source_or_document, (str, Path)):
            source_value = _require_source(source_or_document)
            document = self._parser().parse_document(source_value)
        else:
            if source_or_document is None:
                raise ValueError("document cannot be null")
            document = source_or_document
        result = self._indexer().index_document(document)
        return _success(self.module_name, "index", _to_jsonable(result))

    def delete(self, document_id: str, *, remove_manifest: bool = True) -> dict[str, Any]:
        """Delete all indexed data belonging to one document."""

        cleaned = str(document_id or "").strip()
        if not cleaned:
            raise ValueError("document_id cannot be empty")
        result = self._indexer().delete_document(cleaned, remove_manifest=remove_manifest)
        return _success(self.module_name, "delete", _to_jsonable(result))

    def _parser(self) -> DocumentParser:
        if self._parser_instance is None:
            self._parser_instance = self._parser_factory()
        return self._parser_instance

    def _indexer(self) -> DocumentIndexer:
        if self._indexer_instance is None:
            self._indexer_instance = self._indexer_factory()
        return self._indexer_instance


def build_default_node(
    *,
    index_dir: Path | str = Path("data/index"),
) -> RAGCloudNode:
    """Build a facade for one explicit index directory.

    The library default remains the application's production directory.  The
    customer-facing CLI deliberately passes ``data/acceptance/index`` instead.
    """

    resolved_index_dir = Path(index_dir)
    cache_dir = resolved_index_dir.parent / "parser_cache"
    return RAGCloudNode(
        parser_factory=lambda: _default_parser(cache_dir=cache_dir),
        indexer_factory=lambda: _default_indexer(index_dir=resolved_index_dir),
    )


def _default_parser(*, cache_dir: Path | str = Path("data/parser_cache")) -> DocumentParser:
    from rag_system.parsing import ParsingService

    return ParsingService.with_chunking(
        backend="auto",
        enable_cache=True,
        cache_dir=cache_dir,
    )


def _default_indexer(*, index_dir: Path | str = Path("data/index")) -> DocumentIndexer:
    import logging

    import jieba

    from rag_system.indexing import IndexingService

    jieba.setLogLevel(logging.ERROR)
    return IndexingService(index_dir=index_dir)


def _require_source(source: Path | str) -> Path | str:
    if isinstance(source, Path):
        if not str(source).strip():
            raise ValueError("source cannot be empty")
        return source
    cleaned = str(source or "").strip()
    if not cleaned:
        raise ValueError("source cannot be empty")
    return cleaned


def _success(module: str, operation: str, data: Any) -> dict[str, Any]:
    return {"ok": True, "module": module, "operation": operation, "data": data}


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
