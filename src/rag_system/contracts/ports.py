from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable, Protocol

from rag_system.contracts.result import ModuleResult


class FileModule(Protocol):
    def run(self, input_path: str | Path, **options: Any) -> ModuleResult: ...


class TextModule(Protocol):
    def run(self, text: str, **options: Any) -> ModuleResult: ...


class RetrievalPort(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[Any]: ...


class AnswerPort(Protocol):
    def answer(self, query: str, *, top_k: int | None = None) -> Any: ...

    def stream_answer_events(
        self,
        query: str,
        *,
        top_k: int | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> Iterator[dict[str, Any]]: ...


__all__ = ["AnswerPort", "FileModule", "RetrievalPort", "TextModule"]
