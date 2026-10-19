from __future__ import annotations

from threading import Lock

from rag_system.generation.models import ConversationTurn
from rag_system.settings import DEFAULT_CONVERSATION_MAX_ROUNDS


ROUTER_HISTORY_MAX_ROUNDS = 10


class ConversationHistory:
    def __init__(self, max_rounds: int = DEFAULT_CONVERSATION_MAX_ROUNDS) -> None:
        self._max_rounds = _coerce_max_rounds(max_rounds)
        self._turns: list[ConversationTurn] = []
        self._lock = Lock()

    def set_max_rounds(self, max_rounds: int) -> None:
        with self._lock:
            self._max_rounds = _coerce_max_rounds(max_rounds)
            self._trim_locked()

    def append(self, user: str, assistant: str) -> None:
        user_text = str(user or "").strip()
        assistant_text = str(assistant or "").strip()
        if not user_text or not assistant_text:
            return
        with self._lock:
            self._turns.append(ConversationTurn(user=user_text, assistant=assistant_text))
            self._trim_locked()

    def snapshot(self, *, max_rounds: int | None = None) -> list[ConversationTurn]:
        with self._lock:
            limit = self._max_rounds if max_rounds is None else _coerce_max_rounds(max_rounds)
            return list(self._turns[-limit:])

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._turns)

    def _trim_locked(self) -> None:
        if len(self._turns) > self._max_rounds:
            del self._turns[: len(self._turns) - self._max_rounds]


def _coerce_max_rounds(max_rounds: int) -> int:
    try:
        value = int(max_rounds)
    except (TypeError, ValueError):
        value = DEFAULT_CONVERSATION_MAX_ROUNDS
    return max(1, min(value, 100))
