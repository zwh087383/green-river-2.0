from __future__ import annotations

import os
from pathlib import Path


DEFAULT_TOP_K = 4
MAX_TOP_K = 20


def default_data_dir() -> Path:
    """Resolve persistent RAG state outside the vendor ROS2 SDK installation."""

    configured = os.environ.get("RAG_DATA_DIR", "~/.local/share/green-river")
    return Path(configured).expanduser().resolve()


def normalize_top_k(value: int, *, default: int = DEFAULT_TOP_K) -> int:
    """Apply one bounded top-k policy to every ROS2 entry point."""

    if value <= 0:
        return min(max(1, default), MAX_TOP_K)
    return min(value, MAX_TOP_K)


def nonempty_session_id(value: str) -> str:
    cleaned = str(value or "").strip()
    return cleaned or "robot"
