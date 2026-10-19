from __future__ import annotations

from functools import lru_cache
from typing import Any


def simplify_chinese(text: str) -> str:
    """Convert Chinese ASR output to Simplified Chinese with OpenCC."""
    value = str(text or "")
    if not value:
        return value
    converter = _opencc_converter()
    return str(converter.convert(value))


@lru_cache(maxsize=1)
def _opencc_converter() -> Any:
    try:
        from opencc import OpenCC
    except ImportError as exc:
        raise ImportError(
            'OpenCC is required for Simplified Chinese ASR output; install the "voice" extra'
        ) from exc
    return OpenCC("t2s")
