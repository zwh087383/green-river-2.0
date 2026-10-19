"""Standalone wake-word package with a file-oriented test facade.

The original streaming API remains available from :mod:`rag_system.voice.wakeup`
and is re-exported here for a gradual, backwards-compatible package split.
"""

from rag_system.voice.wakeup import (
    DEFAULT_KEYWORD_SPEC,
    DEFAULT_KEYWORDS_FILENAME,
    DEFAULT_WAKE_WORD,
    WakeWordDependencyError,
    WakeWordEngine,
    WakeWordEvent,
    WakeWordModelError,
    WakeWordSession,
    keyword_spec_for_text,
)

from .node import WakeupFacade, WakeupNode, WakeupResult

__all__ = [
    "DEFAULT_KEYWORD_SPEC",
    "DEFAULT_KEYWORDS_FILENAME",
    "DEFAULT_WAKE_WORD",
    "WakeWordDependencyError",
    "WakeWordEngine",
    "WakeWordEvent",
    "WakeWordModelError",
    "WakeWordSession",
    "WakeupFacade",
    "WakeupNode",
    "WakeupResult",
    "keyword_spec_for_text",
]

