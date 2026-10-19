"""Standalone TTS facade with an explicit MP3 output contract."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from rag_system.tts.node import (
    DEFAULT_AUDIO_FORMAT,
    DEFAULT_AUDIO_SUFFIX,
    DEFAULT_MEDIA_TYPE,
    TTSFacade,
    TTSNode,
    TTSProvider,
)


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DashScopeRealtimeTTS": (
        "rag_system.voice.dashscope_tts",
        "DashScopeRealtimeTTS",
    ),
    "DashScopeTTSConnectionError": (
        "rag_system.voice.dashscope_tts",
        "DashScopeTTSConnectionError",
    ),
    "DashScopeTTSError": ("rag_system.voice.dashscope_tts", "DashScopeTTSError"),
    "DashScopeTTSInputError": (
        "rag_system.voice.dashscope_tts",
        "DashScopeTTSInputError",
    ),
    "TTSStreamSynthesizer": (
        "rag_system.voice.service",
        "TTSStreamSynthesizer",
    ),
    "VoiceDependencyError": ("rag_system.voice.service", "VoiceDependencyError"),
    "VoiceInputError": ("rag_system.voice.service", "VoiceInputError"),
}

__all__ = [
    "DEFAULT_AUDIO_FORMAT",
    "DEFAULT_AUDIO_SUFFIX",
    "DEFAULT_MEDIA_TYPE",
    "TTSFacade",
    "TTSNode",
    "TTSProvider",
    *_LAZY_EXPORTS,
]


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
