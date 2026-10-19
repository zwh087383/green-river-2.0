"""Standalone ASR facade with lazily loaded provider implementations."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from rag_system.asr.node import (
    ASRFacade,
    ASRNode,
    ASRProvider,
    AudioInput,
    RealtimeASRResult,
    RealtimeASRTransport,
)


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DASHSCOPE_ASR_SAMPLE_RATE": (
        "rag_system.voice.dashscope_asr",
        "DASHSCOPE_ASR_SAMPLE_RATE",
    ),
    "DASHSCOPE_ASR_SILENCE_MS": (
        "rag_system.voice.dashscope_asr",
        "DASHSCOPE_ASR_SILENCE_MS",
    ),
    "DASHSCOPE_REALTIME_ASR_MODEL": (
        "rag_system.voice.dashscope_asr",
        "DASHSCOPE_REALTIME_ASR_MODEL",
    ),
    "DashScopeASRConnectionError": (
        "rag_system.voice.dashscope_asr",
        "DashScopeASRConnectionError",
    ),
    "DashScopeASRError": ("rag_system.voice.dashscope_asr", "DashScopeASRError"),
    "DashScopeASREvent": ("rag_system.voice.dashscope_asr", "DashScopeASREvent"),
    "DashScopeASRInputError": (
        "rag_system.voice.dashscope_asr",
        "DashScopeASRInputError",
    ),
    "DashScopeRealtimeASR": (
        "rag_system.voice.dashscope_asr",
        "DashScopeRealtimeASR",
    ),
    "VoiceDependencyError": ("rag_system.voice.service", "VoiceDependencyError"),
    "VoiceInputError": ("rag_system.voice.service", "VoiceInputError"),
}

__all__ = [
    "ASRFacade",
    "ASRNode",
    "ASRProvider",
    "AudioInput",
    "RealtimeASRResult",
    "RealtimeASRTransport",
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
