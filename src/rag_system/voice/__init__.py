"""Compatibility exports for the legacy combined voice package.

New code should import the standalone :mod:`rag_system.asr`,
:mod:`rag_system.tts`, :mod:`rag_system.vad`, or :mod:`rag_system.wakeup`
packages.  Lazy compatibility exports keep existing callers working without
loading every optional voice dependency when only one module is requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "CorrectionResult": ("rag_system.voice.glossary", "CorrectionResult"),
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
    "FsmnVadEngine": ("rag_system.voice.vad", "FsmnVadEngine"),
    "StreamingVadSession": ("rag_system.voice.vad", "StreamingVadSession"),
    "TTSNormalizeResult": ("rag_system.voice.tts_normalize", "TTSNormalizeResult"),
    "TTSStreamSynthesizer": (
        "rag_system.voice.service",
        "TTSStreamSynthesizer",
    ),
    "VadConfig": ("rag_system.voice.vad", "VadConfig"),
    "VadDependencyError": ("rag_system.voice.vad", "VadDependencyError"),
    "VadEvent": ("rag_system.voice.vad", "VadEvent"),
    "VoiceDependencyError": ("rag_system.voice.service", "VoiceDependencyError"),
    "VoiceInputError": ("rag_system.voice.service", "VoiceInputError"),
    "VoiceService": ("rag_system.voice.service", "VoiceService"),
    "VoiceStatus": ("rag_system.voice.service", "VoiceStatus"),
    "WakeWordDependencyError": (
        "rag_system.voice.wakeup",
        "WakeWordDependencyError",
    ),
    "WakeWordEngine": ("rag_system.voice.wakeup", "WakeWordEngine"),
    "WakeWordEvent": ("rag_system.voice.wakeup", "WakeWordEvent"),
    "WakeWordModelError": ("rag_system.voice.wakeup", "WakeWordModelError"),
    "WakeWordSession": ("rag_system.voice.wakeup", "WakeWordSession"),
    "correct_stt_text": ("rag_system.voice.glossary", "correct_stt_text"),
    "get_domain_glossary_text": (
        "rag_system.voice.glossary",
        "get_domain_glossary_text",
    ),
    "keyword_spec_for_text": (
        "rag_system.voice.wakeup",
        "keyword_spec_for_text",
    ),
    "normalize_for_tts": ("rag_system.voice.tts_normalize", "normalize_for_tts"),
    "pcm16_to_wav_bytes": ("rag_system.voice.vad", "pcm16_to_wav_bytes"),
    "simplify_chinese": ("rag_system.voice.chinese", "simplify_chinese"),
    "wrap_ssml": ("rag_system.voice.tts_normalize", "wrap_ssml"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
