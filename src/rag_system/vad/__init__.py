"""Standalone VAD package with a file-oriented test facade.

The original streaming API remains available from :mod:`rag_system.voice.vad`
and is re-exported here for a gradual, backwards-compatible package split.
"""

from rag_system.voice.vad import (
    FsmnVadEngine,
    StreamingVadSession,
    VadConfig,
    VadDependencyError,
    VadEvent,
    pcm16_to_wav_bytes,
)

from .node import VadFacade, VadNode, VadResult

__all__ = [
    "FsmnVadEngine",
    "StreamingVadSession",
    "VadConfig",
    "VadDependencyError",
    "VadEvent",
    "VadFacade",
    "VadNode",
    "VadResult",
    "pcm16_to_wav_bytes",
]

