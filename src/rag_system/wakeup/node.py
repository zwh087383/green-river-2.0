from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from rag_system.contracts import (
    AudioBuffer,
    AudioSpec,
    ModuleResult,
    iter_pcm_chunks,
    load_wav_pcm16,
)
from rag_system.voice.wakeup import WakeWordEngine


AudioInput = str | Path | bytes | bytearray | memoryview


@dataclass(frozen=True, slots=True)
class WakeupResult:
    detected: bool
    keyword: str | None
    sample_rate: int
    audio_ms: int
    processed_ms: int

    def to_dict(self) -> dict[str, str | int | bool | None]:
        return {
            "detected": self.detected,
            "keyword": self.keyword,
        }

    def to_module_result(self) -> ModuleResult:
        return ModuleResult.success(
            "wakeup",
            self.to_dict(),
            metrics={
                "sample_rate": self.sample_rate,
                "audio_ms": self.audio_ms,
                "processed_ms": self.processed_ms,
            },
        )


class WakeupNode:
    """One-file facade around the existing streaming wake-word engine.

    ``engine`` is injectable so this node can be contract-tested without loading
    the real model. With no injected engine, the existing offline
    :class:`~rag_system.voice.wakeup.WakeWordEngine` is used unchanged.
    """

    def __init__(
        self,
        engine: Any | None = None,
        *,
        model_dir: str | Path | None = None,
        wake_word: str | None = None,
        chunk_ms: int = 100,
        engine_factory: Callable[..., Any] = WakeWordEngine,
        **engine_options: Any,
    ) -> None:
        if chunk_ms <= 0:
            raise ValueError("chunk_ms must be positive")
        if engine is not None and (model_dir is not None or wake_word is not None or engine_options):
            raise ValueError(
                "model_dir, wake_word and engine options cannot be used with an injected engine"
            )

        self._owns_engine = engine is None
        if engine is None:
            if wake_word is not None:
                engine_options["keyword_words"] = wake_word
            engine = engine_factory(model_dir, **engine_options)
        self.engine = engine
        self.sample_rate = int(getattr(engine, "sample_rate", 16_000))
        self.chunk_ms = chunk_ms

        # Validate chunk sizing at construction time so CLI failures are immediate.
        tuple(iter_pcm_chunks(AudioBuffer(b"", AudioSpec(self.sample_rate)), chunk_ms=chunk_ms))

    def run(
        self,
        audio: AudioInput,
        **_: Any,
    ) -> ModuleResult:
        """Run a customer-facing WAV test and always return a stable envelope."""

        try:
            return self.detect(audio).to_module_result()
        except Exception as exc:
            return ModuleResult.failure("wakeup", exc)

    def detect(self, audio: AudioInput) -> WakeupResult:
        """Detect a wake word from a WAV path or in-memory WAV bytes."""

        loaded = load_wav_pcm16(audio, target_sample_rate=self.sample_rate)
        return self._detect_buffer(loaded)

    def detect_pcm16(self, pcm: bytes, *, sample_rate: int = 16_000) -> WakeupResult:
        """Detect a wake word from raw little-endian, mono PCM16 bytes."""

        if sample_rate != self.sample_rate:
            raise ValueError(
                f"raw PCM sample rate must be {self.sample_rate} Hz, got {sample_rate}"
            )
        return self._detect_buffer(AudioBuffer(bytes(pcm), AudioSpec(sample_rate)))

    def _detect_buffer(self, loaded: AudioBuffer) -> WakeupResult:
        session = self.engine.create_session()
        keyword: str | None = None
        processed_bytes = 0
        try:
            for chunk in iter_pcm_chunks(
                loaded,
                chunk_ms=self.chunk_ms,
            ):
                processed_bytes += len(chunk)
                event = session.accept_pcm16(chunk)
                keyword = _event_keyword(event)
                if keyword:
                    break
        finally:
            close = getattr(session, "close", None)
            if callable(close):
                close()

        processed_ms = int(
            round((processed_bytes // 2) * 1_000 / loaded.spec.sample_rate)
        )
        return WakeupResult(
            detected=keyword is not None,
            keyword=keyword,
            sample_rate=loaded.spec.sample_rate,
            audio_ms=loaded.duration_ms,
            processed_ms=processed_ms,
        )

    run_wav = run
    run_pcm16 = detect_pcm16

    def close(self) -> None:
        if not self._owns_engine:
            return
        close = getattr(self.engine, "close", None)
        if callable(close):
            close()


def _event_keyword(event: Any) -> str | None:
    if event is None:
        return None
    if isinstance(event, dict):
        value = event.get("keyword")
    else:
        value = getattr(event, "keyword", None)
    cleaned = str(value or "").strip()
    return cleaned or None


WakeupFacade = WakeupNode


__all__ = ["WakeupFacade", "WakeupNode", "WakeupResult"]
