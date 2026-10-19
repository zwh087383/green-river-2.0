from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from rag_system.contracts import (
    AudioBuffer,
    AudioSpec,
    ModuleResult,
    load_wav_pcm16,
    write_wav_file,
)
from rag_system.voice.vad import FsmnVadEngine, VadConfig, VadEvent


AudioInput = str | Path | bytes | bytearray | memoryview


_SPEECH_EVENT_TYPES = frozenset({"speech_start", "speech_end", "forced_end"})


@dataclass(frozen=True, slots=True)
class VadResult:
    events: tuple[Any, ...]
    sample_rate: int
    audio_ms: int

    @property
    def speech_detected(self) -> bool:
        return any(_event_type(event) in _SPEECH_EVENT_TYPES for event in self.events)

    @property
    def utterance_pcm(self) -> bytes | None:
        for event in reversed(self.events):
            if isinstance(event, dict):
                audio = event.get("audio_pcm")
            else:
                audio = getattr(event, "audio_pcm", None)
            if isinstance(audio, (bytes, bytearray, memoryview)):
                return bytes(audio)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "speech_detected": self.speech_detected,
            "events": [_event_payload(event) for event in self.events],
        }

    def to_module_result(
        self,
        *,
        artifacts: dict[str, str] | None = None,
    ) -> ModuleResult:
        return ModuleResult.success(
            "vad",
            self.to_dict(),
            metrics={
                "sample_rate": self.sample_rate,
                "audio_ms": self.audio_ms,
                "event_count": len(self.events),
            },
            artifacts=artifacts,
        )


class VadNode:
    """One-file facade around the existing streaming FSMN VAD engine."""

    def __init__(
        self,
        engine: Any | None = None,
        *,
        model_dir: str | Path | None = None,
        config: VadConfig | None = None,
        end_silence_ms: int = 400,
        engine_factory: Callable[..., Any] = FsmnVadEngine,
        **engine_options: Any,
    ) -> None:
        if config is not None and end_silence_ms != 400:
            raise ValueError("use config or end_silence_ms, not both")
        self.config = config or VadConfig(end_silence_ms=end_silence_ms)
        if engine is not None and (model_dir is not None or engine_options):
            raise ValueError("model_dir and engine options cannot be used with an injected engine")
        self._owns_engine = engine is None
        if engine is None:
            engine = engine_factory(
                model_dir,
                end_silence_ms=self.config.end_silence_ms,
                **engine_options,
            )
        self.engine = engine

    def run(
        self,
        audio: AudioInput,
        *,
        output_wav: str | Path | None = None,
        **_: Any,
    ) -> ModuleResult:
        """Run a customer-facing WAV test and always return a stable envelope."""

        try:
            result = self.segment(audio)
            artifacts: dict[str, str] = {}
            if output_wav is not None and result.utterance_pcm is not None:
                path = write_wav_file(
                    output_wav,
                    AudioBuffer(
                        result.utterance_pcm,
                        AudioSpec(sample_rate=result.sample_rate),
                    ),
                )
                artifacts["utterance_wav"] = str(path)
            return result.to_module_result(artifacts=artifacts)
        except Exception as exc:
            return ModuleResult.failure("vad", exc)

    def segment(self, audio: AudioInput) -> VadResult:
        """Segment a WAV path or in-memory WAV bytes."""

        loaded = load_wav_pcm16(audio, target_sample_rate=self.config.sample_rate)
        return self._segment_buffer(loaded)

    def segment_pcm16(self, pcm: bytes, *, sample_rate: int = 16_000) -> VadResult:
        """Segment raw little-endian, mono PCM16 bytes."""

        if sample_rate != self.config.sample_rate:
            raise ValueError(
                f"raw PCM sample rate must be {self.config.sample_rate} Hz, got {sample_rate}"
            )
        return self._segment_buffer(AudioBuffer(bytes(pcm), AudioSpec(sample_rate)))

    def _segment_buffer(self, loaded: AudioBuffer) -> VadResult:
        session = self.engine.create_session(self.config)
        events: list[Any] = []
        frame_bytes = self.config.frame_bytes

        if loaded.pcm16:
            for offset in range(0, len(loaded.pcm16), frame_bytes):
                chunk = loaded.pcm16[offset : offset + frame_bytes]
                is_final = offset + frame_bytes >= len(loaded.pcm16)
                events.extend(session.feed_pcm16(chunk, is_final=is_final))
                if bool(getattr(session, "finished", False)):
                    break
        else:
            events.extend(session.feed_pcm16(b"", is_final=True))

        return VadResult(
            events=tuple(events),
            sample_rate=loaded.spec.sample_rate,
            audio_ms=loaded.duration_ms,
        )

    run_wav = run
    run_pcm16 = segment_pcm16

    def close(self) -> None:
        if not self._owns_engine:
            return
        close = getattr(self.engine, "close", None)
        if callable(close):
            close()


def _event_type(event: Any) -> str:
    if isinstance(event, dict):
        value = event.get("type")
    else:
        value = getattr(event, "type", None)
    return str(value or "")


def _event_payload(event: Any) -> dict[str, Any]:
    to_dict = getattr(event, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    if isinstance(event, dict):
        return {
            str(key): value
            for key, value in event.items()
            if key != "audio_pcm" and not isinstance(value, (bytes, bytearray, memoryview))
        }
    payload: dict[str, Any] = {"type": _event_type(event)}
    for name in ("start_ms", "end_ms", "reason"):
        value = getattr(event, name, None)
        if value is not None:
            payload[name] = value
    return payload


VadFacade = VadNode


__all__ = ["VadFacade", "VadNode", "VadResult"]
