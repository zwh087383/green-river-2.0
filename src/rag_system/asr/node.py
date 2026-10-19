"""Small, stable ASR facade for module-level tests and applications."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from collections.abc import AsyncIterator
from dataclasses import dataclass
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, Callable, Protocol, TypeAlias, runtime_checkable

from rag_system.contracts import iter_pcm_chunks, load_wav_pcm16


AudioInput: TypeAlias = bytes | bytearray | memoryview | str | Path


@runtime_checkable
class ASRProvider(Protocol):
    """Minimum local/batch ASR provider contract used by :class:`ASRNode`."""

    def speech_to_text(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "recording.wav",
    ) -> str: ...


@runtime_checkable
class RealtimeASRTransport(Protocol):
    """Stable contract for a realtime PCM16 ASR session."""

    @property
    def connected(self) -> bool: ...

    async def append_audio(self, pcm16: bytes) -> None: ...

    async def receive_event(self) -> Any | None: ...

    def events(self) -> AsyncIterator[Any]: ...

    async def finish(self) -> None: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RealtimeASRResult:
    text: str
    partials: tuple[str, ...]
    event_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "partials": list(self.partials),
            "event_count": self.event_count,
        }


class ASRNode:
    """One-node ASR API: WAV bytes/file in, recognized text out.

    The default provider is the existing :class:`~rag_system.voice.VoiceService`,
    so old configuration, model loading, Simplified-Chinese conversion, and
    correction rules remain in effect. Tests and deployments may inject any
    object implementing :class:`ASRProvider`.
    """

    def __init__(
        self,
        provider: ASRProvider | None = None,
        *,
        voice_service: ASRProvider | None = None,
    ) -> None:
        if provider is not None and voice_service is not None:
            raise ValueError("Pass either provider or voice_service, not both")
        self.provider = voice_service if voice_service is not None else provider

    def transcribe(
        self,
        audio: AudioInput,
        *,
        filename: str | None = None,
    ) -> str:
        """Normalize and recognize a complete PCM WAV payload or WAV file."""
        loaded = load_wav_pcm16(audio)
        text = self._provider().speech_to_text(
            loaded.to_wav_bytes(),
            filename=_audio_name(audio, filename=filename),
        )
        cleaned = str(text or "").strip()
        if not cleaned:
            raise RuntimeError("ASR returned no recognized text")
        return cleaned

    def transcribe_file(self, path: str | Path) -> str:
        """Convenience alias intended for simple acceptance tests."""
        return self.transcribe(path)

    async def open_realtime(
        self,
        *,
        end_silence_ms: int = 400,
        force_provider: bool = False,
    ) -> RealtimeASRTransport | None:
        """Open the configured realtime provider, or return ``None`` on fallback."""
        opener = getattr(self._provider(), "open_realtime_asr", None)
        if not callable(opener):
            return None
        parameters = signature(opener).parameters.values()
        accepts_force = any(
            parameter.name == "force" or parameter.kind == Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        if force_provider and accepts_force:
            return await opener(end_silence_ms=end_silence_ms, force=True)
        return await opener(end_silence_ms=end_silence_ms)

    async def release_realtime(self, transport: RealtimeASRTransport) -> None:
        """Release a realtime session through its provider when supported."""
        releaser = getattr(self._provider(), "release_realtime_asr", None)
        if callable(releaser):
            await releaser(transport)
            return
        await transport.close()

    async def transcribe_realtime(
        self,
        audio: AudioInput,
        *,
        chunk_ms: int = 100,
        end_silence_ms: int = 400,
        pace: bool = True,
        timeout_seconds: float = 30.0,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> RealtimeASRResult:
        """Feed one WAV through the configured Qwen realtime transport.

        This is deliberately file-oriented so a customer can verify the cloud
        ASR node without starting the browser or microphone pipeline.
        """

        loaded = load_wav_pcm16(audio)
        transport = await self.open_realtime(
            end_silence_ms=end_silence_ms,
            force_provider=True,
        )
        if transport is None:
            raise RuntimeError(
                "realtime ASR is unavailable; configure DashScope and select the realtime provider"
            )
        receiver = asyncio.create_task(
            self._collect_realtime_events(transport, on_event=on_event)
        )
        try:
            for chunk in iter_pcm_chunks(loaded, chunk_ms=chunk_ms):
                if receiver.done():
                    break
                await transport.append_audio(chunk)
                if pace:
                    await asyncio.sleep(chunk_ms / 1_000)
            if not receiver.done():
                await transport.finish()
            return await asyncio.wait_for(receiver, timeout=timeout_seconds)
        finally:
            if not receiver.done():
                receiver.cancel()
                with suppress(asyncio.CancelledError):
                    await receiver
            await self.release_realtime(transport)

    async def transcribe_realtime_file(
        self,
        path: str | Path,
        **options: Any,
    ) -> RealtimeASRResult:
        return await self.transcribe_realtime(path, **options)

    async def _collect_realtime_events(
        self,
        transport: RealtimeASRTransport,
        *,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> RealtimeASRResult:
        partials: list[str] = []
        final_text = ""
        event_count = 0
        postprocess = getattr(
            self._provider(),
            "postprocess_asr_text",
            lambda value: str(value or "").strip(),
        )
        async for event in transport.events():
            event_count += 1
            event_type = str(getattr(event, "type", ""))
            text = postprocess(getattr(event, "text", ""))
            if on_event is not None and event_type in {"partial", "completed"}:
                on_event({"type": event_type, "text": text})
            if event_type == "partial" and text:
                partials.append(text)
            elif event_type == "completed":
                final_text = text
        if not final_text:
            raise RuntimeError("realtime ASR returned no final transcript")
        return RealtimeASRResult(
            text=final_text,
            partials=tuple(partials),
            event_count=event_count,
        )

    def _provider(self) -> ASRProvider:
        if self.provider is None:
            # Keep ASR importable/testable without loading the combined legacy
            # voice service and its optional TTS dependencies.
            from rag_system.voice.service import VoiceService

            self.provider = VoiceService()
        return self.provider


# Facade is retained as an explicit name for callers that prefer that wording.
ASRFacade = ASRNode


def _audio_name(
    audio: AudioInput,
    *,
    filename: str | None,
) -> str:
    if filename:
        return filename
    if isinstance(audio, (str, Path)):
        return Path(audio).name
    return "recording.wav"
