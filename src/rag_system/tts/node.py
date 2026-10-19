"""Small, stable TTS facade for module-level tests and applications."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

DEFAULT_AUDIO_FORMAT = "mp3"
DEFAULT_AUDIO_SUFFIX = ".mp3"
DEFAULT_MEDIA_TYPE = "audio/mpeg"


@runtime_checkable
class TTSProvider(Protocol):
    """Minimum provider contract used by :class:`TTSNode`."""

    def text_to_speech(self, text: str) -> bytes: ...

    def stream_text_to_speech(self, text: str) -> AsyncIterator[bytes]: ...


class TTSNode:
    """One-node TTS API: text in, MP3 bytes or a file out.

    The current Edge-TTS and DashScope implementations both emit MP3. The
    explicit constants above keep that format visible to test clients instead
    of implying that the bytes are WAV.
    """

    audio_format = DEFAULT_AUDIO_FORMAT
    media_type = DEFAULT_MEDIA_TYPE

    def __init__(
        self,
        provider: TTSProvider | None = None,
        *,
        voice_service: TTSProvider | None = None,
    ) -> None:
        if provider is not None and voice_service is not None:
            raise ValueError("Pass either provider or voice_service, not both")
        self.provider = voice_service if voice_service is not None else provider

    def synthesize(self, text: str) -> bytes:
        """Synthesize one complete MP3 payload."""
        audio = self._provider().text_to_speech(text)
        if not isinstance(audio, bytes) or not audio:
            raise RuntimeError("TTS returned no audio")
        return audio

    def synthesize_to_file(self, text: str, output: str | Path) -> Path:
        """Synthesize text and write the resulting MP3 bytes to ``output``."""
        path = _prepare_output_path(output)
        path.write_bytes(self.synthesize(text))
        return path

    def stream(self, text: str) -> AsyncIterator[bytes]:
        """Return provider audio chunks without buffering the full response."""
        return self._provider().stream_text_to_speech(text)

    async def synthesize_streamed(self, text: str) -> bytes:
        """Collect one streaming synthesis for a file-oriented module test."""
        chunks = [chunk async for chunk in self.stream(text) if chunk]
        audio = b"".join(chunks)
        if not audio:
            raise RuntimeError("streaming TTS returned no audio")
        return audio

    async def synthesize_streamed_to_file(
        self,
        text: str,
        output: str | Path,
    ) -> Path:
        path = _prepare_output_path(output)
        path.write_bytes(await self.synthesize_streamed(text))
        return path

    async def warm(self) -> bool:
        """Warm a persistent provider connection when the provider supports it."""
        warmer = getattr(self._provider(), "warm_tts_connection", None)
        if not callable(warmer):
            return False
        return bool(await warmer())

    async def close(self) -> None:
        """Close a persistent provider connection when one is in use."""
        closer = getattr(self._provider(), "close_tts_connection", None)
        if callable(closer):
            await closer()

    def active_provider(self) -> str | None:
        """Return the effective provider after synthesis when it is observable."""
        status_method = getattr(self._provider(), "status", None)
        if not callable(status_method):
            return None
        status: Any = status_method()
        raw = str(getattr(status, "tts_provider_active", "") or "").strip()
        if raw == "dashscope_realtime":
            return "qwen"
        return raw or None

    def _provider(self) -> TTSProvider:
        if self.provider is None:
            # The legacy composition service is loaded only when a real default
            # provider is requested; fake/module tests stay dependency-isolated.
            from rag_system.voice.service import VoiceService

            self.provider = VoiceService()
        return self.provider


TTSFacade = TTSNode


def _prepare_output_path(output: str | Path) -> Path:
    path = Path(output)
    if not path.suffix:
        path = path.with_suffix(DEFAULT_AUDIO_SUFFIX)
    elif path.suffix.lower() != DEFAULT_AUDIO_SUFFIX:
        raise ValueError(
            "TTS currently outputs MP3; the output file must use the .mp3 suffix"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
