from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import tempfile
import wave
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from rag_system.settings import SettingsStore
from rag_system.voice.chinese import simplify_chinese
from rag_system.voice.dashscope_asr import DashScopeRealtimeASR
from rag_system.voice.dashscope_tts import DashScopeRealtimeTTS
from rag_system.voice.glossary import correct_stt_text
from rag_system.voice.models import asr_model_files_present, resolve_asr_model_dir
from rag_system.voice.tts_normalize import normalize_for_tts, wrap_ssml

MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_TTS_CHARS = 2000
DASHSCOPE_FEMALE_VOICE = "Cherry"

LOGGER = logging.getLogger(__name__)

TTSStreamSynthesizer = Callable[[str, str, str], AsyncIterator[bytes]]
ASRTransportFactory = Callable[..., DashScopeRealtimeASR]


class VoiceDependencyError(RuntimeError):
    pass


class VoiceInputError(ValueError):
    pass


@dataclass(slots=True)
class VoiceStatus:
    voice_available: bool
    stt_available: bool
    tts_available: bool
    stt_model: str
    tts_voice: str
    tts_rate: str
    errors: list[str]
    tts_provider: str = "edge"
    tts_provider_active: str = "edge"
    tts_connection_state: str = "ready"
    tts_model: str | None = None
    tts_streaming: bool = True
    tts_provider_error: str | None = None
    asr_provider: str = "local_whisper"
    asr_provider_active: str = "local_whisper"
    asr_model: str | None = None
    asr_streaming: bool = False
    asr_provider_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VoiceService:
    def __init__(
        self,
        *,
        settings_store: SettingsStore | None = None,
        whisper_model_loader: Callable[[str], Any] | None = None,
        tts_synthesizer: Callable[[str, str, str], bytes] | None = None,
        tts_stream_synthesizer: TTSStreamSynthesizer | None = None,
        dashscope_tts: DashScopeRealtimeTTS | None = None,
        dashscope_asr_factory: ASRTransportFactory | None = None,
        prefer_dashscope_tts: bool = True,
        strict_no_fallback: bool = False,
        stt_correction_enabled: bool | None = None,
        tts_normalize_enabled: bool = True,
        tts_ssml_enabled: bool = False,
    ) -> None:
        self.settings_store = settings_store or SettingsStore()
        self.whisper_model_loader = whisper_model_loader
        self.tts_synthesizer = tts_synthesizer
        self.tts_stream_synthesizer = tts_stream_synthesizer
        self.dashscope_tts = dashscope_tts or DashScopeRealtimeTTS(
            voice=DASHSCOPE_FEMALE_VOICE
        )
        self.prefer_dashscope_tts = prefer_dashscope_tts
        self.strict_no_fallback = strict_no_fallback
        self._dashscope_last_error: str | None = None
        self.dashscope_asr_factory = dashscope_asr_factory or DashScopeRealtimeASR
        self._active_asr_sessions: set[DashScopeRealtimeASR] = set()
        self._active_asr_sessions_lock = Lock()
        self._dashscope_asr_last_error: str | None = None
        self.stt_correction_enabled = stt_correction_enabled
        self._asr_correction_lock = Lock()
        self._asr_correction_config: tuple[bool, tuple[tuple[str, str], ...]] = (
            True,
            (),
        )
        self.refresh_asr_corrections()
        self.tts_normalize_enabled = tts_normalize_enabled
        self.tts_ssml_enabled = tts_ssml_enabled
        self._model_lock = Lock()
        self._whisper_models: dict[str, Any] = {}

    def status(self) -> VoiceStatus:
        settings = self.settings_store.load()
        errors: list[str] = []
        model_dir = resolve_asr_model_dir(settings.stt_model)
        stt_dependency_available = _module_exists("faster_whisper")
        stt_model_available = asr_model_files_present(model_dir)
        stt_simplifier_available = _module_exists("opencc")
        local_stt_available = self.whisper_model_loader is not None or (
            stt_dependency_available and stt_model_available
        )
        if not _module_exists("numpy"):
            local_stt_available = False
            errors.append("numpy is not installed")
        if self.whisper_model_loader is None and not stt_dependency_available:
            errors.append("faster-whisper is not installed")
        if self.whisper_model_loader is None and not stt_model_available:
            errors.append(
                "local ASR model is missing; run: "
                "python -m rag_system.model_setup download --component asr"
            )
        if not stt_simplifier_available:
            local_stt_available = False
            errors.append('OpenCC is not installed; install the project with the "voice" extra')

        asr_provider = str(getattr(settings, "asr_provider", "dashscope_realtime"))
        dashscope_asr_configured = bool(
            asr_provider == "dashscope_realtime" and settings.dashscope_api_key
        )
        if dashscope_asr_configured and self._dashscope_asr_last_error is None:
            asr_provider_active = "dashscope_realtime"
            asr_model = str(
                getattr(settings, "dashscope_asr_model", "qwen3-asr-flash-realtime")
            )
            asr_streaming = True
            stt_available = True
        elif local_stt_available and not self.strict_no_fallback:
            asr_provider_active = "local_whisper"
            asr_model = settings.stt_model
            asr_streaming = False
            stt_available = True
        else:
            asr_provider_active = "unavailable"
            asr_model = None
            asr_streaming = False
            stt_available = False
        if asr_provider == "dashscope_realtime" and not settings.dashscope_api_key:
            errors.append("DashScope API key is not configured; local ASR fallback is active")
        if self._dashscope_asr_last_error:
            errors.append(self._dashscope_asr_last_error)
        edge_tts_available = _module_exists("edge_tts")
        dashscope_configured = bool(
            self.prefer_dashscope_tts and settings.dashscope_api_key
        )
        custom_tts = (
            self.tts_synthesizer is not None
            or self.tts_stream_synthesizer is not None
        )
        tts_available = (
            custom_tts
            or dashscope_configured
            or (edge_tts_available and not self.strict_no_fallback)
        )
        if not tts_available:
            errors.append("No TTS provider is available")

        if custom_tts:
            tts_provider = "custom"
            tts_provider_active = "custom"
            tts_connection_state = "ready"
            tts_voice = settings.tts_voice
            tts_model = None
        elif dashscope_configured:
            dashscope_ready = bool(self.dashscope_tts.connected)
            tts_provider = "dashscope_realtime"
            tts_provider_active = (
                "dashscope_realtime"
                if dashscope_ready
                else "edge"
                if edge_tts_available and not self.strict_no_fallback
                else "dashscope_realtime"
            )
            tts_connection_state = (
                "ready"
                if dashscope_ready
                else "fallback"
                if (
                    edge_tts_available
                    and self._dashscope_last_error
                    and not self.strict_no_fallback
                )
                else "disconnected"
            )
            tts_voice = self.dashscope_tts.voice
            tts_model = self.dashscope_tts.model
        else:
            tts_provider = "edge"
            tts_provider_active = "edge" if edge_tts_available else "unavailable"
            tts_connection_state = "ready" if edge_tts_available else "unavailable"
            tts_voice = settings.tts_voice
            tts_model = None
        return VoiceStatus(
            voice_available=stt_available or tts_available,
            stt_available=stt_available,
            tts_available=tts_available,
            stt_model=settings.stt_model,
            tts_voice=tts_voice,
            tts_rate=settings.tts_rate,
            errors=_dedupe(errors),
            tts_provider=tts_provider,
            tts_provider_active=tts_provider_active,
            tts_connection_state=tts_connection_state,
            tts_model=tts_model,
            tts_streaming=True,
            tts_provider_error=self._dashscope_last_error,
            asr_provider=asr_provider,
            asr_provider_active=asr_provider_active,
            asr_model=asr_model,
            asr_streaming=asr_streaming,
            asr_provider_error=self._dashscope_asr_last_error,
        )

    async def open_realtime_asr(
        self,
        *,
        end_silence_ms: int = 400,
        force: bool = False,
    ) -> DashScopeRealtimeASR | None:
        """Open one cloud ASR transport, returning ``None`` for local fallback."""
        settings = self.settings_store.load()
        if (
            not force
            and str(getattr(settings, "asr_provider", "dashscope_realtime"))
            != "dashscope_realtime"
        ):
            return None
        api_key = settings.dashscope_api_key
        if not api_key:
            self._dashscope_asr_last_error = "DashScope API key is not configured"
            if self.strict_no_fallback:
                raise VoiceDependencyError(self._dashscope_asr_last_error)
            return None
        transport = self.dashscope_asr_factory(
            model=str(
                getattr(settings, "dashscope_asr_model", "qwen3-asr-flash-realtime")
            ),
            silence_duration_ms=end_silence_ms,
        )
        try:
            await transport.connect(api_key)
        except Exception as exc:
            self._dashscope_asr_last_error = str(exc)
            try:
                await transport.close()
            except Exception:
                pass
            LOGGER.warning(
                "DashScope realtime ASR connection failed; using local fallback: %s",
                exc,
            )
            if self.strict_no_fallback:
                raise
            return None
        with self._active_asr_sessions_lock:
            self._active_asr_sessions.add(transport)
        self._dashscope_asr_last_error = None
        return transport

    async def release_realtime_asr(self, transport: DashScopeRealtimeASR) -> None:
        """Close and untrack one browser-scoped realtime ASR transport."""
        with self._active_asr_sessions_lock:
            self._active_asr_sessions.discard(transport)
        await transport.close()

    def note_realtime_asr_failure(self, error: Exception | str) -> None:
        """Expose a mid-session cloud failure through the runtime status API."""
        self._dashscope_asr_last_error = str(error)

    async def close_asr_connections(self) -> None:
        """Release every browser-scoped ASR connection during application shutdown."""
        with self._active_asr_sessions_lock:
            transports = list(self._active_asr_sessions)
            self._active_asr_sessions.clear()
        if transports:
            await asyncio.gather(
                *(transport.close() for transport in transports),
                return_exceptions=True,
            )

    def speech_to_text(self, audio_bytes: bytes, *, filename: str = "recording.wav") -> str:
        _validate_audio(audio_bytes)
        audio = _wav_bytes_to_float32(audio_bytes, filename=filename)
        settings = self.settings_store.load()
        model = self._load_whisper_model(settings.stt_model)
        temperature = getattr(settings, "stt_temperature", 0.0)
        if isinstance(temperature, (int, float)):
            if temperature <= 0:
                temperature = (0.0,)
            elif temperature < 0.1:
                temperature = (temperature, temperature * 2, temperature * 4)
        result = model.transcribe(
            audio,
            language="zh",
            fp16=False,
            temperature=temperature,
            beam_size=int(getattr(settings, "stt_beam_size", 5)),
            best_of=int(getattr(settings, "stt_best_of", 5)),
            patience=float(getattr(settings, "stt_patience", 1.0)),
        )
        raw_text = str(result.get("text") or "").strip() if isinstance(result, dict) else str(result or "").strip()
        return self.postprocess_asr_text(raw_text)

    def refresh_asr_corrections(self) -> None:
        """Reload the immutable ASR correction snapshot after settings change."""
        settings = self.settings_store.load()
        configured_enabled = bool(
            getattr(settings, "asr_correction_enabled", True)
        )
        enabled = (
            configured_enabled
            if self.stt_correction_enabled is None
            else bool(self.stt_correction_enabled)
        )
        rules = tuple(getattr(settings, "asr_correction_rules", ()))
        with self._asr_correction_lock:
            self._asr_correction_config = (enabled, rules)

    def postprocess_asr_text(self, text: str) -> str:
        """Apply the same Simplified-Chinese and glossary rules to every ASR provider."""
        raw_text = str(text or "").strip()
        try:
            raw_text = simplify_chinese(raw_text)
        except ImportError as exc:
            raise VoiceDependencyError(
                'OpenCC is not installed; install the project with the "voice" extra'
            ) from exc

        # ── 领域词表纠错 ──
        with self._asr_correction_lock:
            correction_enabled, custom_rules = self._asr_correction_config
        if correction_enabled and raw_text:
            correction = correct_stt_text(raw_text, custom_rules)
            return correction.corrected
        return raw_text

    def text_to_speech(self, text: str) -> bytes:
        cleaned, voice, rate = self._prepare_tts(text)

        if self.tts_synthesizer is not None:
            return self.tts_synthesizer(cleaned, voice, rate)
        if self.strict_no_fallback:
            raise VoiceDependencyError(
                "Strict no-fallback mode requires the Qwen streaming TTS endpoint"
            )
        return _edge_tts_to_mp3(cleaned, voice=voice, rate=rate)

    def stream_text_to_speech(self, text: str) -> AsyncIterator[bytes]:
        """Return an async iterator that yields MP3 chunks as they arrive.

        Text validation and settings lookup happen before the iterator is
        returned so HTTP callers can report input errors before sending their
        response headers.
        """
        cleaned, voice, rate = self._prepare_tts(text)
        settings = self.settings_store.load()
        dashscope_api_key = (
            settings.dashscope_api_key
            if self.prefer_dashscope_tts
            and self.tts_stream_synthesizer is None
            and self.tts_synthesizer is None
            else None
        )
        if (
            dashscope_api_key is None
            and self.tts_stream_synthesizer is None
            and self.tts_synthesizer is None
        ):
            if self.strict_no_fallback:
                raise VoiceDependencyError(
                    "DashScope API key is required in strict no-fallback mode"
                )
            try:
                import edge_tts  # noqa: F401
            except ImportError as exc:
                raise VoiceDependencyError("edge-tts is not installed") from exc
        return self._stream_prepared_tts(
            cleaned,
            voice=voice,
            rate=rate,
            dashscope_api_key=dashscope_api_key,
        )

    async def warm_tts_connection(self) -> bool:
        """Pre-connect the preferred realtime provider during app startup."""
        if (
            not self.prefer_dashscope_tts
            or self.tts_stream_synthesizer is not None
            or self.tts_synthesizer is not None
        ):
            return False
        api_key = self.settings_store.load().dashscope_api_key
        if not api_key:
            if self.strict_no_fallback:
                raise VoiceDependencyError(
                    "DashScope API key is required in strict no-fallback mode"
                )
            return False
        try:
            await self.dashscope_tts.warm(api_key)
        except Exception as exc:
            self._dashscope_last_error = str(exc)
            raise
        self._dashscope_last_error = None
        return True

    async def close_tts_connection(self) -> None:
        """Release the persistent realtime provider connection."""
        await self.dashscope_tts.close()

    async def _stream_prepared_tts(
        self,
        text: str,
        *,
        voice: str,
        rate: str,
        dashscope_api_key: str | None = None,
    ) -> AsyncIterator[bytes]:
        if self.tts_stream_synthesizer is not None:
            stream = self.tts_stream_synthesizer(text, voice, rate)
            try:
                async for chunk in stream:
                    if isinstance(chunk, bytes) and chunk:
                        yield chunk
            finally:
                await _close_async_iterator(stream)
            return

        # Keep existing injected synchronous providers useful to callers that
        # switch to the streaming API. They naturally produce a single chunk.
        if self.tts_synthesizer is not None:
            # This is an explicitly injected compatibility provider rather than
            # the normal Edge/Qwen streaming path. Calling it directly avoids
            # creating a one-shot default executor whose worker can keep short-
            # lived CLI and test event loops from shutting down on Python 3.12.
            chunk = self.tts_synthesizer(text, voice, rate)
            if isinstance(chunk, bytes) and chunk:
                yield chunk
            return

        if dashscope_api_key:
            dashscope_stream = self.dashscope_tts.stream(text, dashscope_api_key)
            emitted_audio = False
            try:
                async for chunk in dashscope_stream:
                    if isinstance(chunk, bytes) and chunk:
                        emitted_audio = True
                        yield chunk
                if emitted_audio:
                    self._dashscope_last_error = None
                    return
                self._dashscope_last_error = "DashScope realtime TTS returned no audio"
            except Exception as exc:
                self._dashscope_last_error = str(exc)
                if emitted_audio or self.strict_no_fallback:
                    raise
            finally:
                await _close_async_iterator(dashscope_stream)

            if self.strict_no_fallback:
                raise VoiceDependencyError(
                    self._dashscope_last_error
                    or "DashScope realtime TTS returned no audio"
                )

            LOGGER.warning(
                "DashScope realtime TTS failed before first audio; falling back to Edge-TTS: %s",
                self._dashscope_last_error,
            )

        stream = _edge_tts_stream_mp3(text, voice=voice, rate=rate)
        try:
            async for chunk in stream:
                yield chunk
        finally:
            await stream.aclose()

    def _prepare_tts(self, text: str) -> tuple[str, str, str]:
        cleaned = " ".join(str(text or "").split())
        if not cleaned:
            raise VoiceInputError("TTS text cannot be empty")
        if len(cleaned) > MAX_TTS_CHARS:
            raise VoiceInputError(f"TTS text is too long; max {MAX_TTS_CHARS} characters")

        # ── 文本规范化（数字播报、多音字）──
        if self.tts_normalize_enabled:
            norm = normalize_for_tts(cleaned)
            cleaned = norm.normalized

        # Validate the final plain text before optional SSML wrapping. This
        # also catches inputs such as citation-only text that normalization
        # removes entirely, while accepting letters and digits from any
        # Unicode script (including Chinese characters).
        if not any(character.isalnum() for character in cleaned):
            raise VoiceInputError("TTS text must contain at least one letter or number")

        settings = self.settings_store.load()

        # ── SSML 精细发音控制 ──
        if self.tts_ssml_enabled:
            cleaned = wrap_ssml(cleaned, voice=settings.tts_voice, rate=settings.tts_rate)
        return cleaned, settings.tts_voice, settings.tts_rate

    def _load_whisper_model(self, model_name: str) -> Any:
        if self.whisper_model_loader is not None:
            return self.whisper_model_loader(model_name)
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise VoiceDependencyError("faster-whisper is not installed") from exc

        model_dir = resolve_asr_model_dir(model_name)
        if not asr_model_files_present(model_dir):
            raise VoiceDependencyError(
                f"local ASR model is incomplete at {model_dir}; run: "
                "python -m rag_system.model_setup download --component asr"
            )
        cache_key = str(model_dir)
        with self._model_lock:
            if cache_key not in self._whisper_models:
                cpu_threads = max(1, min(8, os.cpu_count() or 4))
                model = WhisperModel(
                    cache_key,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=cpu_threads,
                    local_files_only=True,
                )
                self._whisper_models[cache_key] = _FasterWhisperAdapter(model)
            return self._whisper_models[cache_key]


class _FasterWhisperAdapter:
    def __init__(self, model: Any) -> None:
        self.model = model

    def transcribe(
        self,
        audio: Any,
        *,
        language: str,
        fp16: bool,
        temperature: Any,
        beam_size: int,
        best_of: int,
        patience: float,
    ) -> dict[str, str]:
        del fp16
        segments, _info = self.model.transcribe(
            audio,
            language=language,
            task="transcribe",
            temperature=temperature,
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
        )
        return {"text": "".join(str(segment.text) for segment in segments).strip()}


def _validate_audio(audio_bytes: bytes) -> None:
    if not audio_bytes:
        raise VoiceInputError("Audio file cannot be empty")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise VoiceInputError(f"Audio file is too large; max {MAX_AUDIO_BYTES} bytes")


def _wav_bytes_to_float32(audio_bytes: bytes, *, filename: str) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise VoiceDependencyError("numpy is not installed") from exc

    try:
        with wave.open(BytesIO(audio_bytes), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())
    except (EOFError, wave.Error) as exc:
        raise VoiceInputError(f"{filename} is not a valid WAV file") from exc

    if sample_width != 2:
        raise VoiceInputError("Only 16-bit PCM WAV audio is supported")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio


def _edge_tts_to_mp3(text: str, *, voice: str, rate: str) -> bytes:
    try:
        import edge_tts
    except ImportError as exc:
        raise VoiceDependencyError("edge-tts is not installed") from exc

    async def generate(path: str) -> None:
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save(path)

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as handle:
            temp_path = handle.name
        asyncio.run(generate(temp_path))
        return Path(temp_path).read_bytes()
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


async def _edge_tts_stream_mp3(
    text: str,
    *,
    voice: str,
    rate: str,
) -> AsyncIterator[bytes]:
    try:
        import edge_tts
    except ImportError as exc:
        raise VoiceDependencyError("edge-tts is not installed") from exc

    communicate = edge_tts.Communicate(text, voice, rate=rate)
    stream = communicate.stream()
    try:
        async for event in stream:
            if event.get("type") != "audio":
                continue
            data = event.get("data")
            if isinstance(data, bytes) and data:
                yield data
    finally:
        await _close_async_iterator(stream)


async def _close_async_iterator(stream: Any) -> None:
    close = getattr(stream, "aclose", None)
    if callable(close):
        await close()


def _module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _dedupe(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output
