"""Async DashScope Qwen3 realtime speech recognition transport."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from dataclasses import asdict, dataclass
from typing import Any, Literal

import aiohttp

DASHSCOPE_REALTIME_ASR_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
DASHSCOPE_REALTIME_ASR_MODEL = "qwen3-asr-flash-realtime"
DASHSCOPE_ASR_SAMPLE_RATE = 16_000
DASHSCOPE_ASR_VAD_THRESHOLD = 0.4
DASHSCOPE_ASR_SILENCE_MS = 400


class DashScopeASRError(RuntimeError):
    """Base error raised by the DashScope realtime ASR transport."""


class DashScopeASRConnectionError(DashScopeASRError):
    """The realtime ASR connection failed or closed unexpectedly."""


class DashScopeASRInputError(ValueError):
    """Invalid API key or PCM audio was supplied."""


@dataclass(frozen=True, slots=True)
class DashScopeASREvent:
    """A normalized partial or completed recognition event."""

    type: Literal["partial", "completed"]
    text: str
    confirmed_text: str = ""
    stash: str = ""
    language: str | None = None
    emotion: str | None = None
    item_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DashScopeRealtimeASR:
    """One Qwen3-ASR-Realtime WebSocket session.

    Audio must be raw little-endian 16-bit PCM, mono, at 16 kHz. A transport
    is intended for one browser/client session. After ``finish`` it rejects
    further audio; call ``connect`` again or create a new instance for a new
    session. The clear-text API key is never retained on the instance.
    """

    def __init__(
        self,
        *,
        websocket_url: str = DASHSCOPE_REALTIME_ASR_URL,
        model: str = DASHSCOPE_REALTIME_ASR_MODEL,
        language: str = "zh",
        sample_rate: int = DASHSCOPE_ASR_SAMPLE_RATE,
        vad_threshold: float = DASHSCOPE_ASR_VAD_THRESHOLD,
        silence_duration_ms: int = DASHSCOPE_ASR_SILENCE_MS,
        connect_timeout: float = 10.0,
        receive_timeout: float = 60.0,
        close_timeout: float = 3.0,
        heartbeat: float = 20.0,
        session_factory: Callable[[], Any] | None = None,
        event_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if sample_rate != DASHSCOPE_ASR_SAMPLE_RATE:
            raise DashScopeASRInputError("DashScope ASR requires 16000 Hz audio")
        if not -1.0 <= vad_threshold <= 1.0:
            raise DashScopeASRInputError("DashScope ASR VAD threshold must be between -1 and 1")
        if not 200 <= silence_duration_ms <= 6_000:
            raise DashScopeASRInputError(
                "DashScope ASR silence duration must be between 200 and 6000 ms"
            )

        self.websocket_url = websocket_url
        self.model = model
        self.language = language
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self.silence_duration_ms = silence_duration_ms
        self.connect_timeout = connect_timeout
        self.receive_timeout = receive_timeout
        self.close_timeout = close_timeout
        self.heartbeat = heartbeat
        self._session_factory = session_factory
        self._event_id_factory = event_id_factory or _new_event_id
        self._session: Any | None = None
        self._websocket: Any | None = None
        self._api_key_digest: str | None = None
        self._finishing = False
        self._send_lock = asyncio.Lock()
        self._receive_lock = asyncio.Lock()
        self._connect_lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self._connection_is_open()

    async def connect(self, api_key: str) -> None:
        """Open the WebSocket and configure PCM input plus server VAD."""
        cleaned_key = _validate_api_key(api_key)
        key_digest = _api_key_digest(cleaned_key)
        async with self._connect_lock:
            if (
                self._connection_is_open()
                and self._api_key_digest == key_digest
                and not self._finishing
            ):
                return
            await self._drop_connection()
            await self._connect(cleaned_key, key_digest=key_digest)

    async def append_audio(self, pcm16: bytes) -> None:
        """Append one raw PCM16 mono audio chunk to the server buffer."""
        if not isinstance(pcm16, bytes) or not pcm16:
            raise DashScopeASRInputError("DashScope ASR audio chunk cannot be empty")
        if len(pcm16) % 2:
            raise DashScopeASRInputError("DashScope ASR PCM16 audio length must be even")
        if self._finishing:
            raise DashScopeASRInputError("DashScope ASR session is already finishing")

        event = {
            "event_id": self._event_id_factory(),
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm16).decode("ascii"),
        }
        async with self._send_lock:
            try:
                await self._send_event(event)
            except asyncio.CancelledError:
                await asyncio.shield(self._drop_connection())
                raise
            except DashScopeASRError:
                await self._drop_connection()
                raise

    async def receive_event(self) -> DashScopeASREvent | None:
        """Return the next partial/final event, or ``None`` at session.finished."""
        async with self._receive_lock:
            try:
                while True:
                    event = await self._receive_raw_event()
                    event_type = str(event.get("type") or "")
                    self._raise_for_service_error(event)

                    if event_type == "conversation.item.input_audio_transcription.text":
                        confirmed = event.get("text")
                        stash = event.get("stash")
                        if not isinstance(confirmed, str) or not isinstance(stash, str):
                            raise DashScopeASRError(
                                "DashScope ASR returned an invalid partial transcript"
                            )
                        return DashScopeASREvent(
                            type="partial",
                            text=confirmed + stash,
                            confirmed_text=confirmed,
                            stash=stash,
                            language=_optional_string(event.get("language")),
                            emotion=_optional_string(event.get("emotion")),
                            item_id=_optional_string(event.get("item_id")),
                        )

                    if (
                        event_type
                        == "conversation.item.input_audio_transcription.completed"
                    ):
                        transcript = event.get("transcript")
                        if not isinstance(transcript, str):
                            raise DashScopeASRError(
                                "DashScope ASR returned an invalid final transcript"
                            )
                        return DashScopeASREvent(
                            type="completed",
                            text=transcript,
                            confirmed_text=transcript,
                            language=_optional_string(event.get("language")),
                            emotion=_optional_string(event.get("emotion")),
                            item_id=_optional_string(event.get("item_id")),
                        )

                    if event_type == "session.finished":
                        await self._drop_connection()
                        return None
            except asyncio.CancelledError:
                await asyncio.shield(self._drop_connection())
                raise
            except DashScopeASRError:
                await self._drop_connection()
                raise

    async def events(self) -> AsyncIterator[DashScopeASREvent]:
        """Iterate normalized events until the server finishes the session."""
        while True:
            event = await self.receive_event()
            if event is None:
                return
            yield event

    async def finish(self) -> None:
        """Tell the server no more audio will arrive, preserving final results."""
        if not self._connection_is_open():
            return
        async with self._send_lock:
            if self._finishing:
                return
            self._finishing = True
            try:
                await self._send_event(
                    {
                        "event_id": self._event_id_factory(),
                        "type": "session.finish",
                    }
                )
            except asyncio.CancelledError:
                await asyncio.shield(self._drop_connection())
                raise
            except DashScopeASRError:
                await self._drop_connection()
                raise

    async def close(self) -> None:
        """Finish, briefly drain final events, and always release resources."""
        if not self._connection_is_open():
            await self._drop_connection()
            return
        try:
            await self.finish()

            async def drain_until_finished() -> None:
                while self._connection_is_open():
                    event = await self.receive_event()
                    if event is None:
                        return

            await asyncio.wait_for(drain_until_finished(), timeout=self.close_timeout)
        except (asyncio.TimeoutError, DashScopeASRError):
            pass
        finally:
            await self._drop_connection()

    async def _connect(self, api_key: str, *, key_digest: str) -> None:
        session = self._new_session()
        self._session = session
        self._finishing = False
        try:
            try:
                websocket = await session.ws_connect(
                    self.websocket_url,
                    params={"model": self.model},
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "OpenAI-Beta": "realtime=v1",
                        "User-Agent": "rag-system",
                    },
                    heartbeat=self.heartbeat,
                )
            except aiohttp.WSServerHandshakeError as exc:
                if exc.status in (401, 403):
                    raise DashScopeASRError(
                        "DashScope ASR authentication failed"
                    ) from exc
                raise DashScopeASRConnectionError(
                    f"DashScope ASR WebSocket handshake failed with status {exc.status}"
                ) from exc
            except (aiohttp.ClientError, ConnectionError, OSError) as exc:
                raise DashScopeASRConnectionError(
                    "Could not connect to DashScope realtime ASR"
                ) from exc

            self._websocket = websocket
            await self._wait_for_event("session.created")
            await self._send_event(
                {
                    "event_id": self._event_id_factory(),
                    "type": "session.update",
                    "session": {
                        "input_audio_format": "pcm",
                        "sample_rate": self.sample_rate,
                        "input_audio_transcription": {"language": self.language},
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": self.vad_threshold,
                            "silence_duration_ms": self.silence_duration_ms,
                        },
                    },
                }
            )
            await self._wait_for_event("session.updated")
            self._api_key_digest = key_digest
        except asyncio.CancelledError:
            await asyncio.shield(self._drop_connection())
            raise
        except Exception:
            await self._drop_connection()
            raise

    def _new_session(self) -> Any:
        if self._session_factory is not None:
            return self._session_factory()
        timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=self.connect_timeout,
            sock_read=None,
        )
        return aiohttp.ClientSession(timeout=timeout, trust_env=True)

    async def _wait_for_event(self, expected_type: str) -> dict[str, Any]:
        while True:
            event = await self._receive_raw_event()
            self._raise_for_service_error(event)
            event_type = str(event.get("type") or "")
            if event_type == expected_type:
                return event
            if event_type == "session.finished":
                raise DashScopeASRConnectionError(
                    f"DashScope ASR session closed before {expected_type}"
                )

    async def _send_event(self, event: dict[str, Any]) -> None:
        websocket = self._websocket
        if websocket is None or not self._connection_is_open():
            raise DashScopeASRConnectionError("DashScope ASR WebSocket is not connected")
        try:
            await websocket.send_json(event)
        except (aiohttp.ClientError, ConnectionError, OSError, RuntimeError) as exc:
            raise DashScopeASRConnectionError(
                "DashScope ASR WebSocket closed while sending"
            ) from exc

    async def _receive_raw_event(self) -> dict[str, Any]:
        websocket = self._websocket
        if websocket is None or not self._connection_is_open():
            raise DashScopeASRConnectionError("DashScope ASR WebSocket is not connected")
        try:
            event = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=self.receive_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise DashScopeASRConnectionError(
                "Timed out waiting for DashScope ASR"
            ) from exc
        except (aiohttp.ClientError, ConnectionError, OSError, RuntimeError, TypeError) as exc:
            raise DashScopeASRConnectionError(
                "DashScope ASR WebSocket disconnected"
            ) from exc
        if not isinstance(event, dict):
            raise DashScopeASRError("DashScope ASR returned a non-object event")
        return event

    def _raise_for_service_error(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        if event_type not in (
            "error",
            "conversation.item.input_audio_transcription.failed",
        ):
            return
        error = event.get("error")
        if isinstance(error, dict):
            code = str(error.get("code") or "unknown_error")
            message = str(error.get("message") or "request failed")
        else:
            code = "unknown_error"
            message = "request failed"
        raise DashScopeASRError(f"DashScope ASR error ({code}): {message}")

    def _connection_is_open(self) -> bool:
        websocket = self._websocket
        return websocket is not None and not bool(getattr(websocket, "closed", False))

    async def _drop_connection(self) -> None:
        websocket = self._websocket
        session = self._session
        self._websocket = None
        self._session = None
        self._api_key_digest = None
        self._finishing = False
        if websocket is not None:
            with suppress(Exception):
                await websocket.close()
        if session is not None:
            with suppress(Exception):
                await session.close()


def _validate_api_key(api_key: str) -> str:
    cleaned = str(api_key or "").strip()
    if not cleaned:
        raise DashScopeASRInputError("DashScope API key is required for ASR")
    return cleaned


def _api_key_digest(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _new_event_id() -> str:
    return f"event_{uuid.uuid4().hex}"


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None
