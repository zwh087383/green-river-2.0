"""Persistent DashScope Qwen3 realtime text-to-speech transport."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from typing import Any

import aiohttp

DASHSCOPE_REALTIME_TTS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
DASHSCOPE_REALTIME_TTS_MODEL = "qwen3-tts-flash-realtime"


class DashScopeTTSError(RuntimeError):
    """Base error raised by the DashScope realtime TTS transport."""


class DashScopeTTSConnectionError(DashScopeTTSError):
    """The realtime connection failed or closed unexpectedly."""


class DashScopeTTSInputError(ValueError):
    """The realtime TTS request contains invalid input."""


class DashScopeRealtimeTTS:
    """Reuse one commit-mode Qwen3 TTS WebSocket across serialized segments.

    The API key is used only while opening a WebSocket. The instance retains a
    one-way digest to detect key changes, never the clear-text credential.
    """

    def __init__(
        self,
        *,
        websocket_url: str = DASHSCOPE_REALTIME_TTS_URL,
        model: str = DASHSCOPE_REALTIME_TTS_MODEL,
        voice: str = "Cherry",
        response_format: str = "mp3",
        sample_rate: int = 24_000,
        language_type: str = "Chinese",
        connect_timeout: float = 10.0,
        receive_timeout: float = 60.0,
        heartbeat: float = 20.0,
        session_factory: Callable[[], Any] | None = None,
        event_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.websocket_url = websocket_url
        self.model = model
        self.voice = voice
        self.response_format = response_format
        self.sample_rate = sample_rate
        self.language_type = language_type
        self.connect_timeout = connect_timeout
        self.receive_timeout = receive_timeout
        self.heartbeat = heartbeat
        self._session_factory = session_factory
        self._event_id_factory = event_id_factory or _new_event_id
        self._session: Any | None = None
        self._websocket: Any | None = None
        self._api_key_digest: str | None = None
        self._lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        """Whether an apparently usable persistent WebSocket is available."""
        return self._connection_is_open()

    async def warm(self, api_key: str) -> None:
        """Open and configure the persistent WebSocket without synthesizing."""
        cleaned_key = _validate_api_key(api_key)
        async with self._lock:
            await self._ensure_connection(cleaned_key)

    def stream(self, text: str, api_key: str) -> AsyncIterator[bytes]:
        """Return an async iterator of decoded MP3 chunks for one text segment."""
        cleaned_text = " ".join(str(text or "").split())
        if not cleaned_text:
            raise DashScopeTTSInputError("DashScope TTS text cannot be empty")
        if not any(character.isalnum() for character in cleaned_text):
            raise DashScopeTTSInputError(
                "DashScope TTS text must contain at least one letter or number"
            )
        cleaned_key = _validate_api_key(api_key)
        return self._stream_serialized(cleaned_text, cleaned_key)

    async def close(self) -> None:
        """Gracefully finish and close the persistent session."""
        async with self._lock:
            await self._close_connection(graceful=True)

    async def _stream_serialized(
        self,
        text: str,
        api_key: str,
    ) -> AsyncIterator[bytes]:
        async with self._lock:
            connection_healthy = False
            try:
                for attempt in range(2):
                    emitted_audio = False
                    try:
                        await self._ensure_connection(api_key)
                        async for chunk in self._synthesize_once(text):
                            emitted_audio = True
                            yield chunk
                        connection_healthy = True
                        return
                    except DashScopeTTSConnectionError:
                        await self._close_connection(graceful=False)
                        if attempt == 0 and not emitted_audio:
                            continue
                        raise
            finally:
                # If the consumer cancels or closes the iterator before
                # response.done, unread events make this session unsafe to reuse.
                if not connection_healthy:
                    with suppress(Exception):
                        await asyncio.shield(self._close_connection(graceful=False))

    async def _synthesize_once(self, text: str) -> AsyncIterator[bytes]:
        await self._send_event(
            {
                "event_id": self._event_id_factory(),
                "type": "input_text_buffer.append",
                "text": text,
            }
        )
        await self._send_event(
            {
                "event_id": self._event_id_factory(),
                "type": "input_text_buffer.commit",
            }
        )

        while True:
            event = await self._receive_event()
            event_type = str(event.get("type") or "")
            self._raise_for_service_error(event)

            if event_type == "response.audio.delta":
                delta = event.get("delta")
                if not isinstance(delta, str):
                    raise DashScopeTTSError(
                        "DashScope TTS returned an invalid audio delta"
                    )
                try:
                    chunk = base64.b64decode(delta, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise DashScopeTTSError(
                        "DashScope TTS returned malformed base64 audio"
                    ) from exc
                if chunk:
                    yield chunk
                continue

            if event_type == "response.done":
                response = event.get("response")
                status = response.get("status") if isinstance(response, dict) else None
                if status not in (None, "completed"):
                    raise DashScopeTTSError(
                        f"DashScope TTS response ended with status {status}"
                    )
                return

            if event_type == "session.finished":
                raise DashScopeTTSConnectionError(
                    "DashScope TTS session closed before the response completed"
                )

    async def _ensure_connection(self, api_key: str) -> None:
        key_digest = _api_key_digest(api_key)
        if self._connection_is_open() and self._api_key_digest == key_digest:
            return
        await self._close_connection(graceful=False)
        await self._connect(api_key, key_digest=key_digest)

    async def _connect(self, api_key: str, *, key_digest: str) -> None:
        session = self._new_session()
        self._session = session
        try:
            try:
                websocket = await session.ws_connect(
                    self.websocket_url,
                    params={"model": self.model},
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "User-Agent": "rag-system",
                    },
                    heartbeat=self.heartbeat,
                )
            except aiohttp.WSServerHandshakeError as exc:
                if exc.status in (401, 403):
                    raise DashScopeTTSError(
                        "DashScope TTS authentication failed"
                    ) from exc
                raise DashScopeTTSConnectionError(
                    f"DashScope TTS WebSocket handshake failed with status {exc.status}"
                ) from exc
            except (aiohttp.ClientError, ConnectionError, OSError) as exc:
                raise DashScopeTTSConnectionError(
                    "Could not connect to DashScope realtime TTS"
                ) from exc

            self._websocket = websocket
            await self._wait_for_event("session.created")
            await self._send_event(
                {
                    "event_id": self._event_id_factory(),
                    "type": "session.update",
                    "session": {
                        "voice": self.voice,
                        "mode": "commit",
                        "language_type": self.language_type,
                        "response_format": self.response_format,
                        "sample_rate": self.sample_rate,
                    },
                }
            )
            await self._wait_for_event("session.updated")
            self._api_key_digest = key_digest
        except asyncio.CancelledError:
            await asyncio.shield(self._close_connection(graceful=False))
            raise
        except Exception:
            await self._close_connection(graceful=False)
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
            event = await self._receive_event()
            self._raise_for_service_error(event)
            event_type = str(event.get("type") or "")
            if event_type == expected_type:
                return event
            if event_type == "session.finished":
                raise DashScopeTTSConnectionError(
                    f"DashScope TTS session closed before {expected_type}"
                )

    async def _send_event(self, event: dict[str, Any]) -> None:
        websocket = self._websocket
        if websocket is None or not self._connection_is_open():
            raise DashScopeTTSConnectionError("DashScope TTS WebSocket is not connected")
        try:
            await websocket.send_json(event)
        except (aiohttp.ClientError, ConnectionError, OSError, RuntimeError) as exc:
            raise DashScopeTTSConnectionError(
                "DashScope TTS WebSocket closed while sending"
            ) from exc

    async def _receive_event(self) -> dict[str, Any]:
        websocket = self._websocket
        if websocket is None or not self._connection_is_open():
            raise DashScopeTTSConnectionError("DashScope TTS WebSocket is not connected")
        try:
            event = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=self.receive_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise DashScopeTTSConnectionError(
                "Timed out waiting for DashScope TTS"
            ) from exc
        except (aiohttp.ClientError, ConnectionError, OSError, RuntimeError, TypeError) as exc:
            raise DashScopeTTSConnectionError(
                "DashScope TTS WebSocket disconnected"
            ) from exc
        if not isinstance(event, dict):
            raise DashScopeTTSError("DashScope TTS returned a non-object event")
        return event

    def _raise_for_service_error(self, event: dict[str, Any]) -> None:
        if event.get("type") != "error":
            return
        error = event.get("error")
        if isinstance(error, dict):
            code = str(error.get("code") or "unknown_error")
            message = str(error.get("message") or "request failed")
        else:
            code = "unknown_error"
            message = "request failed"
        raise DashScopeTTSError(f"DashScope TTS error ({code}): {message}")

    def _connection_is_open(self) -> bool:
        websocket = self._websocket
        return websocket is not None and not bool(getattr(websocket, "closed", False))

    async def _close_connection(self, *, graceful: bool) -> None:
        websocket = self._websocket
        session = self._session
        self._websocket = None
        self._session = None
        self._api_key_digest = None

        try:
            if graceful and websocket is not None and not bool(
                getattr(websocket, "closed", False)
            ):
                with suppress(Exception):
                    await websocket.send_json(
                        {
                            "event_id": self._event_id_factory(),
                            "type": "session.finish",
                        }
                    )
        finally:
            if websocket is not None:
                with suppress(Exception):
                    await websocket.close()
            if session is not None:
                with suppress(Exception):
                    await session.close()


def _validate_api_key(api_key: str) -> str:
    cleaned = str(api_key or "").strip()
    if not cleaned:
        raise DashScopeTTSInputError("DashScope API key is required for TTS")
    return cleaned


def _api_key_digest(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _new_event_id() -> str:
    return f"event_{uuid.uuid4().hex}"
