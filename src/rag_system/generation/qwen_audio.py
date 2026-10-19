"""One manually triggered, text-context Qwen Audio reply per connection."""
from __future__ import annotations

import base64
import json
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from concurrent.futures import Future
from threading import BoundedSemaphore, Event, Lock, Thread
from typing import Any
from urllib.parse import urlencode

from rag_system.settings import AppSettings


class AudioReplyError(RuntimeError):
    pass


class ReplyCancelled(Exception):
    pass


_PREPARE_SLOTS = BoundedSemaphore(4)


@dataclass
class QwenAudioReplyClient:
    api_key: str = field(repr=False)
    model: str = "qwen-audio-3.1-realtime-plus"
    url: str = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
    voice: str = "longanqian_v3.1"
    timeout: float = 90.0
    connect: Callable[..., Any] | None = field(default=None, repr=False)

    @classmethod
    def from_settings(cls, settings: AppSettings) -> "QwenAudioReplyClient":
        if not settings.dashscope_api_key:
            raise AudioReplyError("请先配置百炼 API Key")
        return cls(api_key=settings.dashscope_api_key, model=settings.answer_model,
                   url=settings.audio_reply_url, voice=settings.audio_reply_voice)

    def prepare(self, instructions: str, *, audio: bool = True,
                should_cancel: Callable[[], bool] = lambda: False) -> PreparedReply | None:
        """Configure one reply connection while retrieval runs; never generate."""
        if should_cancel() or not _PREPARE_SLOTS.acquire(blocking=False):
            return None
        return PreparedReply(self, instructions, audio, should_cancel)

    def _open_session(self, instructions: str, audio: bool,
                      should_cancel: Callable[[], bool]) -> _ReplySession:
        session = _ReplySession(self, should_cancel)
        try:
            session.configure(instructions, audio)
            return session
        except BaseException:
            session.close()
            raise

    def stream(self, messages: list[dict[str, str]], *, audio: bool = True,
               should_cancel: Callable[[], bool] = lambda: False,
               prepared: PreparedReply | None = None) -> Iterator[dict[str, Any]]:
        if should_cancel():
            if prepared:
                prepared.close()
            return
        instructions = "\n".join(m["content"] for m in messages if m["role"] == "system")
        session = None
        started = False
        completed = False
        response_id = ""
        try:
            session = (prepared.claim(self, instructions, audio, should_cancel) if prepared else
                       self._open_session(instructions, audio, should_cancel))
            send, receive, wait_for = session.send, session.receive, session.wait_for
            for message in messages:
                if message["role"] == "system":
                    continue
                item_id = "ctx_" + uuid.uuid4().hex
                send({"type": "conversation.item.create", "item": {
                    "id": item_id, "type": "message", "role": message["role"],
                    "content": [{"type": "output_text" if message["role"] == "assistant" else "input_text",
                                 "text": message["content"]}],
                }})
                wait_for("conversation.item.created", item_id)
            if should_cancel():
                return
            send({"type": "response.create"})
            started = True
            text_kind = ""
            text_parts: list[str] = []
            audio_bytes = 0
            while True:
                event = receive()
                kind = event.get("type", "")
                if kind == "response.created":
                    if response_id:
                        raise AudioReplyError("Multiple responses in one reply connection")
                    response_id = str(event.get("response", {}).get("id") or "")
                    if not response_id:
                        raise AudioReplyError("Missing response identity")
                    continue
                if not kind.startswith("response."):
                    continue
                owner = event.get("response_id") or event.get("response", {}).get("id")
                if not response_id or owner != response_id:
                    raise AudioReplyError("Reply event response identity mismatch")
                if kind in {"response.audio_transcript.delta", "response.text.delta"}:
                    if text_kind and text_kind != kind:
                        continue  # Some providers emit both representations of the same words.
                    text_kind = kind
                    delta = str(event.get("delta") or "")
                    text_parts.append(delta)
                    if delta:
                        yield {"type": "delta", "text": delta, "response_id": response_id}
                elif kind == "response.audio.delta":
                    data = str(event.get("delta") or "")
                    try:
                        pcm = base64.b64decode(data, validate=True)
                    except ValueError as exc:
                        raise AudioReplyError("Invalid audio payload") from exc
                    if not audio or not pcm or len(pcm) % 2:
                        raise AudioReplyError("Invalid PCM16 reply audio")
                    audio_bytes += len(pcm)
                    yield {"type": "audio", "audio": data, "format": "pcm16",
                           "sample_rate": 24000, "channels": 1, "response_id": response_id}
                elif kind == "response.done":
                    response = event["response"]
                    if response.get("status") != "completed":
                        raise AudioReplyError("Qwen Audio response did not complete")
                    if not "".join(text_parts).strip() or (audio and not audio_bytes):
                        raise AudioReplyError("Qwen Audio completed without the requested text/audio")
                    completed = True
                    yield {"type": "reply_done", "response_id": response_id,
                           "usage": response.get("usage"), "audio_bytes": audio_bytes}
                    return
        except ReplyCancelled:
            return
        finally:
            if session and started and not completed:
                with suppress(Exception):
                    session.send({"type": "response.cancel"})
            if session:
                session.close()
            if prepared:
                prepared.close()


class _ReplySession:
    def __init__(self, client: QwenAudioReplyClient, should_cancel: Callable[[], bool]) -> None:
        from websockets.sync.client import connect
        if should_cancel():
            raise ReplyCancelled()
        self.client = client
        self.should_cancel = should_cancel
        self._close_lock = Lock()
        self._closed = False
        self.connection = (client.connect or connect)(
            client.url + "?" + urlencode({"model": client.model}),
            additional_headers={"Authorization": "Bearer " + client.api_key, "OpenAI-Beta": "realtime=v1"},
            open_timeout=10, close_timeout=2, max_size=4 * 1024 * 1024, proxy=None)
        self.deadline = time.monotonic() + client.timeout

    def send(self, event: dict[str, Any]) -> None:
        event.setdefault("event_id", "evt_" + uuid.uuid4().hex)
        self.connection.send(json.dumps(event, ensure_ascii=False))

    def receive(self) -> dict[str, Any]:
        while True:
            if self.should_cancel():
                raise ReplyCancelled()
            if time.monotonic() >= self.deadline:
                raise AudioReplyError("Qwen Audio reply timed out")
            try:
                raw = self.connection.recv(timeout=0.1)
            except TimeoutError:
                continue
            event = json.loads(raw)
            if event.get("type") == "error":
                raise AudioReplyError("Qwen Audio rejected the request: " + str(event.get("error", {}).get("code", "unknown")))
            return event

    def wait_for(self, kind: str, item_id: str = "") -> dict[str, Any]:
        while True:
            event = self.receive()
            if event.get("type", "").startswith("response."):
                raise AudioReplyError("Unexpected response before context acknowledgement")
            if event.get("type") == kind:
                if item_id and event.get("item", {}).get("id") != item_id:
                    raise AudioReplyError("Context acknowledgement identity mismatch")
                return event

    def configure(self, instructions: str, audio: bool) -> None:
        self.wait_for("session.created")
        modalities = ["text", "audio"] if audio else ["text"]
        self.send({"type": "session.update", "session": {
            "modalities": modalities, "voice": self.client.voice,
            "turn_detection": None, "output_audio_format": "pcm",
            "instructions": instructions, "enable_search": False, "max_history_turns": 50,
        }})
        echo = self.wait_for("session.updated").get("session", {})
        # The live provider omits null turn_detection when confirming manual mode.
        if echo.get("turn_detection") is not None:
            raise AudioReplyError("Qwen Audio did not confirm manual response mode")
        if set(echo.get("modalities", [])) != set(modalities) or echo.get("output_audio_format") != "pcm":
            raise AudioReplyError("Qwen Audio output configuration mismatch")
        if echo.get("model", self.client.model) != self.client.model:
            raise AudioReplyError("Qwen Audio model mismatch")

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self.connection.close()


class PreparedReply:
    """Single-owner, single-use connection; bounded preparation with no queue."""
    def __init__(self, client: QwenAudioReplyClient, instructions: str, audio: bool,
                 should_cancel: Callable[[], bool]) -> None:
        self.client, self.instructions, self.audio = client, instructions, audio
        self._external_cancel = should_cancel
        self._cancel = Event()
        self._ready = Event()
        self._lock = Lock()
        self._slots = _PREPARE_SLOTS
        self._slot_released = False
        self._session: _ReplySession | None = None
        self._claimed = False
        self._future: Future[_ReplySession] = Future()
        try:
            Thread(target=self._run, name="qwen-prepare", daemon=True).start()
        except BaseException:
            _PREPARE_SLOTS.release()
            raise

    def cancelled(self) -> bool:
        return self._cancel.is_set() or self._external_cancel()

    def _run(self) -> None:
        session = None
        try:
            session = self.client._open_session(self.instructions, self.audio, self.cancelled)
            with self._lock:
                if self.cancelled():
                    raise ReplyCancelled()
                self._session = session
                self._future.set_result(session)
        except BaseException as exc:
            try:
                if session:
                    session.close()
            except BaseException as cleanup_error:
                exc = cleanup_error
            self._future.set_exception(exc)
        finally:
            if self.cancelled() or self._future.exception() is not None:
                self._release_slot()
            self._ready.set()

    def _release_slot(self) -> None:
        with self._lock:
            if not self._slot_released:
                self._slot_released = True
                self._slots.release()

    def claim(self, client: QwenAudioReplyClient, instructions: str, audio: bool,
              should_cancel: Callable[[], bool]) -> _ReplySession:
        with self._lock:
            if self._claimed or client is not self.client or instructions != self.instructions or audio != self.audio:
                raise AudioReplyError("Prepared reply ownership/configuration mismatch")
            self._claimed = True
        while True:
            if self.cancelled() or should_cancel():
                raise ReplyCancelled()
            if self._ready.wait(.1):
                session = self._future.result(timeout=.1)
                session.should_cancel = lambda: self.cancelled() or should_cancel()
                return session

    def close(self) -> None:
        self._cancel.set()
        with self._lock:
            session = self._session
        if session:
            try:
                session.close()
            finally:
                self._release_slot()
