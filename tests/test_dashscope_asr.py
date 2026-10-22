from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable
from typing import Any

import pytest

from rag_system.voice.dashscope_asr import (
    DASHSCOPE_REALTIME_ASR_MODEL,
    DASHSCOPE_REALTIME_ASR_URL,
    DashScopeASRError,
    DashScopeASRInputError,
    DashScopeRealtimeASR,
)


class FakeWebSocket:
    def __init__(self, events: list[Any]) -> None:
        self.events = list(events)
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    async def send_json(self, event: dict[str, Any]) -> None:
        if self.closed:
            raise ConnectionError("closed")
        self.sent.append(event)

    async def receive_json(self) -> dict[str, Any]:
        if not self.events:
            raise AssertionError("fake websocket has no queued event")
        event = self.events.pop(0)
        if isinstance(event, BaseException):
            raise event
        return event

    async def close(self) -> None:
        self.closed = True


class BlockingWebSocket(FakeWebSocket):
    async def receive_json(self) -> dict[str, Any]:
        if self.events:
            return await super().receive_json()
        await asyncio.Future()
        raise AssertionError("unreachable")


class FakeSession:
    def __init__(self, websocket: FakeWebSocket) -> None:
        self.websocket = websocket
        self.ws_calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    async def ws_connect(self, url: str, **kwargs: Any) -> FakeWebSocket:
        self.ws_calls.append((url, kwargs))
        return self.websocket

    async def close(self) -> None:
        self.closed = True


def _event_ids() -> Callable[[], str]:
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"event_{counter}"

    return next_id


def _ready_events() -> list[dict[str, str]]:
    return [{"type": "session.created"}, {"type": "session.updated"}]


def test_connect_configures_pcm16_server_vad() -> None:
    websocket = FakeWebSocket(_ready_events())
    session = FakeSession(websocket)
    asr = DashScopeRealtimeASR(
        session_factory=lambda: session,
        event_id_factory=_event_ids(),
    )

    async def scenario() -> None:
        await asr.connect("sk-test-secret")
        assert asr.connected is True

    asyncio.run(scenario())

    assert session.ws_calls == [
        (
            DASHSCOPE_REALTIME_ASR_URL,
            {
                "params": {"model": DASHSCOPE_REALTIME_ASR_MODEL},
                "headers": {
                    "Authorization": "Bearer sk-test-secret",
                    "OpenAI-Beta": "realtime=v1",
                    "User-Agent": "rag-system",
                },
                "heartbeat": 20.0,
            },
        )
    ]
    assert websocket.sent == [
        {
            "event_id": "event_1",
            "type": "session.update",
            "session": {
                "input_audio_format": "pcm",
                "sample_rate": 16_000,
                "input_audio_transcription": {"language": "zh"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.4,
                    "silence_duration_ms": 400,
                },
            },
        }
    ]


def test_append_audio_sends_base64_pcm() -> None:
    websocket = FakeWebSocket(_ready_events())
    session = FakeSession(websocket)
    asr = DashScopeRealtimeASR(
        session_factory=lambda: session,
        event_id_factory=_event_ids(),
    )
    pcm = b"\x00\x01\x02\x03"

    async def scenario() -> None:
        await asr.connect("sk-test")
        await asr.append_audio(pcm)

    asyncio.run(scenario())

    append = websocket.sent[-1]
    assert append["type"] == "input_audio_buffer.append"
    assert base64.b64decode(append["audio"]) == pcm


def test_receive_event_parses_partial_and_completed_transcripts() -> None:
    websocket = FakeWebSocket(
        _ready_events()
        + [
            {
                "type": "conversation.item.input_audio_transcription.text",
                "item_id": "item-1",
                "language": "zh",
                "emotion": "neutral",
                "text": "绿色",
                "stash": "江河",
            },
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item-1",
                "language": "zh",
                "emotion": "neutral",
                "transcript": "绿色江河。",
            },
        ]
    )
    session = FakeSession(websocket)
    asr = DashScopeRealtimeASR(session_factory=lambda: session)

    async def scenario():
        await asr.connect("sk-test")
        return await asr.receive_event(), await asr.receive_event()

    partial, completed = asyncio.run(scenario())

    assert partial is not None
    assert partial.type == "partial"
    assert partial.text == "绿色江河"
    assert partial.confirmed_text == "绿色"
    assert partial.stash == "江河"
    assert partial.item_id == "item-1"
    assert completed is not None
    assert completed.type == "completed"
    assert completed.text == "绿色江河。"
    assert completed.confirmed_text == "绿色江河。"
    assert completed.stash == ""


def test_events_skips_lifecycle_events_and_stops_at_session_finished() -> None:
    websocket = FakeWebSocket(
        _ready_events()
        + [
            {"type": "input_audio_buffer.speech_started"},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "完成",
            },
            {"type": "session.finished"},
        ]
    )
    session = FakeSession(websocket)
    asr = DashScopeRealtimeASR(session_factory=lambda: session)

    async def scenario():
        await asr.connect("sk-test")
        return [event async for event in asr.events()]

    events = asyncio.run(scenario())

    assert [event.text for event in events] == ["完成"]
    assert asr.connected is False
    assert websocket.closed is True
    assert session.closed is True


@pytest.mark.parametrize(
    "server_event",
    [
        {
            "type": "error",
            "error": {"code": "bad_request", "message": "invalid audio"},
        },
        {
            "type": "conversation.item.input_audio_transcription.failed",
            "error": {"code": "decode_failed", "message": "cannot decode"},
        },
    ],
)
def test_server_errors_close_connection_without_exposing_key(server_event) -> None:
    secret = "sk-never-expose-this"
    websocket = FakeWebSocket(_ready_events() + [server_event])
    session = FakeSession(websocket)
    asr = DashScopeRealtimeASR(session_factory=lambda: session)

    async def scenario() -> None:
        await asr.connect(secret)
        with pytest.raises(DashScopeASRError) as caught:
            await asr.receive_event()
        assert secret not in str(caught.value)

    asyncio.run(scenario())
    assert websocket.closed is True
    assert session.closed is True


def test_finish_then_receive_final_and_session_finished() -> None:
    websocket = FakeWebSocket(
        _ready_events()
        + [
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "最后一句",
            },
            {"type": "session.finished"},
        ]
    )
    session = FakeSession(websocket)
    asr = DashScopeRealtimeASR(
        session_factory=lambda: session,
        event_id_factory=_event_ids(),
    )

    async def scenario():
        await asr.connect("sk-test")
        await asr.finish()
        final = await asr.receive_event()
        finished = await asr.receive_event()
        return final, finished

    final, finished = asyncio.run(scenario())

    assert websocket.sent[-1]["type"] == "session.finish"
    assert final is not None and final.text == "最后一句"
    assert finished is None
    assert asr.connected is False


def test_close_sends_finish_and_releases_resources() -> None:
    websocket = FakeWebSocket(_ready_events() + [{"type": "session.finished"}])
    session = FakeSession(websocket)
    asr = DashScopeRealtimeASR(session_factory=lambda: session)

    async def scenario() -> None:
        await asr.connect("sk-test")
        await asr.close()

    asyncio.run(scenario())

    assert websocket.sent[-1]["type"] == "session.finish"
    assert websocket.closed is True
    assert session.closed is True


def test_cancelled_receiver_closes_connection() -> None:
    websocket = BlockingWebSocket(_ready_events())
    session = FakeSession(websocket)
    asr = DashScopeRealtimeASR(session_factory=lambda: session)

    async def scenario() -> None:
        await asr.connect("sk-test")
        receiver = asyncio.create_task(asr.receive_event())
        await asyncio.sleep(0)
        receiver.cancel()
        with pytest.raises(asyncio.CancelledError):
            await receiver

    asyncio.run(scenario())
    assert websocket.closed is True
    assert session.closed is True


@pytest.mark.parametrize("pcm", [b"", b"\x00"])
def test_append_audio_rejects_invalid_pcm(pcm: bytes) -> None:
    asr = DashScopeRealtimeASR(session_factory=lambda: None)

    with pytest.raises(DashScopeASRInputError):
        asyncio.run(asr.append_audio(pcm))


def test_connect_requires_api_key() -> None:
    asr = DashScopeRealtimeASR(session_factory=lambda: None)

    with pytest.raises(DashScopeASRInputError):
        asyncio.run(asr.connect(""))
