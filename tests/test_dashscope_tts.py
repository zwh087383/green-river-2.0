from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable
from typing import Any

import pytest

from rag_system.voice.dashscope_tts import (
    DASHSCOPE_REALTIME_TTS_MODEL,
    DASHSCOPE_REALTIME_TTS_URL,
    DashScopeRealtimeTTS,
    DashScopeTTSError,
    DashScopeTTSInputError,
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


def _audio_event(data: bytes) -> dict[str, str]:
    return {
        "type": "response.audio.delta",
        "delta": base64.b64encode(data).decode("ascii"),
    }


def _done_event() -> dict[str, Any]:
    return {"type": "response.done", "response": {"status": "completed"}}


def test_warm_configures_commit_mode_and_close_finishes_session() -> None:
    websocket = FakeWebSocket(
        [{"type": "session.created"}, {"type": "session.updated"}]
    )
    session = FakeSession(websocket)
    tts = DashScopeRealtimeTTS(
        session_factory=lambda: session,
        event_id_factory=_event_ids(),
    )

    async def scenario() -> None:
        await tts.warm("sk-test-secret")
        assert tts.connected is True
        await tts.close()

    asyncio.run(scenario())

    assert session.ws_calls == [
        (
            DASHSCOPE_REALTIME_TTS_URL,
            {
                "params": {"model": DASHSCOPE_REALTIME_TTS_MODEL},
                "headers": {
                    "Authorization": "Bearer sk-test-secret",
                    "User-Agent": "rag-system",
                },
                "heartbeat": 20.0,
            },
        )
    ]
    update = websocket.sent[0]
    assert update["type"] == "session.update"
    assert update["session"] == {
        "voice": "Cherry",
        "mode": "commit",
        "language_type": "Chinese",
        "response_format": "mp3",
        "sample_rate": 24_000,
    }
    assert websocket.sent[-1]["type"] == "session.finish"
    assert websocket.closed is True
    assert session.closed is True


def test_stream_reuses_connection_and_serializes_segments() -> None:
    websocket = FakeWebSocket(
        [
            {"type": "session.created"},
            {"type": "session.updated"},
            _audio_event(b"first"),
            _done_event(),
            _audio_event(b"second"),
            _done_event(),
        ]
    )
    session = FakeSession(websocket)
    factory_calls = 0

    def session_factory() -> FakeSession:
        nonlocal factory_calls
        factory_calls += 1
        return session

    tts = DashScopeRealtimeTTS(
        session_factory=session_factory,
        event_id_factory=_event_ids(),
    )

    async def collect(text: str) -> bytes:
        return b"".join([chunk async for chunk in tts.stream(text, "sk-test")])

    async def scenario() -> tuple[bytes, bytes]:
        results = await asyncio.gather(collect("第一段"), collect("第二段"))
        await tts.close()
        return results[0], results[1]

    assert asyncio.run(scenario()) == (b"first", b"second")
    assert factory_calls == 1
    request_events = [
        event
        for event in websocket.sent
        if event["type"].startswith("input_text_buffer")
    ]
    assert [event["type"] for event in request_events] == [
        "input_text_buffer.append",
        "input_text_buffer.commit",
        "input_text_buffer.append",
        "input_text_buffer.commit",
    ]
    assert request_events[0]["text"] == "第一段"
    assert request_events[2]["text"] == "第二段"


def test_stream_reconnects_once_when_connection_drops_before_audio() -> None:
    first_websocket = FakeWebSocket(
        [
            {"type": "session.created"},
            {"type": "session.updated"},
            ConnectionError("network dropped"),
        ]
    )
    second_websocket = FakeWebSocket(
        [
            {"type": "session.created"},
            {"type": "session.updated"},
            _audio_event(b"reconnected"),
            _done_event(),
        ]
    )
    sessions = [FakeSession(first_websocket), FakeSession(second_websocket)]
    tts = DashScopeRealtimeTTS(
        session_factory=lambda: sessions.pop(0),
        event_id_factory=_event_ids(),
    )

    async def scenario() -> bytes:
        audio = b"".join(
            [chunk async for chunk in tts.stream("重连测试", "sk-test")]
        )
        await tts.close()
        return audio

    assert asyncio.run(scenario()) == b"reconnected"
    assert first_websocket.closed is True
    assert first_websocket.sent[-1]["type"] == "input_text_buffer.commit"


def test_service_error_drops_connection_without_exposing_api_key() -> None:
    secret = "sk-never-put-this-in-an-error"
    websocket = FakeWebSocket(
        [
            {"type": "session.created"},
            {"type": "session.updated"},
            {
                "type": "error",
                "error": {"code": "invalid_value", "message": "bad request"},
            },
        ]
    )
    session = FakeSession(websocket)
    tts = DashScopeRealtimeTTS(session_factory=lambda: session)

    async def scenario() -> None:
        with pytest.raises(DashScopeTTSError) as caught:
            async for _chunk in tts.stream("测试", secret):
                pass
        assert secret not in str(caught.value)

    asyncio.run(scenario())
    assert websocket.closed is True
    assert session.closed is True
    assert tts.connected is False


def test_cancelling_stream_invalidates_connection() -> None:
    websocket = FakeWebSocket(
        [
            {"type": "session.created"},
            {"type": "session.updated"},
            _audio_event(b"partial"),
            _audio_event(b"unread"),
        ]
    )
    session = FakeSession(websocket)
    tts = DashScopeRealtimeTTS(session_factory=lambda: session)

    async def scenario() -> bytes:
        stream = tts.stream("取消测试", "sk-test")
        first = await anext(stream)
        await stream.aclose()
        return first

    assert asyncio.run(scenario()) == b"partial"
    assert websocket.closed is True
    assert session.closed is True
    assert tts.connected is False


@pytest.mark.parametrize(("text", "api_key"), [("", "sk-test"), ("……", "sk-test"), ("测试", "")])
def test_stream_rejects_invalid_input(text: str, api_key: str) -> None:
    tts = DashScopeRealtimeTTS(session_factory=lambda: None)

    with pytest.raises(DashScopeTTSInputError):
        tts.stream(text, api_key)
