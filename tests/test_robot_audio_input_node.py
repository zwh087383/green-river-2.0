from __future__ import annotations

import importlib
import queue
import sys
from pathlib import Path
from threading import RLock
from types import ModuleType, SimpleNamespace
from typing import Any, Iterator

import pytest


ROOT = Path(__file__).resolve().parents[1]
VOICE = ROOT / "ros2" / "voice"


class _Message:
    def __init__(self) -> None:
        self.header = SimpleNamespace(stamp=None)
        self.request_id = ""
        self.session_id = ""
        self.generation = 0
        self.purpose = ""
        self.command = ""
        self.reason = ""
        self.state = ""
        self.detail = ""
        self.text = ""
        self.top_k = 0


class _String:
    def __init__(self) -> None:
        self.data = ""


class _ClockValue:
    def to_msg(self) -> object:
        return object()


class _Clock:
    def now(self) -> _ClockValue:
        return _ClockValue()


class _Logger:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, detail: str) -> None:
        self.errors.append(detail)

    def warning(self, detail: str) -> None:
        self.warnings.append(detail)


class _RosNode:
    def get_clock(self) -> _Clock:
        return _Clock()

    def get_logger(self) -> _Logger:
        return _Logger()


class _Publisher:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    def publish(self, message: Any) -> None:
        self.messages.append(message)


class _WakeSession:
    def accept_pcm16(self, _pcm16: bytes) -> None:
        return None

    def reset(self) -> None:
        return None

    def close(self) -> None:
        return None


class _WakeEngine:
    def create_session(self) -> _WakeSession:
        return _WakeSession()


class _VadSession:
    def feed_pcm16(self, _pcm16: bytes, *, is_final: bool) -> list[object]:
        assert not is_final
        return []

    def close(self) -> None:
        return None


class _VadEngine:
    def __init__(self) -> None:
        self.sessions: list[_VadSession] = []

    def create_session(self, _config: object) -> _VadSession:
        session = _VadSession()
        self.sessions.append(session)
        return session


class _RecordingPipeline:
    def __init__(self, generation: int, state: str = "armed") -> None:
        self.generation = generation
        self.state = state
        self.feed_calls: list[bytes] = []
        self.begin_answering_calls: list[int] = []

    def feed(self, pcm16: bytes) -> tuple[object, ...]:
        self.feed_calls.append(pcm16)
        return ()

    def begin_answering(self, generation: int) -> bool:
        self.begin_answering_calls.append(generation)
        self.state = "answering"
        return True


class _WakeEventPipeline(_RecordingPipeline):
    def feed(self, pcm16: bytes) -> tuple[object, ...]:
        self.feed_calls.append(pcm16)
        return (
            SimpleNamespace(
                type="wake",
                keyword="小绿小绿",
                generation=self.generation,
            ),
        )


class _InterruptEventPipeline(_RecordingPipeline):
    def feed(self, pcm16: bytes) -> tuple[object, ...]:
        self.feed_calls.append(pcm16)
        return (
            SimpleNamespace(
                type="interrupt",
                keyword="小绿小绿",
                detail="wake word during answer",
                generation=self.generation,
            ),
        )


@pytest.fixture
def audio_input_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.syspath_prepend(str(VOICE))

    rclpy = ModuleType("rclpy")
    rclpy_node = ModuleType("rclpy.node")
    rclpy_node.Node = _RosNode
    rclpy.node = rclpy_node
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.node", rclpy_node)

    std_msgs = ModuleType("std_msgs")
    std_msgs_msg = ModuleType("std_msgs.msg")
    std_msgs_msg.String = _String
    std_msgs.msg = std_msgs_msg
    monkeypatch.setitem(sys.modules, "std_msgs", std_msgs)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", std_msgs_msg)

    rag_interfaces = ModuleType("rag_interfaces")
    rag_interfaces_msg = ModuleType("rag_interfaces.msg")
    rag_interfaces_msg.VoiceControl = _Message
    rag_interfaces_msg.VoicePlayback = _Message
    rag_interfaces_msg.VoiceQuery = _Message
    rag_interfaces.msg = rag_interfaces_msg
    monkeypatch.setitem(sys.modules, "rag_interfaces", rag_interfaces)
    monkeypatch.setitem(sys.modules, "rag_interfaces.msg", rag_interfaces_msg)

    contracts = ModuleType("rag_system.contracts")
    contracts.AudioBuffer = object
    contracts.AudioSpec = object
    settings = ModuleType("rag_system.settings")
    settings.SettingsStore = object
    service = ModuleType("rag_system.voice.service")
    service.VoiceService = object
    vad = ModuleType("rag_system.voice.vad")
    vad.FsmnVadEngine = object
    vad.VadConfig = object
    wakeup = ModuleType("rag_system.voice.wakeup")
    wakeup.WakeWordEngine = object
    monkeypatch.setitem(sys.modules, "rag_system.contracts", contracts)
    monkeypatch.setitem(sys.modules, "rag_system.settings", settings)
    monkeypatch.setitem(sys.modules, "rag_system.voice.service", service)
    monkeypatch.setitem(sys.modules, "rag_system.voice.vad", vad)
    monkeypatch.setitem(sys.modules, "rag_system.voice.wakeup", wakeup)

    previous = sys.modules.pop("voice.audio_input_node", None)
    module = importlib.import_module("voice.audio_input_node")
    try:
        yield module
    finally:
        sys.modules.pop("voice.audio_input_node", None)
        if previous is not None:
            sys.modules["voice.audio_input_node"] = previous


def _make_node(module: ModuleType, pipeline: Any, *, generation: int) -> Any:
    node = module.RobotAudioInputNode.__new__(module.RobotAudioInputNode)
    node._audio_chunks = queue.Queue()
    node._worker_events = queue.SimpleQueue()
    node._epoch_lock = RLock()
    node._epoch = generation
    node._pending_request_id = ""
    node._pending_generation = 0
    node._active_request_id = ""
    node._interrupt_request_id = ""
    node._interrupt_ack_request_id = ""
    node._interrupt_generation = 0
    node._session_id = "robot"
    node._top_k = 4
    node._pipeline = pipeline
    node._query_publisher = _Publisher()
    node._interrupt_publisher = _Publisher()
    node._arm_motion_request_publisher = _Publisher()
    node._status_publisher = _Publisher()
    return node


def _make_pipeline(module: ModuleType) -> Any:
    return module.LocalAudioPipeline(
        wake_engine=_WakeEngine(),
        vad_engine=_VadEngine(),
        vad_config=object(),
    )


def _playback(
    module: ModuleType,
    *,
    request_id: str,
    generation: int,
    purpose: str,
    state: str,
) -> Any:
    message = module.VoicePlayback()
    message.request_id = request_id
    message.session_id = "robot"
    message.generation = generation
    message.purpose = purpose
    message.state = state
    return message


def test_stale_asr_worker_result_cannot_publish_query(
    audio_input_module: ModuleType,
) -> None:
    pipeline = _RecordingPipeline(generation=8, state="transcribing")
    node = _make_node(audio_input_module, pipeline, generation=8)
    node._worker_events.put(
        audio_input_module._WorkerEvent(
            generation=7,
            type="query",
            detail="cancelled question",
        )
    )

    node._flush_worker_events()

    assert node._query_publisher.messages == []
    assert pipeline.begin_answering_calls == []
    assert pipeline.state == "transcribing"


def test_stale_audio_chunk_never_reaches_pipeline(
    audio_input_module: ModuleType,
) -> None:
    pipeline = _RecordingPipeline(generation=5)
    node = _make_node(audio_input_module, pipeline, generation=5)
    node._audio_chunks.put((4, b"stale"))
    node._audio_chunks.put((5, b"current"))

    node._tick()

    assert pipeline.feed_calls == [b"current"]


def test_normal_wake_requests_wave_without_touching_interrupt_path(
    audio_input_module: ModuleType,
) -> None:
    pipeline = _WakeEventPipeline(generation=5)
    node = _make_node(audio_input_module, pipeline, generation=5)
    node._audio_chunks.put((5, b"wake pcm"))

    node._tick()

    assert len(node._arm_motion_request_publisher.messages) == 1
    motion = node._arm_motion_request_publisher.messages[0]
    assert motion.command == "play_motion"
    assert motion.purpose == "wake_wave"
    assert motion.generation == 5
    assert motion.request_id.startswith("wake-")
    assert node._interrupt_publisher.messages == []


def test_busy_interrupt_does_not_request_another_wave(
    audio_input_module: ModuleType,
) -> None:
    pipeline = _InterruptEventPipeline(generation=6, state="answering")
    node = _make_node(audio_input_module, pipeline, generation=6)
    node._active_request_id = "answer"
    node._audio_chunks.put((6, b"interrupt pcm"))

    node._tick()

    assert node._arm_motion_request_publisher.messages == []
    assert len(node._interrupt_publisher.messages) == 1
    assert node._interrupt_publisher.messages[0].command == "interrupt"


def test_stale_playback_generation_cannot_leave_interrupt_ack(
    audio_input_module: ModuleType,
) -> None:
    pipeline = _make_pipeline(audio_input_module)
    assert pipeline.begin_answering(pipeline.generation)
    interrupt = pipeline.interrupt(detail="wake during answer")
    node = _make_node(
        audio_input_module,
        pipeline,
        generation=interrupt.generation,
    )
    node._interrupt_request_id = "cancelled-answer"
    node._interrupt_generation = interrupt.generation
    stale_generation = interrupt.generation - 1

    node._on_playback(
        _playback(
            audio_input_module,
            request_id="cancelled-answer",
            generation=stale_generation,
            purpose="answer",
            state="completed",
        )
    )

    assert pipeline.generation == interrupt.generation
    assert pipeline.state == "interrupt_ack"
    assert node._current_epoch() == interrupt.generation


def test_interrupt_flushes_audio_queue_and_invalidates_active_turn(
    audio_input_module: ModuleType,
) -> None:
    pipeline = _make_pipeline(audio_input_module)
    old_generation = pipeline.generation
    assert pipeline.begin_answering(old_generation)
    interrupt = pipeline.interrupt(keyword="wake", detail="interrupt answer")
    node = _make_node(audio_input_module, pipeline, generation=old_generation)
    node._pending_request_id = "pending"
    node._pending_generation = old_generation
    node._active_request_id = "active"
    node._audio_chunks.put((old_generation, b"old pcm"))
    node._audio_chunks.put((interrupt.generation, b"queued before flush"))

    node._handle_interrupt(interrupt)

    assert node._audio_chunks.empty()
    assert node._current_epoch() == interrupt.generation
    assert node._active_request_id == ""
    assert node._pending_request_id == ""
    assert node._pending_generation == 0
    assert node._interrupt_request_id == "active"
    assert node._interrupt_generation == interrupt.generation
    control = node._interrupt_publisher.messages[0]
    assert control.request_id == "active"
    assert control.generation == interrupt.generation
    assert control.purpose == "interrupt"


def test_matching_interrupt_ack_completion_resumes_listening(
    audio_input_module: ModuleType,
) -> None:
    pipeline = _make_pipeline(audio_input_module)
    old_generation = pipeline.generation
    assert pipeline.begin_answering(old_generation)
    interrupt = pipeline.interrupt(keyword="wake", detail="interrupt answer")
    node = _make_node(audio_input_module, pipeline, generation=old_generation)
    node._active_request_id = "answer"
    node._handle_interrupt(interrupt)
    node._audio_chunks.put((interrupt.generation, b"speaker bleed"))

    node._on_playback(
        _playback(
            audio_input_module,
            request_id="ack",
            generation=interrupt.generation,
            purpose="interrupt_ack",
            state="started",
        )
    )
    assert pipeline.state == "interrupt_ack"

    node._on_playback(
        _playback(
            audio_input_module,
            request_id="ack",
            generation=interrupt.generation,
            purpose="interrupt_ack",
            state="completed",
        )
    )

    assert pipeline.state == "listening"
    assert pipeline.generation > interrupt.generation
    assert node._current_epoch() == pipeline.generation
    assert node._audio_chunks.empty()
    assert node._interrupt_request_id == ""
    assert node._interrupt_generation == 0


def test_interrupt_ack_without_matching_generation_stays_locked(
    audio_input_module: ModuleType,
) -> None:
    pipeline = _make_pipeline(audio_input_module)
    assert pipeline.begin_answering(pipeline.generation)
    interrupt = pipeline.interrupt(keyword="wake", detail="interrupt answer")
    node = _make_node(
        audio_input_module,
        pipeline,
        generation=interrupt.generation,
    )
    node._interrupt_generation = interrupt.generation
    node._interrupt_request_id = "ack"

    node._on_playback(
        _playback(
            audio_input_module,
            request_id="ack",
            generation=0,
            purpose="interrupt_ack",
            state="completed",
        )
    )

    assert pipeline.state == "interrupt_ack"
    assert pipeline.generation == interrupt.generation


def test_interrupt_ack_terminal_requires_matching_started_token(
    audio_input_module: ModuleType,
) -> None:
    pipeline = _make_pipeline(audio_input_module)
    assert pipeline.begin_answering(pipeline.generation)
    interrupt = pipeline.interrupt(keyword="wake", detail="interrupt answer")
    node = _make_node(audio_input_module, pipeline, generation=interrupt.generation)
    node._interrupt_generation = interrupt.generation

    # Cross-topic ordering must not let a terminal packet open VAD before the
    # bridge has established the acknowledgement request token.
    node._on_playback(
        _playback(
            audio_input_module,
            request_id="ack-a",
            generation=interrupt.generation,
            purpose="interrupt_ack",
            state="completed",
        )
    )
    assert pipeline.state == "interrupt_ack"

    node._on_playback(
        _playback(
            audio_input_module,
            request_id="ack-a",
            generation=interrupt.generation,
            purpose="interrupt_ack",
            state="started",
        )
    )
    node._on_playback(
        _playback(
            audio_input_module,
            request_id="ack-b",
            generation=interrupt.generation,
            purpose="interrupt_ack",
            state="completed",
        )
    )
    assert pipeline.state == "interrupt_ack"

    node._on_playback(
        _playback(
            audio_input_module,
            request_id="ack-a",
            generation=interrupt.generation,
            purpose="interrupt_ack",
            state="completed",
        )
    )
    assert pipeline.state == "listening"


def test_normal_playback_requires_nonzero_exact_generation_and_request(
    audio_input_module: ModuleType,
) -> None:
    pipeline = _make_pipeline(audio_input_module)
    generation = pipeline.generation
    assert pipeline.begin_answering(generation)
    node = _make_node(audio_input_module, pipeline, generation=generation)
    node._active_request_id = "answer"
    node._pending_request_id = "answer"

    for request_id, candidate_generation in (
        ("answer", 0),
        ("answer", generation + 1),
        ("wrong", generation),
        ("", generation),
    ):
        node._on_playback(
            _playback(
                audio_input_module,
                request_id=request_id,
                generation=candidate_generation,
                purpose="answer",
                state="completed",
            )
        )
        assert pipeline.state == "answering"

    node._on_playback(
        _playback(
            audio_input_module,
            request_id="answer",
            generation=generation,
            purpose="answer",
            state="completed",
        )
    )
    assert pipeline.state == "armed"
