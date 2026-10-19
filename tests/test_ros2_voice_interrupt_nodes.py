from __future__ import annotations

import importlib.util
import queue
import sys
from pathlib import Path
from threading import RLock
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
VOICE = ROOT / "ros2" / "voice"


class _Header:
    def __init__(self) -> None:
        self.stamp: object | None = None


class _VoiceQuery:
    def __init__(self) -> None:
        self.header = _Header()
        self.request_id = ""
        self.session_id = ""
        self.generation = 0
        self.text = ""
        self.top_k = 0


class _VoiceAnswer:
    def __init__(self) -> None:
        self.header = _Header()
        self.request_id = ""
        self.session_id = ""
        self.generation = 0
        self.text = ""
        self.final = False
        self.success = False
        self.error = ""
        self.model = ""
        self.source_names: list[str] = []
        self.disposition = "answer"


class _VoiceControl:
    def __init__(self) -> None:
        self.header = _Header()
        self.request_id = ""
        self.session_id = ""
        self.generation = 0
        self.purpose = ""
        self.command = ""
        self.reason = ""


class _VoicePlayback:
    def __init__(self) -> None:
        self.header = _Header()
        self.request_id = ""
        self.session_id = ""
        self.generation = 0
        self.purpose = ""
        self.state = ""
        self.detail = ""


class _String:
    def __init__(self) -> None:
        self.data = ""


class _TextToAudio:
    def __init__(self) -> None:
        self.text = ""
        self.filename = ""
        self.is_play = True


class _Ask:
    class Request:
        pass

    class Response:
        pass


class _Publisher:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    def publish(self, message: Any) -> None:
        self.messages.append(message)


class _Logger:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def debug(self, _message: str) -> None:
        pass

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


class _Clock:
    def now(self) -> SimpleNamespace:
        return SimpleNamespace(to_msg=lambda: object())


def _module(name: str, **attributes: Any) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


@pytest.fixture
def ros_node_modules(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Load the ROS wrappers with only their small message/runtime boundary stubbed."""

    monkeypatch.syspath_prepend(str(VOICE))
    node_module = _module("rclpy.node", Node=object)
    monkeypatch.setitem(sys.modules, "rclpy", _module("rclpy", node=node_module))
    monkeypatch.setitem(sys.modules, "rclpy.node", node_module)

    messages = _module(
        "rag_interfaces.msg",
        VoiceQuery=_VoiceQuery,
        VoiceAnswer=_VoiceAnswer,
        VoiceControl=_VoiceControl,
        VoicePlayback=_VoicePlayback,
    )
    services = _module("rag_interfaces.srv", Ask=_Ask)
    interfaces = _module("rag_interfaces", msg=messages, srv=services)
    monkeypatch.setitem(sys.modules, "rag_interfaces", interfaces)
    monkeypatch.setitem(sys.modules, "rag_interfaces.msg", messages)
    monkeypatch.setitem(sys.modules, "rag_interfaces.srv", services)

    std_messages = _module("std_msgs.msg", String=_String)
    monkeypatch.setitem(sys.modules, "std_msgs", _module("std_msgs", msg=std_messages))
    monkeypatch.setitem(sys.modules, "std_msgs.msg", std_messages)
    vendor_messages = _module("wheeled_robot_msgs.msg", TextToAudio=_TextToAudio)
    monkeypatch.setitem(
        sys.modules,
        "wheeled_robot_msgs",
        _module("wheeled_robot_msgs", msg=vendor_messages),
    )
    monkeypatch.setitem(sys.modules, "wheeled_robot_msgs.msg", vendor_messages)

    monkeypatch.setitem(
        sys.modules,
        "rag_system.generation",
        _module("rag_system.generation", AnswerGenerationService=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "rag_system.settings",
        _module("rag_system.settings", SettingsStore=object),
    )

    def load(stem: str) -> ModuleType:
        module_name = f"_test_ros2_{stem}"
        spec = importlib.util.spec_from_file_location(
            module_name,
            VOICE / "voice" / f"{stem}.py",
        )
        assert spec is not None and spec.loader is not None
        loaded = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, module_name, loaded)
        spec.loader.exec_module(loaded)
        return loaded

    return SimpleNamespace(answer=load("answer_node"), bridge=load("bridge_node"))


def _interrupt(*, request_id: str = "", generation: int) -> _VoiceControl:
    message = _VoiceControl()
    message.request_id = request_id
    message.session_id = "robot"
    message.generation = generation
    message.purpose = "interrupt"
    message.command = "interrupt"
    return message


def _answer_node(module: ModuleType) -> Any:
    node = module.RagAnswerNode.__new__(module.RagAnswerNode)
    node._state_lock = RLock()
    node._generation_lock = RLock()
    node._session_generations = {}
    node._session_needs_next_generation = set()
    node._requests = {}
    node._request_metadata = {}
    node._cancelled_request_ids = set()
    node._events = queue.SimpleQueue()
    node._controls = queue.SimpleQueue()
    node._answer_publisher = _Publisher()
    node._control_publisher = _Publisher()
    node.get_clock = lambda: _Clock()
    node.get_logger = lambda: _Logger()
    return node


def _bridge_node(module: ModuleType) -> Any:
    node = module.RobotSpeechBridgeNode.__new__(module.RobotSpeechBridgeNode)
    node._session_id = "robot"
    node._top_k = 4
    node._speak_answers = True
    node._dedup_seconds = 1.0
    node._state_lock = RLock()
    node._generation = 3
    node._awaiting_next_generation = False
    node._active_request_id = "request-1"
    node._active_generation = 3
    node._cancelled_request_ids = set()
    node._completed_request_ids = set()
    node._handled_interrupt_generations = set()
    node._ack_request_id = ""
    node._ack_generation = 0
    node._ack_deadline = 0.0
    node._input_locked = True
    node._last_answer_request_id = "request-1"
    node._last_answer_generation = 3
    node._answer_playback_deadline = 0.0
    node._interrupt_ack_timeout_seconds = 8.0
    node._playback_timeout_seconds = 30.0
    node._tts_publisher = _Publisher()
    node._arm_motion_request_publisher = _Publisher()
    node._playback_publisher = _Publisher()
    node._control_publisher = _Publisher()
    node._display_publisher = _Publisher()
    node.get_clock = lambda: _Clock()
    node.get_logger = lambda: _Logger()
    return node


def test_answer_interrupt_suppresses_queued_and_late_events(ros_node_modules: Any) -> None:
    module = ros_node_modules.answer
    node = _answer_node(module)

    state = node._register_request("queued", "robot", 3)
    assert state is not None
    node._events.put(
        module._AnswerEvent("queued", "robot", "delta", False, True, generation=3)
    )
    node._events.put(
        module._AnswerEvent("queued", "robot", "final", True, True, generation=3)
    )
    node._events.put(
        module._AnswerEvent(
            "queued",
            "robot",
            "",
            True,
            False,
            error="late error",
            generation=3,
        )
    )
    node._controls.put(
        module._ControlEvent(
            "queued",
            "robot",
            "return_to_wakeup",
            "late error",
            generation=3,
        )
    )
    node._on_interrupt(_interrupt(generation=4))
    assert state.cancelled.is_set()
    node._flush_events()
    assert node._answer_publisher.messages == []
    assert node._control_publisher.messages == []

    class InterruptingService:
        def __init__(self, event: dict[str, Any] | None, generation: int) -> None:
            self.event = event
            self.generation = generation

        def stream_answer_events(self, _text: str, *, top_k: int):
            del top_k
            node._on_interrupt(_interrupt(generation=self.generation))
            if self.event is None:
                raise RuntimeError("provider failed after interrupt")
            yield self.event

    late_events = (
        {"type": "delta", "text": "late delta"},
        {
            "type": "final",
            "result": {"answer": "late final", "citations": [], "model": "test"},
        },
        None,
    )
    for offset, event in enumerate(late_events):
        generation = 5 + (offset * 2)
        request = node._register_request(f"late-{generation}", "robot", generation)
        assert request is not None
        node._service_for = lambda _session, value=event, generation=generation + 1: (
            InterruptingService(value, generation)
        )
        node._generate(request, "question", 4)
        node._flush_events()

    assert node._answer_publisher.messages == []
    assert node._control_publisher.messages == []


def test_answer_interrupt_cancels_service_before_history_mutation(
    ros_node_modules: Any,
) -> None:
    node = _answer_node(ros_node_modules.answer)

    class CooperativeService:
        def __init__(self) -> None:
            self.history: list[tuple[str, str]] = []
            self.should_cancel: Any = None

        def stream_answer_events(
            self,
            text: str,
            *,
            top_k: int,
            should_cancel: Any,
        ):
            del top_k
            self.should_cancel = should_cancel
            yield {"type": "delta", "text": "early delta"}
            node._on_interrupt(_interrupt(request_id="cooperative", generation=4))
            if should_cancel():
                return
            self.history.append((text, "late final"))
            yield {
                "type": "final",
                "result": {
                    "answer": "late final",
                    "citations": [],
                    "model": "test",
                },
            }

    service = CooperativeService()
    node._service_for = lambda _session: service
    state = node._register_request("cooperative", "robot", 3)
    assert state is not None

    node._generate(state, "question", 4)
    node._flush_events()

    assert callable(service.should_cancel)
    assert service.should_cancel()
    assert service.history == []
    assert node._answer_publisher.messages == []
    assert node._control_publisher.messages == []


def test_answer_provider_type_error_is_not_retried_as_legacy_call(
    ros_node_modules: Any,
) -> None:
    node = _answer_node(ros_node_modules.answer)

    class FailingService:
        def __init__(self) -> None:
            self.calls = 0
            self.received_should_cancel: list[Any] = []

        def stream_answer_events(
            self,
            _text: str,
            *,
            top_k: int,
            should_cancel: Any = None,
        ):
            del top_k
            self.calls += 1
            self.received_should_cancel.append(should_cancel)
            if should_cancel is not None:
                raise TypeError("provider implementation failed")
            return iter(())

    service = FailingService()
    node._service_for = lambda _session: service
    state = node._register_request("type-error", "robot", 3)
    assert state is not None

    node._generate(state, "question", 4)
    node._flush_events()

    assert service.calls == 1
    assert len(service.received_should_cancel) == 1
    assert callable(service.received_should_cancel[0])
    assert len(node._answer_publisher.messages) == 1
    assert node._answer_publisher.messages[0].error == "provider implementation failed"


def test_answer_node_preserves_history_within_ros_session_and_isolates_sessions(
    ros_node_modules: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = ros_node_modules.answer
    node = _answer_node(module)
    node._session_services = module.OrderedDict()
    node._max_sessions = 4
    node._settings_path = tmp_path / "settings.json"
    node._index_dir = tmp_path / "index"
    services: list[Any] = []

    class StatefulService:
        def __init__(self, *, settings_store: Any, index_dir: Path) -> None:
            del settings_store, index_dir
            self.history: list[tuple[str, str]] = []
            self.calls: list[tuple[str, tuple[tuple[str, str], ...]]] = []
            services.append(self)

        def stream_answer_events(
            self,
            text: str,
            *,
            top_k: int,
            should_cancel: Any,
        ):
            del top_k
            assert not should_cancel()
            self.calls.append((text, tuple(self.history)))
            answer = f"回答{len(self.history) + 1}"
            yield {"type": "delta", "text": answer}
            self.history.append((text, answer))
            yield {
                "type": "final",
                "result": {"answer": answer, "citations": [], "model": "test"},
            }

    monkeypatch.setattr(module, "SettingsStore", lambda path: path)
    monkeypatch.setattr(module, "AnswerGenerationService", StatefulService)

    first = node._register_request("request-1", "robot", 1)
    assert first is not None
    node._generate(first, "第一轮问题", 4)
    node._flush_events()

    follow_up = node._register_request("request-2", "robot", 2)
    assert follow_up is not None
    node._generate(follow_up, "它呢？", 4)
    node._flush_events()

    other_session = node._register_request("request-3", "visitor", 1)
    assert other_session is not None
    node._generate(other_session, "它呢？", 4)
    node._flush_events()

    assert len(services) == 2
    assert services[0].calls == [
        ("第一轮问题", ()),
        ("它呢？", (("第一轮问题", "回答1"),)),
    ]
    assert services[1].calls == [("它呢？", ())]


def test_interrupt_before_query_drops_stale_generation(ros_node_modules: Any) -> None:
    answer = _answer_node(ros_node_modules.answer)
    answer._session_generations["robot"] = 3
    answer._on_interrupt(_interrupt(generation=4))

    assert answer._register_request("stale", "robot", 3) is None
    assert answer._session_generations["robot"] == 4
    assert "robot" in answer._session_needs_next_generation

    fresh = answer._register_request("fresh", "robot", 5)
    assert fresh is not None
    assert fresh.generation == 5

    bridge = _bridge_node(ros_node_modules.bridge)
    bridge._on_interrupt(_interrupt(request_id="request-1", generation=4))
    stale_query = _VoiceQuery()
    stale_query.request_id = "stale"
    stale_query.session_id = "robot"
    stale_query.generation = 3
    assert bridge._track_query(stale_query) is False
    assert bridge._active_request_id == ""
    assert bridge._awaiting_next_generation is True

    fresh_query = _VoiceQuery()
    fresh_query.request_id = "fresh"
    fresh_query.session_id = "robot"
    fresh_query.generation = 5
    assert bridge._track_query(fresh_query) is True
    assert bridge._active_request_id == "fresh"
    assert bridge._active_generation == 5


def test_bridge_interrupt_waits_for_matching_ack_completion(ros_node_modules: Any) -> None:
    module = ros_node_modules.bridge
    node = _bridge_node(module)

    node._on_interrupt(_interrupt(request_id="request-1", generation=4))

    assert [(item.is_play, item.text) for item in node._tts_publisher.messages] == [
        (False, ""),
        (True, "我在"),
    ]
    assert [item.state for item in node._playback_publisher.messages] == ["started"]
    ack_request_id = node._ack_request_id
    assert ack_request_id.startswith("interrupt-ack-")
    assert node._ack_generation == 4
    assert node._ack_deadline > 0

    # A completion from the cancelled answer must not unlock vendor input
    # while the acknowledgement is still playing.
    late_answer_playback = _VoicePlayback()
    late_answer_playback.request_id = "request-1"
    late_answer_playback.session_id = "robot"
    late_answer_playback.generation = 3
    late_answer_playback.purpose = "answer"
    late_answer_playback.state = "completed"
    node._on_playback_feedback(late_answer_playback)
    assert node._input_locked

    wrong = _VoicePlayback()
    wrong.request_id = "different-request"
    wrong.session_id = "robot"
    wrong.generation = 4
    wrong.purpose = "interrupt_ack"
    wrong.state = "completed"
    node._on_playback_feedback(wrong)
    assert node._ack_request_id == ack_request_id

    wrong.request_id = ack_request_id
    wrong.generation = 5
    node._on_playback_feedback(wrong)
    assert node._ack_request_id == ack_request_id

    matching = _VoicePlayback()
    matching.request_id = ack_request_id
    matching.session_id = "robot"
    matching.generation = 4
    matching.purpose = "interrupt_ack"
    matching.state = "completed"
    node._on_playback_feedback(matching)
    assert node._ack_request_id == ""
    assert node._ack_generation == 0
    assert node._ack_deadline == 0.0
    assert not node._input_locked


def test_bridge_ignores_stale_answer_playback_completion(ros_node_modules: Any) -> None:
    node = _bridge_node(ros_node_modules.bridge)
    node._last_answer_request_id = "new-answer"

    completed = _VoicePlayback()
    completed.request_id = "old-answer"
    completed.session_id = "robot"
    completed.generation = 3
    completed.purpose = "answer"
    completed.state = "completed"
    node._on_playback_feedback(completed)
    assert node._input_locked

    # A matching request id is not enough: terminal playback must belong to
    # the same generation as the answer that started it.
    completed.request_id = "new-answer"
    completed.generation = 4
    node._on_playback_feedback(completed)
    assert node._input_locked

    completed.generation = 3
    node._on_playback_feedback(completed)
    assert not node._input_locked


def test_bridge_rejects_answer_with_zero_or_future_generation(
    ros_node_modules: Any,
) -> None:
    node = _bridge_node(ros_node_modules.bridge)
    answer = _VoiceAnswer()
    answer.request_id = "request-1"
    answer.session_id = "robot"
    answer.text = "回答"
    answer.final = True
    answer.success = True

    answer.generation = 0
    node._on_answer(answer)
    answer.generation = 4
    node._on_answer(answer)
    assert node._tts_publisher.messages == []


@pytest.mark.parametrize(
    ("disposition", "purpose"),
    [
        ("answer", "answer_gesture"),
        ("closing", "closing_wave"),
    ],
)
def test_bridge_requests_gesture_before_submitting_final_answer_to_tts(
    ros_node_modules: Any,
    disposition: str,
    purpose: str,
) -> None:
    node = _bridge_node(ros_node_modules.bridge)
    sequence: list[str] = []

    class OrderedPublisher(_Publisher):
        def __init__(self, label: str) -> None:
            super().__init__()
            self.label = label

        def publish(self, message: Any) -> None:
            sequence.append(self.label)
            super().publish(message)

    node._arm_motion_request_publisher = OrderedPublisher("motion")
    node._tts_publisher = OrderedPublisher("tts")
    answer = _VoiceAnswer()
    answer.request_id = "request-1"
    answer.session_id = "robot"
    answer.generation = 3
    answer.text = "这是最终回答"
    answer.final = True
    answer.success = True
    answer.disposition = disposition

    node._on_answer(answer)

    motion = node._arm_motion_request_publisher.messages[0]
    assert motion.command == "play_motion"
    assert motion.purpose == purpose
    assert motion.request_id == "request-1"
    assert motion.generation == 3
    assert sequence[:2] == ["motion", "tts"]


@pytest.mark.parametrize("disposition", ["ignore", ""])
def test_bridge_does_not_request_gesture_for_ignored_or_empty_answers(
    ros_node_modules: Any,
    disposition: str,
) -> None:
    node = _bridge_node(ros_node_modules.bridge)
    answer = _VoiceAnswer()
    answer.request_id = "request-1"
    answer.session_id = "robot"
    answer.generation = 3
    answer.final = True
    answer.success = True
    answer.disposition = disposition or "answer"
    answer.text = "" if not disposition else "忽略"

    node._on_answer(answer)

    assert node._arm_motion_request_publisher.messages == []
    assert node._tts_publisher.messages == []


def test_bridge_does_not_request_gesture_when_speech_is_disabled(
    ros_node_modules: Any,
) -> None:
    node = _bridge_node(ros_node_modules.bridge)
    node._speak_answers = False
    answer = _VoiceAnswer()
    answer.request_id = "request-1"
    answer.session_id = "robot"
    answer.generation = 3
    answer.final = True
    answer.success = True
    answer.text = "回答"

    node._on_answer(answer)

    assert node._arm_motion_request_publisher.messages == []
    assert node._tts_publisher.messages == []


def test_bridge_assigns_generation_to_vendor_transcript(
    ros_node_modules: Any,
) -> None:
    node = _bridge_node(ros_node_modules.bridge)
    node._input_locked = False
    node._query_publisher = _Publisher()
    node._display_publisher = _Publisher()
    node._last_text = ""
    node._last_text_at = 0.0
    transcript = _String()
    transcript.data = "问题"

    node._on_transcript(transcript)

    assert len(node._query_publisher.messages) == 1
    assert node._query_publisher.messages[0].generation > 0


def test_bridge_ack_timeout_returns_to_wakeup(ros_node_modules: Any) -> None:
    module = ros_node_modules.bridge
    node = _bridge_node(module)
    node._query_publisher = _Publisher()
    node._last_text = ""
    node._last_text_at = 0.0
    node._on_interrupt(_interrupt(request_id="request-1", generation=4))
    node._ack_deadline = module.time.monotonic() - 1.0

    node._check_interrupt_ack_timeout()

    assert [item.state for item in node._playback_publisher.messages] == ["started"]
    assert len(node._control_publisher.messages) == 1
    control = node._control_publisher.messages[0]
    assert control.command == "return_to_wakeup"
    assert control.generation == 4
    assert control.purpose == "interrupt_ack_timeout"
    assert node._ack_deadline == 0.0
    assert node._ack_request_id == ""
    assert node._ack_generation == 0
    assert not node._input_locked

    transcript = _String()
    transcript.data = "超时后的下一轮问题"
    node._on_transcript(transcript)
    assert len(node._query_publisher.messages) == 1
    assert node._query_publisher.messages[0].generation > 4

    node._check_interrupt_ack_timeout()
    assert len(node._control_publisher.messages) == 1


def test_bridge_answer_playback_timeout_releases_next_turn(ros_node_modules: Any) -> None:
    module = ros_node_modules.bridge
    node = _bridge_node(module)
    node._query_publisher = _Publisher()
    node._last_text = ""
    node._last_text_at = 0.0
    node._last_answer_request_id = "answer-1"
    node._last_answer_generation = 3
    node._answer_playback_deadline = module.time.monotonic() - 1.0

    node._check_interrupt_ack_timeout()

    assert not node._input_locked
    assert node._last_answer_request_id == ""
    assert node._answer_playback_deadline == 0.0
    control = node._control_publisher.messages[-1]
    assert control.request_id == "answer-1"
    assert control.generation == 3
    assert control.purpose == "answer_playback_timeout"
    assert control.command == "return_to_wakeup"

    transcript = _String()
    transcript.data = "普通回答超时后的下一轮"
    node._on_transcript(transcript)
    assert len(node._query_publisher.messages) == 1
