from __future__ import annotations

import importlib.util
import sys
from collections import deque
from pathlib import Path
from threading import RLock
from types import ModuleType, SimpleNamespace
from typing import Any, Iterator

import pytest


ROOT = Path(__file__).resolve().parents[1]
VOICE = ROOT / "ros2" / "voice"


class _Byte:
    def __init__(self) -> None:
        self._data = b"\x00"

    @property
    def data(self) -> bytes:
        return self._data

    @data.setter
    def data(self, value: bytes) -> None:
        if not isinstance(value, bytes) or len(value) != 1:
            raise TypeError("std_msgs/Byte.data requires exactly one byte")
        self._data = value


class _ArmState:
    def __init__(self, *, left: bool = False, right: bool = False) -> None:
        self.left_arm_moving = left
        self.right_arm_moving = right


class _VoiceControl:
    def __init__(self) -> None:
        self.header = SimpleNamespace(stamp=None)
        self.request_id = ""
        self.session_id = ""
        self.generation = 0
        self.purpose = ""
        self.command = ""
        self.reason = ""


class _Publisher:
    def __init__(self, *, fail: bool = False) -> None:
        self.messages: list[Any] = []
        self.fail = fail

    def publish(self, message: Any) -> None:
        if self.fail:
            raise RuntimeError("publisher unavailable")
        self.messages.append(message)


class _Logger:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def info(self, message: str) -> None:
        self.infos.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)


def _module(name: str, **attributes: Any) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


@pytest.fixture
def arm_motion_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.syspath_prepend(str(VOICE))
    node_module = _module("rclpy.node", Node=object)
    monkeypatch.setitem(sys.modules, "rclpy", _module("rclpy", node=node_module))
    monkeypatch.setitem(sys.modules, "rclpy.node", node_module)

    std_messages = _module("std_msgs.msg", Byte=_Byte)
    monkeypatch.setitem(sys.modules, "std_msgs", _module("std_msgs", msg=std_messages))
    monkeypatch.setitem(sys.modules, "std_msgs.msg", std_messages)
    voice_messages = _module("rag_interfaces.msg", VoiceControl=_VoiceControl)
    monkeypatch.setitem(
        sys.modules,
        "rag_interfaces",
        _module("rag_interfaces", msg=voice_messages),
    )
    monkeypatch.setitem(sys.modules, "rag_interfaces.msg", voice_messages)
    robot_messages = _module("wheeled_robot_msgs.msg", ArmState=_ArmState)
    monkeypatch.setitem(
        sys.modules,
        "wheeled_robot_msgs",
        _module("wheeled_robot_msgs", msg=robot_messages),
    )
    monkeypatch.setitem(sys.modules, "wheeled_robot_msgs.msg", robot_messages)

    module_name = "_test_voice_arm_motion_node"
    spec = importlib.util.spec_from_file_location(
        module_name,
        VOICE / "voice" / "arm_motion_node.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    yield module


def _make_node(module: ModuleType, *, publisher: _Publisher | None = None) -> Any:
    node = module.VoiceArmMotionNode.__new__(module.VoiceArmMotionNode)
    node._enabled = True
    node._session_id = "robot"
    node._motion_codes = {
        "wake_wave": 4,
        "answer_gesture": 5,
        "closing_wave": 4,
    }
    node._default_motion_code = 1
    node._motion_start_timeout = 3.0
    node._motion_finish_timeout = 15.0
    node._reset_start_timeout = 3.0
    node._reset_finish_timeout = 10.0
    node._state_lock = RLock()
    node._phase = module._IDLE
    node._deadline = 0.0
    node._active = None
    node._latest_generation = 0
    node._last_arm_moving = None
    node._seen_tokens = set()
    node._seen_order = deque()
    node._motion_publisher = publisher or _Publisher()
    logger = _Logger()
    node.get_logger = lambda: logger
    node._test_logger = logger
    return node


def _request(
    *,
    purpose: str,
    generation: int = 7,
    request_id: str = "request-1",
) -> _VoiceControl:
    message = _VoiceControl()
    message.request_id = request_id
    message.session_id = "robot"
    message.generation = generation
    message.purpose = purpose
    message.command = "play_motion"
    return message


def _codes(node: Any) -> list[int]:
    return [message.data[0] for message in node._motion_publisher.messages]


@pytest.mark.parametrize(
    ("purpose", "expected_code"),
    [
        ("wake_wave", 4),
        ("answer_gesture", 5),
        ("closing_wave", 4),
    ],
)
def test_sdk_motion_codes_are_published_as_one_byte(
    arm_motion_module: ModuleType,
    purpose: str,
    expected_code: int,
) -> None:
    node = _make_node(arm_motion_module)

    node._on_motion_request(_request(purpose=purpose))

    assert _codes(node) == [expected_code]
    assert node._motion_publisher.messages[0].data == bytes([expected_code])


@pytest.mark.parametrize(
    ("purpose", "expected_code"),
    [
        ("wake_wave", 4),
        ("answer_gesture", 5),
        ("closing_wave", 4),
    ],
)
def test_completed_motion_does_not_publish_redundant_preset(
    arm_motion_module: ModuleType,
    purpose: str,
    expected_code: int,
) -> None:
    node = _make_node(arm_motion_module)
    node._on_motion_request(_request(purpose=purpose))

    node._on_arm_state(_ArmState())
    assert _codes(node) == [expected_code]
    assert node._phase == arm_motion_module._AWAITING_MOTION_START

    node._on_arm_state(_ArmState(left=True))
    assert node._phase == arm_motion_module._MOTION_RUNNING
    node._on_arm_state(_ArmState())

    assert _codes(node) == [expected_code]
    assert node._phase == arm_motion_module._IDLE
    assert node._active is None


def test_other_motion_codes_keep_completion_preset_recovery(
    arm_motion_module: ModuleType,
) -> None:
    node = _make_node(arm_motion_module)
    node._motion_codes["wake_wave"] = 2
    node._on_motion_request(_request(purpose="wake_wave"))
    node._on_arm_state(_ArmState(left=True))
    node._on_arm_state(_ArmState())

    assert _codes(node) == [2, 1]
    assert node._phase == arm_motion_module._AWAITING_RESET_START


def test_unknown_state_is_allowed_but_busy_or_active_requests_are_skipped(
    arm_motion_module: ModuleType,
) -> None:
    node = _make_node(arm_motion_module)
    node._on_motion_request(_request(purpose="answer_gesture"))
    node._on_motion_request(
        _request(purpose="closing_wave", request_id="request-2")
    )
    assert _codes(node) == [5]

    busy = _make_node(arm_motion_module)
    busy._on_arm_state(_ArmState(left=True))
    busy._on_motion_request(_request(purpose="wake_wave"))
    assert _codes(busy) == []


def test_motion_and_reset_timeouts_publish_default_only_once(
    arm_motion_module: ModuleType,
) -> None:
    node = _make_node(arm_motion_module)
    node._on_motion_request(_request(purpose="answer_gesture"))
    node._deadline = arm_motion_module.time.monotonic() - 1.0

    node._check_timeout()
    assert _codes(node) == [5, 1]
    assert node._phase == arm_motion_module._AWAITING_RESET_START

    node._deadline = arm_motion_module.time.monotonic() - 1.0
    node._check_timeout()
    node._check_timeout()
    assert _codes(node) == [5, 1]
    assert node._phase == arm_motion_module._IDLE


def test_running_motion_timeout_still_requests_default_preset(
    arm_motion_module: ModuleType,
) -> None:
    node = _make_node(arm_motion_module)
    node._on_motion_request(_request(purpose="answer_gesture"))
    node._on_arm_state(_ArmState(left=True))
    node._deadline = arm_motion_module.time.monotonic() - 1.0

    node._check_timeout()

    assert _codes(node) == [5, 1]
    assert node._phase == arm_motion_module._AWAITING_RESET_START


def test_interrupt_returns_an_active_gesture_to_preset(
    arm_motion_module: ModuleType,
) -> None:
    node = _make_node(arm_motion_module)
    node._on_motion_request(_request(purpose="answer_gesture", generation=7))

    interrupt = _VoiceControl()
    interrupt.session_id = "robot"
    interrupt.generation = 8
    interrupt.command = "interrupt"
    node._on_control(interrupt)

    assert _codes(node) == [5, 1]
    assert node._phase == arm_motion_module._AWAITING_RESET_START


def test_newer_timeout_control_can_reset_a_wake_gesture_with_another_request_id(
    arm_motion_module: ModuleType,
) -> None:
    node = _make_node(arm_motion_module)
    node._on_motion_request(_request(purpose="wake_wave", generation=7))

    timeout = _VoiceControl()
    timeout.session_id = "robot"
    timeout.generation = 8
    timeout.request_id = "answer-request"
    timeout.command = "return_to_wakeup"
    node._on_control(timeout)

    assert _codes(node) == [4, 1]
    assert node._phase == arm_motion_module._AWAITING_RESET_START


def test_duplicate_stale_and_invalid_requests_are_ignored(
    arm_motion_module: ModuleType,
) -> None:
    node = _make_node(arm_motion_module)
    first = _request(purpose="wake_wave", generation=9)
    node._on_motion_request(first)
    node._on_motion_request(first)
    assert _codes(node) == [4]

    node._phase = arm_motion_module._IDLE
    node._active = None
    node._deadline = 0.0
    node._on_motion_request(
        _request(purpose="answer_gesture", generation=8, request_id="stale")
    )
    node._on_motion_request(
        _request(purpose="unsupported", generation=10, request_id="invalid")
    )
    assert _codes(node) == [4]


def test_publish_failure_releases_coordinator_without_raising(
    arm_motion_module: ModuleType,
) -> None:
    node = _make_node(arm_motion_module, publisher=_Publisher(fail=True))

    node._on_motion_request(_request(purpose="wake_wave"))

    assert node._phase == arm_motion_module._IDLE
    assert node._active is None
    assert node._test_logger.errors
