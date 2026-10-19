from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import Byte

from rag_interfaces.msg import VoiceControl
from wheeled_robot_msgs.msg import ArmState

from voice.runtime import nonempty_session_id


_IDLE = "idle"
_AWAITING_MOTION_START = "awaiting_motion_start"
_MOTION_RUNNING = "motion_running"
_AWAITING_RESET_START = "awaiting_reset_start"
_RESET_RUNNING = "reset_running"

# The vendor's wave and speech motions already finish at the preset pose.
_MOTION_CODES_RETURNING_TO_PRESET = frozenset({4, 5})


@dataclass(frozen=True, slots=True)
class _MotionRequest:
    request_id: str
    session_id: str
    generation: int
    purpose: str
    code: int

    @property
    def token(self) -> tuple[str, str, int, str]:
        return (self.request_id, self.session_id, self.generation, self.purpose)


class VoiceArmMotionNode(Node):
    """Serialize voice gestures and recover to preset after failures or cancellation."""

    def __init__(self) -> None:
        super().__init__("voice_arm_motion_node")
        self.declare_parameter("enabled", True)
        self.declare_parameter("motion_request_topic", "/voice/arm_motion_request")
        self.declare_parameter("interrupt_topic", "/voice/interrupt")
        self.declare_parameter("control_topic", "/voice/control")
        self.declare_parameter("arm_motion_topic", "/arm_control/motion")
        self.declare_parameter("arm_state_topic", "/arm_control/arm_state")
        self.declare_parameter("session_id", "robot")
        self.declare_parameter("wake_motion_code", 4)
        self.declare_parameter("answer_motion_code", 5)
        self.declare_parameter("closing_motion_code", 4)
        self.declare_parameter("default_motion_code", 1)
        self.declare_parameter("motion_start_timeout_seconds", 3.0)
        self.declare_parameter("motion_finish_timeout_seconds", 15.0)
        self.declare_parameter("reset_start_timeout_seconds", 3.0)
        self.declare_parameter("reset_finish_timeout_seconds", 10.0)

        self._enabled = bool(self.get_parameter("enabled").value)
        self._session_id = nonempty_session_id(
            self.get_parameter("session_id").value
        )
        self._motion_codes = {
            "wake_wave": _validated_byte_parameter(self, "wake_motion_code"),
            "answer_gesture": _validated_byte_parameter(self, "answer_motion_code"),
            "closing_wave": _validated_byte_parameter(self, "closing_motion_code"),
        }
        self._default_motion_code = _validated_byte_parameter(
            self, "default_motion_code"
        )
        self._motion_start_timeout = _positive_timeout_parameter(
            self, "motion_start_timeout_seconds"
        )
        self._motion_finish_timeout = _positive_timeout_parameter(
            self, "motion_finish_timeout_seconds"
        )
        self._reset_start_timeout = _positive_timeout_parameter(
            self, "reset_start_timeout_seconds"
        )
        self._reset_finish_timeout = _positive_timeout_parameter(
            self, "reset_finish_timeout_seconds"
        )

        self._state_lock = RLock()
        self._phase = _IDLE
        self._deadline = 0.0
        self._active: _MotionRequest | None = None
        self._latest_generation = 0
        self._last_arm_moving: bool | None = None
        self._seen_tokens: set[tuple[str, str, int, str]] = set()
        self._seen_order: deque[tuple[str, str, int, str]] = deque()

        request_topic = str(self.get_parameter("motion_request_topic").value)
        interrupt_topic = str(self.get_parameter("interrupt_topic").value)
        control_topic = str(self.get_parameter("control_topic").value)
        motion_topic = str(self.get_parameter("arm_motion_topic").value)
        state_topic = str(self.get_parameter("arm_state_topic").value)
        self._motion_publisher = self.create_publisher(Byte, motion_topic, 10)
        self.create_subscription(
            VoiceControl,
            request_topic,
            self._on_motion_request,
            20,
        )
        self.create_subscription(VoiceControl, interrupt_topic, self._on_control, 20)
        if control_topic != interrupt_topic:
            self.create_subscription(VoiceControl, control_topic, self._on_control, 20)
        self.create_subscription(ArmState, state_topic, self._on_arm_state, 20)
        self.create_timer(0.1, self._check_timeout)

        self.get_logger().info(
            "Voice arm motion coordinator ready: "
            f"enabled={self._enabled} requests={request_topic} "
            f"interrupt={interrupt_topic} control={control_topic} "
            f"motion={motion_topic} state={state_topic} "
            f"default_code={self._default_motion_code}"
        )

    def _on_motion_request(self, message: VoiceControl) -> None:
        if not self._enabled:
            return
        if str(getattr(message, "command", "") or "").strip() != "play_motion":
            return
        session_id = str(
            getattr(message, "session_id", "") or self._session_id
        ).strip()
        if session_id != self._session_id:
            return
        request_id = str(getattr(message, "request_id", "") or "").strip()
        generation = _message_generation(message)
        purpose = str(getattr(message, "purpose", "") or "").strip().lower()
        code = self._motion_codes.get(purpose)
        if not request_id or not generation or code is None:
            self.get_logger().warning(
                "Ignored invalid arm motion request: "
                f"request={request_id or '<empty>'} generation={generation} "
                f"purpose={purpose or '<empty>'}"
            )
            return

        request = _MotionRequest(
            request_id=request_id,
            session_id=session_id,
            generation=generation,
            purpose=purpose,
            code=code,
        )
        with self._state_lock:
            if request.token in self._seen_tokens:
                return
            self._remember_token(request.token)
            if generation < self._latest_generation:
                return
            self._latest_generation = generation
            if self._phase != _IDLE:
                self.get_logger().warning(
                    f"Skipped arm motion {purpose}: coordinator is {self._phase}"
                )
                return
            if self._last_arm_moving is True:
                self.get_logger().warning(
                    f"Skipped arm motion {purpose}: an arm is already moving"
                )
                return
            self._active = request
            self._phase = _AWAITING_MOTION_START
            self._deadline = time.monotonic() + self._motion_start_timeout

        if not self._publish_code(code, f"{purpose} request={request_id}"):
            with self._state_lock:
                if self._active == request:
                    self._clear_active_locked()

    def _on_control(self, message: VoiceControl) -> None:
        """Return this node's active gesture to preset on a cancelled voice turn."""

        session_id = str(
            getattr(message, "session_id", "") or self._session_id
        ).strip()
        if session_id != self._session_id:
            return
        command = str(getattr(message, "command", "") or "").strip().lower()
        if command not in {"interrupt", "return_to_wakeup"}:
            return
        generation = _message_generation(message)
        with self._state_lock:
            active = self._active
            if active is None:
                self._latest_generation = max(self._latest_generation, generation)
                return
            if generation and generation < active.generation:
                return
            if command == "return_to_wakeup":
                request_id = str(getattr(message, "request_id", "") or "").strip()
                # A timeout/error control can carry the answer request id even
                # when the active gesture was started by the preceding wake
                # event. A same-or-newer generation is sufficient to retire
                # that gesture; stale generations remain rejected above.
                if (
                    request_id
                    and request_id != active.request_id
                    and not (generation and generation >= active.generation)
                ):
                    return
        self._begin_reset(f"voice {command}")

    def _on_arm_state(self, message: ArmState) -> None:
        moving = bool(
            getattr(message, "left_arm_moving", False)
            or getattr(message, "right_arm_moving", False)
        )
        begin_reset = False
        with self._state_lock:
            self._last_arm_moving = moving
            if self._phase == _AWAITING_MOTION_START and moving:
                self._phase = _MOTION_RUNNING
                self._deadline = time.monotonic() + self._motion_finish_timeout
            elif self._phase == _MOTION_RUNNING and not moving:
                active = self._active
                if (
                    active is not None
                    and active.code in _MOTION_CODES_RETURNING_TO_PRESET
                ):
                    self._clear_active_locked()
                else:
                    begin_reset = True
            elif self._phase == _AWAITING_RESET_START and moving:
                self._phase = _RESET_RUNNING
                self._deadline = time.monotonic() + self._reset_finish_timeout
            elif self._phase == _RESET_RUNNING and not moving:
                self._clear_active_locked()
        if begin_reset:
            self._begin_reset("motion completed")

    def _check_timeout(self) -> None:
        now = time.monotonic()
        with self._state_lock:
            if not self._deadline or now < self._deadline:
                return
            phase = self._phase
            purpose = self._active.purpose if self._active is not None else "unknown"

        if phase == _AWAITING_MOTION_START:
            self.get_logger().warning(
                f"Arm motion {purpose} did not report moving; requesting default preset"
            )
            self._begin_reset("motion start timeout")
        elif phase == _MOTION_RUNNING:
            self.get_logger().warning(
                f"Arm motion {purpose} did not finish before timeout; "
                "requesting default preset"
            )
            self._begin_reset("motion finish timeout")
        elif phase in {_AWAITING_RESET_START, _RESET_RUNNING}:
            self.get_logger().warning(
                "Default arm preset did not report a complete moving cycle; "
                "releasing the coordinator"
            )
            with self._state_lock:
                if self._phase == phase:
                    self._clear_active_locked()

    def _begin_reset(self, reason: str) -> None:
        with self._state_lock:
            if self._phase in {_AWAITING_RESET_START, _RESET_RUNNING, _IDLE}:
                return
            active = self._active
            self._phase = _AWAITING_RESET_START
            self._deadline = time.monotonic() + self._reset_start_timeout
        request_id = active.request_id if active is not None else "unknown"
        if not self._publish_code(
            self._default_motion_code,
            f"default preset after {reason} request={request_id}",
        ):
            with self._state_lock:
                if self._active == active:
                    self._clear_active_locked()

    def _publish_code(self, code: int, detail: str) -> bool:
        try:
            message = Byte()
            # ROS2 Humble's std_msgs/Byte field is one byte, not a Python int.
            message.data = bytes([code])
            self._motion_publisher.publish(message)
        except Exception as exc:
            self.get_logger().error(
                f"Failed to publish arm motion code {code} ({detail}): {exc}"
            )
            return False
        self.get_logger().info(f"Published arm motion code {code}: {detail}")
        return True

    def _remember_token(self, token: tuple[str, str, int, str]) -> None:
        self._seen_tokens.add(token)
        self._seen_order.append(token)
        while len(self._seen_order) > 256:
            expired = self._seen_order.popleft()
            self._seen_tokens.discard(expired)

    def _clear_active_locked(self) -> None:
        self._phase = _IDLE
        self._deadline = 0.0
        self._active = None


def _validated_byte_parameter(node: Node, name: str) -> int:
    value = int(node.get_parameter(name).value)
    if not 0 <= value <= 255:
        raise ValueError(f"{name} must be between 0 and 255")
    return value


def _positive_timeout_parameter(node: Node, name: str) -> float:
    value = float(node.get_parameter(name).value)
    if value <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _message_generation(message: Any) -> int:
    try:
        return max(0, int(getattr(message, "generation", 0) or 0))
    except (TypeError, ValueError):
        return 0


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VoiceArmMotionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
