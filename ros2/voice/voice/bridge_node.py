from __future__ import annotations

import time
import uuid
from threading import RLock
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from rag_interfaces.msg import VoiceAnswer, VoiceControl, VoicePlayback, VoiceQuery
from wheeled_robot_msgs.msg import TextToAudio

from voice.runtime import nonempty_session_id, normalize_top_k


class RobotSpeechBridgeNode(Node):
    """Bridge the vendor speech node to typed RAG query and answer topics."""

    def __init__(self) -> None:
        super().__init__("voice_bridge_node")
        self.declare_parameter("transcript_topic", "/speech_interaction/isr_result")
        self.declare_parameter("query_topic", "/voice/query")
        self.declare_parameter("answer_topic", "/voice/answer")
        self.declare_parameter("interrupt_topic", "/voice/interrupt")
        self.declare_parameter("control_topic", "/voice/control")
        self.declare_parameter(
            "arm_motion_request_topic", "/voice/arm_motion_request"
        )
        self.declare_parameter("text_to_audio_topic", "/speech_interaction/text_to_audio")
        self.declare_parameter("display_topic", "/voice/display_text")
        self.declare_parameter("playback_topic", "/voice/playback")
        self.declare_parameter("session_id", "robot")
        self.declare_parameter("top_k", 4)
        self.declare_parameter("speak_answers", True)
        self.declare_parameter("dedup_seconds", 1.0)
        self.declare_parameter("interrupt_ack_timeout_seconds", 8.0)
        self.declare_parameter("playback_timeout_seconds", 30.0)

        self._session_id = nonempty_session_id(
            self.get_parameter("session_id").value
        )
        self._top_k = normalize_top_k(int(self.get_parameter("top_k").value))
        self._speak_answers = bool(self.get_parameter("speak_answers").value)
        self._dedup_seconds = max(0.0, float(self.get_parameter("dedup_seconds").value))
        self._interrupt_ack_timeout_seconds = max(
            1.0,
            float(self.get_parameter("interrupt_ack_timeout_seconds").value),
        )
        self._playback_timeout_seconds = max(
            1.0,
            float(self.get_parameter("playback_timeout_seconds").value),
        )
        self._last_text = ""
        self._last_text_at = 0.0
        self._state_lock = RLock()
        self._generation = 0
        self._awaiting_next_generation = False
        self._active_request_id = ""
        self._active_generation = 0
        self._cancelled_request_ids: set[str] = set()
        self._completed_request_ids: set[str] = set()
        self._handled_interrupt_generations: set[int] = set()
        self._ack_request_id = ""
        self._ack_generation = 0
        self._ack_deadline = 0.0
        self._input_locked = False
        self._last_answer_request_id = ""
        self._last_answer_generation = 0
        self._answer_playback_deadline = 0.0

        transcript_topic = str(self.get_parameter("transcript_topic").value)
        query_topic = str(self.get_parameter("query_topic").value)
        answer_topic = str(self.get_parameter("answer_topic").value)
        interrupt_topic = str(self.get_parameter("interrupt_topic").value)
        control_topic = str(self.get_parameter("control_topic").value)
        arm_motion_request_topic = str(
            self.get_parameter("arm_motion_request_topic").value
        )
        tts_topic = str(self.get_parameter("text_to_audio_topic").value)
        display_topic = str(self.get_parameter("display_topic").value)
        playback_topic = str(self.get_parameter("playback_topic").value)
        self._query_publisher = self.create_publisher(VoiceQuery, query_topic, 20)
        self._control_publisher = self.create_publisher(VoiceControl, control_topic, 20)
        self._arm_motion_request_publisher = self.create_publisher(
            VoiceControl,
            arm_motion_request_topic,
            20,
        )
        self._tts_publisher = self.create_publisher(TextToAudio, tts_topic, 10)
        self._display_publisher = self.create_publisher(String, display_topic, 20)
        self._playback_publisher = self.create_publisher(VoicePlayback, playback_topic, 20)
        self.create_subscription(String, transcript_topic, self._on_transcript, 20)
        self.create_subscription(VoiceQuery, query_topic, self._on_query, 20)
        self.create_subscription(VoiceAnswer, answer_topic, self._on_answer, 20)
        self.create_subscription(VoicePlayback, playback_topic, self._on_playback, 20)
        self.create_subscription(VoiceControl, interrupt_topic, self._on_interrupt, 20)
        if control_topic != interrupt_topic:
            # Compatibility with the first local-audio release, which used
            # /voice/control for the interrupt packet.
            self.create_subscription(VoiceControl, control_topic, self._on_interrupt, 20)
        self.create_timer(0.1, self._check_interrupt_ack_timeout)

        self.get_logger().info(
            f"Robot speech bridge ready: transcript={transcript_topic} query={query_topic} "
            f"answer={answer_topic} interrupt={interrupt_topic} tts={tts_topic}"
        )

    def _on_transcript(self, message: String) -> None:
        text = str(message.data or "").strip()
        if not text:
            return
        with self._state_lock:
            if self._input_locked:
                self.get_logger().debug("Dropped vendor transcript while voice output is active")
                return
        now = time.monotonic()
        if text == self._last_text and now - self._last_text_at < self._dedup_seconds:
            self.get_logger().debug("Dropped duplicate vendor transcript")
            return
        self._last_text = text
        self._last_text_at = now

        query = VoiceQuery()
        query.header.stamp = self.get_clock().now().to_msg()
        query.request_id = uuid.uuid4().hex
        query.session_id = self._session_id
        query.text = text
        query.top_k = self._top_k
        if not self._track_query(query):
            return
        self._query_publisher.publish(query)
        self._publish_display(f"[query] {text}")

    def _on_query(self, message: VoiceQuery) -> None:
        """Track both vendor-bridge and local-microphone voice requests."""

        self._track_query(message)

    def _track_query(self, message: VoiceQuery) -> bool:
        session_id = str(getattr(message, "session_id", "") or self._session_id).strip()
        if session_id != self._session_id:
            return False
        request_id = str(getattr(message, "request_id", "") or "").strip()
        if not request_id:
            return False
        incoming = _message_generation(message)
        with self._state_lock:
            if self._awaiting_next_generation:
                if incoming and incoming <= self._generation:
                    return False
                self._generation = incoming or (self._generation + 1)
                self._awaiting_next_generation = False
            elif incoming and incoming < self._generation:
                return False
            elif incoming > self._generation:
                self._generation = incoming
            elif not incoming:
                # Vendor transcripts do not carry a generation.  Allocate one
                # before publishing the typed query so every downstream ROS
                # callback can enforce exact generation matching.
                self._generation += 1
            previous = self._active_request_id
            if previous and previous != request_id:
                self._cancelled_request_ids.add(previous)
            self._active_request_id = request_id
            self._active_generation = incoming or self._generation
            self._completed_request_ids.discard(request_id)
            self._input_locked = True
            query_generation = self._active_generation
        _assign_message_field(message, "generation", query_generation)
        return True

    def _on_answer(self, message: VoiceAnswer) -> None:
        if not self._answer_is_current(message):
            self.get_logger().debug(
                f"Dropped stale answer request={str(getattr(message, 'request_id', '') or '')} "
                f"generation={_message_generation(message)}"
            )
            return
        if not message.final:
            if message.text:
                self._publish_display(f"[delta] {message.text}")
            return
        if not message.success:
            self._publish_display(f"[error] {message.error}")
            self._complete_active_request(message)
            with self._state_lock:
                self._input_locked = False
            return
        if str(message.disposition or "") == "ignore":
            self._publish_display("[ignored] no semantic speech")
            self._complete_active_request(message)
            with self._state_lock:
                self._input_locked = False
            return
        answer = str(message.text or "").strip()
        if not answer:
            self._publish_display("[error] empty final answer")
            self._complete_active_request(message)
            with self._state_lock:
                self._input_locked = False
            return
        self._publish_display(f"[answer] {answer}")
        generation = _message_generation(message) or self._active_generation
        if self._speak_answers:
            self._last_answer_request_id = str(message.request_id or "")
            self._last_answer_generation = generation
            estimated_playback_seconds = (len(answer) / 4.0) + 5.0
            self._answer_playback_deadline = time.monotonic() + max(
                self._playback_timeout_seconds,
                estimated_playback_seconds,
            )
            output = TextToAudio()
            output.text = answer
            output.filename = ""
            output.is_play = True
            self._publish_answer_motion_request(message, generation=generation)
            self._tts_publisher.publish(output)
            self._publish_playback(
                message,
                "started",
                "sent to robot TTS",
                generation=generation,
                purpose="answer",
            )
        else:
            self._publish_playback(
                message,
                "completed",
                "speech output disabled",
                generation=generation,
                purpose="answer",
            )
            with self._state_lock:
                self._input_locked = False
        self._complete_active_request(message)

    def _on_interrupt(self, message: VoiceControl) -> None:
        """Stop vendor playback and acknowledge a wake-word interruption."""

        command = str(getattr(message, "command", "") or "").strip().lower()
        if command != "interrupt":
            return
        session_id = str(getattr(message, "session_id", "") or self._session_id).strip()
        if session_id and session_id != self._session_id:
            return
        incoming = _message_generation(message)
        request_id = str(getattr(message, "request_id", "") or "").strip()
        with self._state_lock:
            current = self._generation
            if incoming and incoming < current:
                return
            generation = incoming or (current + 1)
            # A publisher may be connected to both the compatibility control
            # topic and /voice/interrupt.  Do not send two acknowledgements for
            # one generation.
            if generation in self._handled_interrupt_generations:
                return
            self._generation = max(current, generation)
            self._awaiting_next_generation = True
            active = self._active_request_id
            if request_id:
                self._cancelled_request_ids.add(request_id)
            if active:
                self._cancelled_request_ids.add(active)
            # The old answer may already be in vendor playback even though
            # its final answer event has been handled.  Retire that playback
            # token as part of the interruption so a delayed `completed`
            # callback cannot unlock input while the acknowledgement is
            # speaking.
            if self._last_answer_request_id:
                self._cancelled_request_ids.add(self._last_answer_request_id)
                self._last_answer_request_id = ""
                self._last_answer_generation = 0
                self._answer_playback_deadline = 0.0
            self._active_request_id = ""
            self._active_generation = generation
            self._input_locked = True
            self._handled_interrupt_generations.add(generation)
            if len(self._handled_interrupt_generations) > 256:
                self._handled_interrupt_generations = set(
                    sorted(self._handled_interrupt_generations)[-128:]
                )
        # The acknowledgement is a new playback request.  It intentionally
        # does not reuse the cancelled answer request id, so late answer
        # callbacks can never be mistaken for the prompt lifecycle.
        ack_request_id = f"interrupt-ack-{uuid.uuid4().hex}"
        self._publish_display("[interrupt] 我在")

        # The vendor adapter uses an empty, non-playing frame as its stop
        # command.  It must be published before the acknowledgement speech so
        # an old answer cannot overlap the new "我在" prompt.
        stop = TextToAudio()
        stop.text = ""
        stop.filename = ""
        stop.is_play = False
        self._tts_publisher.publish(stop)

        acknowledgement = TextToAudio()
        acknowledgement.text = "我在"
        acknowledgement.filename = ""
        acknowledgement.is_play = True
        self._tts_publisher.publish(acknowledgement)

        playback = VoicePlayback()
        playback.header.stamp = self.get_clock().now().to_msg()
        playback.request_id = ack_request_id
        playback.session_id = self._session_id
        _assign_message_field(playback, "generation", generation)
        _assign_message_field(playback, "purpose", "interrupt_ack")
        playback.state = "started"
        playback.detail = "interruption acknowledgement sent to robot TTS"
        with self._state_lock:
            self._ack_request_id = ack_request_id
            self._ack_generation = generation
            self._ack_deadline = time.monotonic() + self._interrupt_ack_timeout_seconds
        self._playback_publisher.publish(playback)

    def _on_playback_feedback(self, message: VoicePlayback) -> None:
        """Accept a vendor completion for the interruption acknowledgement."""
        purpose = str(getattr(message, "purpose", "") or "").strip().lower()
        state = str(getattr(message, "state", "") or "").strip().lower()
        feedback_request_id = str(getattr(message, "request_id", "") or "").strip()
        if purpose == "answer" and state in {"completed", "failed", "interrupted"}:
            session_id = str(getattr(message, "session_id", "") or self._session_id).strip()
            generation = _message_generation(message)
            if session_id == self._session_id:
                with self._state_lock:
                    if (
                        not self._last_answer_request_id
                        or feedback_request_id != self._last_answer_request_id
                        or not generation
                        or generation != self._last_answer_generation
                    ):
                        return
                    self._input_locked = False
                    self._last_answer_request_id = ""
                    self._last_answer_generation = 0
                    self._answer_playback_deadline = 0.0
            return
        if (
            not purpose
            and feedback_request_id
            and feedback_request_id == self._last_answer_request_id
            and _message_generation(message) == self._last_answer_generation
            and state in {"completed", "failed", "interrupted"}
        ):
            session_id = str(getattr(message, "session_id", "") or self._session_id).strip()
            if session_id == self._session_id:
                with self._state_lock:
                    self._input_locked = False
                    self._last_answer_request_id = ""
                    self._last_answer_generation = 0
                    self._answer_playback_deadline = 0.0
            return
        if purpose not in {"interrupt", "interrupt_ack", "wakeup_ack", "ack"}:
            return
        if state not in {"completed", "failed", "interrupted"}:
            return
        session_id = str(getattr(message, "session_id", "") or self._session_id).strip()
        if session_id != self._session_id:
            return
        generation = _message_generation(message)
        request_id = feedback_request_id
        with self._state_lock:
            if not self._ack_deadline:
                return
            if self._ack_generation and generation != self._ack_generation:
                return
            if self._ack_request_id and request_id != self._ack_request_id:
                return
            self._ack_deadline = 0.0
            self._ack_request_id = ""
            self._ack_generation = 0
            self._input_locked = False
        self._publish_display(
            "[interrupt] 我在播放完成" if state == "completed" else "[interrupt] 我在播放失败"
        )

    # Keep the conventional callback name available to launch/test adapters
    # that previously referred to the bridge's playback subscription directly.
    def _on_playback(self, message: VoicePlayback) -> None:
        self._on_playback_feedback(message)

    def _check_interrupt_ack_timeout(self) -> None:
        now = time.monotonic()
        with self._state_lock:
            ack_expired = bool(self._ack_deadline and now >= self._ack_deadline)
            answer_expired = bool(
                self._answer_playback_deadline
                and now >= self._answer_playback_deadline
            )
            if not ack_expired and not answer_expired:
                return
            if ack_expired:
                generation = self._ack_generation
                request_id = ""
                purpose = "interrupt_ack_timeout"
                self._ack_deadline = 0.0
                self._ack_request_id = ""
                self._ack_generation = 0
            else:
                generation = self._last_answer_generation
                request_id = self._last_answer_request_id
                purpose = "answer_playback_timeout"
                self._answer_playback_deadline = 0.0
                self._last_answer_request_id = ""
                self._last_answer_generation = 0
            # Some vendor deployments never publish a playback terminal
            # callback.  Once the bounded playback window expires, release the
            # bridge input lock instead of dropping every later transcript.
            self._input_locked = False
        if purpose == "interrupt_ack_timeout":
            self.get_logger().error(
                "Interrupt acknowledgement playback timed out; returning to wake-word mode"
            )
            self._publish_display("[interrupt] 我在播放超时，已返回待唤醒")
            reason = "interrupt acknowledgement playback timeout"
        else:
            self.get_logger().warning(
                "Answer playback completion timed out; accepting the next transcript"
            )
            self._publish_display("[answer] 播放回执超时，已恢复下一轮输入")
            reason = "answer playback completion timeout"
        control = VoiceControl()
        control.header.stamp = self.get_clock().now().to_msg()
        control.request_id = request_id
        control.session_id = self._session_id
        _assign_message_field(control, "generation", generation)
        _assign_message_field(control, "purpose", purpose)
        control.command = "return_to_wakeup"
        control.reason = reason
        self._control_publisher.publish(control)

    def _answer_is_current(self, answer: VoiceAnswer) -> bool:
        session_id = str(getattr(answer, "session_id", "") or self._session_id).strip()
        if session_id != self._session_id:
            return False
        request_id = str(getattr(answer, "request_id", "") or "").strip()
        generation = _message_generation(answer)
        with self._state_lock:
            if request_id and request_id in self._cancelled_request_ids:
                return False
            if request_id and request_id in self._completed_request_ids:
                return False
            if generation and generation < self._generation:
                return False
            if not generation or generation != self._active_generation:
                return False
            # Answers are correlated by request id.  An empty id is accepted
            # only while a request is active (useful for old test doubles).
            if not self._active_request_id:
                return False
            if request_id and request_id != self._active_request_id:
                return False
            return True

    def _complete_active_request(self, answer: VoiceAnswer) -> None:
        request_id = str(getattr(answer, "request_id", "") or "").strip()
        with self._state_lock:
            if request_id and request_id == self._active_request_id:
                self._completed_request_ids.add(request_id)
                self._active_request_id = ""

    def _publish_display(self, text: str) -> None:
        message = String()
        message.data = text
        self._display_publisher.publish(message)

    def _publish_answer_motion_request(
        self,
        answer: VoiceAnswer,
        *,
        generation: int,
    ) -> None:
        disposition = str(getattr(answer, "disposition", "") or "").strip().lower()
        purpose = "closing_wave" if disposition == "closing" else "answer_gesture"
        try:
            message = VoiceControl()
            message.header.stamp = self.get_clock().now().to_msg()
            message.request_id = str(answer.request_id or "")
            message.session_id = str(answer.session_id or self._session_id)
            _assign_message_field(message, "generation", generation)
            _assign_message_field(message, "purpose", purpose)
            message.command = "play_motion"
            message.reason = "final answer is being submitted to robot TTS"
            self._arm_motion_request_publisher.publish(message)
        except Exception as exc:
            # Speech output remains primary if the optional gesture path fails.
            self.get_logger().warning(f"Failed to request answer arm motion: {exc}")

    def _publish_playback(
        self,
        answer: VoiceAnswer,
        state: str,
        detail: str,
        *,
        generation: int = 0,
        purpose: str = "answer",
    ) -> None:
        message = VoicePlayback()
        message.header.stamp = self.get_clock().now().to_msg()
        message.request_id = answer.request_id
        message.session_id = answer.session_id
        _assign_message_field(message, "generation", generation)
        _assign_message_field(message, "purpose", purpose)
        message.state = state
        message.detail = detail
        self._playback_publisher.publish(message)


def _message_generation(message: Any) -> int:
    try:
        return max(0, int(getattr(message, "generation", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _assign_message_field(message: Any, name: str, value: Any) -> None:
    """Set interruption fields when generated ROS messages expose them."""

    if hasattr(message, name):
        setattr(message, name, value)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RobotSpeechBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
