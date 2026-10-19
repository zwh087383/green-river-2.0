from __future__ import annotations

import inspect
import queue
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event, RLock
from typing import Any

import rclpy
from rclpy.node import Node

from rag_interfaces.msg import VoiceAnswer, VoiceControl, VoiceQuery
from rag_interfaces.srv import Ask
from rag_system.generation import AnswerGenerationService
from rag_system.settings import SettingsStore

from voice.runtime import default_data_dir, nonempty_session_id, normalize_top_k


@dataclass(frozen=True, slots=True)
class _AnswerEvent:
    request_id: str
    session_id: str
    text: str
    final: bool
    success: bool
    error: str = ""
    model: str = ""
    source_names: tuple[str, ...] = ()
    disposition: str = "answer"
    generation: int = 0


@dataclass(frozen=True, slots=True)
class _ControlEvent:
    request_id: str
    session_id: str
    command: str
    reason: str
    generation: int = 0
    purpose: str = "answer"


@dataclass(slots=True)
class _RequestState:
    """Cancellation token captured by one asynchronous answer request."""

    request_id: str
    session_id: str
    generation: int
    cancelled: Event
    future: Future[Any] | None = None


class RagAnswerNode(Node):
    """Turn typed ROS2 queries into RAG answer events without blocking the executor."""

    def __init__(self) -> None:
        super().__init__("rag_answer_node")
        data_dir = default_data_dir()
        self.declare_parameter("data_dir", str(data_dir))
        self.declare_parameter("settings_path", str(data_dir / "settings.json"))
        self.declare_parameter("index_dir", str(data_dir / "index"))
        self.declare_parameter("query_topic", "/voice/query")
        self.declare_parameter("answer_topic", "/voice/answer")
        self.declare_parameter("control_topic", "/voice/control")
        self.declare_parameter("interrupt_topic", "/voice/interrupt")
        self.declare_parameter("ask_service", "/rag/ask")
        self.declare_parameter("default_top_k", 4)
        self.declare_parameter("worker_count", 2)
        self.declare_parameter("max_sessions", 16)

        self._settings_path = Path(
            self.get_parameter("settings_path").value
        ).expanduser().resolve()
        self._index_dir = Path(self.get_parameter("index_dir").value).expanduser().resolve()
        self._default_top_k = normalize_top_k(
            int(self.get_parameter("default_top_k").value)
        )
        self._max_sessions = max(1, int(self.get_parameter("max_sessions").value))
        workers = max(1, int(self.get_parameter("worker_count").value))
        self._session_services: OrderedDict[str, AnswerGenerationService] = OrderedDict()
        self._generation_lock = RLock()
        self._state_lock = RLock()
        self._session_generations: dict[str, int] = {}
        self._session_needs_next_generation: set[str] = set()
        self._requests: dict[str, _RequestState] = {}
        # Keep lightweight metadata after a worker finishes so an interrupt
        # racing the timer can still invalidate an event already queued for
        # publication.
        self._request_metadata: dict[str, tuple[str, int]] = {}
        self._cancelled_request_ids: set[str] = set()
        self._events: queue.SimpleQueue[_AnswerEvent] = queue.SimpleQueue()
        self._controls: queue.SimpleQueue[_ControlEvent] = queue.SimpleQueue()
        self._workers = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rag-answer")

        query_topic = str(self.get_parameter("query_topic").value)
        answer_topic = str(self.get_parameter("answer_topic").value)
        control_topic = str(self.get_parameter("control_topic").value)
        interrupt_topic = str(self.get_parameter("interrupt_topic").value)
        ask_service = str(self.get_parameter("ask_service").value)
        self._answer_publisher = self.create_publisher(VoiceAnswer, answer_topic, 20)
        self._control_publisher = self.create_publisher(VoiceControl, control_topic, 20)
        self.create_subscription(VoiceQuery, query_topic, self._on_query, 20)
        self.create_subscription(VoiceControl, interrupt_topic, self._on_interrupt, 20)
        # Older audio-input deployments sent the interrupt command on the
        # general control topic.  Keep accepting that form while the dedicated
        # topic rolls out; non-interrupt controls are ignored by the callback.
        if interrupt_topic != control_topic:
            self.create_subscription(VoiceControl, control_topic, self._on_interrupt, 20)
        self.create_service(Ask, ask_service, self._on_ask)
        self.create_timer(0.05, self._flush_events)

        self.get_logger().info(
            f"RAG answer node ready: query={query_topic} answer={answer_topic} index={self._index_dir}"
        )

    def destroy_node(self) -> bool:
        self._workers.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()

    def _on_query(self, message: VoiceQuery) -> None:
        request_id = str(message.request_id or uuid.uuid4().hex)
        session_id = nonempty_session_id(message.session_id)
        text = str(message.text or "").strip()
        if not text:
            self._events.put(
                _AnswerEvent(
                    request_id=request_id,
                    session_id=session_id,
                    text="",
                    final=True,
                    success=False,
                    error="query cannot be empty",
                )
            )
            return
        top_k = normalize_top_k(int(message.top_k), default=self._default_top_k)
        state = self._register_request(request_id, session_id, _message_generation(message))
        if state is None:
            self.get_logger().debug(
                f"Dropped stale query request={request_id} "
                f"generation={_message_generation(message)}"
            )
            return
        self._submit_request(state, text, top_k)

    def _on_ask(self, request: Ask.Request, response: Ask.Response) -> Ask.Response:
        request_id = uuid.uuid4().hex
        session_id = nonempty_session_id(request.session_id)
        text = str(request.query or "").strip()
        if not text:
            response.accepted = False
            response.request_id = request_id
            response.message = "query cannot be empty"
            return response
        top_k = normalize_top_k(int(request.top_k), default=self._default_top_k)
        state = self._register_request(request_id, session_id, 0)
        assert state is not None
        self._submit_request(state, text, top_k)
        response.accepted = True
        response.request_id = request_id
        response.message = "accepted"
        return response

    def _generate(self, state: _RequestState, text: str, top_k: int) -> None:
        request_id = state.request_id
        session_id = state.session_id
        try:
            with self._generation_lock:
                if not self._request_is_current(state):
                    return
                service = self._service_for(session_id)
                self._controls.put(
                    _ControlEvent(
                        request_id=request_id,
                        session_id=session_id,
                        command="lock_listening",
                        reason="正在处理本轮语音请求。",
                        generation=state.generation,
                        purpose="answer",
                    )
                )
                disposition = "answer"
                final_result: dict[str, Any] | None = None
                stream = iter(
                    _stream_answer_events(
                        service,
                        text,
                        top_k=top_k,
                        should_cancel=state.cancelled.is_set,
                    )
                )
                try:
                    for event in stream:
                        # The provider is synchronous and may not be interruptible,
                        # so check the token at every stream boundary.  A late
                        # provider result is then discarded rather than reaching
                        # the bridge or TTS node.
                        if not self._request_is_current(state):
                            return
                        event_type = str(event.get("type") or "")
                        if event_type == "ignored":
                            reason = str(event.get("reason") or "未识别到有效问题。")
                            self._controls.put(
                                _ControlEvent(
                                    request_id=request_id,
                                    session_id=session_id,
                                    command="resume_listening",
                                    reason=reason,
                                    generation=state.generation,
                                    purpose="answer",
                                )
                            )
                            self._events.put(
                                _AnswerEvent(
                                    request_id=request_id,
                                    session_id=session_id,
                                    text="",
                                    final=True,
                                    success=True,
                                    model=str(event.get("model") or ""),
                                    disposition="ignore",
                                    generation=state.generation,
                                )
                            )
                            return
                        if event_type == "meta":
                            retrieval = event.get("retrieval")
                            if (
                                isinstance(retrieval, dict)
                                and retrieval.get("reason") == "end_session"
                            ):
                                disposition = "closing"
                        elif event_type == "delta":
                            delta = str(event.get("text") or "")
                            if delta:
                                self._events.put(
                                    _AnswerEvent(
                                        request_id=request_id,
                                        session_id=session_id,
                                        text=delta,
                                        final=False,
                                        success=True,
                                        disposition=disposition,
                                        generation=state.generation,
                                    )
                                )
                        elif event_type == "final":
                            candidate = event.get("result")
                            if isinstance(candidate, dict):
                                final_result = candidate
                finally:
                    close_stream = getattr(stream, "close", None)
                    if callable(close_stream):
                        try:
                            close_stream()
                        except Exception:
                            pass

            if final_result is None:
                raise RuntimeError("answer provider did not emit a final result")
            if not self._request_is_current(state):
                return
            sources = tuple(
                dict.fromkeys(
                    str(item.get("source_name"))
                    for item in final_result.get("citations", [])
                    if isinstance(item, dict) and item.get("source_name")
                )
            )
            self._events.put(
                _AnswerEvent(
                    request_id=request_id,
                    session_id=session_id,
                    text=str(final_result.get("answer") or ""),
                    final=True,
                    success=True,
                    model=str(final_result.get("model") or ""),
                    source_names=sources,
                    disposition=disposition,
                    generation=state.generation,
                )
            )
        except Exception as exc:
            if not self._request_is_current(state):
                return
            self.get_logger().error(f"RAG request {request_id} failed: {exc}")
            self._controls.put(
                _ControlEvent(
                    request_id=request_id,
                    session_id=session_id,
                    command="return_to_wakeup",
                    reason=str(exc),
                    generation=state.generation,
                    purpose="answer",
                )
            )
            self._events.put(
                _AnswerEvent(
                    request_id=request_id,
                    session_id=session_id,
                    text="",
                    final=True,
                    success=False,
                    error=str(exc),
                    generation=state.generation,
                )
            )
        finally:
            self._finish_request(state)

    def _register_request(
        self, request_id: str, session_id: str, message_generation: int
    ) -> _RequestState | None:
        """Register a request and supersede an older request in the session."""

        with self._state_lock:
            current = self._session_generations.get(session_id, 0)
            needs_next = session_id in self._session_needs_next_generation
            if message_generation and (
                message_generation < current
                or (needs_next and message_generation <= current)
            ):
                return None
            if needs_next and not message_generation:
                current += 1
                self._session_needs_next_generation.discard(session_id)
                self._session_generations[session_id] = current
            if message_generation > current:
                current = int(message_generation)
                self._session_generations[session_id] = current
                self._session_needs_next_generation.discard(session_id)
            # A query without a generation is still a new logical request, but
            # normal ROS deployments use generation=0 as a wildcard because
            # the legacy VoiceQuery message has no such field.
            generation = current
            previous = [
                item
                for item in self._requests.values()
                if item.session_id == session_id and item.request_id != request_id
            ]
            for item in previous:
                item.cancelled.set()
                if item.future is not None:
                    item.future.cancel()
                self._remember_cancelled(item.request_id)
            for known_id, (known_session, _known_generation) in list(
                self._request_metadata.items()
            ):
                if known_session == session_id and known_id != request_id:
                    self._remember_cancelled(known_id)
            state = _RequestState(
                request_id=request_id,
                session_id=session_id,
                generation=generation,
                cancelled=Event(),
            )
            self._requests[request_id] = state
            self._request_metadata[request_id] = (session_id, generation)
            # Reusing a request id is allowed by ROS callers; the new state is
            # authoritative and must not inherit the old cancellation tombstone.
            self._cancelled_request_ids.discard(request_id)
            return state

    def _submit_request(self, state: _RequestState, text: str, top_k: int) -> None:
        """Submit work and retain the Future so queued requests can be cancelled."""

        future = self._workers.submit(self._generate, state, text, top_k)
        with self._state_lock:
            if self._requests.get(state.request_id) is state:
                state.future = future
            else:
                future.cancel()

    def _on_interrupt(self, message: VoiceControl) -> None:
        """Invalidate answer work for the interrupted interaction generation."""

        command = str(getattr(message, "command", "") or "").strip().lower()
        if command != "interrupt":
            return
        session_id = nonempty_session_id(getattr(message, "session_id", ""))
        request_id = str(getattr(message, "request_id", "") or "").strip()
        incoming = _message_generation(message)
        with self._state_lock:
            current = self._session_generations.get(session_id, 0)
            # A lower-generation packet is a delayed interrupt and must not
            # cancel a newer query.  Zero means "next generation" for legacy
            # publishers that cannot populate the field.
            if incoming and incoming < current:
                return
            generation = incoming or (current + 1)
            if generation == current and incoming:
                # Duplicate delivery of the same interrupt is harmless.
                matching = [
                    item
                    for item in self._requests.values()
                    if item.session_id == session_id
                    and (not request_id or item.request_id == request_id)
                    and not item.cancelled.is_set()
                ]
                if not matching:
                    return
            self._session_generations[session_id] = max(current, generation)
            self._session_needs_next_generation.add(session_id)
            for item in list(self._requests.values()):
                if item.session_id != session_id:
                    continue
                if request_id and item.request_id != request_id:
                    continue
                if item.generation <= generation:
                    item.cancelled.set()
                    if item.future is not None:
                        item.future.cancel()
                    self._remember_cancelled(item.request_id)
            for known_id, (known_session, known_generation) in list(
                self._request_metadata.items()
            ):
                if known_session != session_id:
                    continue
                if request_id and known_id != request_id:
                    continue
                if known_generation <= generation:
                    self._remember_cancelled(known_id)

    def _request_is_current(self, state: _RequestState) -> bool:
        if state.cancelled.is_set():
            return False
        with self._state_lock:
            if self._requests.get(state.request_id) is not state:
                return False
            if state.request_id in self._cancelled_request_ids:
                return False
            current = self._session_generations.get(state.session_id, state.generation)
            return current == state.generation

    def _finish_request(self, state: _RequestState) -> None:
        with self._state_lock:
            if self._requests.get(state.request_id) is state:
                self._requests.pop(state.request_id, None)

    def _remember_cancelled(self, request_id: str) -> None:
        if not request_id:
            return
        self._cancelled_request_ids.add(request_id)
        if len(self._cancelled_request_ids) > 2048:
            self._cancelled_request_ids = set(list(self._cancelled_request_ids)[-1024:])
        if len(self._request_metadata) > 4096:
            for old_id in list(self._request_metadata)[:2048]:
                self._request_metadata.pop(old_id, None)

    def _service_for(self, session_id: str) -> AnswerGenerationService:
        existing = self._session_services.pop(session_id, None)
        if existing is not None:
            self._session_services[session_id] = existing
            return existing
        service = AnswerGenerationService(
            settings_store=SettingsStore(self._settings_path),
            index_dir=self._index_dir,
        )
        self._session_services[session_id] = service
        while len(self._session_services) > self._max_sessions:
            self._session_services.popitem(last=False)
        return service

    def _flush_events(self) -> None:
        self._flush_controls()
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                return
            if self._answer_event_is_stale(event):
                continue
            message = VoiceAnswer()
            message.header.stamp = self.get_clock().now().to_msg()
            message.request_id = event.request_id
            message.session_id = event.session_id
            message.text = event.text
            message.final = event.final
            message.success = event.success
            message.error = event.error
            message.model = event.model
            message.source_names = list(event.source_names)
            message.disposition = event.disposition
            _assign_message_field(message, "generation", event.generation)
            self._answer_publisher.publish(message)

    def _answer_event_is_stale(self, event: _AnswerEvent) -> bool:
        with self._state_lock:
            if event.request_id in self._cancelled_request_ids:
                return True
            current = self._session_generations.get(event.session_id, event.generation)
            return bool(event.generation and current > event.generation)

    def _flush_controls(self) -> None:
        while True:
            try:
                event = self._controls.get_nowait()
            except queue.Empty:
                return
            if self._control_event_is_stale(event):
                continue
            message = VoiceControl()
            message.header.stamp = self.get_clock().now().to_msg()
            message.request_id = event.request_id
            message.session_id = event.session_id
            _assign_message_field(message, "generation", event.generation)
            _assign_message_field(message, "purpose", event.purpose)
            message.command = event.command
            message.reason = event.reason
            self._control_publisher.publish(message)

    def _control_event_is_stale(self, event: _ControlEvent) -> bool:
        with self._state_lock:
            if event.request_id in self._cancelled_request_ids:
                return True
            current = self._session_generations.get(event.session_id, event.generation)
            return bool(event.generation and current > event.generation)


def _message_generation(message: Any) -> int:
    try:
        return max(0, int(getattr(message, "generation", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _assign_message_field(message: Any, name: str, value: Any) -> None:
    """Keep source compatibility with generated messages from older builds."""

    if hasattr(message, name):
        setattr(message, name, value)


def _stream_answer_events(
    service: Any,
    text: str,
    *,
    top_k: int,
    should_cancel: Callable[[], bool],
) -> Any:
    """Pass cooperative cancellation when the service signature supports it."""

    stream_answer_events = service.stream_answer_events
    try:
        parameters = inspect.signature(stream_answer_events).parameters
    except (TypeError, ValueError):
        return stream_answer_events(text, top_k=top_k)

    supports_cancellation = "should_cancel" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if supports_cancellation:
        return stream_answer_events(
            text,
            top_k=top_k,
            should_cancel=should_cancel,
        )
    return stream_answer_events(text, top_k=top_k)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RagAnswerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
