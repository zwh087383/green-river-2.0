from __future__ import annotations

import json
import queue
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

_FILLER_RE = re.compile(r'^[嗯啊哦哈呢哟喂哎嗨哼唔噢哇咦哒]+[嗯啊哦哈呢哟喂哎嗨哼唔噢哇咦哒。！？!?]*$')
_MIN_UTTERANCE_MS = 400


def _is_filler(text: str) -> bool:
    t = text.strip()
    return len(t) <= 1 or bool(_FILLER_RE.match(t))

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from rag_interfaces.msg import VoiceControl, VoicePlayback, VoiceQuery
from rag_system.contracts import AudioBuffer, AudioSpec
from rag_system.settings import SettingsStore
from rag_system.voice.service import VoiceService
from rag_system.voice.vad import FsmnVadEngine, VadConfig
from rag_system.voice.wakeup import WakeWordEngine

from voice.audio_pipeline import LocalAudioPipeline
from voice.runtime import default_data_dir, nonempty_session_id, normalize_top_k


@dataclass(frozen=True, slots=True)
class _WorkerEvent:
    """Result from the ASR worker tagged with the interaction generation."""

    generation: int
    type: str
    detail: str


class RobotAudioInputNode(Node):
    """Capture the robot microphone and publish typed queries after local ASR."""

    def __init__(self) -> None:
        super().__init__("robot_audio_input_node")
        data_dir = default_data_dir()
        self.declare_parameter("device", "")
        self.declare_parameter("sample_rate", 16_000)
        self.declare_parameter("block_ms", 100)
        self.declare_parameter("wakeup_enabled", True)
        self.declare_parameter("wake_word", "\u5c0f\u7eff\u5c0f\u7eff")
        self.declare_parameter("end_silence_ms", 400)
        self.declare_parameter("initial_silence_ms", 8_000)
        self.declare_parameter("max_utterance_ms", 30_000)
        self.declare_parameter("pre_pad_ms", 300)
        self.declare_parameter("post_pad_ms", 200)
        self.declare_parameter("settings_path", str(data_dir / "settings.json"))
        self.declare_parameter("query_topic", "/voice/query")
        self.declare_parameter("control_topic", "/voice/control")
        self.declare_parameter("interrupt_topic", "/voice/interrupt")
        self.declare_parameter(
            "arm_motion_request_topic", "/voice/arm_motion_request"
        )
        self.declare_parameter("playback_topic", "/voice/playback")
        self.declare_parameter("status_topic", "/voice/audio_status")
        self.declare_parameter("session_id", "robot")
        self.declare_parameter("top_k", 4)

        self._sample_rate = int(self.get_parameter("sample_rate").value)
        self._block_ms = int(self.get_parameter("block_ms").value)
        if self._sample_rate != 16_000:
            raise ValueError("local wakeup and VAD require a 16000 Hz input stream")
        if self._block_ms != 100:
            raise ValueError("local wakeup and VAD require 100 ms audio blocks")
        self._session_id = nonempty_session_id(
            self.get_parameter("session_id").value
        )
        self._top_k = normalize_top_k(int(self.get_parameter("top_k").value))
        self._audio_chunks: queue.Queue[tuple[int, bytes]] = queue.Queue(maxsize=50)
        self._worker_events: queue.SimpleQueue[_WorkerEvent] = queue.SimpleQueue()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="robot-asr")
        self._stream: Any | None = None
        self._epoch_lock = RLock()
        self._epoch = 0
        self._pending_request_id = ""
        self._pending_generation = 0
        self._active_request_id = ""
        self._interrupt_request_id = ""
        self._interrupt_ack_request_id = ""
        self._interrupt_generation = 0

        query_topic = str(self.get_parameter("query_topic").value)
        control_topic = str(self.get_parameter("control_topic").value)
        interrupt_topic = str(self.get_parameter("interrupt_topic").value)
        arm_motion_request_topic = str(
            self.get_parameter("arm_motion_request_topic").value
        )
        playback_topic = str(self.get_parameter("playback_topic").value)
        status_topic = str(self.get_parameter("status_topic").value)
        self._query_publisher = self.create_publisher(VoiceQuery, query_topic, 20)
        self._interrupt_publisher = self.create_publisher(VoiceControl, interrupt_topic, 20)
        self._arm_motion_request_publisher = self.create_publisher(
            VoiceControl,
            arm_motion_request_topic,
            20,
        )
        self._status_publisher = self.create_publisher(String, status_topic, 20)
        self.create_subscription(VoiceControl, control_topic, self._on_control, 20)
        self.create_subscription(VoicePlayback, playback_topic, self._on_playback, 20)

        settings_path = Path(
            str(self.get_parameter("settings_path").value)
        ).expanduser().resolve()
        self._voice_service = VoiceService(
            settings_store=SettingsStore(settings_path),
            prefer_dashscope_tts=False,
        )
        vad_config = VadConfig(
            sample_rate=self._sample_rate,
            frame_ms=self._block_ms,
            end_silence_ms=int(self.get_parameter("end_silence_ms").value),
            initial_silence_ms=int(
                self.get_parameter("initial_silence_ms").value
            ),
            max_utterance_ms=int(self.get_parameter("max_utterance_ms").value),
            pre_pad_ms=int(self.get_parameter("pre_pad_ms").value),
            post_pad_ms=int(self.get_parameter("post_pad_ms").value),
        )
        self._vad_engine = FsmnVadEngine(
            end_silence_ms=vad_config.end_silence_ms
        )
        wakeup_enabled = bool(self.get_parameter("wakeup_enabled").value)
        self._wake_engine = (
            WakeWordEngine(
                keyword_words=str(self.get_parameter("wake_word").value)
            )
            if wakeup_enabled
            else None
        )
        self._pipeline = LocalAudioPipeline(
            wake_engine=self._wake_engine,
            vad_engine=self._vad_engine,
            vad_config=vad_config,
        )
        self._epoch = self._pipeline.generation

        self._open_stream()
        self.create_timer(0.02, self._tick)
        self._publish_status(self._pipeline.state, "microphone ready")
        self.get_logger().info(
            f"Robot audio input ready: query={query_topic} device={self._device_label()}"
        )

    def destroy_node(self) -> bool:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._worker.shutdown(wait=False, cancel_futures=True)
        self._pipeline.close()
        if self._wake_engine is not None:
            self._wake_engine.close()
        return super().destroy_node()

    def _open_stream(self) -> None:
        try:
            import sounddevice as sound_device
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is unavailable; run scripts/install_all_linux.sh"
            ) from exc

        block_size = self._sample_rate * self._block_ms // 1_000
        self._stream = sound_device.RawInputStream(
            samplerate=self._sample_rate,
            blocksize=block_size,
            device=self._resolved_device(),
            channels=1,
            dtype="int16",
            callback=self._on_audio,
        )
        self._stream.start()

    def _resolved_device(self) -> str | int | None:
        value = str(self.get_parameter("device").value or "").strip()
        if not value:
            return None
        if value.lstrip("-").isdigit():
            return int(value)
        return value

    def _device_label(self) -> str:
        device = self._resolved_device()
        return "default" if device is None else str(device)

    def _on_audio(
        self,
        indata: Any,
        _frames: int,
        _time_info: Any,
        status: Any,
    ) -> None:
        if status:
            self._put_worker_event("warning", str(status))
        chunk = bytes(indata)
        generation = self._current_epoch()
        try:
            self._audio_chunks.put_nowait((generation, chunk))
        except queue.Full:
            try:
                self._audio_chunks.get_nowait()
            except queue.Empty:
                pass
            try:
                self._audio_chunks.put_nowait((generation, chunk))
            except queue.Full:
                pass

    def _tick(self) -> None:
        self._flush_worker_events()
        for _ in range(4):
            try:
                queued = self._audio_chunks.get_nowait()
            except queue.Empty:
                return
            generation, chunk = self._unpack_audio_chunk(queued)
            if generation != self._current_epoch():
                # The callback may have queued a block just before an
                # interruption advanced the epoch.  It belongs to the old
                # turn and must never reach VAD.
                continue
            try:
                events = self._pipeline.feed(chunk)
            except Exception as exc:
                self.get_logger().error(f"Audio pipeline failed: {exc}")
                self._publish_status("error", str(exc))
                self._reset_pipeline()
                return
            for event in events:
                if int(getattr(event, "generation", 0) or 0) > self._current_epoch():
                    self._sync_epoch(
                        int(event.generation),
                        flush_audio=event.type in {"timeout", "error"},
                    )
                if event.type == "wake":
                    self._publish_status("wake_detected", event.keyword)
                    self._publish_arm_motion_request(
                        purpose="wake_wave",
                        generation=int(
                            getattr(event, "generation", 0) or self._current_epoch()
                        ),
                    )
                elif event.type == "interrupt":
                    self._handle_interrupt(event)
                elif event.type == "speech_start":
                    self._publish_status("listening", "speech detected")
                elif event.type == "timeout":
                    self._publish_status(self._pipeline.state, "no speech")
                elif event.type == "error":
                    self._publish_status(self._pipeline.state, event.detail)
                elif event.type == "utterance":
                    if event.generation != self._current_epoch():
                        continue
                    duration_ms = len(event.pcm16) // (self._sample_rate * 2 // 1_000)
                    if duration_ms < _MIN_UTTERANCE_MS:
                        self._pipeline.resume_listening()
                        self._publish_status("listening", f"utterance too short ({duration_ms}ms), ignored")
                        continue
                    self._pipeline.begin_transcribing(event.generation)
                    self._publish_status("transcribing", event.detail)
                    self._worker.submit(self._transcribe, event.pcm16, event.generation)

    def _transcribe(self, pcm16: bytes, generation: int | None = None) -> None:
        work_generation = (
            self._current_epoch() if generation is None else int(generation)
        )
        try:
            wav = AudioBuffer(
                pcm16,
                AudioSpec(sample_rate=self._sample_rate),
            ).to_wav_bytes()
            text = self._voice_service.speech_to_text(
                wav,
                filename="robot-microphone.wav",
            ).strip()
            if not text:
                raise RuntimeError("ASR returned empty text")
            self._put_worker_event("query", text, work_generation)
        except Exception as exc:
            self._put_worker_event("error", str(exc), work_generation)

    def _flush_worker_events(self) -> None:
        while True:
            try:
                event = self._worker_events.get_nowait()
            except queue.Empty:
                return
            event_generation, event_type, detail = self._unpack_worker_event(event)
            if event_generation != self._current_epoch():
                # A late ASR callback from a cancelled turn is harmless.
                continue
            if event_type == "query":
                if _is_filler(detail):
                    self.get_logger().info(f"Filler ignored: {detail!r}")
                    self._pipeline.resume_listening()
                    self._publish_status("listening", f"filler ignored: {detail}")
                    continue
                message = VoiceQuery()
                message.header.stamp = self.get_clock().now().to_msg()
                message.request_id = uuid.uuid4().hex
                message.session_id = self._session_id
                _assign_message_field(message, "generation", event_generation)
                message.text = detail
                message.top_k = self._top_k
                self._query_publisher.publish(message)
                self._active_request_id = message.request_id
                self._pending_request_id = message.request_id
                self._pending_generation = event_generation
                self._pipeline.begin_answering(event_generation)
                self._publish_status("recognized", detail)
            elif event_type == "error":
                self.get_logger().error(f"Local ASR failed: {detail}")
                self._publish_status("error", detail)
                self._reset_pipeline()
            elif event_type == "warning":
                self.get_logger().warning(detail)
                self._publish_status(self._pipeline.state, detail)

    def _on_control(self, message: VoiceControl) -> None:
        if str(message.session_id or "") != self._session_id:
            return
        request_id = str(message.request_id or "")
        generation = self._message_generation(message)
        command = str(message.command or "").strip()
        if not self._accept_message_generation(generation, request_id):
            return
        purpose = str(getattr(message, "purpose", "") or "").strip().lower()
        timeout_control = (
            command == "return_to_wakeup"
            and purpose == "interrupt_ack_timeout"
        )
        if command != "interrupt" and not timeout_control:
            if not request_id or request_id not in {
                self._pending_request_id,
                self._active_request_id,
            }:
                return
        if self._pipeline.state == "interrupt_ack" and command != "interrupt":
            if not (
                command == "return_to_wakeup"
                and purpose == "interrupt_ack_timeout"
            ):
                # Old answer controls may already be queued when KWS wins the
                # race.  Only the acknowledgement timeout may leave this
                # state before matching playback feedback arrives.
                return
        if command == "lock_listening":
            self._pipeline.begin_answering(generation or self._current_epoch())
            self._publish_status("answering", str(message.reason or "answering"))
        elif command == "resume_listening":
            self._pipeline.resume_listening()
            self._clear_active_request()
            self._sync_epoch(flush_audio=True)
            self._publish_status("listening", str(message.reason or "ignored turn"))
        elif command == "return_to_wakeup":
            self._reset_pipeline()
            self._publish_status(self._pipeline.state, str(message.reason or "request failed"))
        elif command == "interrupt":
            # Keep accepting legacy interrupt packets sent on /voice/control.
            # Dedicated deployments use /voice/interrupt and never enter this
            # branch for the node's own publication.
            if (
                generation
                and generation == self._interrupt_generation
                and request_id == self._interrupt_request_id
            ):
                return
            event = self._pipeline.interrupt(
                detail=str(message.reason or "interrupt requested")
            )
            self._handle_interrupt(event, publish_control=False)

    def _on_playback(self, message: VoicePlayback) -> None:
        if str(message.session_id or "") != self._session_id:
            return
        request_id = str(message.request_id or "")
        generation = self._message_generation(message)
        purpose = str(getattr(message, "purpose", "") or "").strip().lower()
        ack_purpose = purpose in {
            "interrupt",
            "interrupt_ack",
            "wakeup_ack",
            "ack",
        }
        if not self._accept_message_generation(generation, request_id):
            return
        if ack_purpose and generation != self._interrupt_generation:
            # The acknowledgement is the gate that opens a fresh VAD turn.
            # Require an explicit matching generation.
            return
        state = str(message.state or "").strip()
        if ack_purpose:
            if state == "started":
                if not request_id:
                    return
                if (
                    self._interrupt_ack_request_id
                    and request_id != self._interrupt_ack_request_id
                ):
                    return
                # The bridge uses a fresh request id for the acknowledgement so
                # a late answer completion cannot be mistaken for its lifecycle.
                self._interrupt_ack_request_id = request_id
            elif (
                not self._interrupt_ack_request_id
                or request_id != self._interrupt_ack_request_id
            ):
                # A terminal packet may arrive on another topic before the
                # bridge's `started` packet.  It cannot open VAD until the exact
                # acknowledgement token has first been established.
                return
        elif not request_id or request_id not in {
            self._pending_request_id,
            self._active_request_id,
        }:
            return
        if state == "started":
            if self._pipeline.state == "interrupt_ack" and not ack_purpose:
                # A completion/started notification from the cancelled answer
                # must not move the new interruption turn back to PLAYING.
                return
            if self._pipeline.playback_started(
                purpose=purpose or "answer",
                generation=generation or self._current_epoch(),
            ):
                self._publish_status(
                    self._pipeline.state,
                    str(message.detail or "playback started"),
                )
            return
        if state not in {"completed", "failed", "interrupted"}:
            return
        if self._pipeline.state == "interrupt_ack" and not ack_purpose:
            return
        if ack_purpose:
            self._pipeline.playback_finished(
                purpose=purpose,
                generation=generation or self._current_epoch(),
                failed=state == "failed",
            )
            self._clear_active_request()
            self._sync_epoch(flush_audio=True)
            self._publish_status(self._pipeline.state, str(message.detail or state))
            return
        self._pipeline.playback_finished(
            purpose=purpose or "answer",
            generation=generation or self._current_epoch(),
            failed=state == "failed",
        )
        self._clear_active_request()
        self._sync_epoch(flush_audio=True)
        self._publish_status(self._pipeline.state, str(message.detail or state))

    def _handle_interrupt(self, event: Any, *, publish_control: bool = True) -> None:
        generation = int(getattr(event, "generation", 0) or self._pipeline.generation)
        self._sync_epoch(generation, flush_audio=True)
        self._interrupt_generation = generation
        self._interrupt_request_id = self._active_request_id or self._pending_request_id
        self._interrupt_ack_request_id = ""
        self._active_request_id = ""
        self._pending_request_id = ""
        self._pending_generation = 0
        if publish_control:
            message = VoiceControl()
            message.header.stamp = self.get_clock().now().to_msg()
            message.request_id = self._interrupt_request_id
            message.session_id = self._session_id
            _assign_message_field(message, "generation", generation)
            _assign_message_field(message, "purpose", "interrupt")
            message.command = "interrupt"
            message.reason = str(
                getattr(event, "detail", "") or "wake word detected during active turn"
            )
            publisher = getattr(self, "_interrupt_publisher", None)
            if publisher is None:
                # Compatibility with pre-interrupt test doubles/launches.
                publisher = getattr(self, "_control_publisher", None)
            if publisher is not None:
                publisher.publish(message)
        self._publish_status("interrupt_ack", str(getattr(event, "keyword", "") or "wake word"))

    def _reset_pipeline(self) -> None:
        self._pipeline.reset()
        self._clear_active_request()
        self._sync_epoch(flush_audio=True)

    def _clear_active_request(self) -> None:
        self._pending_request_id = ""
        self._pending_generation = 0
        self._active_request_id = ""
        self._interrupt_request_id = ""
        self._interrupt_ack_request_id = ""
        self._interrupt_generation = 0

    def _current_epoch(self) -> int:
        with self._epoch_lock:
            return self._epoch

    def _sync_epoch(self, generation: int | None = None, *, flush_audio: bool = False) -> int:
        target = self._pipeline.generation if generation is None else int(generation)
        with self._epoch_lock:
            self._epoch = max(self._epoch, target)
            current = self._epoch
        if flush_audio:
            self._drain_audio_queue()
        return current

    def _drain_audio_queue(self) -> None:
        while True:
            try:
                self._audio_chunks.get_nowait()
            except queue.Empty:
                return

    def _put_worker_event(
        self, event_type: str, detail: str, generation: int | None = None
    ) -> None:
        self._worker_events.put(
            _WorkerEvent(
                generation=self._current_epoch() if generation is None else int(generation),
                type=event_type,
                detail=detail,
            )
        )

    def _unpack_audio_chunk(self, value: Any) -> tuple[int, bytes]:
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], int)
            and isinstance(value[1], (bytes, bytearray, memoryview))
        ):
            return int(value[0]), bytes(value[1])
        # Compatibility for simple test doubles that put raw PCM into the
        # queue directly.
        return self._current_epoch(), bytes(value)

    def _unpack_worker_event(self, value: Any) -> tuple[int, str, str]:
        if isinstance(value, _WorkerEvent):
            return value.generation, value.type, value.detail
        if isinstance(value, tuple):
            if len(value) == 3:
                return int(value[0]), str(value[1]), str(value[2])
            if len(value) == 2:
                return self._current_epoch(), str(value[0]), str(value[1])
        return self._current_epoch(), "warning", str(value)

    @staticmethod
    def _message_generation(message: Any) -> int:
        try:
            return max(0, int(getattr(message, "generation", 0) or 0))
        except (TypeError, ValueError):
            return 0

    def _accept_message_generation(self, generation: int, request_id: str) -> bool:
        """Correlate control/ack packets without letting delayed packets win.

        The ROSIDL layout changed with this protocol, so older generated
        message types are not wire-compatible anyway.  Require the exact
        generation rather than treating zero or an older value as a wildcard.
        """

        del request_id
        return bool(generation and generation == self._current_epoch())

    def _publish_status(self, state: str, detail: str) -> None:
        message = String()
        message.data = json.dumps(
            {"state": state, "detail": detail},
            ensure_ascii=False,
        )
        self._status_publisher.publish(message)

    def _publish_arm_motion_request(self, *, purpose: str, generation: int) -> None:
        try:
            message = VoiceControl()
            message.header.stamp = self.get_clock().now().to_msg()
            message.request_id = f"wake-{uuid.uuid4().hex}"
            message.session_id = self._session_id
            _assign_message_field(message, "generation", generation)
            _assign_message_field(message, "purpose", purpose)
            message.command = "play_motion"
            message.reason = "normal wake word detected"
            self._arm_motion_request_publisher.publish(message)
        except Exception as exc:
            # A mechanical gesture must never interrupt wakeup, VAD, or ASR.
            self.get_logger().warning(f"Failed to request wake arm motion: {exc}")


def _assign_message_field(message: Any, name: str, value: Any) -> None:
    """Set fields added in the interruption protocol on old message doubles."""

    if hasattr(message, name):
        setattr(message, name, value)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RobotAudioInputNode()
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
