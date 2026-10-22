"""Robot microphone/speaker adapter for the same streaming protocol as the Web UI.

Run the local rag_system.web backend first. No vendor TextToAudio reply is used.
"""
from __future__ import annotations

from rag_system.voice_diagnostics import trace_voice
from rag_system.answer_gestures import select_answer_gesture
from rag_system.pcm_probe import PcmProbe

import base64
import io
import json
import queue
import threading
import time
import uuid
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

import httpx
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rag_interfaces.msg import VoicePlayback
from rag_system.robot_motion import MotionClient
from rag_system.hardware_wakeup import SerialWakeSource, DEFAULT_PORT, write_status


class RealtimeVoiceGateway(Node):
    def __init__(self):
        super().__init__("realtime_voice_gateway")
        self.declare_parameter("backend_url", "http://127.0.0.1:8000")
        self.declare_parameter("input_device", "")
        self.declare_parameter("output_device", "")
        self.declare_parameter("session_id", "robot")
        self.declare_parameter("wake_source", "software")
        self.declare_parameter("interrupt_source", "hardware")
        self.declare_parameter("echo_cancel_enabled", False)
        self.declare_parameter("wake_serial_port", DEFAULT_PORT)
        self.declare_parameter("wake_keyword", "小绿小绿")
        self.declare_parameter("playback_guard_ms", 300)
        self._playback_guard_s = float(self.get_parameter("playback_guard_ms").value) / 1000
        if not 0 <= self._playback_guard_s <= 2:
            raise ValueError("playback_guard_ms must be between 0 and 2000")
        self.wake_source = str(self.get_parameter("wake_source").value)
        if self.wake_source not in {"software", "hardware"}:
            raise ValueError("wake_source must be software or hardware")
        self.interrupt_source = str(self.get_parameter("interrupt_source").value)
        if self.interrupt_source not in {"software", "hardware"}:
            raise ValueError("interrupt_source must be software or hardware")
        self.base = str(self.get_parameter("backend_url").value).rstrip("/")
        self.session_id = str(self.get_parameter("session_id").value)
        self.conversation_id = "robot-" + uuid.uuid4().hex
        self.display = self.create_publisher(String, "/voice/display_text", 20)
        self.playback = self.create_publisher(VoicePlayback, "/voice/playback", 20)
        self._motion_client = MotionClient(report=self.get_logger().warning, on_result=self._motion_result)
        self._stop = threading.Event()
        self._capture = threading.Event()
        self._cancel = threading.Event()
        self._overflow = threading.Event()
        self._audio = queue.Queue(maxsize=50)
        self._lock = threading.RLock()
        self._send_lock = threading.RLock()
        self._capture_lock = threading.Lock()
        self._capture_epoch = 0
        self._interrupt_pending = threading.Event()
        self._pcm_probe = PcmProbe("gateway.pcm_forwarded", session_id=self.session_id)
        self._last_hardware_sequence = 0
        self._output_lock = threading.Lock()
        self._generation = time.time_ns() // 1000
        self._socket = None
        self._output = None
        self._response = None
        self._wake_phase = "disconnected"
        self._connection_id = ""
        self._hardware = None
        self._hardware_status = Path(os.environ.get("RAG_DATA_DIR", "data")) / "runtime/hardware-wake.json"
        if self.wake_source == "hardware":
            self._hardware = SerialWakeSource(str(self.get_parameter("wake_serial_port").value),
                self._hardware_wake, keyword=str(self.get_parameter("wake_keyword").value),
                status_path=self._hardware_status, report=self.get_logger().info)
            self._hardware.start()
        else:
            write_status(self._hardware_status, {"source": "software", "connected": False, "error": ""})
        self._workers = ThreadPoolExecutor(max_workers=2, thread_name_prefix="robot-reply")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _device(self, name):
        value = str(self.get_parameter(name).value).strip()
        return int(value) if value.isdigit() else (value or None)

    def _update_voice_phase(self, kind):
        phase = {"armed": "armed", "listening": "listening",
                 "wake_detected": "prompt", "final": "answer"}.get(kind)
        if phase is None:
            return
        with self._lock:
            changed = self._wake_phase != phase
            self._wake_phase = phase
            if self._hardware is not None:
                self._hardware.state["gateway_phase"] = phase
        if changed:
            self.get_logger().info("Voice phase: " + phase)
            if phase == "armed":
                self._show("[idle] 待唤醒，请说小绿小绿")
            elif phase == "listening":
                self._show("[query] 正在聆听，请说话")

    def _send(self, payload):
        with self._send_lock:
            if self._socket is not None:
                self._socket.send(payload if isinstance(payload, bytes) else json.dumps(payload))

    def _hardware_wake(self, event):
        # Nothing is queued across prompts, ASR turns, or socket reconnects.
        # The backend decides normal wake vs interrupt from its current phase.
        with self._send_lock:
            with self._lock:
                sequence = event["sequence"]
                allowed = {"armed", "answer"} if self.interrupt_source == "hardware" else {"armed"}
                reason = ("phase" if self._wake_phase not in allowed else
                          "no_connection" if not self._connection_id else
                          "no_socket" if self._socket is None else
                          "interrupt_pending" if self._interrupt_pending.is_set() else
                          "replay" if sequence <= self._last_hardware_sequence else None)
                trace_voice("gateway.wake_received", sequence=sequence, phase=self._wake_phase,
                            generation=self._generation, connection_id=self._connection_id,
                            decision=reason or "forward")
                if reason is not None:
                    return
                connection_id = self._connection_id
                socket = self._socket
                self._last_hardware_sequence = sequence
                self._interrupt_pending.set()
            try:
                socket.send(json.dumps({"type": "hardware_wake", "connection_id": connection_id,
                                        "sequence": sequence}))
            except Exception:
                # Reconnect instead of leaving a pending handoff latched forever.
                with suppress(Exception):
                    socket.close()
                raise
        trace_voice("gateway.wake_forwarded", sequence=sequence, connection_id=connection_id,
                    generation=self._generation)
        self.get_logger().info("Hardware wake forwarded: sequence=" + str(event["sequence"]))

    def _set_capture(self, enabled):
        # Serialize invalidation with the last send check, but never hold this
        # network lock in the PortAudio callback.
        with self._send_lock:
            with self._capture_lock:
                self._capture.clear()
                self._capture_epoch += 1
                discarded = 0
                while True:
                    try:
                        self._audio.get_nowait()
                        discarded += 1
                    except queue.Empty:
                        break
                self._overflow.clear()
                if enabled:
                    self._capture.set()
                epoch = self._capture_epoch
        self.get_logger().info(f"Audio gate: open={enabled}, epoch={epoch}, discarded={discarded}, "
                               f"generation={self._generation}")

    def _finish_turn(self, control, generation, cancel, *, played):
        # Preserve software interrupt KWS during answer/guard, including when
        # the initial wake uses serial. Do not forward answer PCM to ASR.
        if played and self.wake_source == "hardware":
            self.get_logger().info(f"Playback drained; guard start: generation={generation}, "
                                   f"guard_ms={self._playback_guard_s * 1000:g}")
            deadline = time.monotonic() + self._playback_guard_s
            while True:
                if not self._current(generation, cancel) or self._interrupt_pending.is_set():
                    trace_voice("gateway.guard_cancelled", generation=generation,
                                interrupt_pending=self._interrupt_pending.is_set())
                    self.get_logger().info(f"Playback guard cancelled: generation={generation}")
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                cancel.wait(min(remaining, .02))
        with self._send_lock:
            if not self._current(generation, cancel) or self._interrupt_pending.is_set():
                trace_voice("gateway.control_blocked", generation=generation, control=control,
                            interrupt_pending=self._interrupt_pending.is_set())
                return False
            self._set_capture(False)
            # Close the answer handoff atomically with sending continue/resume.
            # A serial wake already accepted above always wins this race.
            with self._lock:
                self._wake_phase = "transition"
            self._send({"type": control})
        trace_voice("gateway.control_sent", generation=generation, control=control,
                    connection_id=getattr(self, "_connection_id", ""))
        self.get_logger().info(f"Audio handoff: generation={generation}, control={control}")
        return True

    def _show(self, text):
        message = String()
        message.data = text
        self.display.publish(message)

    def _feedback(self, request_id, generation, state, purpose="answer"):
        message = VoicePlayback()
        message.header.stamp = self.get_clock().now().to_msg()
        message.request_id, message.session_id = request_id, self.session_id
        message.generation, message.purpose = generation, purpose
        message.state, message.detail = state, "Qwen PCM local audio device"
        self.playback.publish(message)

    def _advance(self):
        # Generation invalidation and final control sends share a lock. A
        # worker cannot pass its validity check then send after cancellation.
        with self._send_lock:
            with self._lock:
                self._cancel.set()
                response, output = self._response, self._output
                self._generation += 1
                self._cancel = threading.Event()
                generation, cancel = self._generation, self._cancel
        trace_voice("gateway.cancel_start", generation=generation,
                    http_open=response is not None, output_open=output is not None)
        self._motion_client.advance(generation)
        # Closing a socket/device may wait for its worker's finally block.
        # Never hold the state lock while doing either operation.
        if response is not None:
            with suppress(Exception):
                response.close()
        if output is not None:
            with suppress(Exception):
                output.abort()
        self.get_logger().info(f"Reply generation advanced: generation={generation}, "
                               f"http_cancelled={response is not None}, playback_aborted={output is not None}")
        trace_voice("gateway.cancel_complete", generation=generation)
        return generation, cancel

    def _motion_result(self, result):
        self.get_logger().info("[motion-result] " + json.dumps(result, ensure_ascii=True))

    def _gesture(self, generation, purpose, interrupt=False):
        if interrupt:
            self._motion_client.advance(generation)
        else:
            self._motion_client.start_playback(uuid.uuid4().hex, generation, purpose, self.session_id)

    def _current(self, generation, cancel):
        return not cancel.is_set() and not self._stop.is_set() and generation == self._generation

    def _callback(self, data, frames, timing, status):
        with self._capture_lock:
            if self._capture.is_set():
                try:
                    self._audio.put_nowait((self._capture_epoch, bytes(data)))
                except queue.Full:
                    self._overflow.set()

    def _send_audio_frame(self, frame):
        epoch, pcm = frame
        with self._send_lock:
            with self._capture_lock:
                valid = self._capture.is_set() and epoch == self._capture_epoch
            if valid:
                self._send(pcm)
                probe = getattr(self, "_pcm_probe", None)
                if probe is not None and self._wake_phase == "answer":
                    probe.feed(pcm)
            return valid

    def _send_audio(self, connected):
        while connected.is_set() and not self._stop.is_set():
            try:
                pcm = self._audio.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._send_audio_frame(pcm)
            except Exception:
                connected.clear()

    def _handle_voice_event(self, event):
        kind = event.get("type")
        if kind != "partial":
            trace_voice("gateway.voice_event", event=event, generation=self._generation,
                        phase=self._wake_phase, connection_id=getattr(self, "_connection_id", ""))
        self._update_voice_phase(kind)
        if kind in {"armed", "listening"}:
            self._set_capture(kind == "listening" or self.wake_source == "software")
        elif kind == "wake_detected":
            self._set_capture(False)
            # Invalidate the old worker before clearing the
            # pending marker that protects its handoff.
            generation, cancel = self._advance()
            self._interrupt_pending.clear()
            interrupted = event.get("reason") == "interrupt"
            self._gesture(generation, "interrupt" if interrupted else "wake_wave", interrupted)
            self._workers.submit(self._prompt, event.get("reason") == "interrupt", generation, cancel)
        elif kind == "partial":
            self._show("[query] " + event.get("text", ""))
        elif kind == "final":
            # Discard question frames and feed fresh microphone PCM only to
            # the answer-phase software KWS monitor.
            self._set_capture(self.interrupt_source == "software")
            generation, cancel = self._advance()
            self._workers.submit(self._reply, event, generation, cancel)
        elif kind == "no_speech":
            self._finish_turn("resume", self._generation, self._cancel, played=False)
        elif kind == "error":
            raise RuntimeError(str(event.get("detail", "Voice backend error")))

    def _prepare_audio_route(self):
        # AEC is independent of whether wake events come from serial or KWS.
        if bool(self.get_parameter("echo_cancel_enabled").value):
            from rag_system.voice.echo_cancel import prepare_echo_cancel
            route = prepare_echo_cancel()
            self.get_logger().info("Voice echo cancellation: " + json.dumps(route))
            trace_voice("gateway.echo_cancel", **route)
            return route
        return None

    def _run(self):
        import sounddevice as sd
        from websockets.sync.client import connect
        while not self._stop.is_set():
            connected = threading.Event()
            sender = None
            try:
                self._prepare_audio_route()
                ws_url = self.base.replace("http://", "ws://", 1).replace("https://", "wss://", 1)
                with connect(ws_url + "/api/voice/session/stream", proxy=None, close_timeout=2) as ws:
                    with self._lock:
                        self._socket = ws
                        self._connection_id = uuid.uuid4().hex
                        self._wake_phase = "connecting"
                    self._interrupt_pending.clear()
                    self._set_capture(False)
                    self._overflow.clear()
                    connected.set()
                    sender = threading.Thread(target=self._send_audio, args=(connected,), daemon=True)
                    sender.start()
                    self._send({"type": "start", "sample_rate": 16000, "channels": 1,
                                "encoding": "pcm_s16le", "frame_ms": 100,
                                "wake_source": self.wake_source,
                                "interrupt_source": self.interrupt_source,
                                "connection_id": self._connection_id})
                    with sd.RawInputStream(samplerate=16000, channels=1, dtype="int16",
                                           blocksize=1600, device=self._device("input_device"), callback=self._callback):
                        while connected.is_set() and not self._stop.is_set():
                            if self._overflow.is_set():
                                raise RuntimeError("Microphone queue overflow; restarting voice session")
                            try:
                                event = json.loads(ws.recv(timeout=0.2))
                            except TimeoutError:
                                continue
                            self._handle_voice_event(event)
            except Exception as exc:
                self.get_logger().error(f"Realtime voice connection stopped: {exc}")
                self._show("[error] 语音连接中断")
            finally:
                with self._lock:
                    self._wake_phase = "disconnected"
                    self._connection_id = ""
                    if self._hardware is not None:
                        self._hardware.state["gateway_phase"] = "disconnected"
                self._set_capture(False)
                connected.clear()
                if sender:
                    sender.join(timeout=2)
                self._socket = None
                self._advance()
                while not self._audio.empty():
                    try:
                        self._audio.get_nowait()
                    except queue.Empty:
                        break
            self._stop.wait(2)

    def _prompt(self, interrupted, generation, cancel):
        import av
        import sounddevice as sd
        path = "/api/voice/interrupt-acknowledgement" if interrupted else "/api/voice/wakeup-acknowledgement"
        output = None
        try:
            response = httpx.get(self.base + path, timeout=10, trust_env=False)
            response.raise_for_status()
            with self._output_lock:
                if not self._current(generation, cancel):
                    return
                with sd.RawOutputStream(samplerate=24000, channels=1, dtype="int16", device=self._device("output_device")) as output:
                    self._output = output
                    with av.open(io.BytesIO(response.content)) as container:
                        resampler = av.AudioResampler(format="s16", layout="mono", rate=24000)
                        for frame in container.decode(audio=0):
                            if not self._current(generation, cancel):
                                return
                            for pcm in resampler.resample(frame):
                                output.write(pcm.to_ndarray().tobytes())
                        for pcm in resampler.resample(None):
                            output.write(pcm.to_ndarray().tobytes())
                    output.stop()  # PortAudio drains before acknowledging the prompt.
                    self._output = None
            if self._current(generation, cancel):
                self._finish_turn("prompt_finished", generation, cancel, played=True)
        except Exception as exc:
            if self._current(generation, cancel):
                self.get_logger().error(f"Wake prompt playback failed: {exc}")
                if self._socket:
                    self._socket.close()
        finally:
            with self._lock:
                if self._output is output:
                    self._output = None

    def _reply(self, event, generation, cancel):
        import sounddevice as sd
        request_id = uuid.uuid4().hex
        text = str(event.get("text", ""))
        self._show("[query] " + text)
        terminal, closing, started, ignored = False, False, False, False
        reply_prefix, tone = "", "neutral"
        output = response = None
        motion_reason = "playback_failed"
        try:
            with self._output_lock:
                if not self._current(generation, cancel):
                    return
                with httpx.Client(timeout=120, trust_env=False) as client, sd.RawOutputStream(
                    samplerate=24000, channels=1, dtype="int16", device=self._device("output_device")
                ) as output:
                    self._output = output
                    with client.stream("POST", self.base + "/api/answer/stream", json={
                        "query": text, "audio": True, "retrieval_token": event.get("retrieval_token"),
                        "conversation_id": self.conversation_id,
                    }) as response:
                        self._response = response
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if not self._current(generation, cancel):
                                return
                            if not line:
                                continue
                            item = json.loads(line)
                            if item["type"] in {"meta", "final", "ignored", "error", "tool_result"}:
                                result = item.get("result", {})
                                trace_voice("gateway.reply_event", generation=generation,
                                            request_id=request_id, query=text, kind=item["type"],
                                            answer=result.get("answer"),
                                            retrieval={key: value for key, value in
                                                       (item.get("retrieval", result.get("retrieval")) or {}).items()
                                                       if key in {"reason", "turn_control", "response_id"}},
                                            tool_result=result if item["type"] == "tool_result" else None)
                            if item["type"] == "meta":
                                closing = item.get("retrieval", {}).get("reason") == "end_session"
                            elif item["type"] == "emotion":
                                tone = str(item.get("tone", "neutral"))
                                self._show("[tone] " + tone)
                            elif item["type"] == "audio":
                                if (item.get("sample_rate") != 24000 or item.get("channels") != 1
                                        or item.get("format") != "pcm16"):
                                    raise RuntimeError("Unexpected reply audio format")
                                pcm = base64.b64decode(item["audio"], validate=True)
                                if not pcm or len(pcm) % 2:
                                    raise RuntimeError("Malformed reply audio")
                                output.write(pcm)
                                with self._send_lock:
                                    if not self._current(generation, cancel) or self._interrupt_pending.is_set():
                                        return
                                    if not started:
                                        started = True
                                        self._feedback(request_id, generation, "started")
                                        purpose = "closing_gesture" if closing else select_answer_gesture(text, reply_prefix, tone)
                                        trace_voice("gateway.answer_gesture", generation=generation,
                                                    request_id=request_id, purpose=purpose)
                                        self._motion_client.start_playback(request_id, generation, purpose, self.session_id)
                            elif item["type"] == "delta":
                                reply_prefix = (reply_prefix + str(item.get("text", "")))[:1200]
                                self._show("[delta] " + item.get("text", ""))
                            elif item["type"] in {"final", "ignored"}:
                                terminal = True
                                ignored = item["type"] == "ignored"
                                if item["type"] == "final":
                                    closing = item["result"].get("retrieval", {}).get("reason") == "end_session" or closing
                                    self._show("[answer] " + item["result"].get("answer", ""))
                            elif item["type"] == "error":
                                raise RuntimeError(item.get("detail", "Reply failed"))
                    self._response = None
                    if not terminal:
                        raise RuntimeError("Reply stream ended without final result")
                    if not ignored and not started:
                        raise RuntimeError("Reply stream completed without audio")
                    output.stop()
                    motion_reason = "closing_completed" if closing else "playback_completed"
                    self._motion_client.finish_playback(request_id, generation, self.session_id, motion_reason)
                    self._output = None
            if self._current(generation, cancel):
                self._feedback(request_id, generation, "completed")
                self.get_logger().info("Reply drained: end_session=" + str(closing)
                                       + ", next=" + ("armed" if closing else "listening"))
                self._finish_turn("resume" if closing else "continue", generation, cancel, played=started)
        except Exception as exc:
            self._motion_client.finish_playback(request_id, generation, self.session_id, "playback_failed")
            if self._current(generation, cancel):
                self.get_logger().error(f"Reply failed: {exc}")
                self._feedback(request_id, generation, "failed")
                self._finish_turn("resume", generation, cancel, played=started)
        finally:
            self._motion_client.finish_playback(request_id, generation, self.session_id,
                                                "interrupted" if cancel.is_set() else motion_reason)
            with self._lock:
                if self._response is response:
                    self._response = None
                if self._output is output:
                    self._output = None

    def destroy_node(self):
        self._stop.set()
        self._set_capture(False)
        if self._hardware is not None:
            self._hardware.close()
        self._motion_client.close()
        self._advance()
        if self._socket:
            self._socket.close()
        self._workers.shutdown(wait=False, cancel_futures=True)
        self._thread.join(timeout=3)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RealtimeVoiceGateway()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
