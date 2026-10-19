"""Robot microphone/speaker adapter for the same streaming protocol as the Web UI.

Run the local rag_system.web backend first. No vendor TextToAudio reply is used.
"""
from __future__ import annotations

import base64
import io
import json
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

import httpx
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rag_interfaces.msg import VoiceControl, VoicePlayback


class RealtimeVoiceGateway(Node):
    def __init__(self):
        super().__init__("realtime_voice_gateway")
        self.declare_parameter("backend_url", "http://127.0.0.1:8000")
        self.declare_parameter("input_device", "")
        self.declare_parameter("output_device", "")
        self.declare_parameter("session_id", "robot")
        self.base = str(self.get_parameter("backend_url").value).rstrip("/")
        self.session_id = str(self.get_parameter("session_id").value)
        self.conversation_id = "robot-" + uuid.uuid4().hex
        self.display = self.create_publisher(String, "/voice/display_text", 20)
        self.playback = self.create_publisher(VoicePlayback, "/voice/playback", 20)
        self.motion = self.create_publisher(VoiceControl, "/voice/arm_motion_request", 20)
        self.control = self.create_publisher(VoiceControl, "/voice/control", 20)
        self._stop = threading.Event()
        self._capture = threading.Event()
        self._cancel = threading.Event()
        self._overflow = threading.Event()
        self._audio = queue.Queue(maxsize=50)
        self._lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._output_lock = threading.Lock()
        self._generation = time.time_ns() // 1000
        self._socket = None
        self._output = None
        self._response = None
        self._workers = ThreadPoolExecutor(max_workers=2, thread_name_prefix="robot-reply")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _device(self, name):
        value = str(self.get_parameter(name).value).strip()
        return int(value) if value.isdigit() else (value or None)

    def _send(self, payload):
        with self._send_lock:
            if self._socket is not None:
                self._socket.send(payload if isinstance(payload, bytes) else json.dumps(payload))

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
        with self._lock:
            self._cancel.set()
            response, output = self._response, self._output
            self._generation += 1
            self._cancel = threading.Event()
            generation, cancel = self._generation, self._cancel
        # Closing a socket/device may wait for its worker's finally block.
        # Never hold the state lock while doing either operation.
        if response is not None:
            with suppress(Exception):
                response.close()
        if output is not None:
            with suppress(Exception):
                output.abort()
        return generation, cancel

    def _gesture(self, generation, purpose, interrupt=False):
        message = VoiceControl()
        message.header.stamp = self.get_clock().now().to_msg()
        message.request_id, message.session_id = uuid.uuid4().hex, self.session_id
        message.generation, message.purpose = generation, purpose
        message.command = "interrupt" if interrupt else "gesture"
        (self.control if interrupt else self.motion).publish(message)

    def _current(self, generation, cancel):
        return not cancel.is_set() and not self._stop.is_set() and generation == self._generation

    def _callback(self, data, frames, timing, status):
        if self._capture.is_set():
            try:
                self._audio.put_nowait(bytes(data))
            except queue.Full:
                self._overflow.set()

    def _send_audio(self, connected):
        while connected.is_set() and not self._stop.is_set():
            try:
                pcm = self._audio.get(timeout=0.1)
            except queue.Empty:
                continue
            if self._capture.is_set():
                try:
                    self._send(pcm)
                except Exception:
                    connected.clear()

    def _run(self):
        import sounddevice as sd
        from websockets.sync.client import connect
        while not self._stop.is_set():
            connected = threading.Event()
            sender = None
            try:
                ws_url = self.base.replace("http://", "ws://", 1).replace("https://", "wss://", 1)
                with connect(ws_url + "/api/voice/session/stream", proxy=None, close_timeout=2) as ws:
                    self._socket = ws
                    self._overflow.clear()
                    connected.set()
                    sender = threading.Thread(target=self._send_audio, args=(connected,), daemon=True)
                    sender.start()
                    self._send({"type": "start", "sample_rate": 16000, "channels": 1,
                                "encoding": "pcm_s16le", "frame_ms": 100})
                    with sd.RawInputStream(samplerate=16000, channels=1, dtype="int16",
                                           blocksize=1600, device=self._device("input_device"), callback=self._callback):
                        while connected.is_set() and not self._stop.is_set():
                            if self._overflow.is_set():
                                raise RuntimeError("Microphone queue overflow; restarting voice session")
                            try:
                                event = json.loads(ws.recv(timeout=0.2))
                            except TimeoutError:
                                continue
                            kind = event.get("type")
                            if kind in {"armed", "listening"}:
                                self._capture.set()
                            elif kind == "wake_detected":
                                self._capture.clear()
                                generation, cancel = self._advance()
                                interrupted = event.get("reason") == "interrupt"
                                self._gesture(generation, "interrupt" if interrupted else "wake_wave", interrupted)
                                self._workers.submit(self._prompt, event.get("reason") == "interrupt", generation, cancel)
                            elif kind == "partial":
                                self._show("[query] " + event.get("text", ""))
                            elif kind == "final":
                                generation, cancel = self._advance()
                                self._workers.submit(self._reply, event, generation, cancel)
                            elif kind == "no_speech":
                                self._send({"type": "resume"})
                            elif kind == "error":
                                raise RuntimeError(str(event.get("detail", "Voice backend error")))
            except Exception as exc:
                self.get_logger().error(f"Realtime voice connection stopped: {exc}")
                self._show("[error] 语音连接中断")
            finally:
                self._capture.clear()
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
                self._send({"type": "prompt_finished"})
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
        output = response = None
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
                            if item["type"] == "meta":
                                closing = item.get("retrieval", {}).get("reason") == "end_session"
                            elif item["type"] == "audio":
                                if (item.get("sample_rate") != 24000 or item.get("channels") != 1
                                        or item.get("format") != "pcm16"):
                                    raise RuntimeError("Unexpected reply audio format")
                                pcm = base64.b64decode(item["audio"], validate=True)
                                if not pcm or len(pcm) % 2:
                                    raise RuntimeError("Malformed reply audio")
                                output.write(pcm)
                                if not started:
                                    started = True
                                    self._feedback(request_id, generation, "started")
                                    self._gesture(generation, "answer_gesture")
                            elif item["type"] == "delta":
                                self._show("[delta] " + item.get("text", ""))
                            elif item["type"] in {"final", "ignored"}:
                                terminal = True
                                ignored = item["type"] == "ignored"
                                if item["type"] == "final":
                                    self._show("[answer] " + item["result"].get("answer", ""))
                            elif item["type"] == "error":
                                raise RuntimeError(item.get("detail", "Reply failed"))
                    self._response = None
                    if not terminal:
                        raise RuntimeError("Reply stream ended without final result")
                    if not ignored and not started:
                        raise RuntimeError("Reply stream completed without audio")
                    output.stop()
                    self._output = None
            if self._current(generation, cancel):
                self._feedback(request_id, generation, "completed")
                self._send({"type": "resume" if closing else "continue"})
        except Exception as exc:
            if self._current(generation, cancel):
                self.get_logger().error(f"Reply failed: {exc}")
                self._feedback(request_id, generation, "failed")
                self._send({"type": "resume"})
        finally:
            with self._lock:
                if self._response is response:
                    self._response = None
                if self._output is output:
                    self._output = None

    def destroy_node(self):
        self._stop.set()
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
