"""WHEELTEC serial wake events; USB PCM and the existing turn controller stay separate.

Linux stdlib only. This adapter never requests serial audio, starts ASR/TTS, or
reconfigures wake words on reconnect. Provisioning is an explicit CLI operation.
"""
from __future__ import annotations

from rag_system.voice_diagnostics import trace_voice

import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import select
import secrets
import struct
import threading
import time
from typing import Callable

if os.name == "posix":
    import fcntl
    import termios

DEFAULT_PORT = "/dev/serial/by-id/usb-WCH.CN_USB_Single_Serial_0004-if00"
ACK = b"\xa5\x00\x00\x00"


def encode_frame(kind: int, payload: bytes, message_id: int) -> bytes:
    if len(payload) > 10232 or not 0 <= message_id <= 65535:
        raise ValueError("Invalid serial frame")
    data = struct.pack("<BBBHH", 0xA5, 1, kind, len(payload), message_id) + payload
    return data + bytes([(-sum(data)) & 255])


class FrameParser:
    def __init__(self):
        self.buffer = bytearray()
        self.noise_bytes = self.bad_lengths = self.bad_checksums = 0

    def feed(self, data: bytes) -> list[tuple[int, int, bytes]]:
        self.buffer.extend(data)
        frames = []
        while len(self.buffer) >= 8:
            if self.buffer[:2] != b"\xa5\x01":
                self.noise_bytes += 1
                del self.buffer[0]
                continue
            _, _, kind, size, ident = struct.unpack("<BBBHH", self.buffer[:7])
            if size > 10232:
                self.bad_lengths += 1
                del self.buffer[0]
                continue
            if len(self.buffer) < size + 8:
                break
            packet = bytes(self.buffer[:size + 8])
            if sum(packet) & 255:
                self.bad_checksums += 1
                del self.buffer[0]
                continue
            del self.buffer[:size + 8]
            frames.append((kind, ident, packet[7:-1]))
        return frames


def decode_message(payload: bytes) -> dict | None:
    try:
        message = json.loads(payload.decode("utf-8"))
        return message if isinstance(message, dict) else None
    except (ValueError, UnicodeError):
        return None


def wake_metadata(message: dict) -> dict | None:
    if message.get("type") != "aiui_event" or not isinstance(message.get("content"), dict):
        return None
    content = message["content"]
    # The attached v1.0.2 uses eventType=4 and JSON-encoded info.ivw.
    # Newer documented firmware reports beam/angle/confidence directly.
    if "eventType" in content:
        if content["eventType"] not in (4, "4"):
            return None
        info = content.get("info", {})
        if isinstance(info, str):
            info = decode_message(info.encode()) or {}
        values = info.get("ivw", {}) if isinstance(info, dict) else {}
    elif "angle" in content and ("beam" in content or "confidence" in content):
        values = content
    else:
        return None
    if not isinstance(values, dict):
        values = {}
    result = {}
    for name in ("angle", "beam", "confidence"):
        value = values.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            result[name] = value
    return result


class SerialPort:
    def __init__(self, path: str):
        self.path = path
        self.fd: int | None = None
        self.parser = FrameParser()
        self.message_id = secrets.randbelow(65536)

    def __enter__(self):
        if os.name != "posix":
            raise OSError("Hardware wake serial requires Linux")
        self.fd = os.open(self.path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.ioctl(self.fd, termios.TIOCEXCL)
            attrs = termios.tcgetattr(self.fd)
            attrs[0] = attrs[1] = attrs[3] = 0
            attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
            attrs[4] = attrs[5] = termios.B115200
            attrs[6][termios.VMIN] = attrs[6][termios.VTIME] = 0
            termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
            termios.tcflush(self.fd, termios.TCIFLUSH)
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *_args):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def send(self, kind: int, payload: bytes, ident: int):
        data = memoryview(encode_frame(kind, payload, ident))
        deadline = time.monotonic() + 1
        while data:
            if time.monotonic() >= deadline:
                raise TimeoutError("Serial write timed out")
            if select.select([], [self.fd], [], .1)[1]:
                try:
                    data = data[os.write(self.fd, data):]
                except BlockingIOError:
                    continue

    def command(self, message: dict, ident: int):
        self.send(5, json.dumps(message, ensure_ascii=False).encode(), ident)

    def receive(self, timeout: float = .2) -> list[tuple[int, int, bytes]]:
        if not select.select([self.fd], [], [], timeout)[0]:
            return []
        try:
            data = os.read(self.fd, 4096)
        except BlockingIOError:
            return []
        if not data:
            raise ConnectionError("Microphone serial disconnected")
        frames = self.parser.feed(data)
        trace_voice("serial.read", byte_count=len(data), frame_count=len(frames),
                    buffered_bytes=len(self.parser.buffer), noise_bytes=self.parser.noise_bytes,
                    bad_lengths=self.parser.bad_lengths, bad_checksums=self.parser.bad_checksums)
        acknowledged = set()
        for kind, ident, _ in frames:
            if kind == 1:
                if ident not in acknowledged:
                    self.send(255, ACK, ident)
                    acknowledged.add(ident)
        return frames


def write_status(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(state, updated_at=time.time()), ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


class SerialWakeSource:
    def __init__(self, port: str, callback: Callable[[dict], None], *, keyword: str,
                 status_path: Path, report: Callable[[str], None] = print, cooldown_ms: int = 800):
        self.port, self.callback, self.keyword = port, callback, keyword
        self.status_path, self.report = status_path, report
        self.cooldown = max(.1, cooldown_ms / 1000)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="hardware-wake")
        self.state = {"source": "hardware", "port": port, "keyword": keyword, "connected": False,
                      "version": "", "wake_count": 0, "last_wake_at": None, "error": "Starting",
                      "gateway_phase": "disconnected"}
        self._sequence = 0

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)

    def _run(self):
        while not self.stop.is_set():
            try:
                with SerialPort(self.port) as serial:
                    seen = deque(maxlen=128)
                    last_wake = -float("inf")
                    queried = 0.
                    queries = set()
                    opened = time.monotonic()
                    heartbeat = 0.
                    trace_voice("serial.open", port=self.port)
                    self.state.update(connected=False, error="Waiting for device version")
                    while not self.stop.is_set():
                        now = time.monotonic()
                        if not self.state["connected"] and now - queried > 1:
                            serial.message_id = (serial.message_id + 1) % 65536
                            queries.add(serial.message_id)
                            serial.command({"type": "version"}, serial.message_id)
                            queried = now
                        if not self.state["connected"] and now - opened > 6:
                            raise TimeoutError("Microphone version handshake timed out")
                        if now - heartbeat >= 2:
                            trace_voice("serial.heartbeat", phase=self.state["gateway_phase"],
                                        connected=self.state["connected"], sequence=self._sequence)
                            heartbeat = now
                        for kind, ident, payload in serial.receive():
                            # Raw JSON frames precede kind/semantic filtering, dedup and cooldown.
                            raw_message = decode_message(payload) if kind in {4, 5} else None
                            trace_voice("serial.frame", frame_id=ident, kind=kind, size=len(payload),
                                        phase=self.state["gateway_phase"], message=raw_message,
                                        payload_hex=payload.hex() if raw_message is not None else None)
                            if kind not in {4, 5}:
                                continue
                            message = decode_message(payload)
                            if message is None:
                                continue
                            version = message.get("content")
                            if (message.get("type") in {"version", "started"} or ident in queries) and message.get("code") == 0 and isinstance(version, str) and "version" in version.lower():
                                self.state.update(connected=True, version=version, error="")
                                self.report("Hardware wake microphone connected: " + version)
                                continue
                            metadata = wake_metadata(message)
                            key = (ident, payload)
                            reason = ("not_wake" if metadata is None else
                                      "not_connected" if not self.state["connected"] else
                                      "duplicate" if key in seen else None)
                            trace_voice("serial.wake_decision", frame_id=ident,
                                        phase=self.state["gateway_phase"], metadata=metadata,
                                        decision=reason or "cooldown_check")
                            if reason is not None:
                                continue
                            seen.append(key)
                            now = time.monotonic()
                            if now - last_wake < self.cooldown:
                                trace_voice("serial.drop", frame_id=ident, reason="cooldown",
                                            elapsed_ms=(now - last_wake) * 1000,
                                            cooldown_ms=self.cooldown * 1000,
                                            phase=self.state["gateway_phase"])
                                continue
                            last_wake = now
                            self._sequence += 1
                            self.state["wake_count"] += 1
                            self.state["last_wake_at"] = time.time()
                            trace_voice("serial.accept", frame_id=ident, sequence=self._sequence,
                                        phase=self.state["gateway_phase"], metadata=metadata)
                            self.callback(dict(metadata, sequence=self._sequence))
                        write_status(self.status_path, self.state)
            except Exception as exc:
                trace_voice("serial.error", error=str(exc))
                self.state.update(connected=False, error=str(exc))
                self.report("Hardware wake unavailable: " + str(exc))
            finally:
                self.state["connected"] = False
                write_status(self.status_path, self.state)
            self.stop.wait(2)


@dataclass
class WakeEvent:
    keyword: str


class HardwareWakeSession:
    """Connection-local event gate; PCM cannot wake this session."""
    def __init__(self, keyword: str, connection_id: str, cooldown_ms: int = 800):
        self.keyword, self.connection_id = keyword, connection_id
        self.cooldown = cooldown_ms / 1000
        self.sequence = 0
        self.last_wake = -float("inf")

    def accept_pcm16(self, _pcm: bytes):
        return None

    def reset(self):
        pass  # Keep replay protection across ASR turns and prompt playback.

    def accept_control(self, command: dict) -> WakeEvent | None:
        sequence = command.get("sequence")
        reason = ("wrong_type" if command.get("type") != "hardware_wake" else
                  "wrong_connection" if command.get("connection_id") != self.connection_id else
                  "invalid_sequence" if type(sequence) is not int else
                  "replay" if sequence <= self.sequence else None)
        trace_voice("backend.wake_received", sequence=sequence, connection_id=self.connection_id,
                    previous_sequence=self.sequence, decision=reason or "cooldown_check")
        if reason is not None:
            return None
        self.sequence = sequence
        now = time.monotonic()
        if now - self.last_wake < self.cooldown:
            trace_voice("backend.wake_drop", sequence=sequence, connection_id=self.connection_id,
                        reason="cooldown", elapsed_ms=(now - self.last_wake) * 1000,
                        cooldown_ms=self.cooldown * 1000)
            return None
        trace_voice("backend.wake_accept", sequence=sequence, connection_id=self.connection_id)
        self.last_wake = now
        return WakeEvent(self.keyword)


class HardwareWakeEngine:
    def __init__(self, keyword: str, connection_id: str, cooldown_ms: int):
        self.keyword, self.connection_id, self.cooldown_ms = keyword, connection_id, cooldown_ms

    def create_session(self):
        return HardwareWakeSession(self.keyword, self.connection_id, self.cooldown_ms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["version", "configure"])
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--keyword", default="小绿小绿")
    parser.add_argument("--threshold", type=int, default=900)
    args = parser.parse_args()
    if not 500 <= args.threshold <= 1500 or not 1 <= len(args.keyword) <= 30:
        parser.error("Invalid keyword or threshold")
    request = {"type": "version"} if args.action == "version" else {
        "type": "wakeup_keywords", "content": {"keyword": args.keyword, "threshold": str(args.threshold)}}
    with SerialPort(args.port) as serial:
        request_id = serial.message_id
        serial.command(request, request_id)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            for kind, ident, payload in serial.receive():
                if kind == 1:
                    # Some firmware ignores controls sent before the handshake.
                    serial.command(request, request_id)
                message = decode_message(payload) if kind in {4, 5} else None
                if message:
                    print(json.dumps({"kind": kind, "id": ident, "message": message}, ensure_ascii=False), flush=True)
                    if (message.get("type") == request["type"] or message.get("type") is None and ident == request_id) and "code" in message:
                        raise SystemExit(0 if message["code"] == 0 else 1)
        raise SystemExit("No semantic acknowledgement; transport ACK alone is insufficient")


if __name__ == "__main__":
    main()
