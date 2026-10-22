"""Bounded best-effort transport to the native SDK process; no retries or queue."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import uuid
import hashlib
import json
from collections import OrderedDict
import os
import re
import random
import socket
import threading
import time
from pathlib import Path
from typing import Callable

from rag_system.voice_diagnostics import trace_voice
from rag_system.motion_presets import load_catalog

PURPOSE_PRESETS = {"wake_wave": "wave_goodbye", "closing_wave": "wave_goodbye"}
ANSWER_PURPOSES = frozenset({"answer_gesture", "answer_invite", "answer_open_palms"})
ANSWER_EXCLUDED_PRESETS = frozenset({"fly_kiss", "heart", "wave_goodbye"})
CLOSING_PRESETS = ("wave_goodbye", "heart", "fly_kiss")
PURPOSES = frozenset({*PURPOSE_PRESETS, *ANSWER_PURPOSES, "closing_gesture"})


def select_motion_preset(purpose: str, previous: str | None = None) -> str:
    """Legacy answer purposes all share the current eligible custom catalog."""
    if purpose in ANSWER_PURPOSES:
        choices = tuple(preset["id"] for preset in load_catalog()["presets"]
                        if (preset["type"] == "sequence" or preset["id"] == "speech")
                        and preset["id"] not in ANSWER_EXCLUDED_PRESETS)
        if not choices:
            raise ValueError("No eligible custom answer motions configured")
        alternatives = tuple(choice for choice in choices if choice != previous)
        return random.choice(alternatives or choices)
    if purpose == "closing_gesture":
        return random.choice(CLOSING_PRESETS)
    return PURPOSE_PRESETS[purpose]


def motion_socket_path() -> str:
    runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return os.environ.get("GREENRIVER_MOTION_SOCKET", str(Path(runtime) / "greenriver-motion.sock"))


def exchange(command: str, *, path: str | None = None, timeout: float = 0.3) -> dict:
    if "\n" in command or len(command) > 300:
        raise ValueError("Invalid motion command")
    deadline = time.monotonic() + timeout
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
        stream.settimeout(timeout)
        stream.connect(path or motion_socket_path())
        stream.sendall((command + "\n").encode("ascii"))
        data = bytearray()
        while b"\n" not in data:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Motion acknowledgement timed out")
            stream.settimeout(remaining)
            part = stream.recv(1024)
            if not part:
                raise ConnectionError("Motion worker closed without acknowledgement")
            data.extend(part)
            if len(data) > 4096:
                raise ValueError("Oversized motion acknowledgement")
        result = json.loads(data.split(b"\n", 1)[0])
        if not isinstance(result, dict) or not isinstance(result.get("status"), str):
            raise ValueError("Invalid motion acknowledgement")
        return result


def request_home(request_id: str, *, path: str | None = None) -> dict:
    if not request_id or len(request_id) > 128:
        raise ValueError("Invalid home request id")
    token = hashlib.sha256(request_id.encode()).hexdigest()
    return exchange(f"HOME {token}", path=path)


def request_power(request_id: str, power: str, *, path: str | None = None) -> dict:
    if not request_id or len(request_id) > 128 or power not in {"on", "off"}:
        raise ValueError("Invalid arm power request")
    token = hashlib.sha256(f"power:{power}:{request_id}".encode()).hexdigest()
    return exchange(f"POWER {token} {power}", path=path)


def request_all_power(request_id: str, power: str, *, path: str | None = None) -> dict:
    if not request_id or len(request_id)>128 or power not in {"on","off"}:
        raise ValueError("Invalid all-power request")
    token=hashlib.sha256(f"all-power:{power}:{request_id}".encode()).hexdigest()
    return exchange(f"ALLPOWER {token} {power}",path=path)


def request_arm_restart(request_id: str, *, path: str | None = None) -> dict:
    if not request_id or len(request_id) > 128:
        raise ValueError("Invalid arm restart request")
    token = hashlib.sha256(f"power:restart:{request_id}".encode()).hexdigest()
    return exchange(f"POWER {token} restart", path=path)


def request_preset(request_id: str, preset_id: str, *, path: str | None = None) -> dict:
    if (not request_id or len(request_id) > 128
            or not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", preset_id)):
        raise ValueError("Invalid preset request")
    token = hashlib.sha256(f"preset:{preset_id}:{request_id}".encode()).hexdigest()
    return exchange(f"PRESET {token} {preset_id}", path=path)


def request_result(token: str, *, instance_id: str | None = None, path: str | None = None) -> dict:
    """Read a retained result; never execute or retry a motion."""
    for value in (token, instance_id):
        if value is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
            raise ValueError("Invalid motion request token or instance id")
    command = f"RESULT {token}" + (f" {instance_id}" if instance_id else "")
    return exchange(command, path=path)


@dataclass
class PlaybackMotion:
    request_id: str
    generation: int
    purpose: str
    session_id: str
    stop: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    reason: str = "playback_ended"
    finish_current: bool = False


class MotionClient:
    def __init__(self, *, path: str | None = None, report: Callable[[str], None] | None = None,
                 on_result: Callable[[dict], None] | None = None):
        self.path = path
        self.report = report or (lambda _message: None)
        self.on_result = on_result or (lambda _result: None)
        self._lock = threading.Lock()
        self._busy = False
        self._closed = False
        self._generation = 0
        self._wake_token: str | None = None
        self._playbacks: dict[tuple[str, int, str], PlaybackMotion] = {}
        self._thread: threading.Thread | None = None
        self._observer: threading.Thread | None = None
        self._stop = threading.Event()
        self._inflight: set[str] = set()
        self._results: OrderedDict[str, dict] = OrderedDict()
        self._watching: OrderedDict[str, tuple[str | None, float]] = OrderedDict()

    def _wait_for_wake(self, generation: int) -> bool:
        deadline = time.monotonic() + 20
        while True:
            with self._lock:
                if self._closed or generation < self._generation:
                    return False
                wake_token = self._wake_token
            if wake_token is None:
                return True
            state = exchange("STATUS", path=self.path)
            if state.get("state") in {"fault", "powered_off", "initializing"}:
                self.report(f"Answer motion skipped: {state.get('detail') or state.get('state')}")
                return False
            if state.get("request_id") != wake_token or state.get("state") == "ready":
                return True
            if time.monotonic() >= deadline:
                self.report("Answer motion skipped: wake motion did not finish within 20 seconds")
                return False
            time.sleep(.1)

    def advance(self, generation: int) -> None:
        # Legacy/manual requests remain tracked; playback scopes finish safely.
        with self._lock:
            self._generation = max(self._generation, generation)
            for playback in self._playbacks.values():
                if playback.generation < generation:
                    playback.reason = "interrupted"
                    playback.stop.set()

    def start_playback(self, request_id: str, generation: int, purpose: str, session_id: str = "robot") -> bool:
        """Start a serial random gesture loop scoped to real audio playback."""
        if purpose not in PURPOSES or not request_id or generation <= 0:
            return False
        key = (session_id, generation, request_id)
        with self._lock:
            if self._closed or generation < self._generation or key in self._playbacks:
                return False
            # Bound worker count during repeated interrupts and slow safe returns.
            if len(self._playbacks) >= 8:
                self.report("Playback motion skipped: too many finishing scopes")
                return False
            self._generation = generation
            for old in self._playbacks.values():
                if not old.stop.is_set():
                    old.reason = "superseded"
                    old.stop.set()
            playback = PlaybackMotion(request_id, generation, purpose, session_id)
            self._playbacks[key] = playback
            playback.thread = threading.Thread(target=self._playback_loop, args=(key, playback),
                                               daemon=True, name="sdk-playback-motion")
            playback.thread.start()
        return True

    def finish_playback(self, request_id: str, generation: int, session_id: str = "robot",
                        reason: str = "playback_ended") -> None:
        # Only signal here: socket I/O and mechanical return never block audio/control.
        with self._lock:
            playback = self._playbacks.get((session_id, generation, request_id))
            if playback is not None:
                # A normal farewell drain allows its one gesture to finish. A late
                # end_session classification also disables repetition of an answer gesture.
                if reason == "closing_completed" or (playback.purpose in {"closing_gesture", "closing_wave"}
                        and reason in {"playback_completed", "playback_ended"}):
                    playback.finish_current = True
                    return
                playback.reason = reason
                playback.stop.set()

    def _playback_loop(self, key: tuple[str, int, str], playback: PlaybackMotion) -> None:
        scope = instance = token = None
        ended = False
        previous = None
        renew_at = 0.
        deadline = time.monotonic() + 305
        try:
            if playback.stop.is_set():
                return
            state = exchange("STATUS", path=self.path)
            instance = state["instance_id"]
            if playback.stop.is_set():
                return
            opened = exchange(f"OPEN {instance}", path=self.path)
            if opened["status"] != "accepted" or opened.get("instance_id") != instance:
                self.report(f"Playback motion scope unavailable: {opened['status']}")
                return
            scope = opened["scope_id"]
            trace_voice("motion.playback_open", request_id=playback.request_id,
                        generation=playback.generation, scope_id=scope, instance_id=instance)
            while True:
                if playback.stop.is_set() and not ended:
                    result = exchange(f"END {scope} {instance}", path=self.path)
                    ended = True
                    trace_voice("motion.playback_end", request_id=playback.request_id,
                                generation=playback.generation, scope_id=scope, reason=playback.reason,
                                status=result["status"])
                    if result["status"] == "instance_changed":
                        break
                if ended and token is None:
                    break
                if not ended and time.monotonic() >= renew_at:
                    renewed = exchange(f"KEEP {scope} {instance}", path=self.path)
                    if renewed["status"] != "ok":
                        break
                    renew_at = time.monotonic() + 1
                if token is not None:
                    state = request_result(token, instance_id=instance, path=self.path)
                    native = state.get("result")
                    if state["status"] != "ok" or not isinstance(native, dict):
                        self._record(token, result_state="unknown", reason=state["status"])
                        break
                    self._record(token, result_state=native["result_state"], native_result=native)
                    terminal = native["result_state"]
                    if terminal == "verified_completed":
                        with self._lock:
                            self._inflight.discard(token)
                        token = None
                        if playback.purpose not in ANSWER_PURPOSES or playback.finish_current:
                            break  # A farewell finishes once, even after audio has drained.
                    elif terminal in {"failed", "unknown", "cancelled_home"}:
                        break
                elif not ended:
                    if playback.finish_current and previous is not None:
                        break
                    state = exchange("STATUS", path=self.path)
                    if state.get("instance_id") != instance or state.get("state") in {"fault", "powered_off"}:
                        break
                    if state.get("state") == "ready" and state.get("at_preset"):
                        preset = select_motion_preset(playback.purpose, previous)
                        if playback.stop.is_set():
                            continue
                        if playback.finish_current and previous is not None:
                            break
                        token = uuid.uuid4().hex
                        with self._lock:
                            while len(self._results) >= 128:
                                removable = next((item for item in self._results if item not in self._watching
                                                  and item not in self._inflight), None)
                                if removable is None:
                                    raise ValueError("Motion result capacity exhausted")
                                self._results.pop(removable)
                            self._inflight.add(token)
                            self._results[token] = dict(request_id=playback.request_id, token=token,
                                session_id=playback.session_id, generation=playback.generation,
                                purpose=playback.purpose, preset=preset, scope_id=scope,
                                result_state="sending", instance_id=instance)
                        ack = exchange(f"GESTURE {token} {preset} {scope} {instance}", path=self.path)
                        if ack["status"] in {"accepted", "duplicate"}:
                            self._record(token, result_state="accepted", native_result=ack.get("result"))
                            previous = preset
                            deadline = time.monotonic() + 305
                            trace_voice("motion.gesture", request_id=token, playback_request_id=playback.request_id,
                                        generation=playback.generation, scope_id=scope, preset=preset,
                                        purpose=playback.purpose, status=ack["status"])
                        else:
                            self._record(token, result_state="skipped", reason=ack["status"])
                            with self._lock:
                                self._inflight.discard(token)
                            token = None
                            if ack["status"] not in {"busy", "not_at_home"}:
                                break
                if time.monotonic() > deadline and token is not None:
                    self._record(token, result_state="unknown", reason="result_observation_timeout")
                    break
                # END wakes an active loop immediately. After END, poll until safe return is verified.
                if ended:
                    time.sleep(.1)
                else:
                    playback.stop.wait(.1)
                with self._lock:
                    finished = [item for item in self._inflight if item in self._results
                                and self._results[item]["result_state"] in
                                {"verified_completed", "cancelled_home", "failed", "unknown", "skipped"}]
                    self._inflight.difference_update(finished)
        except (OSError, ValueError, KeyError) as exc:
            if token is not None:
                self._record(token, result_state="unknown", reason=str(exc))
                # Lost ACK is resolved read-only. Never submit another physical gesture.
                self._watch(token, instance)
            self.report(f"Playback motion stopped/unknown: {exc}")
        finally:
            if token is not None:
                with self._lock:
                    current = self._results.get(token, {}).get("result_state")
                if current in {"sending", "accepted", "running"}:
                    self._record(token, result_state="unknown", reason="playback_scope_lost")
                    self._watch(token, instance)
            if scope is not None and instance is not None and not ended:
                try:
                    exchange(f"END {scope} {instance}", path=self.path)
                except (OSError, ValueError):
                    pass  # The native ten-second lease independently requests safe finishing.
            with self._lock:
                if token is not None:
                    self._inflight.discard(token)
                self._playbacks.pop(key, None)

    def results(self) -> list[dict]:
        with self._lock:
            return [dict(result) for result in self._results.values()]

    def _record(self, token: str, **changes) -> None:
        with self._lock:
            if token not in self._results:
                return
            previous = self._results[token]
            transition = (previous.get('result_state'), previous.get('reason'), (previous.get('native_result') or {}).get('sdk_returned'))
            self._results[token].update(changes)
            value = dict(self._results[token])
            updated = (value.get('result_state'), value.get('reason'), (value.get('native_result') or {}).get('sdk_returned'))
            if updated == transition:
                return
        try:
            self.on_result(value)
        except Exception as exc:
            self.report(f"Motion result observer failed: {exc}")

    def request(self, request_id: str, generation: int, purpose: str, session_id: str = "robot") -> bool:
        if purpose not in PURPOSES or not request_id or generation <= 0:
            return False
        token = hashlib.sha256(json.dumps([session_id, generation, request_id, purpose]).encode()).hexdigest()
        with self._lock:
            if self._closed:
                return False
            if token in self._results:
                return False
            reason = "client_busy" if self._busy else "stale_generation" if generation < self._generation else None
            while len(self._results) >= 128:
                removable = next((key for key in self._results if key not in self._watching and key not in self._inflight), None)
                if removable is None:
                    return False
                self._results.pop(removable)
            self._results[token] = dict(request_id=request_id, token=token, session_id=session_id,
                                        generation=generation, purpose=purpose, result_state="skipped" if reason else "sending",
                                        reason=reason, instance_id=None)
            if reason is None:
                self._generation = generation
                self._busy = True
                self._inflight.add(token)
                self._thread = threading.Thread(target=self._send, args=(token, generation, purpose),
                                                daemon=True, name="sdk-motion-request")
                self._thread.start()
        if reason:
            self._record(token, reason=reason)
            self.report(f"SDK gesture {purpose} skipped: {reason}; request_id={request_id} token={token}")
        return reason is None

    def _watch(self, token: str, instance_id: str | None) -> None:
        with self._lock:
            if self._closed:
                return
            self._watching[token] = (instance_id, time.monotonic() + 305)
            if self._observer is None:
                self._observer = threading.Thread(target=self._observe, daemon=True, name="sdk-motion-results")
                self._observer.start()

    def _send(self, token: str, generation: int, purpose: str) -> None:
        try:
            with self._lock:
                stale = self._closed or generation < self._generation
            if stale:
                self._record(token, result_state="skipped", reason="stale_generation")
                return
            preset = select_motion_preset(purpose)
            if purpose in ANSWER_PURPOSES and not self._wait_for_wake(generation):
                self._record(token, result_state="skipped", reason="wake_not_completed")
                return
            with self._lock:
                stale = self._closed or generation < self._generation
            if stale:
                self._record(token, result_state="skipped", reason="stale_generation")
                return
            result = exchange(f"PRESET {token} {preset}", path=self.path)
            if purpose == "wake_wave" and result["status"] in {"accepted", "duplicate"}:
                with self._lock:
                    self._wake_token = result.get("request_id", token)
            trace_voice("motion.gesture", request_id=token, generation=generation,
                        purpose=purpose, preset=preset, status=result["status"])
            record = result.get("result") or {}
            if result["status"] in {"accepted", "duplicate"}:
                self._record(token, result_state=record.get("result_state", "accepted"),
                             instance_id=result.get("instance_id"), native_result=record)
                self._watch(token, result.get("instance_id"))
            else:
                self._record(token, result_state="skipped", reason=result["status"])
                self.report(f"SDK gesture {purpose}: {result}; token={token}")
        except (OSError, ValueError) as exc:
            # Ambiguous delivery is resolved through RESULT, never another PLAY.
            self._record(token, result_state="unknown", reason=str(exc))
            self.report(f"SDK gesture {purpose} skipped/unknown: {exc}; token={token}")
            self._watch(token, None)
        finally:
            with self._lock:
                self._busy = False
                self._inflight.discard(token)

    def _observe(self) -> None:
        terminal = {"verified_completed", "cancelled_home", "failed", "unknown"}
        while not self._stop.wait(.2):
            with self._lock:
                watched = list(self._watching.items())
            for token, (instance, deadline) in watched:
                if self._stop.is_set():
                    return
                try:
                    response = request_result(token, instance_id=instance, path=self.path)
                    native = response.get("result")
                    status = response["status"]
                    if status in {"instance_changed", "not_found"}:
                        self._record(token, result_state="unknown", reason=status)
                        done = True
                    elif status == "ok" and isinstance(native, dict):
                        if instance is None and response.get("instance_id"):
                            with self._lock:
                                if token in self._watching:
                                    self._watching[token] = (response["instance_id"], deadline)
                        self._record(token, result_state=native["result_state"],
                                     native_result=native, instance_id=response.get("instance_id"))
                        done = native["result_state"] in terminal
                    else:
                        raise ValueError("Invalid retained motion result")
                except (OSError, ValueError, KeyError) as exc:
                    # A missing service may have restarted; don't upgrade an
                    # ambiguous outcome without the original instance contract.
                    self._record(token, result_state="unknown", reason=str(exc))
                    done = True
                if time.monotonic() > deadline:
                    self._record(token, result_state="unknown", reason="result_observation_timeout")
                    done = True
                if done:
                    with self._lock:
                        self._watching.pop(token, None)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for playback in self._playbacks.values():
                playback.reason = "client_closed"
                playback.stop.set()
            threads = (self._thread, self._observer, *(item.thread for item in self._playbacks.values()))
        self._stop.set()
        for thread in threads:
            if thread is not None:
                thread.join(timeout=0.5)


def main() -> None:
    import uuid
    parser = argparse.ArgumentParser(description="Native SDK motion status / one gesture")
    parser.add_argument("action", choices=["status", "result", *sorted(PURPOSES)], default="status", nargs="?")
    parser.add_argument("token", nargs="?")
    parser.add_argument("--instance-id", default=None)
    parser.add_argument("--socket", default=None)
    args = parser.parse_args()
    if args.action == "result":
        if not args.token: parser.error("result needs a physical token")
        print(json.dumps(request_result(args.token, instance_id=args.instance_id, path=args.socket), ensure_ascii=False))
        return
    if args.action == "status":
        command = "STATUS"
    else:
        preset = select_motion_preset(args.action)
        command = f"PRESET {uuid.uuid4().hex} {preset}"
    print(json.dumps(exchange(command, path=args.socket), ensure_ascii=False))


if __name__ == "__main__":
    main()
