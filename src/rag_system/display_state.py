"""Shared display state used by the Web display shell and ROS2 display node."""
from __future__ import annotations

import json
import os
import argparse
import tempfile
import time
from pathlib import Path
from threading import RLock
from typing import Any

DISPLAY_MODES = frozenset({"emotion", "web"})
EMOTION_STATES = frozenset(
    {"idle", "listening", "thinking", "speaking", "success", "error", "sleep", "shy", "smug"}
)
WEB_DISPLAY_LEASE_SECONDS = 10.0


class DisplayStateStore:
    """Small atomic JSON store shared by separate Web and ROS2 processes."""

    def __init__(self, path: Path | str = Path("data/runtime/display_state.json")) -> None:
        self.path = Path(path)
        self._lock = RLock()
        self._web_sessions: dict[str, dict[str, Any]] = {}
        self._web_clients: dict[str, str] = {}
        self._web_generations: dict[str, int] = {}
        self._retired_sessions: set[str] = set()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            payload = self._read()
            if not payload:
                return self._default()
            current = {**self._default(), **payload}
            if (current.get("source") == "web" and current.get("lease_expires_at", 0)
                    and current["lease_expires_at"] <= time.time()):
                self._change_emotion(current, "idle", reason="web session expired")
                current["lease_expires_at"] = 0
                self._write(current)
            return current

    def set_mode(self, mode: str, *, reason: str = "") -> dict[str, Any]:
        cleaned = str(mode or "").strip().lower()
        if cleaned not in DISPLAY_MODES:
            raise ValueError(f"mode must be one of: {', '.join(sorted(DISPLAY_MODES))}")
        with self._lock:
            current = self.snapshot()
            current.update({"mode": cleaned, "reason": str(reason or "").strip()})
            current["updated_at"] = time.time()
            self._write(current)
            return current

    def set_emotion(
        self,
        state: str,
        *,
        text: str = "",
        reason: str = "",
        source: str = "manual",
    ) -> dict[str, Any]:
        cleaned = str(state or "").strip().lower()
        if cleaned not in EMOTION_STATES:
            raise ValueError(f"unknown emotion state: {cleaned}")
        with self._lock:
            current = self.snapshot()
            current.update(
                {
                    "state": cleaned,
                    "text": str(text or ""),
                    "reason": str(reason or "").strip(),
                    "updated_at": time.time(),
                    "source": source,
                    "web_session_id": "",
                    "web_client_id": "",
                    "lease_expires_at": 0,
                }
            )
            self._write(current)
            return current

    @staticmethod
    def _change_emotion(current: dict[str, Any], state: str, *, reason: str) -> None:
        # Lease renewals must not restart the same animation every heartbeat.
        if current.get("state") != state or current.get("text"):
            current.update(state=state, text="", updated_at=time.time())
        current["reason"] = reason

    def open_web_session(self, client_id: str, session_id: str, generation: int = 0) -> dict[str, Any]:
        """Refresh may replace its own tab, but a passive tab cannot interrupt another."""
        with self._lock:
            if generation < self._web_generations.get(client_id, 0):
                self._retired_sessions.add(session_id)
            if session_id in self._retired_sessions:
                return {**self.snapshot(), "session_status": "retired", "accepted": False}
            if session_id not in self._web_sessions:
                previous = self._web_clients.get(client_id)
                if previous:
                    self._retired_sessions.add(previous)
                    self._web_sessions.pop(previous, None)
                self._web_clients[client_id] = session_id
                self._web_generations[client_id] = generation
                self._web_sessions[session_id] = {"client_id": client_id, "sequence": -1}
                current = self.snapshot()
                other_active = (current.get("source") == "web"
                                and current.get("lease_expires_at", 0) > time.time()
                                and current.get("web_client_id") != client_id)
                if not other_active and current.get("source") != "ros":
                    self._change_emotion(current, "idle", reason="web page ready")
                    current.update(source="web", web_session_id=session_id,
                                   web_client_id=client_id,
                                   lease_expires_at=time.time() + WEB_DISPLAY_LEASE_SECONDS)
                    self._write(current)
            current = self.snapshot()
            return {**current, "session_status": "ready",
                    "accepted": current.get("web_session_id") == session_id}

    def sync_web_session(
        self, session_id: str, sequence: int, state: str, *, claim: bool = False,
    ) -> dict[str, Any]:
        if state not in EMOTION_STATES:
            raise ValueError(f"unknown emotion state: {state}")
        with self._lock:
            current = self.snapshot()
            session = self._web_sessions.get(session_id)
            if session is None:
                status = "retired" if session_id in self._retired_sessions else "unknown"
                return {**current, "session_status": status, "accepted": False}
            if sequence <= session["sequence"]:
                return {**current, "session_status": "ready", "accepted": False}
            session["sequence"] = sequence
            if current.get("web_session_id") != session_id and not claim:
                return {**current, "session_status": "ready", "accepted": False}
            self._change_emotion(current, state, reason="web conversation")
            current.update(source="web", web_session_id=session_id,
                           web_client_id=session["client_id"],
                           lease_expires_at=time.time() + WEB_DISPLAY_LEASE_SECONDS)
            self._write(current)
            return {**current, "session_status": "ready", "accepted": True}

    def release_web_session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            # Remember even an unknown release: a delayed open must not resurrect it.
            self._retired_sessions.add(session_id)
            self._web_sessions.pop(session_id, None)
            current = self.snapshot()
            if current.get("web_session_id") == session_id:
                self._change_emotion(current, "idle", reason="web page closed")
                current.update(web_session_id="", web_client_id="", lease_expires_at=0)
                self._write(current)
            return current

    def _default(self) -> dict[str, Any]:
        return {
            "mode": "emotion",
            "state": "idle",
            "text": "",
            "reason": "startup",
            "updated_at": 0.0,
        }

    def _read(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temp_name, self.path)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def main() -> None:
    """Local display control for Windows and SSH maintenance."""
    parser = argparse.ArgumentParser(description="Control the local robot display")
    parser.add_argument("action", choices=("status", "mode", "emotion"))
    parser.add_argument("value", nargs="?")
    parser.add_argument("--text", default="")
    parser.add_argument(
        "--path",
        default=str(Path(os.environ.get("RAG_DATA_DIR", "data")) / "runtime" / "display_state.json"),
    )
    args = parser.parse_args()
    store = DisplayStateStore(args.path)
    if args.action == "status":
        state = store.snapshot()
    elif args.action == "mode":
        if not args.value:
            parser.error("mode requires emotion or web")
        state = store.set_mode(args.value, reason="local command")
    else:
        if not args.value:
            parser.error("emotion requires a state name")
        state = store.set_emotion(args.value, text=args.text, reason="local command")
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
