"""Opt-in, time-bounded event tracing. Never accepts PCM or credentials."""
from __future__ import annotations

import atexit
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import threading
import time


class EventTrace:
    def __init__(self, directory: str, until: float):
        self.until = until
        self.queue = queue.Queue(maxsize=2048)
        self.dropped = 0
        self.failed = False
        self.thread = None
        if directory and until > time.time():
            try:
                path = Path(directory)
                path.mkdir(parents=True, exist_ok=True)
                self.fd = os.open(path / f"trace-{os.getpid()}.jsonl",
                                  os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            except OSError:
                self.failed = True
                return
            self.thread = threading.Thread(target=self._write, daemon=True, name="voice-trace")
            self.thread.start()
            atexit.register(self.close)

    def emit(self, stage: str, **fields):
        if self.thread is None or self.failed or time.time() >= self.until:
            return
        entry = dict(fields, stage=stage, pid=os.getpid(),
                     at=time.time(), monotonic=time.monotonic(),
                     timestamp=datetime.now(timezone.utc).isoformat(),
                     dropped_before=self.dropped)
        try:
            self.queue.put_nowait(entry)
        except queue.Full:
            self.dropped += 1

    def _write(self):
        try:
            while True:
                try:
                    row = self.queue.get(timeout=.5)
                except queue.Empty:
                    if time.time() >= self.until:
                        return
                    continue
                try:
                    if row is None:
                        return
                    data = memoryview((json.dumps(row, ensure_ascii=False) + "\n").encode())
                    while data:
                        data = data[os.write(self.fd, data):]
                finally:
                    self.queue.task_done()
        except (OSError, TypeError, ValueError):
            self.failed = True
        finally:
            os.close(self.fd)

    def close(self):
        if self.thread is not None and self.thread.is_alive():
            try:
                self.queue.put_nowait(None)
            except queue.Full:
                return
            self.thread.join(timeout=1)


try:
    _until = float(os.environ.get("RAG_VOICE_TRACE_UNTIL", "0"))
except ValueError:
    _until = 0
_trace = EventTrace(os.environ.get("RAG_VOICE_TRACE_DIR", ""), _until)


def trace_voice(stage: str, **fields):
    _trace.emit(stage, **fields)
