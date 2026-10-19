"""Bounded latest-only retrieval for ASR partials, with single-use final tokens."""
from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, BoundedSemaphore, Condition
from typing import Any, Callable


class RetrievalPrefetch:
    def __init__(self, service: Any, *, interval: float = 0.35) -> None:
        self.service = service
        self.interval = interval
        self._lock = Lock()
        self._changed = Condition(self._lock)
        self._pending: str | None = None
        self._inflight: str | None = None
        self._running = False
        self._closed = False
        self._claimed = False
        self._snapshot: tuple[Any, ...] | None = None
        self._last_started = 0.0

    def offer(self, query: str) -> None:
        query = query.strip()
        if len(query) < 2:
            return
        with self._changed:
            if self._closed or self._claimed:
                return
            if self._snapshot and self._snapshot[0] == query:
                self._pending = None
                return
            if self._inflight == query:
                self._pending = None
                return
            self._pending = query
            self._changed.notify_all()
            if self._running:
                return
            if not _SLOTS.acquire(blocking=False):
                return  # Speculation is optional; never queue unbounded work.
            self._running = True
        _WORKERS.submit(self._run)

    def _run(self) -> None:
        try:
            while True:
                with self._lock:
                    if self._closed or self._pending is None:
                        return
                with self._changed:
                    delay = self.interval - (time.monotonic() - self._last_started)
                    while delay > 0 and not self._closed and not self._claimed:
                        self._changed.wait(delay)
                        delay = self.interval - (time.monotonic() - self._last_started)
                    if self._closed:
                        return
                    query, self._pending = self._pending, None
                    if query is None:
                        continue
                    self._inflight = query
                self._last_started = time.monotonic()
                try:
                    revision = self.service.revision()
                    results = self.service.search(query, top_k=4)
                    snapshot = (query, revision, results) if revision == self.service.revision() else None
                except Exception:
                    snapshot = None  # Final retrieval retries and reports failures explicitly.
                with self._changed:
                    self._inflight = None
                    if not self._closed and (self._pending is None or self._pending == query):
                        self._snapshot = snapshot
                        self._pending = None
                    self._changed.notify_all()
        finally:
            with self._changed:
                self._running = False
                self._inflight = None
                _SLOTS.release()
                # An ASR update may arrive between the loop's last pending
                # check and this cleanup. Hand it to a worker atomically.
                if not self._closed and self._pending is not None and _SLOTS.acquire(blocking=False):
                    self._running = True
                    _WORKERS.submit(self._run)
                self._changed.notify_all()

    def take(self, query: str, revision: Any, *,
             should_cancel: Callable[[], bool] = lambda: False,
             timeout: float = 25.0) -> Any | None:
        """Claim a completed or matching in-flight query, without a second search.

        A timeout fails the turn instead of issuing a duplicate while the old
        HTTP request may still be executing. Completed failures may be retried
        by the caller, and changed text/revisions always require fresh results.
        """
        deadline = time.monotonic() + timeout
        with self._changed:
            if self._closed or self._claimed:
                return None
            self._claimed = True
            if self._inflight == query:
                self._pending = None
            self._changed.notify_all()
            try:
                while not self._closed:
                    if should_cancel():
                        return None
                    snapshot = self._snapshot
                    if snapshot and snapshot[0] == query and snapshot[1] == revision:
                        if self.service.revision() == revision:
                            return snapshot[2]
                        return None
                    if not self._running or query not in (self._inflight, self._pending):
                        return None
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Waiting for matching retrieval timed out")
                    self._changed.wait(min(remaining, 0.05))
                return None
            finally:
                self._closed = True
                self._pending = None
                self._snapshot = None
                self._changed.notify_all()

    def close(self) -> None:
        with self._changed:
            self._closed = True
            self._pending = None
            self._snapshot = None
            self._changed.notify_all()


_WORKERS = ThreadPoolExecutor(max_workers=4, thread_name_prefix="asr-retrieval")
_SLOTS = BoundedSemaphore(4)
_LOCK = Lock()
_TURNS: OrderedDict[str, tuple[float, RetrievalPrefetch]] = OrderedDict()


def register(prefetch: RetrievalPrefetch) -> str:
    token = uuid.uuid4().hex
    with _LOCK:
        now = time.monotonic()
        for key, (created, previous) in list(_TURNS.items()):
            if now - created > 120 or len(_TURNS) >= 32:
                previous.close()
                del _TURNS[key]
        _TURNS[token] = (now, prefetch)
    return token


def consume(token: str | None, query: str, revision: Any, *,
            should_cancel: Callable[[], bool] = lambda: False) -> Any | None:
    with _LOCK:
        entry = _TURNS.pop(token, None)
    if entry is None:
        return None
    created, prefetch = entry
    if time.monotonic() - created > 120:
        prefetch.close()
        return None
    return prefetch.take(query, revision, should_cancel=should_cancel)


def discard(token: str) -> None:
    with _LOCK:
        entry = _TURNS.pop(token, None)
    if entry:
        entry[1].close()
