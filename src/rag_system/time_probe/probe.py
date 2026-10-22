"""Function and module timing based on Python's profiling hooks."""

from __future__ import annotations

import dis
import math
import sys
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import FrameType
from typing import Any


Clock = Callable[[], int]
ProfileHook = Callable[[FrameType, str, object], object]

_SUSPEND_OPNAMES = frozenset({"YIELD_FROM", "YIELD_VALUE"})
_FUNCTION_SORT_ALIASES = {
    "average_ms": "avg_ms",
    "average_ns": "avg_ns",
    "call_count": "calls",
    "total_ms": "cumulative_ms",
    "total_ns": "cumulative_ns",
    "total_time_ms": "cumulative_ms",
    "total_time_ns": "cumulative_ns",
}
_MODULE_SORT_ALIASES = {
    **_FUNCTION_SORT_ALIASES,
    "function_count": "functions",
}


@dataclass(frozen=True, slots=True)
class _FunctionKey:
    module: str
    function: str
    filename: str
    line: int

    @property
    def qualified_name(self) -> str:
        return f"{self.module}.{self.function}"


@dataclass(slots=True)
class _FunctionAggregate:
    calls: int = 0
    segments: int = 0
    incomplete_calls: int = 0
    cumulative_ns: int = 0
    active_ns: int = 0
    self_ns: int = 0
    min_ns: int | None = None
    max_ns: int = 0

    def record(
        self,
        *,
        cumulative_ns: int,
        active_ns: int,
        self_ns: int,
        segments: int,
        incomplete: bool,
    ) -> None:
        self.calls += 1
        self.segments += segments
        self.incomplete_calls += int(incomplete)
        self.cumulative_ns += cumulative_ns
        self.active_ns += active_ns
        self.self_ns += self_ns
        if self.min_ns is None or cumulative_ns < self.min_ns:
            self.min_ns = cumulative_ns
        if cumulative_ns > self.max_ns:
            self.max_ns = cumulative_ns


@dataclass(slots=True)
class _Invocation:
    frame_id: int
    key: _FunctionKey
    wall_started_ns: int
    active_ns: int = 0
    self_ns: int = 0
    segments: int = 0
    last_suspended_lasti: int | None = None


@dataclass(slots=True)
class _ActiveSegment:
    invocation: _Invocation
    started_ns: int
    child_active_ns: int = 0


@dataclass(frozen=True, slots=True)
class _StatSnapshot:
    key: _FunctionKey
    calls: int
    segments: int
    incomplete_calls: int
    cumulative_ns: int
    active_ns: int
    self_ns: int
    min_ns: int
    max_ns: int


class TimeProbe:
    """Measure selected Python functions and aggregate them by module.

    ``cumulative`` is wall time from a function's first ``call`` event to its
    final ``return`` event. It therefore includes coroutine and generator
    suspension. ``active`` is the sum of the function's executing segments,
    while ``self`` removes active segments spent in other measured functions.
    Native calls and Python functions outside ``include_modules`` remain part
    of self time.

    The probe installs a profile hook in the calling thread and configures the
    same hook for threads created while it is running. On Python 3.10 and 3.11,
    already-running threads cannot be retrofitted with ``threading.setprofile``.
    """

    def __init__(
        self,
        *,
        include_modules: str | Iterable[str] = ("rag_system",),
        exclude_modules: str | Iterable[str] = ("rag_system.time_probe",),
        clock_ns: Clock | None = None,
    ) -> None:
        self.include_modules = _normalize_prefixes(
            include_modules,
            parameter="include_modules",
            allow_empty=False,
        )
        self.exclude_modules = _normalize_prefixes(
            exclude_modules,
            parameter="exclude_modules",
            allow_empty=True,
        )
        if clock_ns is not None and not callable(clock_ns):
            raise TypeError("clock_ns must be callable")

        self._clock_ns: Clock = clock_ns or time.perf_counter_ns
        self._lock = threading.RLock()
        self._stats: dict[_FunctionKey, _FunctionAggregate] = {}
        self._active_by_thread: dict[int, list[_ActiveSegment]] = {}
        self._invocations: dict[int, _Invocation] = {}
        self._running = False
        self._session_started_ns: int | None = None
        self._elapsed_ns = 0
        self._previous_sys_profile: ProfileHook | None = None
        self._previous_thread_profile: ProfileHook | None = None
        self._callback: ProfileHook = self._profile

    @property
    def running(self) -> bool:
        """Whether the probe is currently collecting events."""

        with self._lock:
            return self._running

    def start(self, *, reset: bool = True) -> TimeProbe:
        """Start collecting profile events and return this probe.

        Repeated calls while running are idempotent. Pass ``reset=False`` to
        add a new session to statistics collected by earlier sessions.
        """

        with self._lock:
            if self._running:
                return self
            if reset:
                self._reset_unlocked()

            started_ns = self._read_clock()
            previous_sys_profile = sys.getprofile()
            previous_thread_profile = _get_thread_profile()
            self._previous_sys_profile = previous_sys_profile
            self._previous_thread_profile = previous_thread_profile
            self._session_started_ns = started_ns
            self._running = True

            try:
                threading.setprofile(self._callback)
                sys.setprofile(self._callback)
            except BaseException:
                self._running = False
                self._session_started_ns = None
                threading.setprofile(previous_thread_profile)
                sys.setprofile(previous_sys_profile)
                self._previous_sys_profile = None
                self._previous_thread_profile = None
                raise
        return self

    def stop(self) -> TimeProbe:
        """Stop collecting, restore earlier hooks, and finalize open calls."""

        with self._lock:
            if not self._running:
                return self

            self._running = False
            previous_sys_profile = self._previous_sys_profile
            previous_thread_profile = self._previous_thread_profile
            sys.setprofile(previous_sys_profile)
            threading.setprofile(previous_thread_profile)

            stopped_ns = self._read_clock()
            self._finalize_open_invocations(stopped_ns)
            if self._session_started_ns is not None:
                self._elapsed_ns += max(0, stopped_ns - self._session_started_ns)
            self._session_started_ns = None
            self._previous_sys_profile = None
            self._previous_thread_profile = None
        return self

    def reset(self) -> TimeProbe:
        """Discard all collected statistics while the probe is stopped."""

        with self._lock:
            if self._running:
                raise RuntimeError("cannot reset a running TimeProbe")
            self._reset_unlocked()
        return self

    def function_stats(
        self,
        *,
        sort_by: str = "cumulative_ms",
        descending: bool = True,
        module_prefix: str | None = None,
        name_contains: str | None = None,
        min_ms: float = 0.0,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return filtered, stably sorted per-function statistics."""

        snapshots, _ = self._snapshot()
        return _function_rows(
            snapshots,
            sort_by=sort_by,
            descending=descending,
            module_prefix=module_prefix,
            name_contains=name_contains,
            min_ms=min_ms,
            limit=limit,
        )

    def module_stats(
        self,
        *,
        sort_by: str = "cumulative_ms",
        descending: bool = True,
        module_prefix: str | None = None,
        min_ms: float = 0.0,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return filtered, stably sorted per-module statistics."""

        snapshots, _ = self._snapshot()
        return _module_rows(
            snapshots,
            sort_by=sort_by,
            descending=descending,
            module_prefix=module_prefix,
            min_ms=min_ms,
            limit=limit,
        )

    def report(
        self,
        *,
        sort_by: str = "cumulative_ms",
        descending: bool = True,
        module_prefix: str | None = None,
        name_contains: str | None = None,
        min_ms: float = 0.0,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Return a JSON-ready snapshot with summary, modules, and functions.

        ``name_contains``, ``min_ms``, and ``limit`` apply to function rows.
        Module rows include all measured functions matching ``module_prefix``.
        Module cumulative and active values are sums of function values and
        can overlap for nested calls; module self time is the non-overlapping
        hotspot metric within the measured module set.
        """

        snapshots, state = self._snapshot()
        functions = _function_rows(
            snapshots,
            sort_by=sort_by,
            descending=descending,
            module_prefix=module_prefix,
            name_contains=name_contains,
            min_ms=min_ms,
            limit=limit,
        )
        module_sort_by = _MODULE_SORT_ALIASES.get(sort_by, sort_by)
        if module_sort_by not in _module_sort_fields():
            module_sort_by = "cumulative_ms"
        modules = _module_rows(
            snapshots,
            sort_by=module_sort_by,
            descending=descending,
            module_prefix=module_prefix,
            min_ms=0.0,
            limit=None,
        )

        total_calls = sum(item.calls for item in snapshots)
        total_segments = sum(item.segments for item in snapshots)
        cumulative_ns = sum(item.cumulative_ns for item in snapshots)
        active_ns = sum(item.active_ns for item in snapshots)
        self_ns = sum(item.self_ns for item in snapshots)
        all_modules = {item.key.module for item in snapshots}
        summary = {
            "running": state["running"],
            "elapsed_ns": state["elapsed_ns"],
            "elapsed_ms": _milliseconds(state["elapsed_ns"]),
            "calls": total_calls,
            "segments": total_segments,
            "incomplete_calls": sum(item.incomplete_calls for item in snapshots),
            "function_count": len(snapshots),
            "module_count": len(all_modules),
            "reported_function_count": len(functions),
            "reported_module_count": len(modules),
            "cumulative_ns": cumulative_ns,
            "cumulative_ms": _milliseconds(cumulative_ns),
            "active_ns": active_ns,
            "active_ms": _milliseconds(active_ns),
            "self_ns": self_ns,
            "self_ms": _milliseconds(self_ns),
            "include_modules": list(self.include_modules),
            "exclude_modules": list(self.exclude_modules),
        }
        return {"summary": summary, "modules": modules, "functions": functions}

    def __enter__(self) -> TimeProbe:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> bool:
        self.stop()
        return False

    def _profile(self, frame: FrameType, event: str, arg: object) -> None:
        if not self._running or (event != "call" and event != "return"):
            return

        module = str(frame.f_globals.get("__name__") or "")
        if not self._includes_module(module):
            return

        now_ns = self._read_clock()
        thread_id = threading.get_ident()
        with self._lock:
            if not self._running:
                return
            if event == "call":
                self._on_call(frame, module, thread_id, now_ns)
            else:
                self._on_return(frame, thread_id, now_ns)

    def _on_call(
        self,
        frame: FrameType,
        module: str,
        thread_id: int,
        now_ns: int,
    ) -> None:
        code = frame.f_code
        function = str(getattr(code, "co_qualname", code.co_name))
        key = _FunctionKey(
            module=module,
            function=function,
            filename=str(code.co_filename),
            line=int(code.co_firstlineno),
        )
        frame_id = id(frame)
        invocation = self._invocations.get(frame_id)
        frame_id_was_reused = (
            invocation is not None
            and invocation.last_suspended_lasti is not None
            and frame.f_lasti < invocation.last_suspended_lasti
        )
        if invocation is not None and (invocation.key != key or frame_id_was_reused):
            self._record_invocation(invocation, now_ns=now_ns, incomplete=True)
            self._invocations.pop(frame_id, None)
            invocation = None
        if invocation is None:
            invocation = _Invocation(
                frame_id=frame_id,
                key=key,
                wall_started_ns=now_ns,
            )
            self._invocations[frame_id] = invocation
        else:
            invocation.last_suspended_lasti = None

        invocation.segments += 1
        self._active_by_thread.setdefault(thread_id, []).append(
            _ActiveSegment(invocation=invocation, started_ns=now_ns)
        )

    def _on_return(self, frame: FrameType, thread_id: int, now_ns: int) -> None:
        stack = self._active_by_thread.get(thread_id)
        if not stack:
            return

        frame_id = id(frame)
        match_index = _find_active_segment(stack, frame_id)
        if match_index is None:
            return

        while len(stack) - 1 > match_index:
            orphan = stack.pop()
            self._finish_segment(orphan, stack, now_ns)
            self._record_invocation(orphan.invocation, now_ns=now_ns, incomplete=True)
            self._invocations.pop(orphan.invocation.frame_id, None)

        segment = stack.pop()
        self._finish_segment(segment, stack, now_ns)
        if not stack:
            self._active_by_thread.pop(thread_id, None)

        if _frame_is_suspended(frame):
            segment.invocation.last_suspended_lasti = frame.f_lasti
            return

        invocation = segment.invocation
        self._record_invocation(invocation, now_ns=now_ns, incomplete=False)
        self._invocations.pop(invocation.frame_id, None)

    def _finish_segment(
        self,
        segment: _ActiveSegment,
        parent_stack: list[_ActiveSegment],
        now_ns: int,
    ) -> None:
        duration_ns = max(0, now_ns - segment.started_ns)
        invocation = segment.invocation
        invocation.active_ns += duration_ns
        invocation.self_ns += max(0, duration_ns - segment.child_active_ns)
        if parent_stack:
            parent_stack[-1].child_active_ns += duration_ns

    def _record_invocation(
        self,
        invocation: _Invocation,
        *,
        now_ns: int,
        incomplete: bool,
    ) -> None:
        cumulative_ns = max(0, now_ns - invocation.wall_started_ns)
        active_ns = min(invocation.active_ns, cumulative_ns)
        self_ns = min(invocation.self_ns, active_ns)
        aggregate = self._stats.setdefault(invocation.key, _FunctionAggregate())
        aggregate.record(
            cumulative_ns=cumulative_ns,
            active_ns=active_ns,
            self_ns=self_ns,
            segments=invocation.segments,
            incomplete=incomplete,
        )

    def _finalize_open_invocations(self, stopped_ns: int) -> None:
        for stack in self._active_by_thread.values():
            while stack:
                segment = stack.pop()
                self._finish_segment(segment, stack, stopped_ns)
        self._active_by_thread.clear()

        for invocation in tuple(self._invocations.values()):
            self._record_invocation(
                invocation,
                now_ns=stopped_ns,
                incomplete=True,
            )
        self._invocations.clear()

    def _includes_module(self, module: str) -> bool:
        for prefix in self.include_modules:
            if _prefix_matches(module, prefix):
                break
        else:
            return False
        for prefix in self.exclude_modules:
            if _prefix_matches(module, prefix):
                return False
        return True

    def _snapshot(self) -> tuple[list[_StatSnapshot], dict[str, int | bool]]:
        with self._lock:
            elapsed_ns = self._elapsed_ns
            if self._running and self._session_started_ns is not None:
                elapsed_ns += max(0, self._read_clock() - self._session_started_ns)
            snapshots = [
                _StatSnapshot(
                    key=key,
                    calls=value.calls,
                    segments=value.segments,
                    incomplete_calls=value.incomplete_calls,
                    cumulative_ns=value.cumulative_ns,
                    active_ns=value.active_ns,
                    self_ns=value.self_ns,
                    min_ns=value.min_ns or 0,
                    max_ns=value.max_ns,
                )
                for key, value in self._stats.items()
            ]
            state: dict[str, int | bool] = {
                "running": self._running,
                "elapsed_ns": elapsed_ns,
            }
        return snapshots, state

    def _read_clock(self) -> int:
        return int(self._clock_ns())

    def _reset_unlocked(self) -> None:
        self._stats.clear()
        self._active_by_thread.clear()
        self._invocations.clear()
        self._elapsed_ns = 0
        self._session_started_ns = None


def _normalize_prefixes(
    values: str | Iterable[str],
    *,
    parameter: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if isinstance(values, str):
        raw_values = (values,)
    else:
        try:
            raw_values = tuple(values)
        except TypeError as exc:
            raise TypeError(f"{parameter} must be a string or iterable of strings") from exc

    normalized: list[str] = []
    for value in raw_values:
        if not isinstance(value, str):
            raise TypeError(f"{parameter} must contain only strings")
        prefix = value.strip().rstrip(".")
        if not prefix:
            raise ValueError(f"{parameter} cannot contain an empty module prefix")
        if prefix not in normalized:
            normalized.append(prefix)
    if not normalized and not allow_empty:
        raise ValueError(f"{parameter} cannot be empty")
    return tuple(normalized)


def _get_thread_profile() -> ProfileHook | None:
    getprofile = getattr(threading, "getprofile", None)
    if getprofile is None:
        return None
    return getprofile()


def _prefix_matches(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def _frame_is_suspended(frame: FrameType) -> bool:
    instruction = frame.f_lasti
    bytecode = frame.f_code.co_code
    if instruction < 0 or instruction >= len(bytecode):
        return False
    if dis.opname[bytecode[instruction]] in _SUSPEND_OPNAMES:
        return True
    # CPython 3.10 reports an await suspension at the instruction preceding
    # YIELD_FROM; 3.11+ reports the actual suspension instruction instead.
    following = instruction + 2
    return (
        sys.version_info[:2] == (3, 10)
        and following < len(bytecode)
        and dis.opname[bytecode[following]] == "YIELD_FROM"
    )


def _find_active_segment(
    stack: list[_ActiveSegment],
    frame_id: int,
) -> int | None:
    for index in range(len(stack) - 1, -1, -1):
        if stack[index].invocation.frame_id == frame_id:
            return index
    return None


def _function_rows(
    snapshots: list[_StatSnapshot],
    *,
    sort_by: str,
    descending: bool,
    module_prefix: str | None,
    name_contains: str | None,
    min_ms: float,
    limit: int | None,
) -> list[dict[str, Any]]:
    minimum_ns = _validate_filters(min_ms=min_ms, limit=limit)
    sort_key = _FUNCTION_SORT_ALIASES.get(sort_by, sort_by)
    rows = [_function_row(item) for item in snapshots]
    if not rows:
        if sort_key not in _function_sort_fields():
            raise ValueError(f"unsupported function sort field: {sort_by}")
        return []
    if sort_key not in rows[0]:
        raise ValueError(f"unsupported function sort field: {sort_by}")

    rows = [
        row
        for row in rows
        if (module_prefix is None or _prefix_matches(row["module"], module_prefix))
        and (name_contains is None or name_contains in row["qualified_name"])
        and row["cumulative_ns"] >= minimum_ns
    ]
    _stable_sort(rows, sort_key=sort_key, name_key="qualified_name", descending=descending)
    if limit is not None:
        rows = rows[:limit]
    return rows


def _module_rows(
    snapshots: list[_StatSnapshot],
    *,
    sort_by: str,
    descending: bool,
    module_prefix: str | None,
    min_ms: float,
    limit: int | None,
) -> list[dict[str, Any]]:
    minimum_ns = _validate_filters(min_ms=min_ms, limit=limit)
    sort_key = _MODULE_SORT_ALIASES.get(sort_by, sort_by)
    if sort_key not in _module_sort_fields():
        raise ValueError(f"unsupported module sort field: {sort_by}")

    grouped: dict[str, list[_StatSnapshot]] = {}
    for item in snapshots:
        if module_prefix is None or _prefix_matches(item.key.module, module_prefix):
            grouped.setdefault(item.key.module, []).append(item)
    rows = [_module_row(module, values) for module, values in grouped.items()]
    rows = [row for row in rows if row["cumulative_ns"] >= minimum_ns]
    _stable_sort(rows, sort_key=sort_key, name_key="module", descending=descending)
    if limit is not None:
        rows = rows[:limit]
    return rows


def _function_row(item: _StatSnapshot) -> dict[str, Any]:
    calls = item.calls
    avg_ns = item.cumulative_ns / calls if calls else 0.0
    active_avg_ns = item.active_ns / calls if calls else 0.0
    self_avg_ns = item.self_ns / calls if calls else 0.0
    return {
        "module": item.key.module,
        "function": item.key.function,
        "qualified_name": item.key.qualified_name,
        "file": item.key.filename,
        "line": item.key.line,
        "calls": calls,
        "segments": item.segments,
        "incomplete_calls": item.incomplete_calls,
        "cumulative_ns": item.cumulative_ns,
        "cumulative_ms": _milliseconds(item.cumulative_ns),
        "active_ns": item.active_ns,
        "active_ms": _milliseconds(item.active_ns),
        "self_ns": item.self_ns,
        "self_ms": _milliseconds(item.self_ns),
        "avg_ns": avg_ns,
        "avg_ms": _milliseconds(avg_ns),
        "active_avg_ns": active_avg_ns,
        "active_avg_ms": _milliseconds(active_avg_ns),
        "self_avg_ns": self_avg_ns,
        "self_avg_ms": _milliseconds(self_avg_ns),
        "min_ns": item.min_ns,
        "min_ms": _milliseconds(item.min_ns),
        "max_ns": item.max_ns,
        "max_ms": _milliseconds(item.max_ns),
    }


def _module_row(module: str, values: list[_StatSnapshot]) -> dict[str, Any]:
    calls = sum(item.calls for item in values)
    cumulative_ns = sum(item.cumulative_ns for item in values)
    active_ns = sum(item.active_ns for item in values)
    self_ns = sum(item.self_ns for item in values)
    avg_ns = cumulative_ns / calls if calls else 0.0
    return {
        "module": module,
        "functions": len(values),
        "calls": calls,
        "segments": sum(item.segments for item in values),
        "incomplete_calls": sum(item.incomplete_calls for item in values),
        "cumulative_ns": cumulative_ns,
        "cumulative_ms": _milliseconds(cumulative_ns),
        "active_ns": active_ns,
        "active_ms": _milliseconds(active_ns),
        "self_ns": self_ns,
        "self_ms": _milliseconds(self_ns),
        "avg_ns": avg_ns,
        "avg_ms": _milliseconds(avg_ns),
    }


def _validate_filters(*, min_ms: float, limit: int | None) -> int:
    if isinstance(min_ms, bool) or not isinstance(min_ms, (int, float)):
        raise TypeError("min_ms must be a non-negative number")
    if min_ms < 0 or not math.isfinite(min_ms):
        raise ValueError("min_ms must be a finite non-negative number")
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be a non-negative integer or None")
        if limit < 0:
            raise ValueError("limit must be a non-negative integer or None")
    return int(min_ms * 1_000_000)


def _stable_sort(
    rows: list[dict[str, Any]],
    *,
    sort_key: str,
    name_key: str,
    descending: bool,
) -> None:
    rows.sort(key=lambda row: row[name_key])
    rows.sort(key=lambda row: row[sort_key], reverse=descending)


def _function_sort_fields() -> frozenset[str]:
    return frozenset(
        {
            "active_avg_ms",
            "active_avg_ns",
            "active_ms",
            "active_ns",
            "avg_ms",
            "avg_ns",
            "calls",
            "cumulative_ms",
            "cumulative_ns",
            "function",
            "incomplete_calls",
            "line",
            "max_ms",
            "max_ns",
            "min_ms",
            "min_ns",
            "module",
            "qualified_name",
            "segments",
            "self_avg_ms",
            "self_avg_ns",
            "self_ms",
            "self_ns",
        }
    )


def _module_sort_fields() -> frozenset[str]:
    return frozenset(
        {
            "active_ms",
            "active_ns",
            "avg_ms",
            "avg_ns",
            "calls",
            "cumulative_ms",
            "cumulative_ns",
            "functions",
            "incomplete_calls",
            "module",
            "segments",
            "self_ms",
            "self_ns",
        }
    )


def _milliseconds(value: int | float) -> float:
    return value / 1_000_000


__all__ = ["TimeProbe"]
