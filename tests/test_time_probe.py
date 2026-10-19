from __future__ import annotations

import asyncio
import gc
import json
import sys
import threading
import time

import pytest

from rag_system.time_probe import TimeProbe


def _inner_work() -> int:
    return sum(range(50))


def _outer_work() -> int:
    return _inner_work() + 1


def _raising_work() -> None:
    raise RuntimeError("expected")


def _streaming_work():
    yield 1
    yield 2


async def _async_work() -> int:
    await asyncio.sleep(0.003)
    return 3


async def _async_streaming_work():
    yield 1
    await asyncio.sleep(0.003)
    yield 2


def _thread_work(started: threading.Barrier) -> int:
    started.wait()
    return 1


def _secret_work(value: str) -> int:
    secret_copy = value
    return len(secret_copy)


def _alpha_work() -> int:
    return 1


def _zeta_work() -> int:
    return 2


def _row(report: dict, suffix: str) -> dict:
    return next(
        item for item in report["functions"] if item["qualified_name"].endswith(suffix)
    )


def test_probe_reports_nested_function_and_module_times_as_json() -> None:
    with TimeProbe(include_modules=(__name__,)) as probe:
        assert _outer_work() == 1226

    report = probe.report()
    outer = _row(report, "._outer_work")
    inner = _row(report, "._inner_work")

    assert outer["calls"] == inner["calls"] == 1
    assert outer["cumulative_ns"] >= outer["active_ns"] >= outer["self_ns"]
    assert inner["cumulative_ns"] >= inner["active_ns"] >= inner["self_ns"]
    assert outer["self_ns"] <= outer["active_ns"] - inner["active_ns"]
    assert report["modules"][0]["cumulative_ns"] >= report["modules"][0]["self_ns"]
    assert report["summary"]["running"] is False
    json.dumps(report)


def test_probe_uses_nanoseconds_for_exact_nested_aggregation() -> None:
    ticks = iter((0, 10_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000))
    probe = TimeProbe(include_modules=(__name__,), clock_ns=lambda: next(ticks))
    probe.start()
    _outer_work()
    probe.stop()

    report = probe.report()
    outer = _row(report, "._outer_work")
    inner = _row(report, "._inner_work")
    module = report["modules"][0]

    assert outer["cumulative_ns"] == 30_000_000
    assert outer["active_ns"] == 30_000_000
    assert outer["self_ns"] == 20_000_000
    assert outer["avg_ms"] == outer["min_ms"] == outer["max_ms"] == 30.0
    assert inner["cumulative_ns"] == inner["self_ns"] == 10_000_000
    assert module["cumulative_ms"] == 40.0
    assert module["active_ms"] == 40.0
    assert module["self_ms"] == 30.0
    assert report["summary"]["elapsed_ms"] == 50.0


def test_probe_never_serializes_arguments_or_locals() -> None:
    secret = "do-not-put-this-api-key-in-the-report"
    with TimeProbe(include_modules=(__name__,)) as probe:
        assert _secret_work(secret) == len(secret)

    assert secret not in json.dumps(probe.report())


def test_probe_finalizes_a_function_that_raises() -> None:
    probe = TimeProbe(include_modules=(__name__,))

    with pytest.raises(RuntimeError, match="expected"):
        with probe:
            _raising_work()

    assert _row(probe.report(), "._raising_work")["calls"] == 1


def test_probe_treats_generator_yields_as_one_wall_clock_call() -> None:
    probe = TimeProbe(include_modules=(__name__,))
    probe.start()
    stream = _streaming_work()
    assert next(stream) == 1
    time.sleep(0.003)
    assert next(stream) == 2
    time.sleep(0.003)
    with pytest.raises(StopIteration):
        next(stream)
    probe.stop()

    row = _row(probe.report(), "._streaming_work")
    assert row["calls"] == 1
    assert row["segments"] == 3
    assert row["cumulative_ns"] > row["active_ns"]


def test_probe_does_not_merge_an_abandoned_generator_with_a_reused_frame() -> None:
    probe = TimeProbe(include_modules=(__name__,))
    probe.start()
    abandoned = _streaming_work()
    assert next(abandoned) == 1
    del abandoned
    gc.collect()
    replacement = _streaming_work()
    assert list(replacement) == [1, 2]
    probe.stop()

    row = _row(probe.report(), "._streaming_work")
    assert row["calls"] == 2
    assert row["incomplete_calls"] == 1


def test_probe_treats_coroutine_await_as_one_wall_clock_call() -> None:
    with TimeProbe(include_modules=(__name__,)) as probe:
        assert asyncio.run(_async_work()) == 3

    row = _row(probe.report(), "._async_work")
    assert row["calls"] == 1
    assert row["segments"] >= 2
    assert row["cumulative_ns"] > row["active_ns"]


def test_probe_treats_async_generator_yields_as_one_wall_clock_call() -> None:
    async def collect() -> list[int]:
        return [item async for item in _async_streaming_work()]

    with TimeProbe(include_modules=(__name__,)) as probe:
        assert asyncio.run(collect()) == [1, 2]

    row = _row(probe.report(), "._async_streaming_work")
    assert row["calls"] == 1
    assert row["segments"] >= 3
    assert row["cumulative_ns"] > row["active_ns"]


def test_probe_separates_thread_stacks() -> None:
    barrier = threading.Barrier(3)
    probe = TimeProbe(include_modules=(__name__,))
    probe.start()
    workers = [threading.Thread(target=_thread_work, args=(barrier,)) for _ in range(2)]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join()
    probe.stop()

    row = _row(probe.report(), "._thread_work")
    assert row["calls"] == 2
    assert row["self_ns"] == row["active_ns"]


def test_probe_restores_profile_hooks_and_can_accumulate_sessions() -> None:
    old_sys_hook = sys.getprofile()
    old_thread_hook = threading.getprofile()

    def previous_hook(frame, event, arg):
        return None

    sys.setprofile(previous_hook)
    threading.setprofile(previous_hook)
    try:
        probe = TimeProbe(include_modules=(__name__,))
        probe.start()
        assert sys.getprofile() == probe._callback
        assert threading.getprofile() == probe._callback
        _inner_work()
        probe.stop()
        assert sys.getprofile() is previous_hook
        assert threading.getprofile() is previous_hook

        probe.start(reset=False)
        _inner_work()
        probe.stop()
        assert _row(probe.report(), "._inner_work")["calls"] == 2

        probe.reset()
        assert probe.report()["functions"] == []
    finally:
        sys.setprofile(old_sys_hook)
        threading.setprofile(old_thread_hook)


def test_probe_validates_filters_even_before_any_calls() -> None:
    probe = TimeProbe(include_modules=("not_imported",))

    with pytest.raises(ValueError, match="sort field"):
        probe.function_stats(sort_by="unknown")
    with pytest.raises(ValueError, match="non-negative"):
        probe.module_stats(min_ms=-1)


def test_probe_breaks_equal_sort_values_by_qualified_name() -> None:
    with TimeProbe(include_modules=(__name__,), clock_ns=lambda: 0) as probe:
        _zeta_work()
        _alpha_work()

    functions = probe.function_stats(sort_by="cumulative_ns", descending=True)
    assert [item["function"] for item in functions] == ["_alpha_work", "_zeta_work"]
