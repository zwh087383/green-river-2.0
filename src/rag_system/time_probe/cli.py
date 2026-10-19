"""Run a Python module under the opt-in RAG system time probe."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import runpy
import sys
import tempfile
import traceback
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from rag_system.time_probe.probe import TimeProbe


DEFAULT_REPORT_PATH = Path("data/runtime/time_probe/latest.json")
_SORT_FIELDS = ("cumulative_ms", "active_ms", "self_ms", "calls")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag_system.time_probe",
        description=(
            "Profile every executed Python function in selected modules and write "
            "function/module timing statistics as JSON."
        ),
        epilog=(
            "example: python -m rag_system.time_probe --output report.json -- "
            "rag_system.module_test rag-cloud parse samples/acceptance/green_river.txt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"JSON report path (default: {DEFAULT_REPORT_PATH})",
    )
    parser.add_argument(
        "--include",
        action="append",
        dest="include_modules",
        metavar="MODULE",
        help="module prefix to profile; repeat as needed (default: rag_system)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        dest="exclude_modules",
        metavar="MODULE",
        help="included module prefix to skip; repeat as needed",
    )
    parser.add_argument(
        "--sort",
        choices=_SORT_FIELDS,
        default="cumulative_ms",
        help="field used by the terminal summary (default: cumulative_ms)",
    )
    parser.add_argument(
        "--limit",
        type=_non_negative_int,
        default=15,
        help="maximum module and function rows in the terminal summary (default: 15)",
    )
    parser.add_argument(
        "--min-ms",
        type=_non_negative_float,
        default=0.0,
        help="hide faster rows from the terminal summary only (default: 0)",
    )
    parser.add_argument(
        "--label",
        help="optional non-sensitive label stored with the report",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="write only the report path to stderr",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stderr: TextIO | None = None,
) -> int:
    """Profile one ``python -m`` style target and preserve its exit code."""

    raw_args = list(sys.argv[1:] if argv is None else argv)
    error_output = stderr or sys.stderr
    parser = build_parser()
    probe_args, target_args = _split_arguments(parser, raw_args)
    options = parser.parse_args(probe_args)
    if not target_args:
        parser.error("a target module is required after --")
    if options.output == Path("-"):
        parser.error("--output - is not supported because target stdout is preserved")

    target_module, *target_argv = target_args
    if not target_module.strip() or target_module.startswith("-"):
        parser.error("the target after -- must be a Python module name")

    report_path = options.output.resolve()
    include_modules = tuple(options.include_modules or ("rag_system",))
    exclude_modules = tuple(
        dict.fromkeys(("rag_system.time_probe", *(options.exclude_modules or ())))
    )
    probe = TimeProbe(
        include_modules=include_modules,
        exclude_modules=exclude_modules,
    )
    discovered_modules = _discover_modules(include_modules)

    exit_code = 0
    status = "ok"
    error_type: str | None = None
    probe.start()
    try:
        exit_code = _run_target_module(target_module, target_argv, stderr=error_output)
        if exit_code != 0:
            status = "failed"
    except KeyboardInterrupt:
        status = "interrupted"
        error_type = "KeyboardInterrupt"
        exit_code = 130
        print("", file=error_output)
    except BaseException as exc:
        status = "error"
        error_type = type(exc).__name__
        exit_code = 1
        traceback.print_exc(file=error_output)
    finally:
        probe.stop()

    report = _build_report(
        probe,
        target_module=target_module,
        target_argument_count=len(target_argv),
        exit_code=exit_code,
        status=status,
        error_type=error_type,
        label=options.label,
        discovered_modules=discovered_modules,
        sort_by=options.sort,
    )
    try:
        _write_json_atomic(report_path, report)
    except Exception as exc:
        print(
            f"[time-probe] could not write report {report_path}: {exc}",
            file=error_output,
        )
        return exit_code or 1

    _print_summary(
        report,
        report_path=report_path,
        sort_by=options.sort,
        limit=options.limit,
        min_ms=options.min_ms,
        quiet=options.quiet,
        output=error_output,
    )
    return exit_code


def _split_arguments(
    parser: argparse.ArgumentParser,
    args: list[str],
) -> tuple[list[str], list[str]]:
    if "--" not in args:
        if any(argument in {"-h", "--help"} for argument in args):
            parser.parse_args(args)
        parser.error("separate probe options from the target module with --")
    separator = args.index("--")
    return args[:separator], args[separator + 1 :]


def _run_target_module(
    module: str,
    argv: list[str],
    *,
    stderr: TextIO,
) -> int:
    previous_argv = sys.argv[:]
    sys.argv = [module, *argv]
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except SystemExit as exc:
        if exc.code is None:
            return 0
        if isinstance(exc.code, int):
            return int(exc.code)
        print(str(exc.code), file=stderr)
        return 1
    finally:
        sys.argv = previous_argv
    return 0


def _build_report(
    probe: TimeProbe,
    *,
    target_module: str,
    target_argument_count: int,
    exit_code: int,
    status: str,
    error_type: str | None,
    label: str | None,
    discovered_modules: set[str],
    sort_by: str,
) -> dict[str, Any]:
    snapshot = probe.report(sort_by=sort_by)
    observed_modules = {row["module"] for row in snapshot["modules"]}
    discovered_count = len(discovered_modules)
    observed_discovered = discovered_modules & observed_modules
    coverage_percent = (
        round(len(observed_discovered) * 100.0 / discovered_count, 2)
        if discovered_count
        else None
    )
    target: dict[str, Any] = {
        "module": target_module,
        "argument_count": target_argument_count,
        "exit_code": exit_code,
        "status": status,
    }
    if label:
        target["label"] = label
    if error_type:
        target["error_type"] = error_type

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": target,
        "inventory": {
            "discovered_module_count": discovered_count,
            "observed_module_count": len(observed_modules),
            "observed_discovered_module_count": len(observed_discovered),
            "module_coverage_percent": coverage_percent,
            "unobserved_modules": sorted(discovered_modules - observed_modules),
            "observed_not_discovered": sorted(observed_modules - discovered_modules),
        },
        **snapshot,
    }


def _discover_modules(prefixes: tuple[str, ...]) -> set[str]:
    modules: set[str] = set()
    for prefix in prefixes:
        try:
            spec = importlib.util.find_spec(prefix)
        except Exception:
            continue
        if spec is None:
            continue
        locations = spec.submodule_search_locations
        if locations is None:
            if spec.origin and spec.origin.endswith(".py"):
                modules.add(prefix)
            continue
        modules.add(prefix)
        for location in locations:
            root = Path(location)
            if not root.is_dir():
                continue
            for source in root.rglob("*.py"):
                relative = source.relative_to(root)
                if "__pycache__" in relative.parts:
                    continue
                parts = list(relative.with_suffix("").parts)
                if parts[-1] == "__init__":
                    parts.pop()
                module = ".".join((prefix, *parts)) if parts else prefix
                modules.add(module)
    return modules


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _print_summary(
    report: dict[str, Any],
    *,
    report_path: Path,
    sort_by: str,
    limit: int,
    min_ms: float,
    quiet: bool,
    output: TextIO,
) -> None:
    target = report["target"]
    summary = report["summary"]
    inventory = report["inventory"]
    if quiet:
        print(f"[time-probe] report={report_path}", file=output)
        return

    print(
        "[time-probe] "
        f"status={target['status']} exit={target['exit_code']} "
        f"elapsed={summary['elapsed_ms']:.3f}ms "
        f"modules={summary['module_count']} functions={summary['function_count']} "
        f"calls={summary['calls']}",
        file=output,
    )
    coverage = inventory["module_coverage_percent"]
    coverage_text = "n/a" if coverage is None else f"{coverage:.2f}%"
    print(
        f"[time-probe] module coverage={coverage_text}; report={report_path}",
        file=output,
    )
    _print_rows(
        "modules",
        report["modules"],
        name_field="module",
        sort_by=sort_by,
        limit=limit,
        min_ms=min_ms,
        output=output,
    )
    _print_rows(
        "functions",
        report["functions"],
        name_field="qualified_name",
        sort_by=sort_by,
        limit=limit,
        min_ms=min_ms,
        output=output,
    )


def _print_rows(
    title: str,
    rows: list[dict[str, Any]],
    *,
    name_field: str,
    sort_by: str,
    limit: int,
    min_ms: float,
    output: TextIO,
) -> None:
    visible = [row for row in rows if row["cumulative_ms"] >= min_ms][:limit]
    if not visible:
        return
    print(f"[time-probe] top {title} by {sort_by}:", file=output)
    for row in visible:
        print(
            "  "
            f"wall={row['cumulative_ms']:10.3f}ms "
            f"active={row['active_ms']:10.3f}ms "
            f"self={row['self_ms']:10.3f}ms "
            f"calls={row['calls']:7d}  {row[name_field]}",
            file=output,
        )


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0 or not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


__all__ = ["DEFAULT_REPORT_PATH", "build_parser", "main"]
