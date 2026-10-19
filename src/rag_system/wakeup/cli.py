from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from rag_system.contracts import ModuleArgumentParser, ModuleCLIUsageError, ModuleResult

from .node import WakeupNode


def build_parser() -> argparse.ArgumentParser:
    parser = ModuleArgumentParser(
        prog="python -m rag_system.wakeup",
        description="Test the offline wake-word module with one 16 kHz mono PCM16 WAV file.",
    )
    parser.add_argument("audio", nargs="?", help="input WAV file")
    parser.add_argument("--input", dest="input_path", help="input WAV file")
    parser.add_argument("--model-dir", type=Path, help="offline KWS model directory")
    parser.add_argument("--wake-word", help="Chinese wake word, for example: 小绿小绿")
    parser.add_argument("--chunk-ms", type=int, default=100, help="stream chunk size (default: 100)")
    parser.add_argument(
        "--require-detected",
        action="store_true",
        help="return exit code 2 when no wake word is detected",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    node_factory: Callable[..., Any] = WakeupNode,
) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except ModuleCLIUsageError as exc:
        return _print_error(str(exc), type(exc).__name__)
    input_path = args.input_path or args.audio
    if not input_path:
        return _print_error("an input WAV file is required; use --input FILE", "InputError")
    if args.input_path and args.audio:
        return _print_error("provide the input file once, either positionally or with --input", "InputError")

    node = None
    try:
        node = node_factory(
            model_dir=args.model_dir,
            wake_word=args.wake_word,
            chunk_ms=args.chunk_ms,
        )
        result = node.run(input_path)
        detected = bool(
            result.ok
            and isinstance(result.output, dict)
            and result.output.get("detected")
        )
        strict_failure = args.require_detected and result.ok and not detected
        if strict_failure:
            result = ModuleResult(
                module="wakeup",
                ok=False,
                output=result.output,
                metrics=result.metrics,
                artifacts=result.artifacts,
                error="wake word was not detected",
            )
        print(json.dumps(result.to_dict(), ensure_ascii=False))
        if strict_failure:
            return 2
        return 0 if result.ok else 1
    except Exception as exc:
        return _print_error(str(exc), type(exc).__name__)
    finally:
        close = getattr(node, "close", None)
        if callable(close):
            close()


def _print_error(message: str, error_type: str) -> int:
    print(
        json.dumps(
            {
                "module": "wakeup",
                "ok": False,
                "error_type": error_type,
                "error": message,
            },
            ensure_ascii=False,
        )
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
