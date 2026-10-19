from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from rag_system.contracts import ModuleArgumentParser, ModuleCLIUsageError, ModuleResult

from .node import VadNode


def build_parser() -> argparse.ArgumentParser:
    parser = ModuleArgumentParser(
        prog="python -m rag_system.vad",
        description="Test the offline VAD module with one 16 kHz mono PCM16 WAV file.",
    )
    parser.add_argument("audio", nargs="?", help="input WAV file")
    parser.add_argument("--input", dest="input_path", help="input WAV file")
    parser.add_argument("--model-dir", type=Path, help="offline FSMN VAD model directory")
    parser.add_argument(
        "--end-silence-ms",
        type=int,
        default=400,
        help="sentence-ending silence interval (default: 400)",
    )
    parser.add_argument("--output-wav", type=Path, help="optional detected speech WAV output")
    parser.add_argument(
        "--require-speech",
        action="store_true",
        help="return exit code 2 when no speech is detected",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    node_factory: Callable[..., Any] = VadNode,
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
            end_silence_ms=args.end_silence_ms,
        )
        result = node.run(input_path, output_wav=args.output_wav)
        speech_detected = bool(
            result.ok
            and isinstance(result.output, dict)
            and result.output.get("speech_detected")
        )
        strict_failure = args.require_speech and result.ok and not speech_detected
        if strict_failure:
            result = ModuleResult(
                module="vad",
                ok=False,
                output=result.output,
                metrics=result.metrics,
                artifacts=result.artifacts,
                error="speech was not detected",
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
                "module": "vad",
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
