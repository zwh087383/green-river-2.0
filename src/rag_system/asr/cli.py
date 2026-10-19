"""Command line entry point for a single ASR node test."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from rag_system.asr.node import ASRNode


class _CLIUsageError(ValueError):
    pass


class _JSONArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _CLIUsageError(message)


def _build_parser() -> argparse.ArgumentParser:
    parser = _JSONArgumentParser(
        prog="python -m rag_system.asr",
        description="Recognize one WAV file and print JSON.",
    )
    parser.add_argument("audio", type=Path, help="input PCM WAV file")
    parser.add_argument(
        "--provider",
        choices=("local", "qwen"),
        default="local",
        help="recognition provider (default: local)",
    )
    parser.add_argument("--chunk-ms", type=int, default=100, help="Qwen audio chunk size")
    parser.add_argument(
        "--end-silence-ms",
        type=int,
        default=400,
        help="Qwen server-VAD sentence interval",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="print Qwen partial/final events as JSON Lines while audio is processed",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    node_factory: Callable[[], ASRNode] = ASRNode,
    stdout: TextIO | None = None,
) -> int:
    """Run one recognition, returning zero on success and JSON in all cases."""
    output = stdout or sys.stdout
    try:
        args = _build_parser().parse_args(argv)
        node = node_factory()
        if args.provider == "qwen":
            def emit_event(event: dict) -> None:
                if not args.stream:
                    return
                print(
                    json.dumps(
                        {
                            "ok": True,
                            "module": "asr",
                            "provider": "qwen",
                            "event": event["type"],
                            "text": event["text"],
                        },
                        ensure_ascii=False,
                    ),
                    file=output,
                    flush=True,
                )

            result = asyncio.run(
                node.transcribe_realtime_file(
                    args.audio,
                    chunk_ms=args.chunk_ms,
                    end_silence_ms=args.end_silence_ms,
                    pace=True,
                    on_event=emit_event,
                )
            )
            payload = {
                "ok": True,
                "module": "asr",
                "provider": "qwen",
                "input": str(args.audio),
                **result.to_dict(),
            }
            if args.stream:
                payload["event"] = "summary"
        else:
            if args.stream:
                raise _CLIUsageError("--stream requires --provider qwen")
            text = node.transcribe_file(args.audio)
            payload = {
                "ok": True,
                "module": "asr",
                "provider": "local",
                "input": str(args.audio),
                "text": text,
            }
        exit_code = 0
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        payload = {
            "ok": False,
            "module": "asr",
            "error": str(exc) or type(exc).__name__,
            "error_type": type(exc).__name__,
        }
        exit_code = 1
    print(json.dumps(payload, ensure_ascii=False), file=output)
    return exit_code
