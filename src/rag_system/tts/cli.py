"""Command line entry point for a single TTS node test."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from rag_system.tts.node import DEFAULT_AUDIO_FORMAT, DEFAULT_MEDIA_TYPE, TTSNode


class _CLIUsageError(ValueError):
    pass


class _JSONArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _CLIUsageError(message)


def _build_parser() -> argparse.ArgumentParser:
    parser = _JSONArgumentParser(
        prog="python -m rag_system.tts",
        description="Synthesize one text value to an MP3 file and print JSON.",
    )
    parser.add_argument("text", help="text to synthesize")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("tts-output.mp3"),
        help="output MP3 path (default: tts-output.mp3)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="exercise the configured streaming provider (Qwen with Edge fallback)",
    )
    parser.add_argument(
        "--require-provider",
        choices=("qwen", "edge"),
        help="fail when the effective provider is not the requested one",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    node_factory: Callable[[], TTSNode] = TTSNode,
    stdout: TextIO | None = None,
) -> int:
    """Run one synthesis, returning zero on success and JSON in all cases."""
    output = stdout or sys.stdout
    try:
        args = _build_parser().parse_args(argv)
        node = node_factory()
        if args.stream:
            path, active_provider = asyncio.run(
                _stream_to_file_and_close(node, args.text, args.output)
            )
        else:
            path = node.synthesize_to_file(args.text, args.output)
            active_provider = node.active_provider()
        payload = {
            "ok": True,
            "module": "tts",
            "output": str(path),
            "format": DEFAULT_AUDIO_FORMAT,
            "media_type": DEFAULT_MEDIA_TYPE,
            "bytes": path.stat().st_size,
        }
        if active_provider:
            payload["provider"] = active_provider
        if args.stream:
            payload["streaming"] = True
        provider_mismatch = bool(
            args.require_provider and active_provider != args.require_provider
        )
        if provider_mismatch:
            payload.update(
                {
                    "ok": False,
                    "error_type": "ProviderMismatch",
                    "error": (
                        f"required provider {args.require_provider}, "
                        f"but active provider was {active_provider or 'unknown'}"
                    ),
                }
            )
            exit_code = 2
        else:
            exit_code = 0
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        payload = {
            "ok": False,
            "module": "tts",
            "format": DEFAULT_AUDIO_FORMAT,
            "error": str(exc) or type(exc).__name__,
            "error_type": type(exc).__name__,
        }
        exit_code = 1
    print(json.dumps(payload, ensure_ascii=False), file=output)
    return exit_code


async def _stream_to_file_and_close(
    node: TTSNode,
    text: str,
    output: Path,
) -> tuple[Path, str | None]:
    try:
        path = await node.synthesize_streamed_to_file(text, output)
        return path, node.active_provider()
    finally:
        await node.close()
