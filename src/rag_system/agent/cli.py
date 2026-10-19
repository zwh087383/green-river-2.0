from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

from rag_system.contracts import ModuleArgumentParser, ModuleCLIUsageError
from rag_system.agent.node import AgentNode, build_default_node


def build_parser() -> ModuleArgumentParser:
    parser = ModuleArgumentParser(
        prog="python -m rag_system.agent",
        description="Agent single-node tester",
    )
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON")
    commands = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("answer", "return one complete answer"),
        ("stream", "stream answer events as JSON Lines"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("query", help="user question")
        command.add_argument("--top-k", type=int, default=None, help="retrieval result limit")
        command.add_argument(
            "--index-dir",
            type=Path,
            default=Path("data/acceptance/index"),
            help="isolated index directory (default: data/acceptance/index)",
        )
        command.add_argument(
            "--verbose",
            action="store_true",
            help="include retrieval, evidence, and routing diagnostics",
        )
    return parser


def main(argv: Sequence[str] | None = None, *, node: AgentNode | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except ModuleCLIUsageError as exc:
        _print_json(
            {
                "ok": False,
                "module": "agent",
                "operation": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            pretty=False,
        )
        return 1

    active_node = node or build_default_node(index_dir=args.index_dir)
    try:
        if args.command == "answer":
            _print_json(
                active_node.answer(
                    args.query,
                    top_k=args.top_k,
                    include_details=args.verbose,
                ),
                pretty=args.pretty,
            )
        else:
            for event in active_node.stream(
                args.query,
                top_k=args.top_k,
                include_details=args.verbose,
            ):
                # A stream is a JSON Lines protocol: one complete JSON object per line.
                _print_json(event, pretty=False)
    except Exception as exc:
        _print_json(
            {
                "ok": False,
                "module": "agent",
                "operation": args.command,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            pretty=args.pretty if args.command == "answer" else False,
        )
        return 1
    return 0


def _print_json(payload: object, *, pretty: bool, file=None) -> None:
    output = sys.stdout if file is None else file
    print(
        json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None),
        file=output,
    )
