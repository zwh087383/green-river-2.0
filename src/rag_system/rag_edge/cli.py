from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from rag_system.contracts import ModuleArgumentParser, ModuleCLIUsageError
from rag_system.rag_edge.node import RAGEdgeNode, build_default_node


def build_parser() -> ModuleArgumentParser:
    parser = ModuleArgumentParser(
        prog="python -m rag_system.rag_edge",
        description="RAG edge/read-side single-node tester",
    )
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON")
    commands = parser.add_subparsers(dest="command", required=True)

    search_command = commands.add_parser("search", help="search the local knowledge base")
    search_command.add_argument("query", help="question or search phrase")
    search_command.add_argument("--top-k", type=int, default=5, help="maximum result count")
    search_command.add_argument(
        "--verbose",
        action="store_true",
        help="include retrieval metadata and ranking diagnostics",
    )
    _add_index_dir_argument(search_command)
    list_command = commands.add_parser("list", help="list indexed documents")
    _add_index_dir_argument(list_command)
    stats_command = commands.add_parser("stats", help="show index statistics")
    _add_index_dir_argument(stats_command)
    return parser


def main(argv: Sequence[str] | None = None, *, node: RAGEdgeNode | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except ModuleCLIUsageError as exc:
        _print_json(
            {
                "ok": False,
                "module": "rag-edge",
                "operation": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            pretty=False,
        )
        return 1

    active_node = node or build_default_node(index_dir=args.index_dir)
    logging.getLogger("jieba").setLevel(logging.ERROR)
    try:
        if args.command == "search":
            payload = active_node.search(
                args.query,
                top_k=args.top_k,
                include_details=args.verbose,
            )
        elif args.command == "list":
            payload = active_node.list()
        else:
            payload = active_node.stats()
    except Exception as exc:
        _print_json(
            {
                "ok": False,
                "module": "rag-edge",
                "operation": args.command,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            pretty=args.pretty,
        )
        return 1
    _print_json(payload, pretty=args.pretty)
    return 0


def _add_index_dir_argument(parser: ModuleArgumentParser) -> None:
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/acceptance/index"),
        help="isolated index directory (default: data/acceptance/index)",
    )


def _print_json(payload: object, *, pretty: bool, file=None) -> None:
    output = sys.stdout if file is None else file
    print(
        json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None),
        file=output,
    )
