from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from rag_system.contracts import ModuleArgumentParser, ModuleCLIUsageError
from rag_system.rag_cloud.node import RAGCloudNode, build_default_node


def build_parser() -> ModuleArgumentParser:
    parser = ModuleArgumentParser(
        prog="python -m rag_system.rag_cloud",
        description="RAG cloud/write-side single-node tester",
    )
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON")
    commands = parser.add_subparsers(dest="command", required=True)

    parse_command = commands.add_parser("parse", help="parse one document")
    parse_command.add_argument("source", help="document path")
    _add_index_dir_argument(parse_command)

    index_command = commands.add_parser("index", help="parse and index one document")
    index_command.add_argument("source", help="document path")
    _add_index_dir_argument(index_command)

    delete_command = commands.add_parser("delete", help="delete one indexed document")
    delete_command.add_argument("document_id", help="document ID")
    delete_command.add_argument(
        "--keep-manifest",
        action="store_true",
        help="keep the document manifest row",
    )
    _add_index_dir_argument(delete_command)
    return parser


def main(argv: Sequence[str] | None = None, *, node: RAGCloudNode | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except ModuleCLIUsageError as exc:
        _print_json(
            {
                "ok": False,
                "module": "rag-cloud",
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
        if args.command == "parse":
            payload = active_node.parse(args.source)
        elif args.command == "index":
            payload = active_node.index(args.source)
        else:
            payload = active_node.delete(args.document_id, remove_manifest=not args.keep_manifest)
    except Exception as exc:
        _print_json(
            {
                "ok": False,
                "module": "rag-cloud",
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
