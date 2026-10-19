from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Sequence
from typing import TextIO

from rag_system.module_test.catalog import MODULE_BY_COMMAND, MODULE_SPECS
from rag_system.module_test.doctor import build_doctor_report, report_exit_code


USAGE = """usage: python -m rag_system.module_test <module> [module arguments]

modules:
  doctor     offline readiness checks
  wakeup     WAV -> wakeup event
  vad        WAV -> speech segments
  asr        WAV -> text
  tts        text -> MP3
  rag-cloud  document -> parse/index result
  rag-edge   query -> retrieval result
  agent      query -> answer

Run `python -m rag_system.module_test list` for machine-readable metadata.
Run `python -m rag_system.module_test doctor` before customer acceptance tests.
Run `python -m rag_system.module_test <module> --help` for one module's options.
"""


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    output = stdout or sys.stdout
    if not args or args[0] in {"-h", "--help", "help"}:
        print(USAGE, file=output, end="")
        return 0
    if args[0] == "list":
        print(
            json.dumps(
                {"ok": True, "modules": [spec.to_dict() for spec in MODULE_SPECS]},
                ensure_ascii=False,
            ),
            file=output,
        )
        return 0
    if args[0] == "doctor":
        if len(args) != 1:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "command": "doctor",
                        "error_type": "InputError",
                        "error": "doctor does not accept additional arguments",
                    },
                    ensure_ascii=False,
                ),
                file=output,
            )
            return 1
        try:
            report = build_doctor_report()
        except Exception as exc:
            report = {
                "ok": False,
                "command": "doctor",
                "offline": True,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        print(json.dumps(report, ensure_ascii=False), file=output)
        return report_exit_code(report)

    command = args.pop(0).strip().lower().replace("_", "-")
    spec = MODULE_BY_COMMAND.get(command)
    if spec is None:
        print(
            json.dumps(
                {
                    "ok": False,
                    "module": command,
                    "error": f"unknown module: {command}",
                    "available": [item.command for item in MODULE_SPECS],
                },
                ensure_ascii=False,
            ),
            file=output,
        )
        return 1

    cli = importlib.import_module(f"{spec.package}.cli")
    return int(cli.main(args))


__all__ = ["USAGE", "main"]
