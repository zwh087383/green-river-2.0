from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rag_system.parsing.model_assets import download_docling_models, verify_docling_models
from rag_system.voice.models import download_asr_model, verify_asr_model
from rag_system.voice.vad_models import download_vad_model, verify_vad_model
from rag_system.voice.wakeup_models import download_wakeup_model, verify_wakeup_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage local models downloaded from ModelScope.")
    parser.add_argument("action", choices=("download", "verify"))
    parser.add_argument(
        "--component",
        choices=("all", "asr", "vad", "wakeup", "docling"),
        default="all",
    )
    parser.add_argument("--target", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.target is not None and args.component == "all":
        raise SystemExit("--target requires a specific --component")

    results: dict[str, dict[str, object]] = {}
    if args.component in {"all", "asr"}:
        asr_result = (
            download_asr_model(args.target)
            if args.action == "download"
            else verify_asr_model(args.target)
        )
        results["asr"] = asr_result.to_dict()
    if args.component in {"all", "vad"}:
        vad_result = (
            download_vad_model(args.target)
            if args.action == "download"
            else verify_vad_model(args.target)
        )
        results["vad"] = vad_result.to_dict()
    if args.component in {"all", "wakeup"}:
        wakeup_result = (
            download_wakeup_model(args.target)
            if args.action == "download"
            else verify_wakeup_model(args.target)
        )
        results["wakeup"] = wakeup_result.to_dict()
    if args.component in {"all", "docling"}:
        docling_result = (
            download_docling_models(args.target)
            if args.action == "download"
            else verify_docling_models(args.target)
        )
        results["docling"] = docling_result.to_dict()

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(bool(result["ready"]) for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
