from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rag_system.parsing.heavy_parsers import DoclingParser


def main() -> int:
    parser = argparse.ArgumentParser(description="Internal Docling parser worker.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdf-do-ocr", action="store_true")
    parser.add_argument("--rapidocr-backend", default="onnxruntime")
    args = parser.parse_args()

    sections = DoclingParser(
        pdf_do_ocr=args.pdf_do_ocr,
        rapidocr_backend=args.rapidocr_backend,
    ).parse(args.path)
    args.output.write_text(
        json.dumps([asdict(section) for section in sections], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
