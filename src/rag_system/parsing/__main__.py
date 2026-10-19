from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rag_system.parsing.service import ParsingService


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse documents into sections or chunks.")
    parser.add_argument("path", type=Path, help="File or directory to parse")
    parser.add_argument("--sections", action="store_true", help="Output parsed sections instead of chunks")
    parser.add_argument("--chunk-size", type=int, default=2400)
    parser.add_argument("--chunk-overlap", type=int, default=300)
    parser.add_argument(
        "--backend",
        choices=["light", "docling", "unstructured", "auto"],
        default="light",
        help="Parser backend. Heavy backends require optional dependencies.",
    )
    parser.add_argument("--document", action="store_true", help="Output rich ParsedDocument payload")
    parser.add_argument("--cache", action="store_true", help="Enable parser result cache")
    parser.add_argument("--ocr-languages", default="eng+chi_sim", help="Tesseract OCR language list")
    parser.add_argument("--no-recursive", action="store_true", help="Do not recurse when path is a directory")
    args = parser.parse_args()

    service = ParsingService.with_chunking(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        backend=args.backend,
        enable_cache=args.cache,
        ocr_languages=args.ocr_languages,
    )

    if args.document:
        if args.path.is_dir():
            payload = asdict(service.parse_many_documents([args.path], recursive=not args.no_recursive))
        else:
            payload = asdict(service.parse_document(args.path))
    elif args.sections:
        if args.path.is_dir():
            payload = []
            for file_path in service._iter_files([args.path], recursive=not args.no_recursive):
                payload.extend(asdict(section) for section in service.parse_file(file_path))
        else:
            payload = [asdict(section) for section in service.parse_file(args.path)]
    else:
        if args.path.is_dir():
            chunks = service.parse_many_to_chunks([args.path], recursive=not args.no_recursive)
        else:
            chunks = service.parse_to_chunks(args.path)
        payload = [asdict(chunk) for chunk in chunks]

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
