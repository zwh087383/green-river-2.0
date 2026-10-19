from __future__ import annotations

import hashlib
import re
from pathlib import Path

from rag_system.parsing.models import ParsedDocument


def document_id_for_document(document: ParsedDocument) -> str:
    payload = f"{document.source_name}:{file_sha256_for_document(document)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def document_id_for_file(path: Path | str, file_sha256: str) -> str:
    resolved = Path(path)
    payload = f"{resolved.name}:{file_sha256}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def file_sha256_for_document(document: ParsedDocument) -> str:
    value = document.metadata.get("file_sha256")
    if isinstance(value, str) and value:
        return value
    chunk_text = "\n".join(chunk.text for chunk in document.chunks)
    return hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    normalised = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha1(normalised.encode("utf-8")).hexdigest()[:16]
