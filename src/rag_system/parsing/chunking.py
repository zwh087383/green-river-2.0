from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from rag_system.parsing.cleaning import normalize_text
from rag_system.parsing.models import DocumentChunk, ParsedSection


@dataclass(slots=True)
class ChunkingConfig:
    """Character-based chunking defaults.

    A tokenizer-aware implementation can replace this later without changing
    parser outputs or indexing inputs.
    """

    chunk_size: int = 2400
    chunk_overlap: int = 300

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")


class DocumentChunker:
    """Turn parsed sections into index-ready chunks."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk_sections(
        self,
        sections: list[ParsedSection],
        source_path: Path,
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []

        for section_index, section in enumerate(sections):
            text_parts = self._split_text(section.text)
            for chunk_index, chunk_text in enumerate(text_parts):
                metadata = {
                    **section.metadata,
                    "source": str(source_path),
                    "source_name": source_path.name,
                    "section_index": section_index,
                    "chunk_index": chunk_index,
                    "chunk_size": self.config.chunk_size,
                    "chunk_overlap": self.config.chunk_overlap,
                }
                chunks.append(
                    DocumentChunk(
                        id=self._build_chunk_id(source_path, section_index, chunk_index, chunk_text),
                        text=chunk_text,
                        metadata=metadata,
                    )
                )

        return chunks

    def _split_text(self, text: str) -> list[str]:
        text = normalize_text(text)
        if not text:
            return []
        if len(text) <= self.config.chunk_size:
            return [text]

        chunks: list[str] = []
        current = ""

        for paragraph in text.split("\n\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(paragraph) > self.config.chunk_size:
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.extend(self._hard_split(paragraph))
                continue

            candidate = paragraph if not current else f"{current}\n\n{paragraph}"
            if len(candidate) <= self.config.chunk_size:
                current = candidate
                continue

            chunks.append(current.strip())
            current = self._with_overlap(current, paragraph)

        if current:
            chunks.append(current.strip())

        return [chunk for chunk in chunks if chunk]

    def _hard_split(self, text: str) -> list[str]:
        parts: list[str] = []
        start = 0
        step = self.config.chunk_size - self.config.chunk_overlap
        while start < len(text):
            part = text[start : start + self.config.chunk_size].strip()
            if part:
                parts.append(part)
            start += step
        return parts

    def _with_overlap(self, previous: str, next_paragraph: str) -> str:
        overlap = previous[-self.config.chunk_overlap :].strip()
        return f"{overlap}\n\n{next_paragraph}" if overlap else next_paragraph

    def _build_chunk_id(
        self,
        source_path: Path,
        section_index: int,
        chunk_index: int,
        text: str,
    ) -> str:
        digest_input = f"{source_path.resolve()}:{section_index}:{chunk_index}:{text[:120]}"
        digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:12]
        return f"{source_path.name}:s{section_index}:c{chunk_index}:{digest}"
