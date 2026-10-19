from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rag_system.generation.models import Citation, EvidenceChunk
from rag_system.indexing.models import SearchResult


def build_evidence_chunks(
    results: Iterable[SearchResult],
    *,
    max_items: int = 4,
    max_chars_per_chunk: int = 3000,
) -> list[EvidenceChunk]:
    evidence: list[EvidenceChunk] = []
    for index, result in enumerate(list(results)[:max_items], start=1):
        metadata = dict(result.metadata)
        text = _truncate_text(result.text, max_chars=max_chars_per_chunk)
        evidence.append(
            EvidenceChunk(
                id=index,
                search_result_id=result.id,
                text=text,
                source_name=str(metadata.get("source_name") or metadata.get("source") or "unknown"),
                chunk_id=str(metadata.get("chunk_id") or metadata.get("vector_id") or result.id),
                heading=str(metadata.get("heading") or ""),
                heading_path=str(metadata.get("heading_path") or ""),
                page=_first_present(metadata, ["page_number", "page", "pages"]),
                section_index=_optional_int(metadata.get("section_index")),
                chunk_index=_optional_int(metadata.get("chunk_index")),
                score=float(result.score),
                metadata=metadata,
            )
        )
    return evidence


def build_context_text(evidence: list[EvidenceChunk]) -> str:
    blocks: list[str] = []
    for item in evidence:
        location_parts = [
            f"文件：{item.source_name}",
            f"chunk_id：{item.chunk_id}",
        ]
        if item.heading_path:
            location_parts.append(f"标题路径：{item.heading_path}")
        elif item.heading:
            location_parts.append(f"标题：{item.heading}")
        if item.page not in (None, ""):
            location_parts.append(f"页码：{item.page}")
        if item.section_index is not None:
            location_parts.append(f"section_index：{item.section_index}")
        if item.chunk_index is not None:
            location_parts.append(f"chunk_index：{item.chunk_index}")
        blocks.append(
            "\n".join(
                [
                    f"[{item.id}]",
                    *location_parts,
                    "内容：",
                    item.text,
                ]
            )
        )
    return "\n\n".join(blocks)


def citation_from_evidence(item: EvidenceChunk) -> Citation:
    metadata = item.metadata
    return Citation(
        id=item.id,
        source_name=item.source_name,
        chunk_id=item.chunk_id,
        heading=item.heading,
        heading_path=item.heading_path,
        page=item.page,
        section_index=item.section_index,
        chunk_index=item.chunk_index,
        score=item.score,
        rrf_score=_optional_float(metadata.get("rrf_score")),
        rerank_score=_optional_float(metadata.get("rerank_score")),
        retrieval_source=str(metadata.get("retrieval_source") or ""),
        matched_token_ratio=_optional_float(metadata.get("matched_token_ratio")),
        matched_token_count=_optional_int(metadata.get("matched_token_count")),
        route_count=_optional_int(metadata.get("route_count")),
        quote=_truncate_text(item.text, max_chars=260),
    )


def _first_present(metadata: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = metadata.get(key)
        if value not in (None, ""):
            return value
    return None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truncate_text(text: str, *, max_chars: int) -> str:
    cleaned = str(text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[:max_chars].rstrip()}\n...[truncated]"
