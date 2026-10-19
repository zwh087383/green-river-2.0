"""Re-embed legacy Chroma text in an isolated namespace, preserving source IDs."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from rag_system.indexing.bm25 import BM25Index
from rag_system.indexing.embeddings import EmbeddingGemmaClient
from rag_system.indexing.manifest import IndexManifest
from rag_system.indexing.store import ChromaVectorStore
from rag_system.settings import SettingsStore


def migrate(index_dir: Path, settings_path: Path) -> dict:
    import chromadb

    base = index_dir.resolve()
    if not (base / "chroma" / "chroma.sqlite3").is_file():
        raise RuntimeError("Legacy Chroma database not found; index parsed documents normally")
    if not (base / "index.sqlite3").is_file():
        raise RuntimeError("Legacy document manifest is missing; cannot preserve document state")
    legacy = chromadb.PersistentClient(path=str(base / "chroma")).get_collection("rag_chunks")
    data = legacy.get(include=["documents", "metadatas"])
    rows = sorted(zip(data["ids"], data["documents"], data["metadatas"]), key=lambda r: r[0])
    if not rows:
        raise RuntimeError("Legacy collection is empty")
    fingerprint = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    target = base / "embeddinggemma2_768_v1"
    target.mkdir(parents=True, exist_ok=True)
    marker = target / "migration-in-progress.json"
    receipt = target / "migration.json"
    if receipt.exists() and not marker.exists():
        existing = json.loads(receipt.read_text(encoding="utf-8"))
        if existing.get("source_fingerprint") == fingerprint:
            return existing
        raise RuntimeError("Target already migrated from a different source; refusing to overwrite it")
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8"))["source_fingerprint"] != fingerprint:
        raise RuntimeError("Legacy source changed during migration; refusing to mix snapshots")
    if not marker.exists() and (target / "chroma" / "chroma.sqlite3").exists():
        active_client = chromadb.PersistentClient(path=str(target / "chroma"))
        collections = {c.name for c in active_client.list_collections()}
        if "rag_chunks__embeddinggemma2_768_v1" in collections:
            if active_client.get_collection("rag_chunks__embeddinggemma2_768_v1").count():
                raise RuntimeError("Target already contains independently indexed data; refusing to overwrite it")
    marker.write_text(json.dumps({"source_fingerprint": fingerprint}), encoding="utf-8")
    client = EmbeddingGemmaClient.from_settings(SettingsStore(settings_path).load())
    store = ChromaVectorStore(target / "chroma", "rag_chunks__embeddinggemma2_768_v1")
    bm25 = BM25Index(target / "bm25.sqlite3")
    manifest = IndexManifest(target / "index.sqlite3")
    counts: Counter[str] = Counter()
    for start in range(0, len(rows), 16):
        batch = rows[start:start + 16]
        ids = [r[0] for r in batch]
        texts = [r[1] for r in batch]
        metadata = [dict(r[2] or {}, model=client.model, dimensions=client.dimensions) for r in batch]
        vectors = client.embed_texts(texts)
        store.upsert(ids, texts, vectors, metadata)
        bm25.upsert_chunks(ids=ids, texts=texts, metadatas=metadata)
        counts.update(str(m.get("document_id", "")) for m in metadata)
    old_manifest = IndexManifest(base / "index.sqlite3")
    for document in old_manifest.list_documents():
        count = counts.get(document.document_id, 0)
        if count:
            manifest.upsert_document(replace(document, model=client.model, dimensions=768,
                vector_count=count, chunk_count=count, bm25_count=count, sparse_count=0))
        else:
            # Preserve parsed-only and image-only document records as well.
            manifest.upsert_document(replace(document, vector_count=0, bm25_count=0,
                                             sparse_count=0, model="", dimensions=0))
    migrated = store.list_all()
    if {r.id: r.text for r in migrated} != {r[0]: r[1] for r in rows}:
        raise RuntimeError("Migration verification failed: chunk IDs or text mismatch")
    source_disabled = old_manifest.disabled_document_ids()
    if manifest.disabled_document_ids() != source_disabled:
        raise RuntimeError("Migration verification failed: enabled state mismatch")
    # Verify the original collection was not edited while we encoded it.
    latest = legacy.get(include=["documents", "metadatas"])
    latest_rows = sorted(zip(latest["ids"], latest["documents"], latest["metadatas"]), key=lambda r: r[0])
    if latest_rows != rows:
        raise RuntimeError("Legacy collection changed during migration")
    result = {"model": client.model, "dimensions": 768, "vector_count": len(rows),
              "document_count": len(counts), "disabled_document_count": len(source_disabled),
              "source_fingerprint": fingerprint, "source": str(base / "chroma"),
              "target": str(target), "completed_at": datetime.now(timezone.utc).isoformat()}
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    marker.unlink()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=Path("data/index"))
    parser.add_argument("--settings", type=Path, default=Path("data/settings.json"))
    args = parser.parse_args()
    print(json.dumps(migrate(args.index_dir, args.settings), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
