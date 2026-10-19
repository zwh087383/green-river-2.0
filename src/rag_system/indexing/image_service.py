from __future__ import annotations

import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_system.indexing.document_ids import document_id_for_document
from rag_system.indexing.image_embeddings import ImageEmbeddingClient, QwenImageEmbeddingClient
from rag_system.indexing.manifest import IndexManifest
from rag_system.indexing.models import ImageIndexStats, ImageIndexingResult, ImageSearchResult
from rag_system.indexing.store import ChromaVectorStore, VectorStore
from rag_system.parsing.models import ParsedDocument
from rag_system.settings import SettingsStore


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class ImageIndexingService:
    def __init__(
        self,
        settings_store: SettingsStore | None = None,
        vector_store: VectorStore | None = None,
        embedding_client: ImageEmbeddingClient | None = None,
        manifest: IndexManifest | None = None,
        index_dir: Path | str = Path("data/index"),
        collection_name: str = "rag_images",
    ) -> None:
        self.settings_store = settings_store or SettingsStore()
        self.index_dir = Path(index_dir)
        self.collection_name = collection_name
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        # Text and image management share the active document enabled state;
        # image vectors themselves keep their existing independent collection.
        manifest_dir = self.index_dir / "embeddinggemma2_768_v1" if vector_store is None else self.index_dir
        self.manifest = manifest or IndexManifest(manifest_dir / "index.sqlite3")

    def index_document_images(self, document: ParsedDocument) -> ImageIndexingResult:
        candidates = self._image_candidates(document)
        if not candidates:
            raise ValueError(f"No images found to index for {document.source_name}")

        embedding_client = self._embedding_client()
        image_paths = [Path(candidate["image_path"]) for candidate in candidates]
        vectors = embedding_client.embed_images(image_paths)
        if len(vectors) != len(candidates):
            raise RuntimeError(f"Image embedding count mismatch: expected {len(candidates)}, got {len(vectors)}")
        for vector in vectors:
            if len(vector) != embedding_client.dimensions:
                raise RuntimeError(
                    f"Image embedding dimension mismatch: expected {embedding_client.dimensions}, got {len(vector)}"
                )

        ids = [str(candidate["image_id"]) for candidate in candidates]
        documents = [str(candidate.get("ocr_text") or candidate.get("caption") or candidate["image_path"]) for candidate in candidates]
        metadatas = [
            {
                **candidate,
                "model": embedding_client.model,
                "dimensions": embedding_client.dimensions,
            }
            for candidate in candidates
        ]
        vector_store = self._vector_store()
        vector_store.upsert(ids=ids, texts=documents, embeddings=vectors, metadatas=metadatas)
        indexed_at = datetime.now(timezone.utc).isoformat()
        document_id = str(candidates[0].get("document_id") or "")
        if document_id:
            self.manifest.update_image_index(
                document_id,
                image_count=len(candidates),
                image_vector_count=len(vectors),
                status="success",
            )
        return ImageIndexingResult(
            status="success",
            image_count=len(candidates),
            vector_count=len(vectors),
            model=embedding_client.model,
            dimensions=embedding_client.dimensions,
            collection=vector_store.collection_name,
            indexed_at=indexed_at,
            images=metadatas,
            message=f"Indexed {len(vectors)} image vectors",
        )

    def search_by_image(self, image_path: Path | str, top_k: int = 5) -> list[ImageSearchResult]:
        embedding_client = self._embedding_client()
        vectors = embedding_client.embed_images([Path(image_path)])
        if not vectors:
            return []
        expanded_top_k = min(max(top_k * 4, top_k), 80)
        return self._to_image_results(self._vector_store().query(vectors[0], top_k=expanded_top_k), top_k=top_k)

    def search_by_text(self, query: str, top_k: int = 5) -> list[ImageSearchResult]:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Image search query cannot be empty")
        embedding_client = self._embedding_client()
        vectors = embedding_client.embed_texts([cleaned])
        if not vectors:
            return []
        expanded_top_k = min(max(top_k * 4, top_k), 80)
        return self._to_image_results(self._vector_store().query(vectors[0], top_k=expanded_top_k), top_k=top_k)

    def stats(self) -> ImageIndexStats:
        settings = self.settings_store.load()
        return ImageIndexStats(
            image_count=self._safe_vector_count(),
            vector_count=self._safe_vector_count(),
            collection=self.collection_name,
            model=settings.image_embedding_model,
            dimensions=settings.image_embedding_dimensions,
        )

    def delete_document_images(self, document_id: str) -> dict[str, Any]:
        deleted = {"document_id": document_id, "image_vectors": 0}
        try:
            deleted["image_vectors"] = self._vector_store().delete_where({"document_id": document_id})
        except RuntimeError:
            deleted["image_vectors"] = 0
        self.manifest.update_image_index(
            document_id,
            image_count=0,
            image_vector_count=0,
            status="deleted",
        )
        return deleted

    def list_document_images(self, document_id: str) -> list[dict[str, Any]]:
        try:
            items = self._vector_store().list_all()
        except RuntimeError:
            return []
        images = []
        for item in items:
            if str(item.metadata.get("document_id") or "") != document_id:
                continue
            images.append(
                {
                    "id": item.id,
                    "text": item.text,
                    "metadata": item.metadata,
                    "image_data_url": None,
                }
            )
        return images

    def _image_candidates(self, document: ParsedDocument) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        direct_path = Path(document.source)
        if direct_path.suffix.lower() in IMAGE_SUFFIXES and direct_path.exists():
            candidates.append(
                self._candidate_from_path(
                    direct_path,
                    document=document,
                    origin="uploaded_image",
                    metadata={"ocr_text": "\n\n".join(section.text for section in document.sections if section.text)},
                )
            )

        for asset in document.assets:
            if not asset.path:
                continue
            asset_path = Path(asset.path)
            if not asset_path.exists() or asset_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            candidates.append(
                self._candidate_from_path(
                    asset_path,
                    document=document,
                    origin=str(asset.metadata.get("origin") or asset.kind),
                    metadata={
                        "asset_id": asset.asset_id,
                        "caption": asset.caption or "",
                        "page": asset.page or "",
                        "ocr_text": self._ocr_text_for_asset(document, asset.path),
                        "embedded_path": str(asset.metadata.get("embedded_path") or ""),
                    },
                )
            )

        deduped: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            deduped.setdefault(str(candidate["image_sha256"]), candidate)
        return list(deduped.values())

    def _candidate_from_path(
        self,
        image_path: Path,
        *,
        document: ParsedDocument,
        origin: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        image_sha256 = _file_sha256(image_path)
        document_id = document_id_for_document(document)
        return {
            "image_id": _image_id(document, image_sha256),
            "document_id": document_id,
            "image_sha256": image_sha256,
            "image_path": str(image_path),
            "source": document.source,
            "source_name": document.source_name,
            "document_backend": document.backend,
            "origin": origin,
            **metadata,
        }

    def _ocr_text_for_asset(self, document: ParsedDocument, image_path: str) -> str:
        for section in document.sections:
            if str(section.metadata.get("image_path") or "") == image_path:
                return section.text
        return ""

    def _embedding_client(self) -> ImageEmbeddingClient:
        if self.embedding_client is None:
            self.embedding_client = QwenImageEmbeddingClient.from_settings(self.settings_store.load())
        return self.embedding_client

    def _vector_store(self) -> VectorStore:
        if self.vector_store is None:
            self.vector_store = ChromaVectorStore(
                path=self.index_dir / "chroma",
                collection_name=self.collection_name,
            )
        return self.vector_store

    def _safe_vector_count(self) -> int:
        try:
            return self._vector_store().count()
        except RuntimeError:
            return 0

    def _to_image_results(self, results: list[Any], *, top_k: int) -> list[ImageSearchResult]:
        disabled_document_ids = self.manifest.disabled_document_ids()
        image_results: list[ImageSearchResult] = []
        for result in results:
            document_id = str(result.metadata.get("document_id") or "")
            if document_id and document_id in disabled_document_ids:
                continue
            image_path = str(result.metadata.get("image_path") or "")
            image_results.append(
                ImageSearchResult(
                    image_id=str(result.metadata.get("image_id") or result.id),
                    image_path=image_path,
                    score=result.score,
                    metadata=dict(result.metadata),
                )
            )
        return image_results[: max(1, top_k)]


def image_indexing_result_to_dict(result: ImageIndexingResult) -> dict[str, Any]:
    return asdict(result)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_id(document: ParsedDocument, image_sha256: str) -> str:
    raw = f"{document_id_for_document(document)}:{image_sha256}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]
