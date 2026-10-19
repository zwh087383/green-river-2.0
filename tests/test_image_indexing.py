from __future__ import annotations

from pathlib import Path

from PIL import Image

from rag_system.indexing import ImageIndexingService, InMemoryVectorStore
from rag_system.parsing.models import ParsedAsset, ParsedDocument, ParsedSection
from rag_system.settings import SettingsStore


class FakeImageEmbeddingClient:
    model = "qwen3-vl-embedding"
    dimensions = 2048

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        return [self._embed(path.read_bytes()) for path in image_paths]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text.encode("utf-8")) for text in texts]

    def _embed(self, data: bytes) -> list[float]:
        first = float(data[0] if data else 0)
        return [first, 1.0] + [0.0] * (self.dimensions - 2)


def _write_image(path: Path, color: str) -> None:
    image = Image.new("RGB", (40, 40), color)
    image.save(path)


def test_image_index_direct_image_and_search(tmp_path) -> None:
    image_path = tmp_path / "red.png"
    _write_image(image_path, "red")
    service = ImageIndexingService(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        vector_store=InMemoryVectorStore(collection_name="rag_images"),
        embedding_client=FakeImageEmbeddingClient(),
    )
    document = ParsedDocument(
        source=str(image_path),
        source_name=image_path.name,
        backend="auto",
        sections=[ParsedSection("RED OCR", {"parser": "image"})],
    )

    result = service.index_document_images(document)
    hits = service.search_by_image(image_path, top_k=1)

    assert result.status == "success"
    assert result.image_count == 1
    assert result.vector_count == 1
    assert result.model == "qwen3-vl-embedding"
    assert result.dimensions == 2048
    assert hits[0].metadata["source_name"] == "red.png"
    assert hits[0].image_path == str(image_path)


def test_image_index_embedded_asset_deduplicates_by_sha(tmp_path) -> None:
    image_path = tmp_path / "embedded.png"
    _write_image(image_path, "blue")
    service = ImageIndexingService(
        settings_store=SettingsStore(tmp_path / "settings.json"),
        vector_store=InMemoryVectorStore(collection_name="rag_images"),
        embedding_client=FakeImageEmbeddingClient(),
    )
    document = ParsedDocument(
        source=str(tmp_path / "demo.docx"),
        source_name="demo.docx",
        backend="auto",
        sections=[
            ParsedSection(
                "BLUE IMAGE OCR",
                {"parser": "embedded_image_ocr", "image_path": str(image_path)},
            )
        ],
        assets=[
            ParsedAsset(asset_id="asset-1", kind="image", path=str(image_path)),
            ParsedAsset(asset_id="asset-2", kind="image", path=str(image_path)),
        ],
    )

    result = service.index_document_images(document)

    assert result.image_count == 1
    assert result.images[0]["ocr_text"] == "BLUE IMAGE OCR"
    assert result.images[0]["source_name"] == "demo.docx"
