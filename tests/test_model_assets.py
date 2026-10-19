from __future__ import annotations

from rag_system.parsing.model_assets import (
    DOCLING_LAYOUT_MODEL_ID,
    default_docling_artifacts_dir,
    verify_docling_models,
)


def test_default_docling_artifacts_dir_can_be_overridden(monkeypatch, tmp_path) -> None:
    target = tmp_path / "docling-models"
    monkeypatch.setenv("RAG_SYSTEM_DOCLING_ARTIFACTS_DIR", str(target))

    assert default_docling_artifacts_dir() == target.resolve()


def test_verify_docling_models_reports_missing_files(tmp_path) -> None:
    result = verify_docling_models(tmp_path / "missing")

    assert result.ready is False
    assert DOCLING_LAYOUT_MODEL_ID in result.model_ids
    assert result.size_bytes == 0
    assert any("model.safetensors" in error for error in result.errors)
