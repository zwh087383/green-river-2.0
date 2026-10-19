from __future__ import annotations

import hashlib

from rag_system.voice import vad_models


def test_default_vad_model_dir_can_be_overridden(monkeypatch, tmp_path) -> None:
    target = tmp_path / "models" / "vad"
    monkeypatch.setenv("RAG_SYSTEM_VAD_MODEL_DIR", str(target))

    assert vad_models.default_vad_model_dir() == target.resolve()


def test_verify_vad_model_reports_missing_runtime_files(tmp_path) -> None:
    result = vad_models.verify_vad_model(tmp_path / "missing")

    assert result.ready is False
    assert result.model_id == vad_models.VAD_MODEL_ID
    assert result.revision == vad_models.VAD_MODEL_REVISION
    assert result.size_bytes == 0
    assert "missing file: am.mvn" in result.errors
    assert "missing file: config.yaml" in result.errors
    assert "missing file: model_quant.onnx" in result.errors


def test_verify_vad_model_accepts_complete_verified_model(monkeypatch, tmp_path) -> None:
    model_bytes = b"fake-quantized-vad"
    (tmp_path / "am.mvn").write_bytes(b"mvn")
    (tmp_path / "config.yaml").write_text("model: fsmn-vad", encoding="utf-8")
    (tmp_path / vad_models.VAD_MODEL_FILE).write_bytes(model_bytes)
    monkeypatch.setattr(vad_models, "VAD_MODEL_SIZE", len(model_bytes))
    monkeypatch.setattr(
        vad_models,
        "VAD_MODEL_SHA256",
        hashlib.sha256(model_bytes).hexdigest(),
    )

    result = vad_models.verify_vad_model(tmp_path)

    assert result.ready is True
    assert result.errors == []
    assert result.size_bytes == len(model_bytes)


def test_model_setup_parser_accepts_vad_component() -> None:
    from rag_system.model_setup import _build_parser

    args = _build_parser().parse_args(["verify", "--component", "vad"])

    assert args.action == "verify"
    assert args.component == "vad"
