from __future__ import annotations

import hashlib

from rag_system.voice import wakeup_models


def test_default_wakeup_model_dir_can_be_overridden(monkeypatch, tmp_path) -> None:
    target = tmp_path / "models" / "wakeup"
    monkeypatch.setenv("RAG_SYSTEM_WAKEUP_MODEL_DIR", str(target))

    assert wakeup_models.default_wakeup_model_dir() == target.resolve()


def test_verify_wakeup_model_reports_every_missing_runtime_file(tmp_path) -> None:
    result = wakeup_models.verify_wakeup_model(tmp_path / "missing")

    assert result.ready is False
    assert result.model_id == wakeup_models.WAKEUP_MODEL_ID
    assert result.revision == wakeup_models.WAKEUP_MODEL_REVISION
    assert result.size_bytes == 0
    for filename in wakeup_models.REQUIRED_WAKEUP_FILES:
        assert f"missing file: {filename}" in result.errors


def test_verify_wakeup_model_checks_all_int8_assets(monkeypatch, tmp_path) -> None:
    contents = {
        wakeup_models.WAKEUP_ENCODER_FILE: b"fake encoder",
        wakeup_models.WAKEUP_DECODER_FILE: b"fake decoder",
        wakeup_models.WAKEUP_JOINER_FILE: b"fake joiner",
        wakeup_models.WAKEUP_TOKENS_FILE: b"<blk> 0\n",
    }
    manifest = {
        filename: (len(content), hashlib.sha256(content).hexdigest())
        for filename, content in contents.items()
    }
    monkeypatch.setattr(wakeup_models, "WAKEUP_FILE_MANIFEST", manifest)
    for filename, content in contents.items():
        (tmp_path / filename).write_bytes(content)

    result = wakeup_models.verify_wakeup_model(tmp_path)

    assert result.ready is True
    assert result.errors == []
    assert result.size_bytes == len(contents[wakeup_models.WAKEUP_ENCODER_FILE])


def test_verify_wakeup_model_rejects_corrupt_non_encoder_asset(
    monkeypatch,
    tmp_path,
) -> None:
    contents = {
        wakeup_models.WAKEUP_ENCODER_FILE: b"fake encoder",
        wakeup_models.WAKEUP_DECODER_FILE: b"fake decoder",
        wakeup_models.WAKEUP_JOINER_FILE: b"fake joiner",
        wakeup_models.WAKEUP_TOKENS_FILE: b"<blk> 0\n",
    }
    manifest = {
        filename: (len(content), hashlib.sha256(content).hexdigest())
        for filename, content in contents.items()
    }
    decoder_size, _ = manifest[wakeup_models.WAKEUP_DECODER_FILE]
    manifest[wakeup_models.WAKEUP_DECODER_FILE] = (decoder_size, "0" * 64)
    monkeypatch.setattr(wakeup_models, "WAKEUP_FILE_MANIFEST", manifest)
    for filename, content in contents.items():
        (tmp_path / filename).write_bytes(content)

    result = wakeup_models.verify_wakeup_model(tmp_path)

    assert result.ready is False
    assert any(
        error.startswith(
            f"unexpected sha256 for {wakeup_models.WAKEUP_DECODER_FILE}:"
        )
        for error in result.errors
    )


def test_checked_in_wakeup_model_manifest_matches_local_download() -> None:
    model_dir = wakeup_models.default_wakeup_model_dir()
    if not wakeup_models.wakeup_model_files_present(model_dir):
        return

    result = wakeup_models.verify_wakeup_model(model_dir)

    assert result.ready, result.errors
