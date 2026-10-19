from __future__ import annotations

import hashlib
import os
from pathlib import Path

from rag_system.voice.models import ModelVerification, project_root


VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-onnx"
VAD_MODEL_REVISION = "v2.0.6"
VAD_MODEL_FILE = "model_quant.onnx"
VAD_MODEL_SIZE = 506_744
VAD_MODEL_SHA256 = "5289eb2aa3c9af2d7a4284bcfa7c3ceb81d360814ed4203239b6c5d0569da8a1"
VAD_MODEL_DIR_ENV = "RAG_SYSTEM_VAD_MODEL_DIR"
REQUIRED_VAD_FILES = (
    "am.mvn",
    "config.yaml",
    VAD_MODEL_FILE,
)


def default_vad_model_dir() -> Path:
    configured = str(os.environ.get(VAD_MODEL_DIR_ENV) or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (project_root() / "data" / "models" / "vad" / "fsmn-vad-onnx").resolve()


def vad_model_files_present(model_dir: Path | str | None = None) -> bool:
    root = Path(model_dir or default_vad_model_dir())
    return all((root / filename).is_file() for filename in REQUIRED_VAD_FILES)


def verify_vad_model(model_dir: Path | str | None = None) -> ModelVerification:
    root = Path(model_dir or default_vad_model_dir()).expanduser().resolve()
    errors: list[str] = []
    for filename in REQUIRED_VAD_FILES:
        if not (root / filename).is_file():
            errors.append(f"missing file: {filename}")

    model_path = root / VAD_MODEL_FILE
    size_bytes = model_path.stat().st_size if model_path.is_file() else 0
    digest = _sha256(model_path) if model_path.is_file() else ""
    if model_path.is_file() and size_bytes != VAD_MODEL_SIZE:
        errors.append(f"unexpected model size: {size_bytes}")
    if model_path.is_file() and digest != VAD_MODEL_SHA256:
        errors.append(f"unexpected model sha256: {digest}")

    return ModelVerification(
        model_id=VAD_MODEL_ID,
        revision=VAD_MODEL_REVISION,
        path=str(root),
        ready=not errors,
        size_bytes=size_bytes,
        sha256=digest,
        errors=errors,
    )


def download_vad_model(model_dir: Path | str | None = None) -> ModelVerification:
    try:
        from modelscope_hub.api import HubApi
        from modelscope_hub.constants import RepoType
    except ImportError as exc:
        raise RuntimeError(
            'modelscope-hub is not installed; install with the "vad" or "voice" extra'
        ) from exc

    root = Path(model_dir or default_vad_model_dir()).expanduser().resolve()
    HubApi(endpoint="https://modelscope.cn").download_repo(
        VAD_MODEL_ID,
        RepoType.MODEL,
        revision=VAD_MODEL_REVISION,
        local_dir=root,
        allow_patterns=list(REQUIRED_VAD_FILES),
        max_workers=4,
    )
    verification = verify_vad_model(root)
    if not verification.ready:
        details = "; ".join(verification.errors)
        raise RuntimeError(f"downloaded VAD model failed verification: {details}")
    return verification


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
