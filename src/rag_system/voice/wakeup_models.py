from __future__ import annotations

import hashlib
import os
from pathlib import Path

from rag_system.voice.models import ModelVerification, project_root


WAKEUP_MODEL_ID = (
    "pkufool/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
)
WAKEUP_MODEL_REVISION = "master"
WAKEUP_MODEL_DIR_ENV = "RAG_SYSTEM_WAKEUP_MODEL_DIR"

WAKEUP_ENCODER_FILE = "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
WAKEUP_DECODER_FILE = "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
WAKEUP_JOINER_FILE = "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
WAKEUP_TOKENS_FILE = "tokens.txt"

# Values published by the ModelScope repository at WAKEUP_MODEL_REVISION.
WAKEUP_FILE_MANIFEST: dict[str, tuple[int, str]] = {
    WAKEUP_ENCODER_FILE: (
        4_807_159,
        "017af32f2c0138f931d05fbc009ee864295e910aff304f77d2f563815fc834fb",
    ),
    WAKEUP_DECODER_FILE: (
        181_025,
        "fe53b8d6a07bc5373d1770649025a5a985c8fb3dab70323386a1b73aacba0546",
    ),
    WAKEUP_JOINER_FILE: (
        65_208,
        "431de10b554f134ef8af320feea2db337e641290449a3d3f6cb6e5f5fd2c9c3d",
    ),
    WAKEUP_TOKENS_FILE: (
        1_627,
        "72316508d9119696145abc6f1f8cdc46287535c34e5ce7e595f845cb1499cf2e",
    ),
}
REQUIRED_WAKEUP_FILES = tuple(WAKEUP_FILE_MANIFEST)


def default_wakeup_model_dir() -> Path:
    configured = str(os.environ.get(WAKEUP_MODEL_DIR_ENV) or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        project_root()
        / "data"
        / "models"
        / "wakeup"
        / "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
    ).resolve()


def wakeup_model_files_present(model_dir: Path | str | None = None) -> bool:
    root = Path(model_dir or default_wakeup_model_dir())
    return all((root / filename).is_file() for filename in REQUIRED_WAKEUP_FILES)


def verify_wakeup_model(model_dir: Path | str | None = None) -> ModelVerification:
    """Verify every CPU INT8 runtime asset against the ModelScope manifest."""

    root = Path(model_dir or default_wakeup_model_dir()).expanduser().resolve()
    errors: list[str] = []

    for filename, (expected_size, expected_sha256) in WAKEUP_FILE_MANIFEST.items():
        path = root / filename
        if not path.is_file():
            errors.append(f"missing file: {filename}")
            continue

        size = path.stat().st_size
        if size != expected_size:
            errors.append(f"unexpected size for {filename}: {size}")
            continue

        digest = _sha256(path)
        if digest != expected_sha256:
            errors.append(f"unexpected sha256 for {filename}: {digest}")

    # ModelVerification historically reports the primary neural-network file.
    # The remaining required assets are still checked individually above.
    encoder_path = root / WAKEUP_ENCODER_FILE
    encoder_size = encoder_path.stat().st_size if encoder_path.is_file() else 0
    encoder_sha256 = _sha256(encoder_path) if encoder_path.is_file() else ""
    return ModelVerification(
        model_id=WAKEUP_MODEL_ID,
        revision=WAKEUP_MODEL_REVISION,
        path=str(root),
        ready=not errors,
        size_bytes=encoder_size,
        sha256=encoder_sha256,
        errors=errors,
    )


def download_wakeup_model(model_dir: Path | str | None = None) -> ModelVerification:
    """Download only the four files needed by the CPU INT8 KWS runtime."""

    try:
        from modelscope_hub.api import HubApi
        from modelscope_hub.constants import RepoType
    except ImportError as exc:
        raise RuntimeError(
            'modelscope-hub is not installed; install with the "wakeup" or "voice" extra'
        ) from exc

    root = Path(model_dir or default_wakeup_model_dir()).expanduser().resolve()
    HubApi(endpoint="https://modelscope.cn").download_repo(
        WAKEUP_MODEL_ID,
        RepoType.MODEL,
        revision=WAKEUP_MODEL_REVISION,
        local_dir=root,
        allow_patterns=list(REQUIRED_WAKEUP_FILES),
        max_workers=4,
    )
    verification = verify_wakeup_model(root)
    if not verification.ready:
        details = "; ".join(verification.errors)
        raise RuntimeError(f"downloaded wake-up model failed verification: {details}")
    return verification


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "REQUIRED_WAKEUP_FILES",
    "WAKEUP_DECODER_FILE",
    "WAKEUP_ENCODER_FILE",
    "WAKEUP_FILE_MANIFEST",
    "WAKEUP_JOINER_FILE",
    "WAKEUP_MODEL_DIR_ENV",
    "WAKEUP_MODEL_ID",
    "WAKEUP_MODEL_REVISION",
    "WAKEUP_TOKENS_FILE",
    "default_wakeup_model_dir",
    "download_wakeup_model",
    "verify_wakeup_model",
    "wakeup_model_files_present",
]
