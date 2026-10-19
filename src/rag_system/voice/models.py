from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


ASR_MODEL_FILE = "model.bin"
ASR_MODEL_DIR_ENV = "RAG_SYSTEM_ASR_MODEL_DIR"
REQUIRED_ASR_FILES = (
    "config.json",
    ASR_MODEL_FILE,
    "tokenizer.json",
    "vocabulary.txt",
)


@dataclass(frozen=True, slots=True)
class ASRModelSpec:
    name: str
    model_id: str
    revision: str
    directory_name: str
    model_size: int
    model_sha256: str


ASR_MODEL_SPECS = {
    "base": ASRModelSpec(
        name="base",
        model_id="Systran/faster-whisper-base",
        revision="f445defbb81556b9d64c07284683f4cdc2cc17f5",
        directory_name="faster-whisper-base",
        model_size=145_217_532,
        model_sha256="d01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9",
    ),
    "medium": ASRModelSpec(
        name="medium",
        model_id="Systran/faster-whisper-medium",
        revision="4425f9d17a866befa63d7400866fdb136304b04f",
        directory_name="faster-whisper-medium",
        model_size=1_527_906_378,
        model_sha256="9b45e1009dcc4ab601eff815b61d80e60ce3fd8c74c1a14f4a282258286b51ae",
    ),
}

ASR_MODEL_NAME = "medium"
ASR_MODEL_ID = ASR_MODEL_SPECS[ASR_MODEL_NAME].model_id
ASR_MODEL_REVISION = ASR_MODEL_SPECS[ASR_MODEL_NAME].revision
ASR_MODEL_SIZE = ASR_MODEL_SPECS[ASR_MODEL_NAME].model_size
ASR_MODEL_SHA256 = ASR_MODEL_SPECS[ASR_MODEL_NAME].model_sha256


@dataclass(frozen=True, slots=True)
class ModelVerification:
    model_id: str
    revision: str
    path: str
    ready: bool
    size_bytes: int
    sha256: str
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_asr_model_dir() -> Path:
    configured = str(os.environ.get(ASR_MODEL_DIR_ENV) or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        project_root()
        / "data"
        / "models"
        / "asr"
        / ASR_MODEL_SPECS[ASR_MODEL_NAME].directory_name
    ).resolve()


def _model_spec(model_name: str) -> ASRModelSpec | None:
    cleaned = str(model_name or "").strip()
    for spec in ASR_MODEL_SPECS.values():
        if cleaned in {spec.name, spec.model_id, spec.directory_name}:
            return spec
    return None


def resolve_asr_model_dir(model_name: str) -> Path:
    cleaned = str(model_name or "").strip()
    spec = _model_spec(cleaned)
    if spec is not None and spec.name == ASR_MODEL_NAME:
        return default_asr_model_dir()
    if spec is not None:
        return (
            project_root() / "data" / "models" / "asr" / spec.directory_name
        ).resolve()
    return Path(cleaned).expanduser().resolve()


def asr_model_files_present(model_dir: Path | str | None = None) -> bool:
    root = Path(model_dir or default_asr_model_dir())
    return all((root / filename).is_file() for filename in REQUIRED_ASR_FILES)


def verify_asr_model(model_dir: Path | str | None = None) -> ModelVerification:
    spec = ASR_MODEL_SPECS[ASR_MODEL_NAME]
    root = Path(model_dir or default_asr_model_dir()).expanduser().resolve()
    errors: list[str] = []
    for filename in REQUIRED_ASR_FILES:
        if not (root / filename).is_file():
            errors.append(f"missing file: {filename}")

    model_path = root / ASR_MODEL_FILE
    size_bytes = model_path.stat().st_size if model_path.is_file() else 0
    digest = _sha256(model_path) if model_path.is_file() else ""
    if model_path.is_file() and size_bytes != spec.model_size:
        errors.append(f"unexpected model size: {size_bytes}")
    if model_path.is_file() and digest != spec.model_sha256:
        errors.append(f"unexpected model sha256: {digest}")

    return ModelVerification(
        model_id=spec.model_id,
        revision=spec.revision,
        path=str(root),
        ready=not errors,
        size_bytes=size_bytes,
        sha256=digest,
        errors=errors,
    )


def download_asr_model(model_dir: Path | str | None = None) -> ModelVerification:
    try:
        from modelscope_hub.api import HubApi
        from modelscope_hub.constants import RepoType
    except ImportError as exc:
        raise RuntimeError(
            'modelscope-hub is not installed; install with the "asr-local" or "voice" extra'
        ) from exc

    spec = ASR_MODEL_SPECS[ASR_MODEL_NAME]
    root = Path(model_dir or default_asr_model_dir()).expanduser().resolve()
    HubApi(endpoint="https://modelscope.cn").download_repo(
        spec.model_id,
        RepoType.MODEL,
        revision=spec.revision,
        local_dir=root,
        max_workers=4,
    )
    verification = verify_asr_model(root)
    if not verification.ready:
        details = "; ".join(verification.errors)
        raise RuntimeError(f"downloaded ASR model failed verification: {details}")
    return verification


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage offline voice models from ModelScope.")
    parser.add_argument("action", choices=("download", "verify"))
    parser.add_argument("--target", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = (
        download_asr_model(args.target)
        if args.action == "download"
        else verify_asr_model(args.target)
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
