from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path


DOCLING_LAYOUT_MODEL_ID = "ds4sd/docling-layout-heron"
DOCLING_TABLE_MODEL_ID = "AI-ModelScope/docling-models"
DOCLING_RAPIDOCR_MODEL_ID = "RapidAI/RapidOCR"
DOCLING_RAPIDOCR_REVISION = "v3.9.0"
DOCLING_ARTIFACTS_DIR_ENV = "RAG_SYSTEM_DOCLING_ARTIFACTS_DIR"

DOCLING_FILE_MANIFEST: dict[str, tuple[int, str]] = {
    "docling-project--docling-layout-heron/config.json": (
        3_268,
        "fdea30805ce2f5666b147fca941dcdd27ad468e27d6ed21902207d3da056a97d",
    ),
    "docling-project--docling-layout-heron/model.safetensors": (
        171_658_996,
        "00333a43451945aaf89db8ca9c0a17e75d1537c17db60fdb91aa95f4c7929e0c",
    ),
    "docling-project--docling-layout-heron/preprocessor_config.json": (
        444,
        "cd38cd59999e7a95d68e487fbe5132df3d4e5c32a0836add57e6126ba0c4eaf1",
    ),
    "docling-project--docling-models/model_artifacts/tableformer/accurate/tm_config.json": (
        7_060,
        "984e122ceb8ccf84d84c9d2882f6f2302a44b4f1e577babd6289892c36f3cffd",
    ),
    (
        "docling-project--docling-models/model_artifacts/tableformer/accurate/"
        "tableformer_accurate.safetensors"
    ): (
        212_758_388,
        "2a7d6c924b3cd12fb99a09280ca9c33a89c5d60b93253617d2e088c1a40374d9",
    ),
    "RapidOcr/onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx": (
        9_929_594,
        "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
    ),
    "RapidOcr/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx": (
        585_532,
        "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c",
    ),
    "RapidOcr/onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx": (
        21_234_383,
        "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
    ),
    "RapidOcr/resources/fonts/FZYTK.TTF": (
        3_241_748,
        "4065a23df6823c8e2b69a0e76d02f02a6470b8774a5e91086609701ad95cc33f",
    ),
}


@dataclass(frozen=True, slots=True)
class DoclingModelVerification:
    model_ids: list[str]
    path: str
    ready: bool
    size_bytes: int
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_docling_artifacts_dir() -> Path:
    configured = str(os.environ.get(DOCLING_ARTIFACTS_DIR_ENV) or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (project_root() / "data" / "models" / "docling").resolve()


def docling_model_files_present(model_dir: Path | str | None = None) -> bool:
    root = Path(model_dir or default_docling_artifacts_dir())
    return all((root / relative_path).is_file() for relative_path in DOCLING_FILE_MANIFEST)


def verify_docling_models(model_dir: Path | str | None = None) -> DoclingModelVerification:
    root = Path(model_dir or default_docling_artifacts_dir()).expanduser().resolve()
    errors: list[str] = []
    total_size = 0
    for relative_path, (expected_size, expected_hash) in DOCLING_FILE_MANIFEST.items():
        path = root / relative_path
        if not path.is_file():
            errors.append(f"missing file: {relative_path}")
            continue
        actual_size = path.stat().st_size
        total_size += actual_size
        if actual_size != expected_size:
            errors.append(f"unexpected size for {relative_path}: {actual_size}")
            continue
        digest = _sha256(path)
        if digest != expected_hash:
            errors.append(f"unexpected sha256 for {relative_path}: {digest}")

    return DoclingModelVerification(
        model_ids=[
            DOCLING_LAYOUT_MODEL_ID,
            DOCLING_TABLE_MODEL_ID,
            f"{DOCLING_RAPIDOCR_MODEL_ID}@{DOCLING_RAPIDOCR_REVISION}",
        ],
        path=str(root),
        ready=not errors,
        size_bytes=total_size,
        errors=errors,
    )


def download_docling_models(model_dir: Path | str | None = None) -> DoclingModelVerification:
    try:
        from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel
        from modelscope_hub.api import HubApi
        from modelscope_hub.constants import RepoType
    except ImportError as exc:
        raise RuntimeError(
            'Docling model setup requires the "heavy" project extra'
        ) from exc

    root = Path(model_dir or default_docling_artifacts_dir()).expanduser().resolve()
    api = HubApi(endpoint="https://modelscope.cn")
    api.download_repo(
        DOCLING_LAYOUT_MODEL_ID,
        RepoType.MODEL,
        revision="master",
        local_dir=root / "docling-project--docling-layout-heron",
        max_workers=4,
    )
    api.download_repo(
        DOCLING_TABLE_MODEL_ID,
        RepoType.MODEL,
        revision="master",
        local_dir=root / "docling-project--docling-models",
        allow_patterns=[
            "config.json",
            "configuration.json",
            "model_artifacts/tableformer/accurate/**",
        ],
        max_workers=4,
    )
    RapidOcrModel.download_models(
        backend="onnxruntime",
        lang="chinese",
        local_dir=root / "RapidOcr",
        progress=True,
    )

    verification = verify_docling_models(root)
    if not verification.ready:
        details = "; ".join(verification.errors)
        raise RuntimeError(f"downloaded Docling models failed verification: {details}")
    return verification


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
