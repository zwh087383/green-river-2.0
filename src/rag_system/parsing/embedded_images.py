from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_system.parsing.base import OCRBackend
from rag_system.parsing.cleaning import normalize_text
from rag_system.parsing.models import ParsedSection, ParsingDiagnostic
from rag_system.parsing.parsers import ImageParser


SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

OFFICE_MEDIA_PREFIXES = {
    ".docx": ("word/media/",),
    ".docm": ("word/media/",),
    ".pptx": ("ppt/media/",),
    ".pptm": ("ppt/media/",),
    ".xlsx": ("xl/media/",),
    ".xlsm": ("xl/media/",),
}


@dataclass(slots=True)
class EmbeddedImageOCRResult:
    sections: list[ParsedSection] = field(default_factory=list)
    diagnostics: list[ParsingDiagnostic] = field(default_factory=list)


def extract_embedded_image_ocr_sections(
    source_path: Path,
    *,
    output_dir: Path | str = Path("data/extracted_assets"),
    ocr_languages: str = "eng+chi_sim",
    ocr_backend: OCRBackend | None = None,
) -> EmbeddedImageOCRResult:
    suffix = source_path.suffix.lower()
    if suffix in OFFICE_MEDIA_PREFIXES:
        return _extract_office_images(
            source_path,
            Path(output_dir),
            ocr_languages=ocr_languages,
            ocr_backend=ocr_backend,
        )
    if suffix == ".pdf":
        return _extract_pdf_images(
            source_path,
            Path(output_dir),
            ocr_languages=ocr_languages,
            ocr_backend=ocr_backend,
        )
    return EmbeddedImageOCRResult()


def _extract_office_images(
    source_path: Path,
    output_dir: Path,
    *,
    ocr_languages: str,
    ocr_backend: OCRBackend | None,
) -> EmbeddedImageOCRResult:
    result = EmbeddedImageOCRResult()
    if not zipfile.is_zipfile(source_path):
        return result

    prefixes = OFFICE_MEDIA_PREFIXES[source_path.suffix.lower()]
    try:
        with zipfile.ZipFile(source_path) as archive:
            names = [
                name
                for name in archive.namelist()
                if name.lower().startswith(prefixes)
                and Path(name).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ]
            seen_hashes: set[str] = set()
            for image_index, name in enumerate(sorted(names)):
                data = archive.read(name)
                image_hash = hashlib.sha256(data).hexdigest()
                if image_hash in seen_hashes:
                    continue
                seen_hashes.add(image_hash)
                image_path = _write_image_bytes(source_path, output_dir, name, data)
                section = _ocr_image_to_section(
                    image_path,
                    source_path=source_path,
                    image_index=image_index,
                    image_hash=image_hash,
                    embedded_path=name,
                    origin="office_embedded_image",
                    ocr_languages=ocr_languages,
                    ocr_backend=ocr_backend,
                    extra_metadata={},
                )
                if section is not None:
                    result.sections.append(section)
    except Exception as exc:
        result.diagnostics.append(
            ParsingDiagnostic(
                level="warning",
                code="embedded_image_extraction_failed",
                message=f"Could not extract embedded images from {source_path.name}: {exc}",
                metadata={"source_name": source_path.name, "content_type": source_path.suffix.lower().lstrip(".")},
            )
        )

    if result.sections:
        result.diagnostics.append(
            ParsingDiagnostic(
                level="info",
                code="embedded_image_ocr",
                message=f"OCR extracted text from {len(result.sections)} embedded image(s).",
                metadata={"source_name": source_path.name, "image_count": len(result.sections)},
            )
        )
    return result


def _extract_pdf_images(
    source_path: Path,
    output_dir: Path,
    *,
    ocr_languages: str,
    ocr_backend: OCRBackend | None,
) -> EmbeddedImageOCRResult:
    result = EmbeddedImageOCRResult()
    try:
        from pypdf import PdfReader
    except ImportError:
        return result

    try:
        reader = PdfReader(str(source_path))
        seen_hashes: set[str] = set()
        image_counter = 0
        for page_index, page in enumerate(reader.pages):
            for image in getattr(page, "images", []):
                data = _pdf_image_bytes(image)
                if not data:
                    continue
                image_hash = hashlib.sha256(data).hexdigest()
                if image_hash in seen_hashes:
                    continue
                seen_hashes.add(image_hash)
                image_name = _pdf_image_name(image, page_index, image_counter)
                image_path = _write_image_bytes(source_path, output_dir, image_name, data)
                section = _ocr_image_to_section(
                    image_path,
                    source_path=source_path,
                    image_index=image_counter,
                    image_hash=image_hash,
                    embedded_path=image_name,
                    origin="pdf_embedded_image",
                    ocr_languages=ocr_languages,
                    ocr_backend=ocr_backend,
                    extra_metadata={"page": page_index + 1},
                )
                if section is not None:
                    result.sections.append(section)
                image_counter += 1
    except Exception as exc:
        result.diagnostics.append(
            ParsingDiagnostic(
                level="warning",
                code="pdf_embedded_image_extraction_failed",
                message=f"Could not extract embedded PDF images from {source_path.name}: {exc}",
                metadata={"source_name": source_path.name, "content_type": "pdf"},
            )
        )

    if result.sections:
        result.diagnostics.append(
            ParsingDiagnostic(
                level="info",
                code="embedded_image_ocr",
                message=f"OCR extracted text from {len(result.sections)} embedded image(s).",
                metadata={"source_name": source_path.name, "image_count": len(result.sections)},
            )
        )
    return result


def _ocr_image_to_section(
    image_path: Path,
    *,
    source_path: Path,
    image_index: int,
    image_hash: str,
    embedded_path: str,
    origin: str,
    ocr_languages: str,
    ocr_backend: OCRBackend | None,
    extra_metadata: dict[str, Any],
) -> ParsedSection | None:
    sections = ImageParser(ocr_backend=ocr_backend, ocr_languages=ocr_languages).parse(image_path)
    if not sections:
        return None
    text = normalize_text("\n\n".join(section.text for section in sections))
    if not text:
        return None
    effective_languages = sections[0].metadata.get("ocr_languages", ocr_languages)
    return ParsedSection(
        text=text,
        metadata={
            "parser": "embedded_image_ocr",
            "content_type": "embedded_image",
            "element_type": "EmbeddedImageOCR",
            "origin": origin,
            "source_name": source_path.name,
            "parent_content_type": source_path.suffix.lower().lstrip("."),
            "embedded_path": embedded_path,
            "image_path": str(image_path),
            "image_index": image_index,
            "image_sha256": image_hash,
            "ocr_languages": effective_languages,
            **extra_metadata,
        },
    )


def _write_image_bytes(source_path: Path, output_dir: Path, embedded_path: str, data: bytes) -> Path:
    file_digest = _file_sha256(source_path)[:16]
    image_digest = hashlib.sha256(data).hexdigest()[:16]
    extension = Path(embedded_path).suffix.lower()
    if extension not in SUPPORTED_IMAGE_EXTENSIONS:
        extension = ".bin"
    target_dir = output_dir / file_digest
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = _safe_stem(Path(embedded_path).stem)
    target = target_dir / f"{safe_stem}-{image_digest}{extension}"
    if not target.exists():
        target.write_bytes(data)
    return target


def _pdf_image_bytes(image: object) -> bytes | None:
    data = getattr(image, "data", None)
    if isinstance(data, bytes):
        return data
    image_object = getattr(image, "image", None)
    if image_object is not None:
        try:
            from io import BytesIO

            handle = BytesIO()
            image_object.save(handle, format="PNG")
            return handle.getvalue()
        except Exception:
            return None
    return None


def _pdf_image_name(image: object, page_index: int, image_index: int) -> str:
    name = getattr(image, "name", None)
    if isinstance(name, str) and Path(name).suffix:
        return name
    return f"page-{page_index + 1}-image-{image_index + 1}.png"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_stem(stem: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem).strip("._-")
    return cleaned or "embedded-image"
