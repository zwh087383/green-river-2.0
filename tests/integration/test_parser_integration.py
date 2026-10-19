from __future__ import annotations

import os
from pathlib import Path

import pytest

from rag_system.parsing import ParsingService


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_PARSER_INTEGRATION") != "1",
    reason="Set RUN_PARSER_INTEGRATION=1 to run parser integration samples.",
)


@pytest.mark.parametrize(
    ("sample", "min_sections"),
    [
        ("samples/online/python_tutorial_introduction.html", 1),
        ("samples/online/controlled_table_sample.xlsx", 1),
        ("samples/online/controlled_ocr_text.png", 1),
        ("samples/online/controlled_chinese_ocr_text_large.png", 1),
        ("samples/online/w3c_dummy.pdf", 1),
        ("samples/online/controlled_email_sample.eml", 1),
    ],
)
def test_auto_parser_real_samples(sample: str, min_sections: int, tmp_path: Path) -> None:
    service = ParsingService.with_chunking(
        backend="auto",
        enable_cache=True,
        cache_dir=tmp_path / "cache",
    )

    document = service.parse_document(Path(sample))

    assert len(document.sections) >= min_sections
    assert len(document.chunks) >= 1
    assert document.diagnostics


def test_auto_parser_extracts_xlsx_table(tmp_path: Path) -> None:
    service = ParsingService.with_chunking(
        backend="auto",
        enable_cache=True,
        cache_dir=tmp_path / "cache",
    )

    document = service.parse_document(Path("samples/online/controlled_table_sample.xlsx"))

    assert len(document.tables) == 1
    assert document.tables[0].columns[:2] == ["Document Type", "Parser"]


def test_auto_parser_chinese_ocr_sample(tmp_path: Path) -> None:
    service = ParsingService.with_chunking(
        backend="auto",
        enable_cache=True,
        cache_dir=tmp_path / "cache",
    )

    document = service.parse_document(Path("samples/online/controlled_chinese_ocr_text_large.png"))

    assert "文档" in document.sections[0].text
    assert "missing_chinese_ocr_language" not in {item.code for item in document.diagnostics}
