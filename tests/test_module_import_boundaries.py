from __future__ import annotations

import json
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("package", "forbidden"),
    [
        ("rag_system.asr", "rag_system.voice.service"),
        ("rag_system.tts", "rag_system.voice.service"),
        ("rag_system.wakeup", "rag_system.voice.service"),
        ("rag_system.vad", "rag_system.voice.service"),
        ("rag_system.rag_cloud", "rag_system.indexing.service"),
        ("rag_system.rag_edge", "rag_system.indexing.service"),
        ("rag_system.agent", "rag_system.generation.service"),
    ],
)
def test_importing_one_public_package_does_not_construct_its_legacy_service(
    package: str,
    forbidden: str,
) -> None:
    code = (
        "import importlib,json,sys;"
        f"importlib.import_module({package!r});"
        f"print(json.dumps({{'loaded': {forbidden!r} in sys.modules}}))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["loaded"] is False
