from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_system.display_state import DisplayStateStore


def test_display_state_defaults_to_emotion_and_can_switch_modes(tmp_path: Path) -> None:
    store = DisplayStateStore(tmp_path / "runtime" / "display.json")

    assert store.snapshot()["mode"] == "emotion"
    assert store.snapshot()["state"] == "idle"

    thinking = store.set_emotion("thinking", text="检索中")
    assert thinking["state"] == "thinking"
    assert thinking["text"] == "检索中"

    web = store.set_mode("web", reason="ssh maintenance")
    assert web["mode"] == "web"
    assert web["reason"] == "ssh maintenance"


def test_display_state_rejects_unknown_values(tmp_path: Path) -> None:
    store = DisplayStateStore(tmp_path / "display.json")

    with pytest.raises(ValueError):
        store.set_mode("browser")
    with pytest.raises(ValueError):
        store.set_emotion("unknown")


def test_display_state_recovers_from_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "display.json"
    path.write_text("not json", encoding="utf-8")
    store = DisplayStateStore(path)

    assert store.snapshot()["mode"] == "emotion"
    store.set_mode("web")
    assert json.loads(path.read_text(encoding="utf-8"))["mode"] == "web"
