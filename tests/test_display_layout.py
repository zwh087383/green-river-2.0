"""Keep the shared eye anchors and safe bounds in sync with shipped PNGs."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ASSET_DIR = Path(__file__).resolve().parents[1] / "src" / "rag_system" / "web" / "assets" / "emotions"


def test_display_layout_contains_all_pixels_and_anchors_idle_eyes() -> None:
    manifest = json.loads((ASSET_DIR / "manifest.json").read_text(encoding="utf-8"))
    width = manifest["canvas"]["width"]
    height = manifest["canvas"]["height"]
    half_width = width // 2
    layout = manifest["layout"]
    eye_bounds = {
        state: [[width, height, -1, -1] for _ in range(2)]
        for state in ("idle", "thinking")
    }

    for state, config in manifest["states"].items():
        for filename in config["frames"]:
            with Image.open(ASSET_DIR / filename) as source:
                image = source.convert("RGB")
            assert image.size == (width, height)
            for side in range(2):
                source_x = side * half_width
                visible = image.crop((source_x, 0, source_x + half_width, height)).getbbox()
                if visible is None:
                    continue
                left, top, right, bottom = layout["content_bounds"][side]
                assert left <= source_x + visible[0]
                assert top <= visible[1]
                assert source_x + visible[2] - 1 <= right
                assert visible[3] - 1 <= bottom
                if state in eye_bounds:
                    state_bounds = eye_bounds[state][side]
                    state_bounds[0] = min(state_bounds[0], source_x + visible[0])
                    state_bounds[1] = min(state_bounds[1], visible[1])
                    state_bounds[2] = max(state_bounds[2], source_x + visible[2] - 1)
                    state_bounds[3] = max(state_bounds[3], visible[3] - 1)

    for state, bounds in eye_bounds.items():
        anchors = manifest["states"].get(state, {}).get("eye_anchors", layout["eye_anchors"])
        for side, (left, top, right, bottom) in enumerate(bounds):
            assert anchors[side] == [(left + right) // 2, (top + bottom) // 2]
