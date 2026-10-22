"""Shared keyword scoring adapted from Project N.E.K.O. (Apache-2.0).

Copyright 2025-2026 Project N.E.K.O. Team
Copyright (c) 2025 Hongzhi Wen
Source: main_routers/system_router/emotion.py, revision
30c6d7556d0a5f04ce30dd0b2aaec75e39f387de. See licenses/neko/.
Changes: shared configurable vocabulary, eight outward tones, bounded overlap
scores and Chinese/English clause-local negation for the Green River guide.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

RULES: dict[str, Any] = json.loads(
    (Path(__file__).parent / "web/assets/emotion_rules.json").read_text(encoding="utf-8")
)
_CLAUSE = re.compile(r"[，。！？；,.!?;\n：:]")
_DEGREE = r"(?:真的|特别|非常|有点|很|太|那么|这么|感到|觉得|让人|令人|值得|需要|会|能|\s)*"


def _negated(text: str, start: int, end: int) -> bool:
    """Keep negation inside its clause/contrast; fixed affirmative phrases win."""
    head = _CLAUSE.split(text[max(0, start - 24):start])[-1]
    for contrast in RULES["contrasts"]:
        head = head.rsplit(contrast, 1)[-1]
    for phrase in RULES["negation_exceptions"]:
        head = head.replace(phrase, "")
    markers = "|".join(re.escape(word) for word in RULES["negations"])
    if re.search(r"(?:" + markers + r")" + _DEGREE + r"$", head):
        return True
    return any(text[end:].startswith(word) for word in RULES["negative_suffixes"])


def score_emotion(text: str, *, interaction: bool = False) -> tuple[str, int]:
    """Count non-overlapping evidence per tone, cap repeated filler evidence."""
    text = re.sub(r"\[[^\]]*\]", "", str(text or "")).lower()[:1200]
    groups = RULES["interactions"] if interaction else RULES["tones"]
    scores: dict[str, int] = {}
    for tone, cues in groups.items():
        hits: list[tuple[int, int, int]] = []
        for cue, weight in cues.items():
            pattern = re.escape(cue)
            if cue.isascii() and cue.replace(" ", "").isalpha():
                pattern = r"(?<![a-zA-Z])" + pattern + r"(?![a-zA-Z])"
            for match in re.finditer(pattern, text):
                if not _negated(text, match.start(), match.end()):
                    hits.append((match.start(), match.end(), weight))
        # A long cue and its component word are one piece of evidence.
        occupied: list[tuple[int, int]] = []
        score = 0
        for start, end, weight in sorted(hits, key=lambda hit: (-hit[2], hit[0], -(hit[1] - hit[0]))):
            if any(start < old_end and end > old_start for old_start, old_end in occupied):
                continue
            occupied.append((start, end))
            score += weight
        scores[tone] = min(score, 6)
    best = max(scores, key=scores.get) if scores else "neutral"
    return (best, scores[best]) if scores.get(best, 0) else ("neutral", 0)
