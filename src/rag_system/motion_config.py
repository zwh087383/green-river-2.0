"""Validate the active vendor motion config without executing SDK code."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

MOTION_KEYS = {1: "A", 4: "Y", 5: "LB"}


def active_motion_config(sdk: Path) -> Path:
    path = sdk / "install/product/config/arm_motion.json"
    if not path.is_file():
        raise ValueError(f"Active product motion config missing: {path}; select the correct product in the vendor installation")
    return path


def load_motion_targets(sdk: Path) -> dict[int, list[dict[str, list[float]]]]:
    config = json.loads(active_motion_config(sdk).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Vendor motion config must be an object")
    result = {}
    for code, key in MOTION_KEYS.items():
        sequence = config.get(key)
        if not isinstance(sequence, list) or not sequence or len(sequence) > 256:
            raise ValueError(f"Invalid vendor motion sequence: {key}")
        steps = []
        for step in sequence:
            if not isinstance(step, dict):
                raise ValueError(f"Invalid vendor motion step: {key}")
            targets = {}
            for side in ("left_arm", "right_arm"):
                if side not in step:
                    continue
                pose = step[side]
                if (not isinstance(pose, list) or len(pose) != 7 or
                        any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                            not math.isfinite(v) for v in pose)):
                    raise ValueError(f"Invalid SDK joint positions: {key}/{side}")
                targets[side] = [math.radians(v) for v in pose]
            if targets:
                steps.append(targets)
        if not steps:
            raise ValueError(f"No arm targets in vendor sequence: {key}")
        if code == 1 and {side for step in steps for side in step} != {"left_arm", "right_arm"}:
            raise ValueError("Preset must define both arms")
        result[code] = steps
    return result


def serialize_motion_targets(targets: dict[int, list[dict[str, list[float]]]]) -> str:
    """Small numeric protocol, parsed by the native worker without a JSON library."""
    words = []
    for code, steps in targets.items():
        words.extend((str(code), str(len(steps))))
        for step in steps:
            mask = int("left_arm" in step) | (int("right_arm" in step) << 1)
            words.append(str(mask))
            for side in ("left_arm", "right_arm"):
                if side in step:
                    words.extend(format(value, ".17g") for value in step[side])
    return " ".join(words)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
