"""Validated, file-backed catalog shared by the UI and native SDK worker."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from typing import Any

PRESET_ID = re.compile(r"[a-z][a-z0-9_]{0,39}\Z")


def catalog_path() -> Path:
    return Path(os.environ.get("GREENRIVER_MOTION_PRESETS", str(
        Path(__file__).resolve().parents[2] / "config/motion_presets.json")))


def number(value: Any, low: float, high: float) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    raw = (path or catalog_path()).read_bytes()
    if len(raw) > 262144:
        raise ValueError("动作目录过大")
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("动作目录版本无效")
    presets = data.get("presets")
    if not isinstance(presets, list) or not 1 <= len(presets) <= 64:
        raise ValueError("动作数量必须在 1 到 64 之间")
    seen: set[str] = set()
    for item in presets:
        if not isinstance(item, dict):
            raise ValueError("动作配置必须是对象")
        ident = item.get("id", "")
        if not isinstance(ident, str) or not PRESET_ID.fullmatch(ident) or ident in seen:
            raise ValueError("动作 ID 无效或重复")
        seen.add(ident)
        if not isinstance(item.get("name"), str) or not 1 <= len(item["name"]) <= 60:
            raise ValueError("动作名称无效")
        if not number(item.get("timeout_s", 30), 5, 120):
            raise ValueError("动作超时必须在 5 到 120 秒之间")
        if item.get("type") == "builtin":
            code = item.get("sdk_code")
            if type(code) is not int or not 0 <= code <= 6:
                raise ValueError("SDK 动作编号无效")
        elif item.get("type") == "sequence":
            frames = item.get("frames")
            if not isinstance(frames, list) or not 2 <= len(frames) <= 32:
                raise ValueError("动作需要 2 到 32 个关键帧")
            for frame in frames:
                if not isinstance(frame, dict) or not number(frame.get("scale"), .05, 1):
                    raise ValueError("关键帧速度倍率必须在 0.05 到 1 之间")
                if type(frame.get("wait_ms", 0)) is not int or not 0 <= frame.get("wait_ms", 0) <= 5000:
                    raise ValueError("关键帧停留时间必须在 0 到 5000 毫秒之间")
                if frame.get("home") is True:
                    if "left_arm" in frame or "right_arm" in frame:
                        raise ValueError("归位帧不能包含额外关节目标")
                else:
                    for side in ("left_arm", "right_arm"):
                        angles = frame.get(side)
                        if (not isinstance(angles, list) or len(angles) != 7
                                or not all(number(v, -180, 180) for v in angles)):
                            raise ValueError("每只手臂需要 7 个有效关节角，单位为度")
            if frames[-1].get("home") is not True:
                raise ValueError("测试动作必须以归位帧结束")
        else:
            raise ValueError("动作类型无效")
    return {"version": 1, "revision": hashlib.sha256(raw).hexdigest(), "presets": presets}


def validate_vendor_limits(preset: dict[str, Any], sdk: Path | None = None) -> None:
    if preset["type"] != "sequence":
        return
    sdk = sdk or Path(os.environ.get("ROBOT_WS", str(Path.home() / "wheeled_robot_pkg")))
    model = ET.parse(sdk / 'install/share/robot_description/urdf/wheel_robot.urdf')
    for frame in preset["frames"]:
        if frame.get("home"):
            continue
        for key, side in (("left_arm", "L"), ("right_arm", "R")):
            for index, angle in enumerate(frame[key], 1):
                limit = model.find(f".//joint[@name='AR5_{side}joint_{index}']/limit")
                if limit is None:
                    raise ValueError("厂家关节限位缺失")
                radians = math.radians(angle)
                if not float(limit.attrib["lower"]) <= radians <= float(limit.attrib["upper"]):
                    raise ValueError(f"{key} 第 {index} 关节超出厂家范围")
