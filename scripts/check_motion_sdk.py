#!/usr/bin/env python3
"""Compile/link capability probes only. Never load the SDK or connect a device."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_system.motion_config import active_motion_config, file_sha256, load_motion_targets

REQUIRED = {
    "connectDev": "c.connectDev()", "disconnectDev": "c.disconnectDev()",
    "isArmMoving": "c.isArmMoving(C::Left)", "getArmPosition": "c.getArmPosition(C::Left, p)",
}
OPTIONAL = {
    "getPowerState": "c.getPowerState(C::Left)", "isArmConnected": "c.isArmConnected(C::Left)",
    "isArmPowerOn": "c.isArmPowerOn(C::Left)",
    "isArmEmergencyStopped": "c.isArmEmergencyStopped(C::Left)",
    "getJointsVelocity": "c.getJointsVelocity(C::Left, p)",
}


def inventory(sdk: Path) -> dict:
    files = {name: sdk / "install" / relative for name, relative in {
        "arm_control_header": "include/arm_control.hpp", "arm_motion_header": "include/arm_motion.hpp",
        "library": "lib/libarm_control.so", "motion_config": "product/config/arm_motion.json",
    }.items()}
    for name,relative in {
        "trunk_library":"lib/libtrunk_control.so", "harmonic_library":"lib/libeu_harmonic.so",
        "joint_model":"share/robot_description/urdf/wheel_robot.urdf",
    }.items():
        extra=sdk/"install"/relative
        if extra.is_file(): files[name]=extra
    teaching=sdk/"teach_script/teach_arm_config_init_pose.yaml"
    if teaching.is_file():files["trunk_home_config"]=teaching
    catalog=Path(__file__).resolve().parents[1]/"config/motion_presets.json"
    if catalog.is_file():files["motion_catalog"]=catalog
    for path in files.values():
        if not path.is_file():
            raise ValueError(f"SDK file missing: {path}")
    load_motion_targets(sdk)
    return {"schema_version": 1, "sdk_root": str(sdk.resolve()),
            "files": {key: {"path": str(path.resolve()), "sha256": file_sha256(path)}
                      for key, path in files.items()},
            "product_path": str((sdk / "install/product").resolve()),
            "ros_distro": os.environ.get("ROS_DISTRO", "unknown")}


def include_flags(sdk: Path) -> list[str]:
    roots = [sdk / "install"]
    roots.extend(Path(p) for p in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep) if p)
    flags = []
    for root in roots:
        folder = root / "include"
        if folder.is_dir():
            flags += ["-I", str(folder)]
            for child in sorted(folder.iterdir()):
                if child.is_dir():
                    flags += ["-I", str(child)]
    return flags


def probe(sdk: Path, report: dict, *, link: bool = True) -> dict:
    compiler = shlex.split(os.environ.get("CXX", "g++"))
    flags = ["-std=c++17", *include_flags(sdk)]
    prefix = '#include <string>\n#include "arm_motion.hpp"\nusing C = wheeled_robot::ArmControl;\n'
    capabilities = {}
    for name, expr in {**REQUIRED, **OPTIONAL}.items():
        source = prefix + f"void probe(C& c, std::array<double,7>& p) {{ (void)({expr}); }}\n"
        result = subprocess.run([*compiler, *flags, "-fsyntax-only", "-x", "c++", "-"],
                                input=source, capture_output=True, text=True, timeout=30)
        capabilities[name] = result.returncode == 0
        if name in REQUIRED and result.returncode:
            raise ValueError(f"Required SDK capability failed: {name}\n{result.stderr[-3000:]}")
    if not capabilities["getPowerState"] and not all(capabilities[n] for n in
            ("isArmConnected", "isArmPowerOn", "isArmEmergencyStopped")):
        raise ValueError("SDK needs getPowerState or all three boolean connection/power/emergency feedback methods")
    # Even optional methods compiled into the adapter must exist in the library.
    symbols = subprocess.run(["nm", "-DC", "--defined-only", str(sdk / "install/lib/libarm_control.so")],
                             capture_output=True, text=True, check=True, timeout=30).stdout
    for name, present in capabilities.items():
        if present and f"wheeled_robot::ArmControl::{name}(" not in symbols:
            raise ValueError(f"SDK header/library mismatch: missing exported {name}")
    for name in ("initialize", "update"):
        if f"wheeled_robot::ArmMotion::{name}(" not in symbols:
            raise ValueError(f"SDK library lacks ArmMotion::{name}")
    if link:
        with tempfile.TemporaryDirectory(prefix="greenriver-sdk-probe-") as temp:
            source = Path(temp) / "probe.cpp"
            body = ";\n".join(f"(void)({expr})" for name, expr in {**REQUIRED, **OPTIONAL}.items()
                                if capabilities[name])
            source.write_text(prefix + "void probe(C& c, wheeled_robot::ArmMotion& m, std::array<double,7>& p) {\n" +
                              body + "; m.initialize(&c); m.update(wheeled_robot::ArmMotion::PresetPosition); }\n",
                              encoding="utf-8")
            result = subprocess.run([*compiler, *flags, "-fPIC", "-shared", str(source),
                                     str(sdk / "install/lib/libarm_control.so"), "-Wl,-z,defs", f"-Wl,-rpath-link,{sdk / 'install/lib'}",
                                     "-o", str(Path(temp) / "probe.so")],
                                    capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise ValueError(f"SDK link probe failed (check ROS environment and matching libraries):\n{result.stderr[-4000:]}")
    report.update(capabilities=capabilities, power_source="sdk_enum" if capabilities["getPowerState"] else "sdk_boolean",
                  hardware_timestamp_available=False, link_verified=link)
    return report


def verify_manifest(sdk: Path, path: Path) -> dict:
    saved = json.loads(path.read_text(encoding="utf-8"))
    current = inventory(sdk)
    if saved.get("schema_version") != 1 or not saved.get("link_verified") or saved.get("ros_distro") != "humble":
        raise ValueError("SDK contract was not validated by a Humble compile/link build")
    for key, value in current["files"].items():
        if saved.get("files", {}).get(key, {}).get("sha256") != value["sha256"]:
            raise ValueError(f"SDK contract changed: {key}; inspect the deployment and rebuild before takeover")
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ws", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-manifest", type=Path)
    parser.add_argument("--inventory-only", action="store_true", help="Read hashes/config only; not a build authorization")
    args = parser.parse_args()
    try:
        if args.verify_manifest:
            report = verify_manifest(args.robot_ws, args.verify_manifest)
        else:
            report = inventory(args.robot_ws)
            if not args.inventory_only:
                if report["ros_distro"] != "humble":
                    raise ValueError(f"Motion plugin requires ROS Humble; active ROS_DISTRO={report['ros_distro']}")
                report = probe(args.robot_ws, report)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Motion SDK preflight failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
