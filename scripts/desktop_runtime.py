#!/usr/bin/env python3
"""Desktop start/stop controls for the robot's single managed runtime."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

from robot_runtime import ancestors, counts, select_processes, snapshot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ("supervisor", "voice_launcher", "web", "embedding", "voice_gateway",
            "display", "knowledge", "motion_supervisor", "arm_sdk_host", "kiosk")


def project_counts(root: Path) -> dict[str, int]:
    processes = snapshot()
    selected = select_processes(processes, (root, Path.home() / "Voice"),
                                ancestors(os.getpid(), processes))
    return counts(selected, (root, Path.home() / "Voice"))


def complete(summary: dict[str, int]) -> bool:
    return all(summary.get(kind) == 1 for kind in EXPECTED) and all(
        kind in EXPECTED and count == 1
        for kind, count in summary.items() if kind != "browser_child")


def healthy() -> bool:
    try:
        with urlopen("http://127.0.0.1:8001/api/health", timeout=2) as response:
            return json.load(response).get("ok") is True
    except (OSError, ValueError):
        return False


def start(root: Path) -> None:
    if complete(project_counts(root)) and healthy():
        print("项目已经运行，各组件只有一份，无需重复启动。", flush=True)
        print("控制页：http://127.0.0.1:8001/", flush=True)
        return
    runtime = root / "data/runtime"
    log_path = runtime / "desktop-runtime.log"
    env = dict(os.environ, RAG_DATA_DIR=str(root / "data"), RAG_SYSTEM_PORT="8001",
               RAG_SYSTEM_HOST="127.0.0.1", RAG_ROBOT_WAKE_SOURCE="hardware")
    env.pop("GREENRIVER_INSTANCE_ID", None)
    print("正在统一启动项目；SDK 会先给双臂上电并归位，请保持活动区域畅通。", flush=True)
    with log_path.open("ab", buffering=0) as log:
        supervisor = subprocess.Popen(
            ["/bin/bash", str(root / "scripts/run_ipc_voice.sh")], cwd=root, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if supervisor.poll() is not None:
            raise RuntimeError(f"启动失败，请查看日志：{log_path}")
        manifest = runtime / "runtime-instance.json"
        try:
            current = json.loads(manifest.read_text(encoding="utf-8"))
            ready = current.get("supervisor_pid") == supervisor.pid
        except (OSError, ValueError):
            ready = False
        if ready and complete(project_counts(root)) and healthy():
            print("全部项目进程已启动，各组件只有一份。", flush=True)
            print("控制页：http://127.0.0.1:8001/", flush=True)
            print(f"日志：{log_path}", flush=True)
            return
        time.sleep(.5)
    raise RuntimeError(f"启动尚未完成，请查看日志：{log_path}")


def stop(root: Path) -> None:
    print("正在停止全部项目进程……", flush=True)
    subprocess.run([
        "/usr/bin/python3", str(root / "scripts/robot_runtime.py"), "stop",
        "--project-root", str(root), "--legacy-root", str(Path.home() / "Voice"),
    ], cwd=root, check=True)
    if project_counts(root):
        raise RuntimeError("仍有项目进程残留，请检查进程状态。")
    print("全部项目进程已停止。", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="小绿机器人项目桌面控制")
    parser.add_argument("action", choices=("start", "stop"))
    args = parser.parse_args()
    runtime = PROJECT_ROOT / "data/runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    result = 0
    try:
        with (runtime / "desktop-action.lock").open("a") as lock:
            print("正在准备操作（另一操作进行中时会等待）……", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            (start if args.action == "start" else stop)(PROJECT_ROOT)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"操作失败：{exc}", file=sys.stderr, flush=True)
        result = 1
    if sys.stdin.isatty():
        try:
            input("\n按回车关闭窗口……")
        except EOFError:
            pass
    return result


if __name__ == "__main__":
    raise SystemExit(main())
