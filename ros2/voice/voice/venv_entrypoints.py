from __future__ import annotations

import os
import sys
from pathlib import Path


def full_python() -> Path:
    configured = os.environ.get("RAG_SYSTEM_PYTHON")
    path = (
        Path(configured).expanduser()
        if configured
        else Path.home() / "Voice" / ".venv-linux" / "bin" / "python"
    )
    # A venv's python binary is normally a symlink. Resolving it would bypass
    # pyvenv.cfg and silently start the system interpreter instead.
    path = Path(os.path.abspath(path))
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(
            f"Full Linux environment is missing: {path}. "
            "Run scripts/install_all_linux.sh first."
        )
    return path


def _exec_module(module: str) -> None:
    python = full_python()
    statement = (
        "import sys; "
        "print(f'[voice-runtime] executable={sys.executable} prefix={sys.prefix}', "
        "flush=True); "
        f"from {module} import main; main()"
    )
    os.execv(str(python), [str(python), "-c", statement, *sys.argv[1:]])


def bridge_main() -> None:
    _exec_module("voice.bridge_node")


def arm_main() -> None:
    _exec_module("voice.arm_motion_node")


def answer_main() -> None:
    _exec_module("voice.answer_node")


def knowledge_main() -> None:
    _exec_module("voice.knowledge_node")


def display_main() -> None:
    _exec_module("voice.display_node")


__all__ = [
    "answer_main",
    "arm_main",
    "bridge_main",
    "full_python",
    "knowledge_main",
    "display_main",
]
