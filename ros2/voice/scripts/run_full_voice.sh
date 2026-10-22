#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Work both from the source tree and the installed ROS share directory.
PROJECT_ROOT="$SCRIPT_DIR"
while [[ ! -f "$PROJECT_ROOT/scripts/run_ipc_voice.sh" ]]; do
    if [[ "$PROJECT_ROOT" == / ]]; then
        echo "找不到 4.0 统一启动器" >&2
        exit 1
    fi
    PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
done
FULL_PYTHON="${RAG_SYSTEM_PYTHON:-${PROJECT_ROOT}/.venv-linux/bin/python}"
export RAG_SYSTEM_PYTHON="${FULL_PYTHON}"
export RAG_DATA_DIR="${RAG_DATA_DIR:-${PROJECT_ROOT}/data}"
exec bash "${PROJECT_ROOT}/scripts/run_ipc_voice.sh" "$@"
